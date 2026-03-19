"""Background scheduler for triggering scheduled calls."""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .supabase_client import get_supabase_client
from .push_service import send_push_notification, get_user_push_token, schedule_receipt_check

logger = logging.getLogger(__name__)


class CallScheduler:
    """
    Background scheduler that checks for and triggers scheduled calls.
    
    Runs every 5 minutes to check the scheduled_calls table for pending calls
    that are due, then triggers the agent to make each call.
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.trigger_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._running = False

    def set_trigger_callback(self, callback: Callable[[str], Awaitable[None]]):
        """
        Set the callback function to trigger a call.
        
        Args:
            callback: Async function that takes a user_id and initiates a call
        """
        self.trigger_callback = callback

    async def check_and_trigger_calls(self):
        """
        Check for pending scheduled calls and trigger them.
        
        This runs every 5 minutes and:
        1. Queries scheduled_calls where scheduled_for <= now AND status = 'pending'
        2. For each, updates status to 'processing'
        3. Triggers the agent to make the call
        4. Handles failures (increment attempt_count, reschedule if < max_attempts)
        """
        if not self.trigger_callback:
            logger.warning("No trigger callback set, skipping scheduled call check")
            return

        try:
            db = get_supabase_client()
            pending_calls = await db.get_pending_scheduled_calls()
            
            if not pending_calls:
                logger.debug("No pending scheduled calls")
                return
            
            logger.info(f"Found {len(pending_calls)} pending scheduled calls")
            
            for scheduled_call in pending_calls:
                await self._process_scheduled_call(scheduled_call)
                
        except Exception as e:
            logger.error(f"Error checking scheduled calls: {e}")

    async def _process_scheduled_call(self, scheduled_call: dict):
        """
        Process a single scheduled call.
        
        Args:
            scheduled_call: The scheduled call record with user info
        """
        call_id = scheduled_call["id"]
        user_id = scheduled_call["user_id"]
        attempt_count = scheduled_call.get("attempt_count", 0)
        max_attempts = scheduled_call.get("max_attempts", 3)
        
        db = get_supabase_client()
        
        try:
            # Mark as processing
            await db.update_scheduled_call(call_id, {
                "status": "processing",
                "last_attempt_at": datetime.utcnow().isoformat(),
                "attempt_count": attempt_count + 1
            })
            
            # Check if user still has calls enabled
            user_settings = scheduled_call.get("user_settings", {})
            if not user_settings.get("calls_enabled", True):
                logger.info(f"Calls disabled for user {user_id}, skipping")
                await db.update_scheduled_call(call_id, {"status": "skipped"})
                return
            
            # Check if user has a verified phone number
            phone_number = user_settings.get("phone_number")
            if not phone_number or not user_settings.get("phone_verified", False):
                logger.warning(f"No verified phone for user {user_id}, skipping")
                await db.update_scheduled_call(call_id, {"status": "skipped"})
                return
            
            # Only send push on the first attempt — not on retries
            if attempt_count == 0:
                push_token = await get_user_push_token(user_id)
                if push_token:
                    ticket_id = await send_push_notification(
                        push_token=push_token,
                        title="Check-in Call Coming Up",
                        body="Your Praxa check-in is starting now. Get ready!",
                        data={"notificationType": "call_reminder"},
                    )
                    if ticket_id:
                        schedule_receipt_check(ticket_id, user_id)

            # Trigger the call
            logger.info(f"Triggering call for user {user_id}")
            result = await self.trigger_callback(user_id)
            
            # Schedule the next call for this user based on their checkin_schedule
            if result:
                try:
                    await self._schedule_next_for_user(user_id, user_settings)
                except Exception as e:
                    logger.error(f"Failed to schedule next call for user {user_id}: {e}")
            
            # Note: The call log completion will be handled by the agent
            # The scheduled_call will be marked complete when the call ends
            
        except Exception as e:
            logger.error(f"Error processing scheduled call {call_id}: {e}")
            
            # Handle failure
            if attempt_count + 1 >= max_attempts:
                # Max attempts reached, mark as failed
                await db.update_scheduled_call(call_id, {
                    "status": "failed"
                })
                logger.warning(f"Scheduled call {call_id} failed after {max_attempts} attempts")
            else:
                # Reset to pending for retry
                await db.update_scheduled_call(call_id, {
                    "status": "pending"
                })
                logger.info(f"Scheduled call {call_id} will retry (attempt {attempt_count + 1}/{max_attempts})")

    async def _schedule_next_for_user(self, user_id: str, user_settings: dict):
        """
        Schedule the next call for a user after completing the current one.
        
        Args:
            user_id: The UUID of the user
            user_settings: The user's settings including checkin_schedule
        """
        db = get_supabase_client()
        
        try:
            checkin_schedule = user_settings.get("checkin_schedule", [])
            checkin_enabled = user_settings.get("checkin_enabled", True)
            timezone = user_settings.get("timezone", "America/New_York")
            
            if not checkin_enabled or not checkin_schedule:
                logger.info(f"Checkins disabled or no schedule for user {user_id}")
                return
            
            # Schedule the next call
            await db.schedule_next_call(
                user_id=user_id,
                checkin_schedule=checkin_schedule,
                timezone=timezone,
                checkin_enabled=checkin_enabled
            )
            logger.info(f"Successfully scheduled next call for user {user_id}")
        except Exception as e:
            logger.error(f"Error scheduling next call for user {user_id}: {e}")

    async def _run_task_notifications(self):
        """
        Hourly job that sends daily task-related push notifications to users.

        For each user with a push token, checks if it is currently 9–10am in their
        local timezone. If so — and if no notification has been sent for them today —
        it sends up to three summarised pushes:
          1. Tasks due today  (task_due_soon)
          2. Overdue tasks    (task_overdue)
          3. Sprint ending today with open tasks  (sprint_deadline)

        Uses an in-memory set to avoid duplicate sends within the same process run.
        """
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        db = get_supabase_client()
        users = await db.get_all_users_with_push_tokens()
        if not users:
            return

        logger.info(f"[TaskNotifications] Checking {len(users)} users for daily task nudges")

        for user in users:
            user_id = user.get("user_id")
            push_token = user.get("push_token")
            timezone_str = user.get("timezone") or "UTC"

            if not user_id or not push_token:
                continue

            try:
                user_tz = ZoneInfo(timezone_str)
            except (ZoneInfoNotFoundError, Exception):
                user_tz = ZoneInfo("UTC")

            now_local = datetime.now(user_tz)
            today_key = f"{user_id}:{now_local.date().isoformat()}"

            # Only send once per day, in the 9–10am window
            if now_local.hour != 9:
                continue
            if today_key in self._notified_today:
                continue

            self._notified_today.add(today_key)

            try:
                # 1 — Tasks due today
                due_today = await db.get_tasks_due_today(user_id, timezone_str)
                if due_today:
                    count = len(due_today)
                    body = f"You have {count} task{'s' if count != 1 else ''} due today"
                    ticket_id = await send_push_notification(
                        push_token=push_token,
                        title="Tasks due today",
                        body=body,
                        data={"type": "task_due_soon"},
                    )
                    if ticket_id:
                        schedule_receipt_check(ticket_id, user_id)

                # 2 — Overdue tasks
                overdue = await db.get_overdue_tasks(user_id)
                if overdue:
                    count = len(overdue)
                    body = f"You have {count} overdue task{'s' if count != 1 else ''}"
                    ticket_id = await send_push_notification(
                        push_token=push_token,
                        title="Overdue tasks",
                        body=body,
                        data={"type": "task_overdue"},
                    )
                    if ticket_id:
                        schedule_receipt_check(ticket_id, user_id)

                # 3 — Sprint ending today
                sprint_cadence = user.get("sprint_cadence") or "weekly"
                last_reset_str = user.get("last_sprint_reset_at")
                sprint_ends_today = self._is_sprint_end_today(
                    now_local, sprint_cadence, last_reset_str
                )
                if sprint_ends_today:
                    sprint_tasks = await db.get_this_week_tasks(user_id)
                    if sprint_tasks:
                        count = len(sprint_tasks)
                        body = f"{count} task{'s' if count != 1 else ''} still open in this sprint"
                        ticket_id = await send_push_notification(
                            push_token=push_token,
                            title="Sprint ending today",
                            body=body,
                            data={"type": "sprint_deadline"},
                        )
                        if ticket_id:
                            schedule_receipt_check(ticket_id, user_id)

                logger.info(f"[TaskNotifications] Sent daily nudges for user {user_id}")
            except Exception as e:
                logger.error(f"[TaskNotifications] Error processing user {user_id}: {e}")

    def _is_sprint_end_today(
        self, now_local: datetime, cadence: str, last_reset_str: Optional[str]
    ) -> bool:
        """Return True if today is the last day of the user's current sprint."""
        try:
            weekday = now_local.weekday()  # 0=Mon … 6=Sun
            if cadence == "weekly":
                return weekday == 6  # Sunday = last day of week
            if cadence == "daily":
                return True  # every day is the last day of a daily sprint
            if cadence == "monthly":
                import calendar
                last_day = calendar.monthrange(now_local.year, now_local.month)[1]
                return now_local.day == last_day
        except Exception:
            pass
        return False

    def start(self):
        """Start the background scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._notified_today: set[str] = set()
        
        # Add the job to run every 5 minutes
        self.scheduler.add_job(
            self.check_and_trigger_calls,
            trigger=IntervalTrigger(minutes=5),
            id="check_scheduled_calls",
            name="Check and trigger scheduled calls",
            replace_existing=True
        )

        # Add daily memory consolidation at 3am UTC
        from apscheduler.triggers.cron import CronTrigger
        self.scheduler.add_job(
            self._run_memory_consolidation,
            trigger=CronTrigger(hour=3, minute=0),
            id="consolidate_memories",
            name="Nightly memory consolidation",
            replace_existing=True,
        )

        # Hourly job that sends task push notifications at 9am local time per user
        self.scheduler.add_job(
            self._run_task_notifications,
            trigger=IntervalTrigger(hours=1),
            id="task_notifications",
            name="Daily task push notifications",
            replace_existing=True,
        )
        
        self.scheduler.start()
        self._running = True
        logger.info("Call scheduler started (checking every 5 minutes)")

    async def _run_memory_consolidation(self):
        """Run nightly memory consolidation for all users."""
        try:
            from services.memory_service import consolidate_all_users_memories
            logger.info("[Scheduler] Starting nightly memory consolidation")
            await consolidate_all_users_memories()
            logger.info("[Scheduler] Nightly memory consolidation complete")
        except Exception as e:
            logger.error(f"[Scheduler] Memory consolidation failed: {e}", exc_info=True)

    def stop(self):
        """Stop the background scheduler."""
        if not self._running:
            return
        
        self.scheduler.shutdown(wait=False)
        self._running = False
        logger.info("Call scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return self._running


# Singleton instance
_scheduler: Optional[CallScheduler] = None


def get_call_scheduler() -> CallScheduler:
    """Get or create the call scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = CallScheduler()
    return _scheduler

