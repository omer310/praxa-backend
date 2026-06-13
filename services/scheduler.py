"""Background scheduler for triggering scheduled calls."""

import os
import asyncio
import logging
from datetime import datetime, timezone
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

        Flow:
          1. Mark as processing, increment attempt_count.
          2. Validate user settings (calls enabled, phone verified).
          3. Trigger the call via LiveKit/Twilio.
             - Success: record stays in 'processing'; the Twilio status webhook
               will advance it to next week (completed) or reset to pending (missed).
             - Failure (exception or None result): retry up to max_attempts.
               After exhausting attempts, advance the record to next week.
        """
        call_id = scheduled_call["id"]
        user_id = scheduled_call["user_id"]
        attempt_count = scheduled_call.get("attempt_count", 0)
        max_attempts = scheduled_call.get("max_attempts", 3)

        db = get_supabase_client()

        try:
            await db.update_scheduled_call(call_id, {
                "status": "processing",
                "last_attempt_at": datetime.now(timezone.utc).isoformat(),
                "attempt_count": attempt_count + 1,
            })

            user_settings = scheduled_call.get("user_settings", {})
            if not user_settings.get("calls_enabled", True):
                logger.info(f"Calls disabled for user {user_id}, skipping")
                await db.update_scheduled_call(call_id, {"status": "skipped"})
                return

            phone_number = user_settings.get("phone_number")
            if not phone_number or not user_settings.get("phone_verified", False):
                logger.warning(f"No verified phone for user {user_id}, skipping")
                await db.update_scheduled_call(call_id, {"status": "skipped"})
                return

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

            logger.info(f"Triggering call for user {user_id} (attempt {attempt_count + 1}/{max_attempts})")
            result = await self.trigger_callback(user_id)

            if not result:
                raise Exception(f"Call initiation returned no result for user {user_id}")

            logger.info(f"Call initiated for user {user_id} — scheduled_call {call_id} waiting for Twilio outcome")

        except Exception as e:
            logger.error(f"Error initiating scheduled call {call_id}: {e}")

            current_attempt = attempt_count + 1
            if current_attempt >= max_attempts:
                logger.warning(f"Scheduled call {call_id} exhausted {max_attempts} attempts, advancing to next week")
                try:
                    await db.advance_scheduled_call(call_id)
                except Exception as advance_err:
                    logger.error(f"Failed to advance scheduled call {call_id} after max attempts: {advance_err}")
                    await db.update_scheduled_call(call_id, {"status": "failed"})
            else:
                await db.update_scheduled_call(call_id, {"status": "pending"})
                logger.info(f"Scheduled call {call_id} will retry (attempt {current_attempt}/{max_attempts})")

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

        # Action dispatcher safety-net poll: picks up actions queued by any
        # surface (incl. the out-of-process voice agent) and confirmed actions.
        self.scheduler.add_job(
            self._run_action_dispatcher,
            trigger=IntervalTrigger(seconds=30),
            id="dispatch_pending_actions",
            name="Dispatch pending integration actions",
            replace_existing=True,
        )

        # Integration context enrichment: fetch content + embeddings for newly
        # synced Notion rows (n8n does the lightweight metadata pass).
        self.scheduler.add_job(
            self._run_integration_ingest,
            trigger=IntervalTrigger(minutes=10),
            id="enrich_integration_context",
            name="Enrich integration context with embeddings",
            replace_existing=True,
        )

        from apscheduler.triggers.cron import CronTrigger

        # Proactive daily briefing: runs hourly, fires per-user at their local
        # BRIEFING_HOUR. Kill-switch PROACTIVE_BRIEFING_ENABLED (default on).
        self.scheduler.add_job(
            self._run_daily_briefing,
            trigger=CronTrigger(minute=10),
            id="daily_briefing",
            name="Proactive daily briefing",
            replace_existing=True,
        )

        # Proactive initiative loop: scans for attention emails without a draft
        # and queues confirm-only reply drafts. Kill-switch PROACTIVE_LOOP_ENABLED.
        self.scheduler.add_job(
            self._run_initiative_loop,
            trigger=IntervalTrigger(minutes=20),
            id="initiative_loop",
            name="Proactive initiative loop",
            replace_existing=True,
        )

        # Relationship linker: refresh VIP + link contacts to Notion/Slack daily.
        self.scheduler.add_job(
            self._run_relationship_linker,
            trigger=CronTrigger(hour=4, minute=20),
            id="relationship_linker",
            name="Relationship graph linker",
            replace_existing=True,
        )

        # Follow-up detector: nudge users about unanswered threads and due tasks.
        self.scheduler.add_job(
            self._run_follow_up_detector,
            trigger=CronTrigger(hour=4, minute=30),
            id="follow_up_detector",
            name="Daily follow-up detector",
            replace_existing=True,
        )

        # Add daily memory consolidation at 3am UTC
        self.scheduler.add_job(
            self._run_memory_consolidation,
            trigger=CronTrigger(hour=3, minute=0),
            id="consolidate_memories",
            name="Nightly memory consolidation",
            replace_existing=True,
        )

        # Add nightly skill proposals at 3:30am UTC (after memory consolidation)
        self.scheduler.add_job(
            self._run_skill_proposals,
            trigger=CronTrigger(hour=3, minute=30),
            id="propose_skills",
            name="Nightly agent skills proposal",
            replace_existing=True,
        )

        # Add nightly skill consolidation at 3:45am UTC (after proposals)
        self.scheduler.add_job(
            self._run_skill_consolidation,
            trigger=CronTrigger(hour=3, minute=45),
            id="consolidate_skills",
            name="Nightly agent skills consolidation",
            replace_existing=True,
        )

        # Background reasoning agent: LLM pass over world state every 30 min.
        self.scheduler.add_job(
            self._run_background_agent,
            trigger=IntervalTrigger(minutes=30),
            id="background_reasoning_agent",
            name="Background reasoning agent",
            replace_existing=True,
        )

        # Autonomy learner: daily job to propose user_autonomy_rules from
        # approval patterns recorded in action_approval_log.
        self.scheduler.add_job(
            self._run_autonomy_learner,
            trigger=CronTrigger(hour=5, minute=0),
            id="autonomy_learner",
            name="Autonomy pattern learner",
            replace_existing=True,
        )

        # World state safety-net refresh: rebuild snapshots for all users.
        self.scheduler.add_job(
            self._run_world_state_refresh,
            trigger=IntervalTrigger(minutes=30),
            id="world_state_refresh",
            name="World state snapshot refresh",
            replace_existing=True,
        )

        # Daily task nudges are handled by the Supabase Edge Function
        # send-daily-notifications (learned digest hour, multi-channel, dedupe).
        # _run_task_notifications is kept for reference but not scheduled here.

        self.scheduler.start()
        self._running = True
        logger.info("Call scheduler started (checking every 5 minutes)")

    async def _run_action_dispatcher(self):
        """Poll the integration_actions queue and dispatch any due actions."""
        try:
            from services.action_dispatcher import run_due_actions, notify_pending_approvals
            await notify_pending_approvals()
            await run_due_actions()
        except Exception as e:
            logger.error(f"[Scheduler] Action dispatcher poll failed: {e}", exc_info=True)

    async def _run_integration_ingest(self):
        """Enrich newly synced integration_context rows with content + embeddings."""
        try:
            from services.integration_ingest import enrich_integration_context
            await enrich_integration_context()
        except Exception as e:
            logger.error(f"[Scheduler] Integration ingest failed: {e}", exc_info=True)

    async def _run_daily_briefing(self):
        """Generate per-user daily briefs at their local BRIEFING_HOUR."""
        try:
            from services.briefing import run_daily_briefing
            await run_daily_briefing()
        except Exception as e:
            logger.error(f"[Scheduler] Daily briefing failed: {e}", exc_info=True)

    async def _run_initiative_loop(self):
        """Scan attention emails and queue confirm-only reply drafts."""
        try:
            from services.initiative_loop import run_initiative_loop
            await run_initiative_loop()
        except Exception as e:
            logger.error(f"[Scheduler] Initiative loop failed: {e}", exc_info=True)

    async def _run_relationship_linker(self):
        """Refresh VIP status and link contacts to Notion/Slack identities."""
        try:
            from services.relationship_linker import run_relationship_linker
            await run_relationship_linker()
        except Exception as e:
            logger.error(f"[Scheduler] Relationship linker failed: {e}", exc_info=True)

    async def _run_follow_up_detector(self):
        """Run daily follow-up detection: unanswered threads + due-soon tasks."""
        try:
            from services.follow_up_detector import run_follow_up_detector
            await run_follow_up_detector()
        except Exception as e:
            logger.error(f"[Scheduler] Follow-up detector failed: {e}", exc_info=True)

    async def _run_memory_consolidation(self):
        """Run nightly memory consolidation for all users."""
        try:
            from services.memory_service import consolidate_all_users_memories
            logger.info("[Scheduler] Starting nightly memory consolidation")
            await consolidate_all_users_memories()
            logger.info("[Scheduler] Nightly memory consolidation complete")
        except Exception as e:
            logger.error(f"[Scheduler] Memory consolidation failed: {e}", exc_info=True)

    async def _run_skill_proposals(self):
        """Run nightly agent skill proposals for all users."""
        try:
            from services.skill_service import propose_skills_for_all_users
            logger.info("[Scheduler] Starting nightly skill proposals")
            await propose_skills_for_all_users()
            logger.info("[Scheduler] Nightly skill proposals complete")
        except Exception as e:
            logger.error(f"[Scheduler] Skill proposals failed: {e}", exc_info=True)

    async def _run_skill_consolidation(self):
        """Run nightly agent skill consolidation/deduplication for all users."""
        try:
            from services.skill_service import consolidate_skills_for_all_users
            logger.info("[Scheduler] Starting nightly skill consolidation")
            await consolidate_skills_for_all_users()
            logger.info("[Scheduler] Nightly skill consolidation complete")
        except Exception as e:
            logger.error(f"[Scheduler] Skill consolidation failed: {e}", exc_info=True)

    async def _run_background_agent(self):
        """Run a proactive reasoning pass for all active users."""
        try:
            from services.background_agent import run_reasoning_pass_all_users
            await run_reasoning_pass_all_users()
        except Exception as e:
            logger.error(f"[Scheduler] Background agent failed: {e}", exc_info=True)

    async def _run_autonomy_learner(self):
        """Daily scan of action_approval_log to propose user_autonomy_rules."""
        try:
            from services.autonomy_learner import learn_autonomy_patterns_all_users
            logger.info("[Scheduler] Starting autonomy pattern learning")
            await learn_autonomy_patterns_all_users()
            logger.info("[Scheduler] Autonomy learning complete")
        except Exception as e:
            logger.error(f"[Scheduler] Autonomy learner failed: {e}", exc_info=True)

    async def _run_world_state_refresh(self):
        """Refresh world state snapshots for all users (safety-net pass)."""
        try:
            from services.supabase_client import get_supabase_client
            from services.world_state import refresh_world_state
            db = get_supabase_client()
            resp = await asyncio.to_thread(
                lambda: db.client.table("users").select("id").eq("ai_enabled", True).execute()
            )
            user_ids = [r["id"] for r in (resp.data or [])]
            for uid in user_ids:
                try:
                    await refresh_world_state(uid)
                except Exception:
                    pass
            logger.debug(f"[Scheduler] World state refreshed for {len(user_ids)} users")
        except Exception as e:
            logger.error(f"[Scheduler] World state refresh failed: {e}", exc_info=True)

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

