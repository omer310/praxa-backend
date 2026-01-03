"""Background scheduler for triggering scheduled calls."""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .supabase_client import get_supabase_client

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

    def start(self):
        """Start the background scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        # Add the job to run every 5 minutes
        self.scheduler.add_job(
            self.check_and_trigger_calls,
            trigger=IntervalTrigger(minutes=5),
            id="check_scheduled_calls",
            name="Check and trigger scheduled calls",
            replace_existing=True
        )
        
        self.scheduler.start()
        self._running = True
        logger.info("Call scheduler started (checking every 5 minutes)")

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

