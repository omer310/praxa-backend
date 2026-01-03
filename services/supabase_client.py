"""Supabase database client for all database operations."""

import os
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
import logging

from supabase import create_client, Client

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Client for interacting with Supabase database."""

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        
        self.client: Client = create_client(url, key)

    # ==================== User & Settings ====================

    async def get_user_with_settings(self, user_id: str) -> Optional[dict]:
        """
        Fetch user settings by user_id.
        """
        try:
            # Query by the 'user_id' column (the FK to the user)
            settings_response = self.client.table("user_settings").select("*").eq("user_id", user_id).execute()
            
            if not settings_response.data or len(settings_response.data) == 0:
                logger.warning(f"User settings not found for: {user_id}")
                return None
            
            settings = settings_response.data[0]  # Get first (and only) result
            
            # Create a synthetic user object from settings for compatibility
            return {
                "user": {
                    "id": settings.get("id"),
                    "email": settings.get("email", ""),
                    "name": settings.get("name", ""),
                },
                "settings": settings
            }
        except Exception as e:
            logger.error(f"Error fetching user settings: {e}")
            return None  # Return None instead of raising, so endpoint can handle gracefully

    async def get_users_due_for_call(self) -> list[dict]:
        """
        Get all users who are due for a scheduled call.
        
        Returns:
            List of scheduled calls with user settings
        """
        try:
            # Use timezone-aware datetime for PostgREST comparison
            from datetime import timezone
            now = datetime.now(timezone.utc).isoformat()
            
            # Query scheduled_calls only (no JOIN needed)
            response = self.client.table("scheduled_calls").select(
                "*"
            ).eq("status", "pending").lte("scheduled_for", now).execute()
            
            # For each scheduled call, fetch the user settings separately
            scheduled_calls = response.data or []
            for call in scheduled_calls:
                if call.get("user_id"):
                    try:
                        settings_response = self.client.table("user_settings").select(
                            "*"
                        ).eq("user_id", call["user_id"]).execute()
                        if settings_response.data:
                            call["user_settings"] = settings_response.data[0]
                    except Exception as e:
                        logger.warning(f"Could not fetch user_settings for scheduled call {call['id']}: {e}")
                        call["user_settings"] = {}
            
            return scheduled_calls
        except Exception as e:
            logger.error(f"Error fetching users due for call: {e}")
            raise

    # ==================== Buckets & Tasks ====================

    async def get_user_buckets_with_loops(self, user_id: str) -> list[dict]:
        """
        Fetch all buckets for a user with their associated tasks (loops).
        
        Args:
            user_id: The UUID of the user
            
        Returns:
            List of buckets with nested loops
        """
        try:
            response = self.client.table("buckets").select(
                "*, loops(*)"
            ).eq("user_id", user_id).eq("archived", False).execute()
            
            return response.data or []
        except Exception as e:
            logger.error(f"Error fetching buckets with loops: {e}")
            raise

    async def get_this_week_tasks(self, user_id: str) -> list[dict]:
        """
        Get tasks marked for this week's focus.
        
        Args:
            user_id: The UUID of the user
            
        Returns:
            List of tasks marked is_this_week=True that aren't done
        """
        try:
            response = self.client.table("loops").select(
                "*, buckets(name, color)"
            ).eq("user_id", user_id).eq("is_this_week", True).neq("status", "done").execute()
            
            # Flatten bucket info
            tasks = []
            for task in response.data or []:
                if task.get("buckets"):
                    task["bucket_name"] = task["buckets"]["name"]
                    task["bucket_color"] = task["buckets"]["color"]
                    del task["buckets"]
                tasks.append(task)
            
            return tasks
        except Exception as e:
            logger.error(f"Error fetching this week's tasks: {e}")
            raise

    async def get_overdue_tasks(self, user_id: str) -> list[dict]:
        """
        Get overdue tasks for a user.
        
        Args:
            user_id: The UUID of the user
            
        Returns:
            List of tasks past their due date that aren't done
        """
        try:
            now = datetime.utcnow().isoformat()
            
            response = self.client.table("loops").select(
                "*, buckets(name, color)"
            ).eq("user_id", user_id).neq("status", "done").lt("due_date", now).execute()
            
            # Flatten bucket info
            tasks = []
            for task in response.data or []:
                if task.get("buckets"):
                    task["bucket_name"] = task["buckets"]["name"]
                    task["bucket_color"] = task["buckets"]["color"]
                    del task["buckets"]
                tasks.append(task)
            
            return tasks
        except Exception as e:
            logger.error(f"Error fetching overdue tasks: {e}")
            raise

    async def get_recently_completed_tasks(self, user_id: str, days: int = 7) -> list[dict]:
        """
        Get recently completed tasks.
        
        Args:
            user_id: The UUID of the user
            days: Number of days to look back
            
        Returns:
            List of recently completed tasks
        """
        try:
            since = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            response = self.client.table("loops").select(
                "*, buckets(name, color)"
            ).eq("user_id", user_id).eq("status", "done").gte("updated_at", since).execute()
            
            # Flatten bucket info
            tasks = []
            for task in response.data or []:
                if task.get("buckets"):
                    task["bucket_name"] = task["buckets"]["name"]
                    task["bucket_color"] = task["buckets"]["color"]
                    del task["buckets"]
                tasks.append(task)
            
            return tasks
        except Exception as e:
            logger.error(f"Error fetching recently completed tasks: {e}")
            raise

    # ==================== Task Updates ====================

    async def mark_task_complete(self, loop_id: str) -> dict:
        """
        Mark a task as complete.
        
        Args:
            loop_id: The UUID of the task/loop
            
        Returns:
            Updated task data
        """
        try:
            response = self.client.table("loops").update({
                "status": "done",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", loop_id).execute()
            
            logger.info(f"Marked task {loop_id} as complete")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error marking task complete: {e}")
            raise

    async def add_task_note(self, loop_id: str, note: str) -> dict:
        """
        Add or append a note to a task.
        
        Args:
            loop_id: The UUID of the task/loop
            note: The note text to add
            
        Returns:
            Updated task data
        """
        try:
            # First get existing note
            existing = self.client.table("loops").select("notes").eq("id", loop_id).single().execute()
            
            existing_notes = existing.data.get("notes", "") if existing.data else ""
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            
            if existing_notes:
                new_notes = f"{existing_notes}\n\n[{timestamp} - Praxa Call] {note}"
            else:
                new_notes = f"[{timestamp} - Praxa Call] {note}"
            
            response = self.client.table("loops").update({
                "notes": new_notes,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", loop_id).execute()
            
            logger.info(f"Added note to task {loop_id}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error adding note to task: {e}")
            raise

    async def create_task(
        self,
        user_id: str,
        bucket_id: str,
        title: str,
        description: Optional[str] = None,
        priority: str = "medium",
        due_date: Optional[str] = None,
        is_this_week: bool = False
    ) -> dict:
        """
        Create a new task.
        
        Args:
            user_id: The UUID of the user
            bucket_id: The UUID of the bucket to add the task to
            title: Task title
            description: Optional task description
            priority: Task priority (low, medium, high)
            due_date: Optional due date in ISO format
            is_this_week: Whether to mark for this week's focus
            
        Returns:
            Created task data
        """
        try:
            task_data = {
                "user_id": user_id,
                "bucket_id": bucket_id,
                "title": title,
                "status": "open",
                "priority": priority,
                "is_this_week": is_this_week,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if description:
                task_data["description"] = description
            if due_date:
                task_data["due_date"] = due_date
            
            response = self.client.table("loops").insert(task_data).execute()
            
            logger.info(f"Created new task: {title}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            raise

    async def update_task_due_date(self, loop_id: str, due_date: str) -> dict:
        """
        Update a task's due date.
        
        Args:
            loop_id: The UUID of the task/loop
            due_date: New due date in ISO format
            
        Returns:
            Updated task data
        """
        try:
            response = self.client.table("loops").update({
                "due_date": due_date,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", loop_id).execute()
            
            logger.info(f"Updated due date for task {loop_id}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error updating task due date: {e}")
            raise

    async def update_task_status(self, loop_id: str, status: str) -> dict:
        """
        Update a task's status.
        
        Args:
            loop_id: The UUID of the task/loop
            status: New status (open, in_progress, done)
            
        Returns:
            Updated task data
        """
        try:
            response = self.client.table("loops").update({
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", loop_id).execute()
            
            logger.info(f"Updated status for task {loop_id} to {status}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            raise

    # ==================== Call Management ====================

    async def create_call_log(
        self,
        user_id: str,
        phone_number: str,
        livekit_room_name: str,
        scheduled_at: Optional[str] = None
    ) -> dict:
        """
        Create a new call log entry.
        
        Args:
            user_id: The UUID of the user
            phone_number: The phone number being called
            livekit_room_name: The LiveKit room name for this call
            scheduled_at: When the call was scheduled for
            
        Returns:
            Created call log data
        """
        try:
            call_data = {
                "user_id": user_id,
                "phone_number": phone_number,
                "livekit_room_name": livekit_room_name,
                "status": "initiated",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if scheduled_at:
                call_data["scheduled_at"] = scheduled_at
            
            response = self.client.table("call_logs").insert(call_data).execute()
            
            logger.info(f"Created call log for user {user_id}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error creating call log: {e}")
            raise

    async def update_call_log(self, call_log_id: str, updates: dict) -> dict:
        """
        Update a call log entry.
        
        Args:
            call_log_id: The UUID of the call log
            updates: Dictionary of fields to update
            
        Returns:
            Updated call log data
        """
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            response = self.client.table("call_logs").update(updates).eq("id", call_log_id).execute()
            
            logger.info(f"Updated call log {call_log_id}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error updating call log: {e}")
            raise

    async def get_call_log_by_room(self, room_name: str) -> Optional[dict]:
        """
        Get call log by LiveKit room name.
        
        Args:
            room_name: The LiveKit room name
            
        Returns:
            Call log data or None
        """
        try:
            response = self.client.table("call_logs").select("*").eq("livekit_room_name", room_name).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching call log by room: {e}")
            return None

    async def get_call_log_by_sid(self, call_sid: str) -> Optional[dict]:
        """
        Get call log by Twilio call SID.
        
        Args:
            call_sid: The Twilio call SID
            
        Returns:
            Call log data or None
        """
        try:
            response = self.client.table("call_logs").select("*").eq("call_sid", call_sid).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching call log by SID: {e}")
            return None

    async def create_all_scheduled_calls(
        self,
        user_id: str,
        checkin_schedule: list,
        timezone: str,
        checkin_enabled: bool = True
    ) -> list[dict]:
        """
        Create all upcoming scheduled calls based on the checkin_schedule.
        This creates calls for the next occurrence of each day in the schedule.
        
        Args:
            user_id: The UUID of the user
            checkin_schedule: List of schedule entries [{"day": 1, "time": "09:40", "label": "Monday"}, ...]
            timezone: The user's timezone
            checkin_enabled: Whether checkins are enabled
            
        Returns:
            List of created scheduled calls
        """
        if not checkin_enabled or not checkin_schedule:
            logger.info(f"Calls disabled or no schedule for user {user_id}")
            return []
        
        try:
            from zoneinfo import ZoneInfo
            
            # Get current time in user's timezone
            user_tz = ZoneInfo(timezone)
            now_local = datetime.now(user_tz)
            
            created_calls = []
            
            # Create a scheduled call for each entry
            for schedule_entry in checkin_schedule:
                day = schedule_entry.get("day")
                time_str = schedule_entry.get("time")
                label = schedule_entry.get("label", f"Day {day}")
                
                if day is None or not time_str:
                    logger.warning(f"Invalid schedule entry: {schedule_entry}")
                    continue
                
                # Parse time
                try:
                    hour, minute = map(int, time_str.split(":"))
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid time format: {time_str}")
                    continue
                
                # Calculate next occurrence
                current_weekday = now_local.weekday()
                if day == 0:  # Sunday
                    target_weekday = 6
                else:
                    target_weekday = day - 1
                
                days_ahead = (target_weekday - current_weekday) % 7
                next_call_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
                next_call_local += timedelta(days=days_ahead)
                
                if next_call_local <= now_local:
                    next_call_local += timedelta(days=7)
                
                # Convert to UTC
                next_call_utc = next_call_local.astimezone(ZoneInfo("UTC"))
                
                # Determine time window
                if hour < 12:
                    time_window = "morning"
                elif hour < 17:
                    time_window = "afternoon"
                else:
                    time_window = "evening"
                
                # Create scheduled call
                scheduled_call_data = {
                    "user_id": user_id,
                    "scheduled_for": next_call_utc.isoformat(),
                    "time_window": time_window,
                    "status": "pending",
                    "attempt_count": 0,
                    "max_attempts": 3,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                response = self.client.table("scheduled_calls").insert(scheduled_call_data).execute()
                if response.data:
                    created_calls.append(response.data[0])
                    logger.info(f"Created scheduled call for {label} at {time_str} -> {next_call_utc.isoformat()} UTC")
            
            # Update user_settings with the earliest next scheduled call
            if created_calls:
                earliest = min(created_calls, key=lambda x: x["scheduled_for"])
                try:
                    self.client.table("user_settings").update({
                        "next_scheduled_call": earliest["scheduled_for"],
                        "updated_at": datetime.utcnow().isoformat()
                    }).eq("user_id", user_id).execute()
                except Exception as e:
                    logger.warning(f"Could not update user_settings.next_scheduled_call: {e}")
            
            return created_calls
        except Exception as e:
            logger.error(f"Error creating scheduled calls: {e}", exc_info=True)
            raise

    # ==================== Scheduled Calls ====================

    async def get_pending_scheduled_calls(self) -> list[dict]:
        """
        Get all pending scheduled calls that are due.
        
        Returns:
            List of scheduled calls that are ready to be processed
        """
        try:
            # Use timezone-aware datetime for PostgREST comparison
            from datetime import timezone
            now = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Querying scheduled_calls with now={now}")
            
            # Query scheduled_calls only (no JOIN needed)
            response = self.client.table("scheduled_calls").select(
                "*"
            ).eq("status", "pending").lte("scheduled_for", now).order("scheduled_for").execute()
            
            # For each scheduled call, fetch the user settings separately
            scheduled_calls = response.data or []
            logger.info(f"Found {len(scheduled_calls)} pending scheduled calls")
            for call in scheduled_calls:
                if call.get("user_id"):
                    try:
                        settings_response = self.client.table("user_settings").select(
                            "*"
                        ).eq("user_id", call["user_id"]).execute()
                        if settings_response.data:
                            call["user_settings"] = settings_response.data[0]
                    except Exception as e:
                        logger.warning(f"Could not fetch user_settings for scheduled call {call['id']}: {e}")
                        call["user_settings"] = {}
            
            return scheduled_calls
        except Exception as e:
            logger.error(f"Error fetching pending scheduled calls: {e}")
            raise

    async def update_scheduled_call(self, scheduled_call_id: str, updates: dict) -> dict:
        """
        Update a scheduled call entry.
        
        Args:
            scheduled_call_id: The UUID of the scheduled call
            updates: Dictionary of fields to update
            
        Returns:
            Updated scheduled call data
        """
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            response = self.client.table("scheduled_calls").update(updates).eq("id", scheduled_call_id).execute()
            
            logger.info(f"Updated scheduled call {scheduled_call_id}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error updating scheduled call: {e}")
            raise

    async def mark_scheduled_call_complete(self, scheduled_call_id: str, call_log_id: str) -> dict:
        """
        Mark a scheduled call as completed.
        
        Args:
            scheduled_call_id: The UUID of the scheduled call
            call_log_id: The UUID of the associated call log
            
        Returns:
            Updated scheduled call data
        """
        return await self.update_scheduled_call(scheduled_call_id, {
            "status": "completed",
            "call_log_id": call_log_id
        })

    async def schedule_next_call(
        self,
        user_id: str,
        checkin_schedule: list,
        timezone: str,
        checkin_enabled: bool = True
    ) -> Optional[dict]:
        """
        Schedule the next call for a user based on their checkin_schedule.
        
        Args:
            user_id: The UUID of the user
            checkin_schedule: List of schedule entries [{"day": 1, "time": "09:40", "label": "Monday"}, ...]
                            where day is 0-6 (Sunday=0, Monday=1, ..., Saturday=6)
            timezone: The user's timezone (e.g., "America/New_York")
            checkin_enabled: Whether checkins are enabled
            
        Returns:
            Created scheduled call data, or None if calls are disabled
        """
        if not checkin_enabled or not checkin_schedule:
            logger.info(f"Calls disabled or no schedule for user {user_id}")
            return None
        
        try:
            from zoneinfo import ZoneInfo
            from datetime import datetime as dt
            
            # Get current time in user's timezone
            user_tz = ZoneInfo(timezone)
            now_local = datetime.now(user_tz)
            
            # Find the next scheduled time
            next_call_local = None
            closest_schedule = None
            
            # Check each schedule entry and find the next occurrence
            for schedule_entry in checkin_schedule:
                day = schedule_entry.get("day")  # 0=Sunday, 1=Monday, etc.
                time_str = schedule_entry.get("time")  # "HH:MM" format
                
                if day is None or not time_str:
                    logger.warning(f"Invalid schedule entry: {schedule_entry}")
                    continue
                
                # Parse time
                try:
                    hour, minute = map(int, time_str.split(":"))
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid time format: {time_str}")
                    continue
                
                # Calculate days until this day of week
                current_weekday = now_local.weekday()
                # Convert to same format (0=Sunday in schedule, but weekday() uses 0=Monday)
                # schedule: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
                # weekday(): 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
                
                # Convert schedule day to weekday format
                if day == 0:  # Sunday
                    target_weekday = 6
                else:
                    target_weekday = day - 1
                
                # Calculate days ahead
                days_ahead = (target_weekday - current_weekday) % 7
                
                # Create the candidate datetime
                candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
                candidate += timedelta(days=days_ahead)
                
                # If this time has already passed today, add 7 days
                if candidate <= now_local:
                    candidate += timedelta(days=7)
                
                # Check if this is the earliest next occurrence
                if next_call_local is None or candidate < next_call_local:
                    next_call_local = candidate
                    closest_schedule = schedule_entry
            
            if next_call_local is None:
                logger.error(f"Could not calculate next call time for user {user_id}")
                return None
            
            # Convert to UTC for storage
            next_call_utc = next_call_local.astimezone(ZoneInfo("UTC"))
            
            # Determine time window for backward compatibility
            hour_local = next_call_local.hour
            if hour_local < 12:
                time_window = "morning"
            elif hour_local < 17:
                time_window = "afternoon"
            else:
                time_window = "evening"
            
            scheduled_call_data = {
                "user_id": user_id,
                "scheduled_for": next_call_utc.isoformat(),
                "time_window": time_window,
                "status": "pending",
                "attempt_count": 0,
                "max_attempts": 3,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("scheduled_calls").insert(scheduled_call_data).execute()
            
            # Also update user_settings with next_scheduled_call
            try:
                self.client.table("user_settings").update({
                    "next_scheduled_call": next_call_utc.isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("user_id", user_id).execute()
            except Exception as e:
                logger.warning(f"Could not update user_settings.next_scheduled_call: {e}")
            
            schedule_label = closest_schedule.get("label", "scheduled day") if closest_schedule else "scheduled day"
            logger.info(f"Scheduled next call for user {user_id} on {schedule_label} at {time_str} ({timezone}) -> {next_call_utc.isoformat()} UTC")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error scheduling next call: {e}", exc_info=True)
            raise

    # ==================== Bucket Operations ====================

    async def get_bucket_by_name(self, user_id: str, bucket_name: str) -> Optional[dict]:
        """
        Find a bucket by name (case-insensitive).
        
        Args:
            user_id: The UUID of the user
            bucket_name: The name of the bucket to find
            
        Returns:
            Bucket data or None
        """
        try:
            response = self.client.table("buckets").select("*").eq(
                "user_id", user_id
            ).eq("archived", False).ilike("name", bucket_name).execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error finding bucket by name: {e}")
            return None

    async def get_user_bucket_names(self, user_id: str) -> list[str]:
        """
        Get list of bucket names for a user.
        
        Args:
            user_id: The UUID of the user
            
        Returns:
            List of bucket names
        """
        try:
            response = self.client.table("buckets").select("name").eq(
                "user_id", user_id
            ).eq("archived", False).execute()
            
            return [b["name"] for b in response.data] if response.data else []
        except Exception as e:
            logger.error(f"Error fetching bucket names: {e}")
            return []


# Singleton instance
_client: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get or create the Supabase client singleton."""
    global _client
    if _client is None:
        _client = SupabaseClient()
    return _client

