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
        Fetch user settings directly by ID.
        For MVP without auth, we query user_settings by id (not user_id).
        """
        try:
            # For MVP: Query by the 'id' column directly (since user_id is NULL)
            settings_response = self.client.table("user_settings").select("*").eq("id", user_id).single().execute()
            
            if not settings_response.data:
                logger.warning(f"User settings not found for: {user_id}")
                return None
            
            settings = settings_response.data
            
            # Create a synthetic user object from settings for compatibility
            return {
                "user": {
                    "id": settings.get("id"),
                    "email": "",
                    "name": "",
                },
                "settings": settings
            }
        except Exception as e:
            logger.error(f"Error fetching user settings: {e}")
            raise

    async def get_users_due_for_call(self) -> list[dict]:
        """
        Get all users who are due for a scheduled call.
        
        Returns:
            List of scheduled calls with user settings
        """
        try:
            now = datetime.utcnow().isoformat()
            
            # Query scheduled_calls that are due and pending
            # Join with user_settings only (no users table for MVP)
            response = self.client.table("scheduled_calls").select(
                "*, user_settings!inner(*)"
            ).eq("status", "pending").lte("scheduled_for", now).execute()
            
            return response.data or []
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

    # ==================== Scheduled Calls ====================

    async def get_pending_scheduled_calls(self) -> list[dict]:
        """
        Get all pending scheduled calls that are due.
        
        Returns:
            List of scheduled calls that are ready to be processed
        """
        try:
            now = datetime.utcnow().isoformat()
            
            # Query without users table - just scheduled_calls and user_settings
            response = self.client.table("scheduled_calls").select(
                "*, user_settings!inner(*)"
            ).eq("status", "pending").lte("scheduled_for", now).order("scheduled_for").execute()
            
            return response.data or []
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
        frequency: str,
        timezone: str,
        time_window: str = "afternoon"
    ) -> Optional[dict]:
        """
        Schedule the next call for a user based on their frequency preference.
        
        Args:
            user_id: The UUID of the user
            frequency: The checkin frequency (once_per_week, twice_per_week, off)
            timezone: The user's timezone
            time_window: Preferred time window (morning, afternoon, evening)
            
        Returns:
            Created scheduled call data, or None if calls are disabled
        """
        if frequency == "off":
            logger.info(f"Calls disabled for user {user_id}")
            return None
        
        try:
            # Calculate next call time based on frequency
            now = datetime.utcnow()
            
            if frequency == "once_per_week":
                # Schedule for 7 days from now
                next_call = now + timedelta(days=7)
            elif frequency == "twice_per_week":
                # Schedule for 3-4 days from now
                next_call = now + timedelta(days=3)
            else:
                # Default to weekly
                next_call = now + timedelta(days=7)
            
            # Adjust time based on time window (rough approximation in UTC)
            # In production, you'd want to properly handle timezone conversion
            if time_window == "morning":
                next_call = next_call.replace(hour=14, minute=0, second=0)  # ~9am EST
            elif time_window == "afternoon":
                next_call = next_call.replace(hour=18, minute=0, second=0)  # ~1pm EST
            else:  # evening
                next_call = next_call.replace(hour=22, minute=0, second=0)  # ~5pm EST
            
            scheduled_call_data = {
                "user_id": user_id,
                "scheduled_for": next_call.isoformat(),
                "time_window": time_window,
                "status": "pending",
                "attempt_count": 0,
                "max_attempts": 3,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("scheduled_calls").insert(scheduled_call_data).execute()
            
            # Also update user_settings with next_scheduled_call
            # Try updating by user_id first, then by id if that fails
            try:
                self.client.table("user_settings").update({
                    "next_scheduled_call": next_call.isoformat(),
                    "last_call_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("user_id", user_id).execute()
            except Exception:
                # Fallback to updating by id
                self.client.table("user_settings").update({
                    "next_scheduled_call": next_call.isoformat(),
                    "last_call_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", user_id).execute()
            
            logger.info(f"Scheduled next call for user {user_id} at {next_call}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error scheduling next call: {e}")
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

