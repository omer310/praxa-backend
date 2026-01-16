"""LiveKit Agent for Praxa voice AI conversations."""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID
import json

# Reduce logging noise from third-party libraries to avoid Railway rate limits
logging.getLogger("livekit").setLevel(logging.ERROR)
logging.getLogger("livekit.agents").setLevel(logging.ERROR)
logging.getLogger("livekit.rtc").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("deepgram").setLevel(logging.ERROR)
logging.getLogger("hpack").setLevel(logging.ERROR)
logging.getLogger("hpack.hpack").setLevel(logging.ERROR)
logging.getLogger("hpack.table").setLevel(logging.ERROR)
logging.getLogger("h2").setLevel(logging.ERROR)
logging.getLogger("h11").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.ERROR)
logging.getLogger("aiohttp").setLevel(logging.ERROR)

# Silence ALL debug logging except our own
for name in logging.root.manager.loggerDict:
    if not name.startswith("__main__") and not name.startswith("services") and not name.startswith("agent"):
        logging.getLogger(name).setLevel(logging.ERROR)

# Ensure the parent directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livekit import rtc, api as livekit_api
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    function_tool,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

# LiveKit SIP configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
LIVEKIT_SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")

from services.supabase_client import get_supabase_client
from agent.prompts import SYSTEM_PROMPT, get_user_context_prompt, get_opening_message, get_closing_message

# For calendar access
import httpx

logger = logging.getLogger(__name__)

# Nylas API configuration
NYLAS_API_KEY = os.getenv("NYLAS_API_KEY", "")


class PraxaAgent:
    """
    The Praxa voice AI agent that handles phone conversations.
    
    This agent:
    1. Connects to a LiveKit room where the phone call audio is routed
    2. Uses Deepgram for speech-to-text
    3. Uses OpenAI for conversation intelligence
    4. Uses ElevenLabs for text-to-speech
    5. Can take actions like marking tasks complete, adding notes, creating tasks
    """

    def __init__(self, user_id: str, call_log_id: str, calendar_grant_id: Optional[str] = None):
        self.user_id = user_id
        self.call_log_id = call_log_id
        self.calendar_grant_id = calendar_grant_id
        self.db = get_supabase_client()
        
        # Call tracking
        self.transcript: list[dict] = []
        self.tasks_discussed: list[str] = []
        self.tasks_completed: list[str] = []
        self.tasks_created: list[str] = []
        self.goals_updated: list[dict] = []
        self.call_started_at: Optional[datetime] = None
        
        # User context (loaded on start)
        self.user_context: Optional[dict] = None
        self.user_settings: Optional[dict] = None
        self.buckets: list[dict] = []
        self.this_week_tasks: list[dict] = []
        self.overdue_tasks: list[dict] = []
        self.recently_completed: list[dict] = []
        
        # Calendar context (loaded if grant ID available)
        self.calendar_events: list[dict] = []
        self.calendar_busy_count: int = 0

    async def load_user_context(self):
        """Load all user data needed for the conversation."""
        try:
            # Get user and settings
            user_data = await self.db.get_user_with_settings(self.user_id)
            if not user_data:
                raise ValueError(f"User not found: {self.user_id}")
            
            self.user_context = user_data["user"]
            self.user_settings = user_data["settings"]
            
            # Get buckets with tasks
            self.buckets = await self.db.get_user_buckets_with_loops(self.user_id)
            
            # Get this week's tasks
            self.this_week_tasks = await self.db.get_this_week_tasks(self.user_id)
            
            # Get overdue tasks
            self.overdue_tasks = await self.db.get_overdue_tasks(self.user_id)
            
            # Get recently completed
            self.recently_completed = await self.db.get_recently_completed_tasks(self.user_id)
            
            # Load calendar if grant ID available
            if self.calendar_grant_id:
                try:
                    await self._load_calendar_context()
                except Exception as e:
                    logger.warning(f"Error loading calendar context: {e}")
                    # Don't fail the call if calendar fails
            
            logger.info(
                f"Loaded context for user {self.user_id}: "
                f"{len(self.buckets)} buckets, "
                f"{len(self.this_week_tasks)} this week tasks, "
                f"{len(self.overdue_tasks)} overdue, "
                f"{len(self.recently_completed)} recently completed, "
                f"{len(self.calendar_events)} calendar events"
            )
            
        except Exception as e:
            logger.error(f"Error loading user context: {e}")
            raise

    async def _load_calendar_context(self):
        """Load calendar events for the upcoming week."""
        if not NYLAS_API_KEY or not self.calendar_grant_id:
            return
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get primary calendar
                calendars_url = f"https://api.us.nylas.com/v3/grants/{self.calendar_grant_id}/calendars"
                cal_response = await client.get(
                    calendars_url,
                    headers={
                        "Authorization": f"Bearer {NYLAS_API_KEY}",
                        "Accept": "application/json"
                    }
                )
                
                if cal_response.status_code != 200:
                    logger.warning(f"Failed to fetch calendars: {cal_response.status_code}")
                    return
                
                calendars = cal_response.json().get("data", [])
                if not calendars:
                    return
                
                calendar_id = calendars[0].get("id")
                
                # Get events for next 7 days
                events_url = f"https://api.us.nylas.com/v3/grants/{self.calendar_grant_id}/events"
                events_response = await client.get(
                    events_url,
                    headers={
                        "Authorization": f"Bearer {NYLAS_API_KEY}",
                        "Accept": "application/json"
                    },
                    params={"calendar_id": calendar_id, "limit": 50}
                )
                
                if events_response.status_code == 200:
                    self.calendar_events = events_response.json().get("data", [])
                    
                    # Count how many events this week (simple metric for "busy-ness")
                    self.calendar_busy_count = len(self.calendar_events)
                    
                    logger.info(f"Loaded {len(self.calendar_events)} calendar events")
                else:
                    logger.warning(f"Failed to fetch events: {events_response.status_code}")
        
        except Exception as e:
            logger.warning(f"Error fetching calendar data: {e}")

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt with user context."""
        user_name = self.user_context.get("name") if self.user_context else None
        checkin_frequency = self.user_settings.get("checkin_frequency", "once_per_week") if self.user_settings else "once_per_week"
        
        context_prompt = get_user_context_prompt(
            user_name=user_name,
            buckets=self.buckets,
            this_week_tasks=self.this_week_tasks,
            overdue_tasks=self.overdue_tasks,
            recently_completed=self.recently_completed,
            checkin_frequency=checkin_frequency,
            calendar_events=self.calendar_events if self.calendar_grant_id else None,
            calendar_busy_count=self.calendar_busy_count
        )
        
        return f"{SYSTEM_PROMPT}\n\n--- USER CONTEXT ---\n{context_prompt}"

    def _get_opening_message(self) -> str:
        """Get the opening message for the call."""
        user_name = self.user_context.get("name") if self.user_context else None
        return get_opening_message(
            user_name=user_name,
            this_week_count=len(self.this_week_tasks),
            recently_completed_count=len(self.recently_completed)
        )

    def _find_task_by_title(self, title: str) -> Optional[dict]:
        """Find a task by title (fuzzy match)."""
        title_lower = title.lower().strip()
        
        # Check this week's tasks first
        for task in self.this_week_tasks:
            if title_lower in task["title"].lower():
                return task
        
        # Check overdue tasks
        for task in self.overdue_tasks:
            if title_lower in task["title"].lower():
                return task
        
        # Check all tasks in buckets
        for bucket in self.buckets:
            for task in bucket.get("loops", []):
                if title_lower in task["title"].lower():
                    return task
        
        return None

    async def on_call_started(self):
        """Called when the call is connected."""
        self.call_started_at = datetime.utcnow()
        
        # Update call log
        await self.db.update_call_log(self.call_log_id, {
            "status": "in_progress",
            "started_at": self.call_started_at.isoformat()
        })
        
        logger.info(f"Call started for user {self.user_id}")

    async def on_call_ended(self):
        """Called when the call ends."""
        try:
            ended_at = datetime.utcnow()
            duration = int((ended_at - self.call_started_at).total_seconds()) if self.call_started_at else 0
            
            # CRITICAL DEBUG: Log transcript status (use print to bypass Railway log rate limits)
            transcript_count = len(self.transcript)
            print(f"\n{'='*60}", flush=True)
            print(f"[TRANSCRIPT DEBUG] on_call_ended called - transcript has {transcript_count} messages", flush=True)
            logger.info(f"[TRANSCRIPT DEBUG] on_call_ended called - transcript has {transcript_count} messages")
            
            if transcript_count > 0:
                print(f"[TRANSCRIPT DEBUG] First message: {self.transcript[0]}", flush=True)
                print(f"[TRANSCRIPT DEBUG] Last message: {self.transcript[-1]}", flush=True)
                logger.info(f"[TRANSCRIPT DEBUG] First message: {self.transcript[0]}")
                logger.info(f"[TRANSCRIPT DEBUG] Last message: {self.transcript[-1]}")
            else:
                print(f"[TRANSCRIPT DEBUG] WARNING: Transcript is EMPTY!", flush=True)
                logger.warning(f"[TRANSCRIPT DEBUG] WARNING: Transcript is EMPTY!")
            print(f"{'='*60}\n", flush=True)
            
            # Generate summary using OpenAI
            summary = await self._generate_summary()
            logger.info(f"[TRANSCRIPT DEBUG] Generated summary: {summary[:100]}...")
            
            # Prepare update data
            update_data = {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": duration,
                "transcript": self.transcript,  # This is a list of dicts
                "summary": summary,
                "tasks_discussed": list(set(self.tasks_discussed)),
                "tasks_completed": list(set(self.tasks_completed)),
                "tasks_created": list(set(self.tasks_created)),
                "goals_updated": self.goals_updated
            }
            
            logger.info(f"[TRANSCRIPT DEBUG] About to update call_log {self.call_log_id} with {len(self.transcript)} transcript messages")
            
            # Update call log
            result = await self.db.update_call_log(self.call_log_id, update_data)
            
            logger.info(f"[TRANSCRIPT DEBUG] Call log updated successfully. Result: {result.get('id', 'unknown')}")
            
            # Schedule next call
            if self.user_settings:
                await self.db.schedule_next_call(
                    user_id=self.user_id,
                    checkin_schedule=self.user_settings.get("checkin_schedule", []),
                    timezone=self.user_settings.get("timezone", "America/New_York"),
                    checkin_enabled=self.user_settings.get("checkin_enabled", True)
                )
            
            logger.info(
                f"Call ended for user {self.user_id}: "
                f"{duration}s, {transcript_count} transcript messages, "
                f"{len(self.tasks_completed)} completed, "
                f"{len(self.tasks_created)} created"
            )
            
        except Exception as e:
            logger.error(f"Error handling call end: {e}", exc_info=True)

    async def _generate_summary(self) -> str:
        """Generate a summary of the call using OpenAI."""
        try:
            if not self.transcript:
                return "No conversation recorded."
            
            # Format transcript for summarization
            transcript_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.transcript
            ])
            
            summary_prompt = f"""Summarize this productivity check-in call in 2-3 sentences. 
Focus on: what was discussed, tasks completed, and any new commitments made.

Transcript:
{transcript_text}"""
            
            # Use the OpenAI API directly for summarization
            import openai as openai_api
            
            openai_client = openai_api.OpenAI()
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200
            )
            
            return response.choices[0].message.content or "Call completed."
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Call completed. {len(self.tasks_completed)} tasks marked done, {len(self.tasks_created)} tasks created."

    def on_transcript_update(self, role: str, content: str):
        """Called when there's new transcript content."""
        transcript_entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.transcript.append(transcript_entry)
        logger.info(f"[TRANSCRIPT DEBUG] Added to transcript - {role}: {content[:50]}... (total: {len(self.transcript)} messages)")

    # Tool functions for the agent
    async def mark_task_complete(self, task_title: str) -> str:
        """Mark a task as complete."""
        try:
            task = self._find_task_by_title(task_title)
            if not task:
                return f"I couldn't find a task called '{task_title}'. Could you tell me the exact name?"
            
            task_id = task["id"]
            await self.db.mark_task_complete(task_id)
            self.tasks_completed.append(task_id)
            self.tasks_discussed.append(task_id)
            
            logger.info(f"Marked task complete: {task_title}")
            return f"Done! I've marked '{task_title}' as complete."
        except Exception as e:
            logger.error(f"Error marking task complete: {e}")
            return "Sorry, I had trouble updating that task. Let me note it down and we'll fix it."

    async def add_task_note(self, task_title: str, note: str) -> str:
        """Add a note to a task."""
        try:
            task = self._find_task_by_title(task_title)
            if not task:
                return f"I couldn't find a task called '{task_title}'."
            
            task_id = task["id"]
            await self.db.add_task_note(task_id, note)
            self.tasks_discussed.append(task_id)
            
            logger.info(f"Added note to task: {task_title}")
            return f"Got it! I've added that note to '{task_title}'."
        except Exception as e:
            logger.error(f"Error adding task note: {e}")
            return "Sorry, I had trouble adding that note."

    async def create_task(self, title: str, bucket_name: str, is_this_week: bool = False) -> str:
        """Create a new task."""
        try:
            bucket = await self.db.get_bucket_by_name(self.user_id, bucket_name)
            if not bucket:
                bucket_names = await self.db.get_user_bucket_names(self.user_id)
                return f"I don't see a bucket called '{bucket_name}'. Your buckets are: {', '.join(bucket_names)}. Which one should I use?"
            
            new_task = await self.db.create_task(
                user_id=self.user_id,
                bucket_id=bucket["id"],
                title=title,
                is_this_week=is_this_week
            )
            
            if new_task:
                self.tasks_created.append(new_task["id"])
                logger.info(f"Created new task: {title} in {bucket_name}")
                week_note = " and marked it for this week" if is_this_week else ""
                return f"Done! I've added '{title}' to your {bucket_name} bucket{week_note}."
            else:
                return "Sorry, I had trouble creating that task."
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return "Sorry, something went wrong creating that task."

    async def update_task_due_date(self, task_title: str, due_date: str) -> str:
        """Update a task's due date."""
        try:
            task = self._find_task_by_title(task_title)
            if not task:
                return f"I couldn't find a task called '{task_title}'."
            
            task_id = task["id"]
            await self.db.update_task_due_date(task_id, due_date)
            self.tasks_discussed.append(task_id)
            
            logger.info(f"Updated due date for task: {task_title} to {due_date}")
            return f"Updated! '{task_title}' is now due on {due_date}."
        except Exception as e:
            logger.error(f"Error updating due date: {e}")
            return "Sorry, I had trouble updating that due date."

    async def list_buckets(self) -> str:
        """List the user's buckets."""
        try:
            bucket_names = await self.db.get_user_bucket_names(self.user_id)
            if bucket_names:
                return f"Your buckets are: {', '.join(bucket_names)}"
            else:
                return "You don't have any buckets set up yet."
        except Exception as e:
            logger.error(f"Error listing buckets: {e}")
            return "Sorry, I had trouble getting your buckets."

    async def get_calendar_overview(self) -> str:
        """Get a brief overview of the calendar for this week."""
        if not self.calendar_grant_id or not self.calendar_events:
            return "I don't have access to your calendar right now."
        
        try:
            from datetime import datetime, timedelta
            
            # Group events by day
            today = datetime.now().date()
            week_days = {}
            
            for event in self.calendar_events:
                when = event.get("when", {})
                start_time_str = when.get("start_time") or when.get("date")
                
                if not start_time_str:
                    continue
                
                try:
                    # Parse the date
                    event_date = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')).date()
                    
                    # Only include events in next 7 days
                    if event_date < today or event_date > today + timedelta(days=7):
                        continue
                    
                    day_name = event_date.strftime("%A")
                    if day_name not in week_days:
                        week_days[day_name] = 0
                    week_days[day_name] += 1
                except:
                    continue
            
            if not week_days:
                return "Your calendar looks pretty open this week."
            
            # Find busiest and lightest days
            busiest_day = max(week_days.items(), key=lambda x: x[1])
            lightest_day = min(week_days.items(), key=lambda x: x[1])
            
            total_events = sum(week_days.values())
            
            response = f"You have {total_events} events this week. "
            
            if busiest_day[1] > 3:
                response += f"{busiest_day[0]} is your busiest day with {busiest_day[1]} meetings. "
            
            if lightest_day[1] <= 2 and lightest_day[0] != busiest_day[0]:
                response += f"{lightest_day[0]} looks lighter with only {lightest_day[1]} meetings - good day for focused work."
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting calendar overview: {e}")
            return "I had trouble analyzing your calendar."

    async def get_todays_calendar(self) -> str:
        """Get today's calendar events."""
        if not self.calendar_grant_id or not self.calendar_events:
            return "I don't have access to your calendar right now."
        
        try:
            from datetime import datetime
            
            today = datetime.now().date()
            today_events = []
            
            for event in self.calendar_events:
                when = event.get("when", {})
                start_time_str = when.get("start_time") or when.get("date")
                
                if not start_time_str:
                    continue
                
                try:
                    event_date = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')).date()
                    if event_date == today:
                        title = event.get("title", "Untitled")
                        time_str = start_time_str.split("T")[1][:5] if "T" in start_time_str else ""
                        today_events.append(f"{title} at {time_str}" if time_str else title)
                except:
                    continue
            
            if not today_events:
                return "You don't have any events scheduled for today."
            
            return f"Today you have: {', '.join(today_events)}"
            
        except Exception as e:
            logger.error(f"Error getting today's calendar: {e}")
            return "I had trouble getting today's events."


# Global agent instance for the current session
_current_agent: Optional[PraxaAgent] = None


def create_praxa_agent_class(praxa: PraxaAgent):
    """Create a custom Agent class with the Praxa tools."""
    
    class PraxaVoiceAgent(Agent):
        def __init__(self):
            super().__init__(
                instructions=praxa._build_system_prompt(),
            )
        
        @function_tool
        async def mark_task_complete(self, task_title: str) -> str:
            """Mark a task as complete. Use this when the user says they finished a task.
            
            Args:
                task_title: The title of the task to mark complete
            """
            return await praxa.mark_task_complete(task_title)
        
        @function_tool
        async def add_task_note(self, task_title: str, note: str) -> str:
            """Add a note to an existing task. Use this to record updates or details about a task.
            
            Args:
                task_title: The title of the task to add a note to
                note: The note to add to the task
            """
            return await praxa.add_task_note(task_title, note)
        
        @function_tool
        async def create_task(self, title: str, bucket_name: str, is_this_week: bool = False) -> str:
            """Create a new task. Ask the user which bucket/initiative to add it to.
            
            Args:
                title: The title of the new task
                bucket_name: The name of the bucket/initiative to add the task to
                is_this_week: Whether to mark this for the current week's focus
            """
            return await praxa.create_task(title, bucket_name, is_this_week)
        
        @function_tool
        async def update_task_due_date(self, task_title: str, due_date: str) -> str:
            """Update the due date of a task. Use when user wants to reschedule.
            
            Args:
                task_title: The title of the task to update
                due_date: The new due date in YYYY-MM-DD format
            """
            return await praxa.update_task_due_date(task_title, due_date)
        
        @function_tool
        async def list_buckets(self) -> str:
            """Get the list of user's buckets/initiatives. Use when you need to know what buckets exist."""
            return await praxa.list_buckets()
        
        @function_tool
        async def get_calendar_overview(self) -> str:
            """Get an overview of the user's calendar for this week. Shows busy days and best days for focused work."""
            return await praxa.get_calendar_overview()
        
        @function_tool
        async def get_todays_calendar(self) -> str:
            """Get today's calendar events. Use when user asks about today's schedule."""
            return await praxa.get_todays_calendar()
    
    return PraxaVoiceAgent


async def entrypoint(ctx: JobContext):
    """
    Main entry point for the LiveKit agent.
    
    This is called by LiveKit when a new room is created for a call.
    The agent will:
    1. Connect to the room first
    2. Read room metadata to get phone number
    3. Dial out via SIP
    4. Wait for phone user to connect
    5. Start the conversation
    """
    global _current_agent
    
    # Use print for debugging (bypasses log rate limits)
    import sys
    print("=" * 50, flush=True)
    print("AGENT ENTRYPOINT CALLED", flush=True)
    print("=" * 50, flush=True)
    
    room_name = ctx.room.name
    print(f"Room name: {room_name}", flush=True)
    logger.info(f"Agent entrypoint called for room: {room_name}")
    
    # Connect to the room FIRST (metadata is available after connection)
    print("Connecting to room...", flush=True)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print(f"Connected to room: {room_name}", flush=True)
    logger.info(f"Agent connected to room: {room_name}")
    
    # NOW parse room metadata (available after connection)
    try:
        # Access metadata from the connected room
        room_metadata = ctx.room.metadata
        print(f"Room metadata: {room_metadata}", flush=True)
        logger.info(f"Room metadata: {room_metadata}")
        
        metadata = json.loads(room_metadata) if room_metadata else {}
        user_id = metadata.get("user_id")
        call_log_id = metadata.get("call_log_id")
        phone_number = metadata.get("phone_number")
        calendar_grant_id = metadata.get("calendar_grant_id")
        
        # Fallback to parsing room name if metadata is missing
        if not user_id:
            parts = room_name.split("-")
            if len(parts) >= 4 and parts[0] == "praxa" and parts[1] == "call":
                user_id = parts[2]
                call_log_id = parts[3] if len(parts) > 3 else None
        
        if not user_id:
            logger.error(f"Could not get user_id from room: {room_name}")
            return
            
        if not phone_number:
            logger.error(f"No phone_number in room metadata: {room_name}, metadata was: {room_metadata}")
            return
        
        # Log calendar availability
        if calendar_grant_id:
            print(f"Calendar grant ID available: {calendar_grant_id[:20]}...", flush=True)
            logger.info(f"Calendar grant ID available for user {user_id}")
        else:
            print("No calendar grant ID - calendar features will be unavailable", flush=True)
            logger.info("No calendar grant ID - agent will skip calendar features")
            
    except Exception as e:
        logger.error(f"Error parsing room metadata: {e}")
        return
    
    # Create the Praxa agent instance with calendar access if available
    praxa = PraxaAgent(user_id=user_id, call_log_id=call_log_id or "unknown", calendar_grant_id=calendar_grant_id)
    _current_agent = praxa
    
    # Load user context
    await praxa.load_user_context()
    
    # Dial out to the phone number via SIP
    participant = None
    try:
        print(f"LIVEKIT_SIP_TRUNK_ID = {LIVEKIT_SIP_TRUNK_ID}", flush=True)
        
        if not LIVEKIT_SIP_TRUNK_ID:
            print("ERROR: LIVEKIT_SIP_TRUNK_ID not configured!", flush=True)
            logger.error("LIVEKIT_SIP_TRUNK_ID not configured - cannot make outbound calls")
            logger.error("Please configure a SIP trunk in LiveKit Cloud: https://cloud.livekit.io")
            return
        
        print(f"Dialing out to {phone_number} via SIP...", flush=True)
        logger.info(f"Dialing out to {phone_number} via SIP...")
        
        lk_api = livekit_api.LiveKitAPI(
            LIVEKIT_URL,
            LIVEKIT_API_KEY,
            LIVEKIT_API_SECRET
        )
        
        # Create SIP participant (dial out)
        sip_response = await lk_api.sip.create_sip_participant(
            livekit_api.CreateSIPParticipantRequest(
                sip_trunk_id=LIVEKIT_SIP_TRUNK_ID,
                sip_call_to=phone_number,
                room_name=room_name,
                participant_identity="phone-user",
                participant_name="Phone User",
                play_ringtone=True,
            )
        )
        
        await lk_api.aclose()
        
        logger.info(f"SIP call initiated to {phone_number}")
        
        # Update call log
        if call_log_id:
            await praxa.db.update_call_log(call_log_id, {
                "status": "ringing",
                "call_sid": sip_response.sip_call_id if sip_response else None
            })
        
        # Wait for the phone participant to connect
        print("Waiting for phone participant to connect...", flush=True)
        participant = await ctx.wait_for_participant()
        print(f"Phone participant connected: {participant.identity}", flush=True)
        logger.info(f"Phone participant connected: {participant.identity}")
        
    except Exception as e:
        logger.error(f"Failed to dial out via SIP: {e}")
        if call_log_id:
            await praxa.db.update_call_log(call_log_id, {
                "status": "failed",
                "failure_reason": f"SIP dial failed: {str(e)}"
            })
        return
    
    # Mark call as started
    await praxa.on_call_started()
    logger.info("Call marked as started, creating agent session...")
    
    try:
        # Initialize voice pipeline components
        logger.info("Initializing voice pipeline components...")
        
        vad = silero.VAD.load()
        logger.info("VAD loaded")
        
        stt = deepgram.STT()
        logger.info("Deepgram STT initialized")
        
        llm_instance = openai.LLM(model="gpt-4o")
        logger.info("OpenAI LLM initialized")
        
        tts = elevenlabs.TTS(
            api_key=os.getenv("ELEVEN_LABS_API_KEY") or os.getenv("ELEVEN_API_KEY"),
            voice_id=os.getenv("ELEVEN_LABS_VOICE_ID", "Pcfg2Zc6kmNWQ9ji3J5F"),
        )
        logger.info("ElevenLabs TTS initialized")
        
        # Create the Praxa agent with function tools for task management
        PraxaVoiceAgent = create_praxa_agent_class(praxa)
        agent = PraxaVoiceAgent()
        logger.info("PraxaVoiceAgent created with function tools")
        
        # Create AgentSession with pipeline components
        session = AgentSession(
            vad=vad,
            stt=stt,
            llm=llm_instance,
            tts=tts,
        )
        logger.info("AgentSession created successfully")
        
        # Track transcripts using LiveKit Agents v1.0+ patterns
        logger.info("[TRANSCRIPT DEBUG] Setting up transcript event listeners...")
        print("[TRANSCRIPT DEBUG] Setting up transcript event listeners...", flush=True)
        
        # Track logged messages to avoid duplicates
        praxa._logged_messages = set()
        
        # Listen to conversation_item_added - the main event for conversation tracking
        @session.on("conversation_item_added")
        def on_conversation_item(item):
            try:
                role = item.role if hasattr(item, 'role') else 'assistant'
                content = item.content if hasattr(item, 'content') else str(item)
                
                # Create unique ID to prevent duplicates
                message_id = f"{role}:{content[:50]}"
                if message_id not in praxa._logged_messages:
                    print(f"[{'🎤 USER' if role == 'user' else '🤖 AGENT'}] {content[:80]}...", flush=True)
                    logger.info(f"[TRANSCRIPT] conversation_item_added - {role}: {content[:50]}...")
                    praxa.on_transcript_update(role, content)
                    praxa._logged_messages.add(message_id)
            except Exception as e:
                logger.error(f"[TRANSCRIPT] Error in conversation_item_added handler: {e}")
        
        # Also listen to user_input_transcribed for user speech
        @session.on("user_input_transcribed")  
        def on_user_transcribed(transcription):
            try:
                content = transcription.text if hasattr(transcription, 'text') else str(transcription)
                message_id = f"user:{content[:50]}"
                if message_id not in praxa._logged_messages:
                    print(f"[🎤 USER TRANSCRIBED] {content[:80]}...", flush=True)
                    logger.info(f"[TRANSCRIPT] user_input_transcribed: {content[:50]}...")
                    praxa.on_transcript_update("user", content)
                    praxa._logged_messages.add(message_id)
            except Exception as e:
                logger.error(f"[TRANSCRIPT] Error in user_input_transcribed handler: {e}")
        
        logger.info("[TRANSCRIPT DEBUG] Event listeners registered: conversation_item_added, user_input_transcribed")
        print("[TRANSCRIPT DEBUG] Event listeners registered successfully", flush=True)
        
        # Start the session
        logger.info(f"Starting session with participant: {participant.identity}")
        
        # v1.0 uses room= and agent=
        try:
            await session.start(room=ctx.room, agent=agent)
        except TypeError as e:
            logger.info(f"First start attempt failed: {e}, trying without agent")
            try:
                await session.start(room=ctx.room)
            except TypeError as e2:
                logger.info(f"Second start attempt failed: {e2}, trying with no params")
                await session.start()
        
        logger.info("Session started!")
        
        # Say the opening message
        opening_msg = praxa._get_opening_message()
        logger.info(f"Saying opening message: {opening_msg[:50]}...")
        await session.say(opening_msg, allow_interruptions=True)
        logger.info("Opening message sent!")
        
        # The session is now running in the background managed by LiveKit
        # We DON'T need to wait() - LiveKit handles the session lifecycle
        # The session will stay active until the room disconnects
        # Just like the working Praxa-voice-agent code!
        logger.info("Session running - LiveKit managing lifecycle. Job complete!")
        print("Session running - agent will remain active until room disconnect", flush=True)
        
    except Exception as e:
        logger.error(f"Error in agent session: {e}", exc_info=True)
    finally:
        # Extract final transcript from session.history before ending
        logger.info("Extracting final transcript from session...")
        print("=" * 60, flush=True)
        print("EXTRACTING FINAL TRANSCRIPT", flush=True)
        print("=" * 60, flush=True)
        
        try:
            # Try to get conversation history from session.history (LiveKit v1.0+ API)
            if hasattr(session, 'history'):
                history = session.history
                logger.info(f"[TRANSCRIPT FINAL] Found session.history with {len(history)} items")
                print(f"[TRANSCRIPT FINAL] session.history has {len(history)} items", flush=True)
                
                for item in history:
                    role = item.role if hasattr(item, 'role') else 'assistant'
                    content = item.content if hasattr(item, 'content') else str(item)
                    
                    # Only add if not already in transcript
                    message_id = f"{role}:{content[:50]}"
                    if not hasattr(praxa, '_logged_messages') or message_id not in praxa._logged_messages:
                        print(f"[FINAL EXTRACT] {role}: {content[:80]}...", flush=True)
                        praxa.on_transcript_update(role, content)
                        if hasattr(praxa, '_logged_messages'):
                            praxa._logged_messages.add(message_id)
            else:
                logger.warning("[TRANSCRIPT FINAL] session.history not available")
                print("[TRANSCRIPT FINAL] session.history not available", flush=True)
        except Exception as e:
            logger.error(f"[TRANSCRIPT FINAL] Error extracting from session.history: {e}")
            print(f"[TRANSCRIPT FINAL] Error: {e}", flush=True)
        
        print(f"[TRANSCRIPT FINAL] Total messages captured: {len(praxa.transcript)}", flush=True)
        print("=" * 60, flush=True)
        
        # ALWAYS call on_call_ended, even if there was an error
        logger.info("Calling on_call_ended...")
        try:
            await praxa.on_call_ended()
            logger.info("on_call_ended completed successfully")
        except Exception as e:
            logger.error(f"Error in on_call_ended: {e}")


def run_agent():
    """Run the LiveKit agent worker."""
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )


if __name__ == "__main__":
    run_agent()
