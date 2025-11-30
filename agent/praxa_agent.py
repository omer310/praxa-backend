"""LiveKit Agent for Praxa voice AI conversations."""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID
import json

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    RoomInputOptions,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

from services.supabase_client import get_supabase_client
from .prompts import SYSTEM_PROMPT, get_user_context_prompt, get_opening_message, get_closing_message

logger = logging.getLogger(__name__)


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

    def __init__(self, user_id: str, call_log_id: str):
        self.user_id = user_id
        self.call_log_id = call_log_id
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
            
            logger.info(
                f"Loaded context for user {self.user_id}: "
                f"{len(self.buckets)} buckets, "
                f"{len(self.this_week_tasks)} this week tasks, "
                f"{len(self.overdue_tasks)} overdue, "
                f"{len(self.recently_completed)} recently completed"
            )
            
        except Exception as e:
            logger.error(f"Error loading user context: {e}")
            raise

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
            checkin_frequency=checkin_frequency
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
            
            # Generate summary using OpenAI
            summary = await self._generate_summary()
            
            # Update call log
            await self.db.update_call_log(self.call_log_id, {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": duration,
                "transcript": self.transcript,
                "summary": summary,
                "tasks_discussed": list(set(self.tasks_discussed)),
                "tasks_completed": list(set(self.tasks_completed)),
                "tasks_created": list(set(self.tasks_created)),
                "goals_updated": self.goals_updated
            })
            
            # Schedule next call
            if self.user_settings:
                await self.db.schedule_next_call(
                    user_id=self.user_id,
                    frequency=self.user_settings.get("checkin_frequency", "once_per_week"),
                    timezone=self.user_settings.get("timezone", "America/New_York"),
                    time_window=self.user_settings.get("checkin_time_window", "afternoon")
                )
            
            logger.info(
                f"Call ended for user {self.user_id}: "
                f"{duration}s, {len(self.tasks_completed)} completed, "
                f"{len(self.tasks_created)} created"
            )
            
        except Exception as e:
            logger.error(f"Error handling call end: {e}")

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
        self.transcript.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

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


# Global agent instance for the current session
_current_agent: Optional[PraxaAgent] = None


def create_praxa_agent_class(praxa: PraxaAgent):
    """Create a custom Agent class with the Praxa tools."""
    
    class PraxaVoiceAgent(Agent):
        def __init__(self):
            super().__init__(
                instructions=praxa._build_system_prompt(),
            )
        
        @llm.function_tool
        async def mark_task_complete(self, task_title: str) -> str:
            """Mark a task as complete. Use this when the user says they finished a task.
            
            Args:
                task_title: The title of the task to mark complete
            """
            return await praxa.mark_task_complete(task_title)
        
        @llm.function_tool
        async def add_task_note(self, task_title: str, note: str) -> str:
            """Add a note to an existing task. Use this to record updates or details about a task.
            
            Args:
                task_title: The title of the task to add a note to
                note: The note to add to the task
            """
            return await praxa.add_task_note(task_title, note)
        
        @llm.function_tool
        async def create_task(self, title: str, bucket_name: str, is_this_week: bool = False) -> str:
            """Create a new task. Ask the user which bucket/initiative to add it to.
            
            Args:
                title: The title of the new task
                bucket_name: The name of the bucket/initiative to add the task to
                is_this_week: Whether to mark this for the current week's focus
            """
            return await praxa.create_task(title, bucket_name, is_this_week)
        
        @llm.function_tool
        async def update_task_due_date(self, task_title: str, due_date: str) -> str:
            """Update the due date of a task. Use when user wants to reschedule.
            
            Args:
                task_title: The title of the task to update
                due_date: The new due date in YYYY-MM-DD format
            """
            return await praxa.update_task_due_date(task_title, due_date)
        
        @llm.function_tool
        async def list_buckets(self) -> str:
            """Get the list of user's buckets/initiatives. Use when you need to know what buckets exist."""
            return await praxa.list_buckets()
    
    return PraxaVoiceAgent


async def entrypoint(ctx: JobContext):
    """
    Main entry point for the LiveKit agent.
    
    This is called by LiveKit when a new room is created for a call.
    The room name should contain the call information.
    """
    global _current_agent
    
    room_name = ctx.room.name
    logger.info(f"Agent entrypoint called for room: {room_name}")
    
    # Parse room name to get IDs
    # Room name format: praxa-call-{user_id}-{call_log_id}
    try:
        parts = room_name.split("-")
        if len(parts) >= 4 and parts[0] == "praxa" and parts[1] == "call":
            user_id = parts[2]
            call_log_id = parts[3]
        else:
            # Try to get from job metadata
            metadata = json.loads(ctx.room.metadata) if ctx.room.metadata else {}
            user_id = metadata.get("user_id")
            call_log_id = metadata.get("call_log_id")
            
            if not user_id or not call_log_id:
                logger.error(f"Could not parse room name: {room_name}")
                return
    except Exception as e:
        logger.error(f"Error parsing room name: {e}")
        return
    
    # Create the Praxa agent instance
    praxa = PraxaAgent(user_id=user_id, call_log_id=call_log_id)
    _current_agent = praxa
    
    # Load user context
    await praxa.load_user_context()
    
    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Wait for a participant (the phone caller)
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant connected: {participant.identity}")
    
    # Mark call as started
    await praxa.on_call_started()
    
    # Create the agent class with tools
    PraxaVoiceAgent = create_praxa_agent_class(praxa)
    
    # Create the agent session
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=elevenlabs.TTS(
            voice_id=os.getenv("ELEVEN_LABS_VOICE_ID", "Pcfg2Zc6kmNWQ9ji3J5F"),
            model="eleven_turbo_v2"
        ),
    )
    
    # Track transcript via session events
    @session.on("user_message")
    def on_user_message(msg):
        praxa.on_transcript_update("user", msg.content)
    
    @session.on("agent_message")
    def on_agent_message(msg):
        praxa.on_transcript_update("assistant", msg.content)
    
    # Start the session with the agent
    agent = PraxaVoiceAgent()
    session.start(
        room=ctx.room,
        participant=participant,
        agent=agent,
    )
    
    # Say the opening message
    await session.say(praxa._get_opening_message(), allow_interruptions=True)
    
    # Wait for the session to end
    await session.wait()
    
    # Handle call end
    await praxa.on_call_ended()


def prequit(proc: JobProcess):
    """Called before the worker process exits."""
    logger.info("Agent worker shutting down")


def run_agent():
    """Run the LiveKit agent worker."""
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prequit_fnc=prequit,
        )
    )


if __name__ == "__main__":
    run_agent()
