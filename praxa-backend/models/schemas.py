"""Pydantic models for data validation and serialization."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CallStatus(str, Enum):
    """Status values for call logs."""
    SCHEDULED = "scheduled"
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    CANCELED = "canceled"


class ScheduledCallStatus(str, Enum):
    """Status values for scheduled calls."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TimeWindow(str, Enum):
    """Time window preferences for calls."""
    MORNING = "morning"      # 8am - 12pm
    AFTERNOON = "afternoon"  # 12pm - 5pm
    EVENING = "evening"      # 5pm - 8pm


class CheckinFrequency(str, Enum):
    """How often the user wants check-in calls."""
    ONCE_PER_WEEK = "once_per_week"
    TWICE_PER_WEEK = "twice_per_week"
    OFF = "off"


class TaskStatus(str, Enum):
    """Status values for tasks (loops)."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Database Models

class User(BaseModel):
    """User account from the database."""
    id: UUID
    email: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserSettings(BaseModel):
    """User settings including phone and call preferences."""
    id: UUID
    user_id: UUID
    sprint_cadence: str = "weekly"
    checkin_frequency: str = "once_per_week"
    checkin_time_window: str = "afternoon"
    phone_number: Optional[str] = None
    phone_country_code: str = "+1"
    phone_verified: bool = False
    calls_enabled: bool = True
    next_scheduled_call: Optional[datetime] = None
    last_call_at: Optional[datetime] = None
    timezone: str = "America/New_York"
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @property
    def full_phone_number(self) -> Optional[str]:
        """Get the full phone number in E.164 format."""
        if not self.phone_number:
            return None
        # If phone_number already has country code, return as is
        if self.phone_number.startswith("+"):
            return self.phone_number
        return f"{self.phone_country_code}{self.phone_number}"


class Bucket(BaseModel):
    """A goal category/initiative."""
    id: UUID
    user_id: UUID
    name: str
    description: Optional[str] = None
    color: str
    icon: str
    goal: Optional[str] = None
    archived: bool = False
    created_at: datetime
    updated_at: datetime
    # Nested loops when fetched with joins
    loops: list["Loop"] = Field(default_factory=list)

    class Config:
        from_attributes = True


class Loop(BaseModel):
    """A task within a bucket."""
    id: UUID
    bucket_id: UUID
    user_id: UUID
    title: str
    description: Optional[str] = None
    status: str = "open"
    priority: str = "medium"
    due_date: Optional[datetime] = None
    is_this_week: bool = False
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    # Nested bucket name when fetched with joins
    bucket_name: Optional[str] = None
    bucket_color: Optional[str] = None

    class Config:
        from_attributes = True


class CallLog(BaseModel):
    """Record of a phone call."""
    id: UUID
    user_id: UUID
    call_sid: Optional[str] = None
    livekit_room_name: Optional[str] = None
    phone_number: str
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    status: str = "scheduled"
    failure_reason: Optional[str] = None
    transcript: Optional[list[dict]] = None
    summary: Optional[str] = None
    tasks_discussed: Optional[list[str]] = None
    tasks_completed: Optional[list[str]] = None
    tasks_created: Optional[list[str]] = None
    goals_updated: Optional[list[dict]] = None
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    user_feedback: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ScheduledCall(BaseModel):
    """A scheduled call in the queue."""
    id: UUID
    user_id: UUID
    scheduled_for: datetime
    time_window: str
    status: str = "pending"
    call_log_id: Optional[UUID] = None
    attempt_count: int = 0
    max_attempts: int = 3
    last_attempt_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# API Request/Response Models

class TriggerCallRequest(BaseModel):
    """Request body for manually triggering a call."""
    user_id: UUID


class TriggerCallResponse(BaseModel):
    """Response for trigger call endpoint."""
    success: bool
    message: str
    call_log_id: Optional[UUID] = None
    livekit_room_name: Optional[str] = None


class TwilioWebhookRequest(BaseModel):
    """Twilio status callback webhook payload."""
    CallSid: str
    CallStatus: str
    To: Optional[str] = None
    From_: Optional[str] = Field(None, alias="From")
    Direction: Optional[str] = None
    CallDuration: Optional[str] = None
    Timestamp: Optional[str] = None

    class Config:
        populate_by_name = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    environment: str
    timestamp: datetime


# Data Transfer Objects for Agent

class UserContext(BaseModel):
    """Complete user context for the agent."""
    user_id: UUID
    email: str
    phone_number: str
    timezone: str
    checkin_frequency: str
    buckets: list[Bucket]
    this_week_tasks: list[Loop]
    overdue_tasks: list[Loop]
    recently_completed: list[Loop]


class TaskUpdate(BaseModel):
    """An update to be made to a task."""
    loop_id: UUID
    action: str  # "complete", "add_note", "reschedule", "update_status"
    value: Optional[str] = None  # Note text, new due date, etc.


class NewTask(BaseModel):
    """A new task to be created."""
    bucket_id: UUID
    title: str
    description: Optional[str] = None
    priority: str = "medium"
    due_date: Optional[datetime] = None
    is_this_week: bool = False


class CallSummary(BaseModel):
    """Summary of a completed call."""
    transcript: list[dict]
    summary: str
    tasks_discussed: list[UUID]
    tasks_completed: list[UUID]
    tasks_created: list[UUID]
    goals_updated: list[dict]
    duration_seconds: int

