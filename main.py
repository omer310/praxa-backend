"""
Praxa Backend - FastAPI Server

This is the main entry point for the Praxa backend service.
It provides HTTP endpoints for triggering calls, receiving webhooks,
and manages the background scheduler for automated calls.
"""

import os
import asyncio
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from livekit import api as livekit_api

from models.schemas import (
    TriggerCallRequest,
    TriggerCallResponse,
    ScheduleCallRequest,
    HealthResponse,
    CallStatus,
)
from services.supabase_client import get_supabase_client, SupabaseClient
from services.scheduler import get_call_scheduler, CallScheduler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Reduce noise from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("livekit").setLevel(logging.WARNING)

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
PORT = int(os.getenv("PORT", 8000))
BASE_URL = os.getenv("BASE_URL", f"http://localhost:{PORT}")

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Supabase configuration for JWT validation
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")


async def verify_jwt_token(request: Request) -> dict:
    """
    Verify Supabase JWT token using Supabase client.
    
    This properly verifies the JWT signature using Supabase's auth system.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Dict with 'user_id' if token is valid
        
    Raises:
        HTTPException with 401 status if token is invalid or missing
    """
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = auth_header.split(" ")[1]
    
    try:
        # Use Supabase client to verify the token properly
        supabase_client = get_supabase_client()
        
        # Verify token by getting user - this validates the JWT signature
        response = supabase_client.client.auth.get_user(token)
        
        if not response or not response.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return {"user_id": response.user.id}
        
    except Exception as e:
        logger.error(f"JWT verification error: {e}")
        raise HTTPException(status_code=401, detail="Unauthorized")


async def trigger_call_for_user(user_id: str) -> Optional[dict]:
    """
    Trigger a call for a specific user.
    
    This function:
    1. Fetches user data from Supabase
    2. Creates a call log entry
    3. Creates a LiveKit room with phone number in metadata
    4. LiveKit dispatches the agent to the room
    5. The agent dials out via SIP after joining
    
    Args:
        user_id: The UUID of the user to call
        
    Returns:
        Dict with call_log_id and room_name if successful, None if failed
    """
    db = get_supabase_client()
    
    try:
        # Get user with settings
        user_data = await db.get_user_with_settings(user_id)
        
        if not user_data:
            logger.error(f"User not found: {user_id}")
            return None
        
        user = user_data["user"]
        settings = user_data["settings"]
        
        if not settings:
            logger.error(f"No settings found for user: {user_id}")
            return None
        
        # Check if calls are enabled
        if not settings.get("calls_enabled", True):
            logger.info(f"Calls disabled for user: {user_id}")
            return None
        
        # Get phone number
        phone_number = settings.get("phone_number")
        if not phone_number:
            logger.error(f"No phone number for user: {user_id}")
            return None
        
        # Format phone number (ensure E.164)
        if not phone_number.startswith("+"):
            country_code = settings.get("phone_country_code", "+1")
            phone_number = f"{country_code}{phone_number}"
        
        # Check if phone is verified
        if not settings.get("phone_verified", False):
            logger.warning(f"Phone not verified for user: {user_id}")
            # In production, you might want to skip unverified phones
        
        # Generate room name
        call_id = str(uuid4())[:8]
        room_name = f"praxa-call-{user_id}-{call_id}"
        
        # Get Nylas calendar grant ID (optional - agent will work without it)
        calendar_grant_id = None
        try:
            # Query nylas_oauth_tokens for calendar grant
            calendar_token = db.client.table("nylas_oauth_tokens").select("grant_id").eq(
                "user_id", user_id
            ).eq("integration_type", "calendar").limit(1).execute()
            
            if calendar_token.data and len(calendar_token.data) > 0:
                calendar_grant_id = calendar_token.data[0].get("grant_id")
                logger.info(f"Found calendar grant ID for user {user_id}: {calendar_grant_id[:20]}...")
            else:
                logger.info(f"No calendar grant ID found for user {user_id} - agent will skip calendar features")
        except Exception as e:
            logger.warning(f"Error fetching calendar grant ID: {e}")
        
        # Create call log entry
        call_log = await db.create_call_log(
            user_id=user_id,
            phone_number=phone_number,
            livekit_room_name=room_name
        )
        
        if not call_log:
            logger.error(f"Failed to create call log for user: {user_id}")
            return None
        
        call_log_id = call_log["id"]
        
        # Create LiveKit room with phone number in metadata
        # The agent will be dispatched to this room and will dial out via SIP
        try:
            lk_api = livekit_api.LiveKitAPI(
                LIVEKIT_URL,
                LIVEKIT_API_KEY,
                LIVEKIT_API_SECRET
            )
            
            # Include metadata for agent (phone_number, calendar_grant_id)
            metadata_dict = {
                "user_id": user_id,
                "call_log_id": call_log_id,
                "phone_number": phone_number,
            }
            
            # Add calendar grant ID if available
            if calendar_grant_id:
                metadata_dict["calendar_grant_id"] = calendar_grant_id
            
            room_metadata = json.dumps(metadata_dict)
            
            await lk_api.room.create_room(
                livekit_api.CreateRoomRequest(
                    name=room_name,
                    empty_timeout=300,  # 5 minutes
                    max_participants=3,  # agent + phone user + buffer
                    metadata=room_metadata
                )
            )
            
            await lk_api.aclose()
            
            logger.info(f"Created LiveKit room: {room_name} - agent will dial {phone_number}")
            
            # Update call log status
            await db.update_call_log(call_log_id, {
                "status": "initiated"
            })
            
            return {"call_log_id": call_log_id, "room_name": room_name}
            
        except Exception as e:
            logger.error(f"Failed to create LiveKit room: {e}")
            await db.update_call_log(call_log_id, {
                "status": "failed",
                "failure_reason": f"Failed to initiate call: {str(e)}"
            })
            return None
            
    except Exception as e:
        logger.error(f"Error triggering call for user {user_id}: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Start the background scheduler on startup
    - Stop the scheduler on shutdown
    """
    # Startup
    logger.info(f"Starting Praxa Backend ({ENVIRONMENT})")
    
    # Initialize scheduler
    scheduler = get_call_scheduler()
    scheduler.set_trigger_callback(trigger_call_for_user)
    scheduler.start()
    
    logger.info("Praxa Backend started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Praxa Backend")
    scheduler.stop()
    logger.info("Praxa Backend shut down")


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

# Create FastAPI app
app = FastAPI(
    title="Praxa Backend",
    description="Backend service for Praxa productivity assistant voice calls",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS with environment-specific origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []

# In development, allow localhost for testing
if ENVIRONMENT == "development":
    ALLOWED_ORIGINS.extend(["http://localhost:8081", "http://localhost:19000", "http://127.0.0.1:8081"])
    logger.info(f"Development mode: CORS origins = {ALLOWED_ORIGINS}")
else:
    # Production: mobile apps don't need CORS (native requests), only for admin tools
    logger.info(f"Production mode: CORS origins = {ALLOWED_ORIGINS}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else [],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Railway deployment.
    
    Returns:
        Health status with environment and timestamp
    """
    return HealthResponse(
        status="healthy",
        environment=ENVIRONMENT,
        timestamp=datetime.utcnow()
    )


@app.post("/trigger-call", response_model=TriggerCallResponse)
@limiter.limit("10/minute")  # Max 10 calls per minute per IP
async def trigger_call(
    request: Request,
    call_request: TriggerCallRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(verify_jwt_token)
):
    """
    Manually trigger a call for a user.
    
    **Requires Authentication**: Supabase JWT token in Authorization header
    
    This endpoint is useful for:
    - Testing the call flow
    - Manually initiating a check-in call
    - Admin-triggered calls
    
    Args:
        request: Contains the user_id to call
        background_tasks: FastAPI background tasks handler
        auth: Authenticated user info from JWT token (dependency)
        
    Returns:
        TriggerCallResponse with call details
    """
    user_id = str(call_request.user_id)
    authenticated_user_id = auth["user_id"]
    
    # Security: Only allow users to trigger calls for themselves
    if user_id != authenticated_user_id:
        raise HTTPException(status_code=403, detail="Cannot trigger calls for other users")
    
    logger.info(f"Trigger call request for user: {user_id} (authenticated: {authenticated_user_id})")
    
    # Verify user exists
    db = get_supabase_client()
    user_data = await db.get_user_with_settings(user_id)
    
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not user_data.get("settings"):
        raise HTTPException(status_code=400, detail="User has no settings configured")
    
    settings = user_data["settings"]
    
    if not settings.get("phone_number"):
        raise HTTPException(status_code=400, detail="User has no phone number configured")
    
    # Trigger the call
    result = await trigger_call_for_user(user_id)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to initiate call")
    
    return TriggerCallResponse(
        success=True,
        message="Call initiated successfully",
        call_log_id=UUID(result["call_log_id"]),
        livekit_room_name=result["room_name"]
    )


@app.post("/webhook/twilio")
@limiter.limit("100/minute")  # Allow high volume from Twilio, but prevent abuse
async def twilio_webhook(request: Request):
    """
    Webhook endpoint for Twilio call status updates.
    
    Twilio sends POST requests here when call status changes:
    - initiated: Call is being set up
    - ringing: Phone is ringing
    - answered: Call was answered
    - completed: Call ended normally
    - busy: Line was busy
    - no-answer: No answer
    - failed: Call failed
    - canceled: Call was canceled
    
    This endpoint updates the call_log with the new status.
    """
    try:
        # Parse form data from Twilio
        form_data = await request.form()
        
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        call_duration = form_data.get("CallDuration")
        
        if not call_sid:
            logger.warning("Twilio webhook received without CallSid")
            return JSONResponse({"status": "ok"})
        
        logger.info(f"Twilio webhook: {call_sid} -> {call_status}")
        
        # Map Twilio status to our status
        status_map = {
            "initiated": CallStatus.INITIATED,
            "ringing": CallStatus.RINGING,
            "in-progress": CallStatus.IN_PROGRESS,
            "completed": CallStatus.COMPLETED,
            "busy": CallStatus.BUSY,
            "no-answer": CallStatus.NO_ANSWER,
            "failed": CallStatus.FAILED,
            "canceled": CallStatus.CANCELED,
        }
        
        our_status = status_map.get(call_status, call_status)
        
        # Update call log
        db = get_supabase_client()
        call_log = await db.get_call_log_by_sid(call_sid)
        
        if call_log:
            updates = {"status": our_status}
            
            # Add duration if call completed
            if call_duration:
                try:
                    updates["duration_seconds"] = int(call_duration)
                except ValueError:
                    pass
            
            # Add ended_at for terminal statuses
            if our_status in [CallStatus.COMPLETED, CallStatus.FAILED, 
                             CallStatus.NO_ANSWER, CallStatus.BUSY, 
                             CallStatus.CANCELED]:
                updates["ended_at"] = datetime.utcnow().isoformat()
            
            # Add failure reason for failed calls
            if our_status in [CallStatus.FAILED, CallStatus.NO_ANSWER, CallStatus.BUSY]:
                updates["failure_reason"] = f"Call {call_status}"
            
            await db.update_call_log(call_log["id"], updates)
            logger.info(f"Updated call log {call_log['id']} to status {our_status}")
        else:
            logger.warning(f"No call log found for SID: {call_sid}")
        
        return JSONResponse({"status": "ok"})
        
    except Exception as e:
        logger.error(f"Error processing Twilio webhook: {e}")
        # Still return 200 to prevent Twilio retries
        return JSONResponse({"status": "error", "message": str(e)})


@app.get("/scheduled-calls")
async def list_scheduled_calls():
    """
    List all pending scheduled calls.
    
    Returns:
        List of pending scheduled calls with user info
    """
    db = get_supabase_client()
    calls = await db.get_pending_scheduled_calls()
    
    return {
        "count": len(calls),
        "scheduled_calls": calls
    }


@app.post("/schedule-call")
@limiter.limit("20/minute")  # Max 20 schedule requests per minute per IP
async def schedule_call(request: Request, schedule_request: ScheduleCallRequest):
    """
    Manually schedule a call for a user.
    
    This creates a new entry in scheduled_calls that will be
    picked up by the scheduler.
    
    Args:
        user_id: The UUID of the user to schedule a call for
        
    Returns:
        The created scheduled call entry
    """
    db = get_supabase_client()
    
    # Get user settings
    user_data = await db.get_user_with_settings(str(schedule_request.user_id))
    
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    settings = user_data.get("settings")
    if not settings:
        raise HTTPException(status_code=400, detail="User has no settings configured")
    
    # Schedule the call
    scheduled = await db.schedule_next_call(
        user_id=str(schedule_request.user_id),
        checkin_schedule=settings.get("checkin_schedule", []),
        timezone=settings.get("timezone", "America/New_York"),
        checkin_enabled=settings.get("checkin_enabled", True)
    )
    
    if not scheduled:
        raise HTTPException(status_code=400, detail="Calls are disabled for this user")
    
    return {
        "success": True,
        "scheduled_call": scheduled
    }


@app.post("/sync-scheduled-calls")
async def sync_scheduled_calls(
    user_id: str,
    auth: dict = Depends(verify_jwt_token)
):
    """
    Sync scheduled calls for a user based on their current checkin_schedule.
    This should be called when the user updates their schedule in the app.
    
    It will:
    1. Cancel all pending scheduled calls for this user
    2. Create new scheduled calls based on the current checkin_schedule
    
    **Requires Authentication**: Supabase JWT token in Authorization header
    
    Args:
        user_id: The UUID of the user
        auth: Authenticated user info from JWT token (dependency)
        
    Returns:
        List of newly created scheduled calls
    """
    authenticated_user_id = auth["user_id"]
    
    # Security: Only allow users to sync their own calls
    if user_id != authenticated_user_id:
        raise HTTPException(status_code=403, detail="Cannot sync calls for other users")
    
    db = get_supabase_client()
    
    # Get user settings
    user_data = await db.get_user_with_settings(user_id)
    
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    settings = user_data.get("settings")
    if not settings:
        raise HTTPException(status_code=400, detail="User has no settings configured")
    
    try:
        checkin_schedule = settings.get("checkin_schedule", [])
        timezone_str = settings.get("timezone", "America/New_York")
        checkin_enabled = settings.get("checkin_enabled", True)
        
        if not checkin_enabled or not checkin_schedule:
            # Cancel all pending calls if checkin is disabled or no schedule
            db.client.table("scheduled_calls").update({
                "status": "cancelled",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", user_id).eq("status", "pending").execute()
            
            logger.info(f"Cancelled all pending calls for user {user_id} (checkin disabled or no schedule)")
            return {"success": True, "scheduled_calls": [], "count": 0}
        
        # Cancel all old pending calls and create fresh ones
        # This ensures we always have the correct set of calls matching the current schedule
        db.client.table("scheduled_calls").update({
            "status": "cancelled",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("user_id", user_id).eq("status", "pending").execute()
        
        logger.info(f"Cancelled all pending scheduled calls for user {user_id}")
        
        # Create new scheduled calls for ALL days in the schedule
        created_calls = await db.create_all_scheduled_calls(
            user_id=user_id,
            checkin_schedule=checkin_schedule,
            timezone=timezone_str,
            checkin_enabled=checkin_enabled
        )
        
        return {
            "success": True,
            "scheduled_calls": created_calls,
            "count": len(created_calls)
        }
    except Exception as e:
        logger.error(f"Error syncing scheduled calls: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync scheduled calls")


@app.get("/call-logs/{user_id}")
async def get_user_call_logs(user_id: str, limit: int = 10):
    """
    Get call history for a user.
    
    Args:
        user_id: The UUID of the user
        limit: Maximum number of records to return
        
    Returns:
        List of call logs for the user
    """
    db = get_supabase_client()
    
    try:
        response = db.client.table("call_logs").select("*").eq(
            "user_id", user_id
        ).order("created_at", desc=True).limit(limit).execute()
        
        # Add debug info about transcripts
        if response.data:
            for log in response.data:
                transcript = log.get("transcript")
                if transcript:
                    log["_transcript_debug"] = {
                        "type": str(type(transcript)),
                        "length": len(transcript) if isinstance(transcript, (list, dict, str)) else None,
                        "is_empty": not bool(transcript)
                    }
        
        return {
            "count": len(response.data) if response.data else 0,
            "call_logs": response.data or []
        }
    except Exception as e:
        logger.error(f"Error fetching call logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch call logs")


@app.get("/call-log/{call_log_id}/debug")
async def debug_call_log(call_log_id: str):
    """
    Debug endpoint to inspect a specific call log in detail.
    
    Args:
        call_log_id: The UUID of the call log
        
    Returns:
        Detailed call log information including transcript debug data
    """
    db = get_supabase_client()
    
    try:
        response = db.client.table("call_logs").select("*").eq("id", call_log_id).single().execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Call log not found")
        
        call_log = response.data
        transcript = call_log.get("transcript")
        
        # Detailed transcript analysis
        debug_info = {
            "transcript_exists": transcript is not None,
            "transcript_type": str(type(transcript)),
            "transcript_length": len(transcript) if isinstance(transcript, (list, dict, str)) else 0,
            "transcript_is_empty": not bool(transcript),
            "transcript_sample": transcript[:2] if isinstance(transcript, list) and len(transcript) > 0 else transcript,
            "summary_exists": bool(call_log.get("summary")),
            "status": call_log.get("status"),
            "duration_seconds": call_log.get("duration_seconds"),
            "tasks_completed_count": len(call_log.get("tasks_completed", [])),
            "tasks_created_count": len(call_log.get("tasks_created", [])),
        }
        
        return {
            "call_log_id": call_log_id,
            "debug_info": debug_info,
            "full_call_log": call_log
        }
    except Exception as e:
        logger.error(f"Error fetching call log debug: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch call log")


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=ENVIRONMENT == "development"
    )

