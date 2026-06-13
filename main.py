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
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
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
from services.push_service import send_push_notification, get_user_push_token, schedule_receipt_check

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

# Shared secret for internal service-to-service calls (background agent → trigger call)
PRAXA_INTERNAL_SECRET = os.getenv("PRAXA_INTERNAL_SECRET", "")


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


async def trigger_call_for_user(user_id: str, reason: Optional[str] = None) -> Optional[dict]:
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

        if not await db.is_ai_enabled(user_id):
            logger.warning(f"AI disabled for user {user_id} — skipping call")
            return None

        # Get phone number
        phone_number = settings.get("phone_number")
        if not phone_number:
            logger.error(f"No phone number for user: {user_id}")
            return None
        
        # Normalize to strict E.164 format (digits only after +)
        # Handles formats like +1(646) 847-2984, (646) 847-2984, 6468472984, etc.
        import re
        digits = re.sub(r'\D', '', phone_number)
        if phone_number.startswith("+"):
            phone_number = f"+{digits}"
        elif digits and len(digits) == 10:
            country_code = settings.get("phone_country_code", "+1")
            phone_number = f"{country_code}{digits}"
        elif digits and len(digits) == 11 and digits.startswith("1"):
            phone_number = f"+{digits}"
        else:
            country_code = settings.get("phone_country_code", "+1")
            phone_number = f"{country_code}{digits}"
        
        logger.info(f"Normalized phone number to E.164: {phone_number}")
        
        # Check if phone is verified
        if not settings.get("phone_verified", False):
            logger.warning(f"Phone not verified for user: {user_id}")
            # In production, you might want to skip unverified phones
        
        # Generate room name
        call_id = str(uuid4())[:8]
        room_name = f"praxa-call-{user_id}-{call_id}"
        
        # Get Nylas grant IDs (calendar + email) — optional, agent works without them
        calendar_grant_id = None
        email_grant_id = None
        try:
            nylas_tokens = db.client.table("nylas_oauth_tokens").select("grant_id,integration_type").eq(
                "user_id", user_id
            ).execute()
            
            for token in (nylas_tokens.data or []):
                if token.get("integration_type") == "calendar" and not calendar_grant_id:
                    calendar_grant_id = token.get("grant_id")
                elif token.get("integration_type") == "email" and not email_grant_id:
                    email_grant_id = token.get("grant_id")
            
            if calendar_grant_id:
                logger.info(f"Found calendar grant for user {user_id}: {calendar_grant_id[:20]}...")
            if email_grant_id:
                logger.info(f"Found email grant for user {user_id}: {email_grant_id[:20]}...")
            if not calendar_grant_id and not email_grant_id:
                logger.info(f"No Nylas grants found for user {user_id} - agent will skip email/calendar features")
        except Exception as e:
            logger.warning(f"Error fetching Nylas grant IDs: {e}")
        
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
            
            if calendar_grant_id:
                metadata_dict["calendar_grant_id"] = calendar_grant_id
            if email_grant_id:
                metadata_dict["email_grant_id"] = email_grant_id
            if reason:
                metadata_dict["reason"] = reason
            
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
        timestamp=datetime.now(timezone.utc)
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

    if not await db.is_ai_enabled(user_id):
        raise HTTPException(status_code=403, detail="AI features are disabled for this account")

    allowed, used, limit = await db.check_and_record_voice_rate_limit(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Weekly voice call limit of {limit} reached ({used}/{limit} used). Resets after 7 days."
        )

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
                updates["ended_at"] = datetime.now(timezone.utc).isoformat()
            
            # Add failure reason for failed calls
            if our_status in [CallStatus.FAILED, CallStatus.NO_ANSWER, CallStatus.BUSY]:
                updates["failure_reason"] = f"Call {call_status}"
            
            await db.update_call_log(call_log["id"], updates)
            logger.info(f"Updated call log {call_log['id']} to status {our_status}")

            # Update the scheduled_call record that triggered this call
            user_id = call_log.get("user_id")
            if user_id and our_status in [
                CallStatus.COMPLETED, CallStatus.FAILED,
                CallStatus.NO_ANSWER, CallStatus.BUSY, CallStatus.CANCELED,
            ]:
                try:
                    sc = await db.get_processing_scheduled_call_for_user(user_id)
                    if sc:
                        sc_id = sc["id"]
                        sc_attempts = sc.get("attempt_count", 1)
                        sc_max = sc.get("max_attempts", 3)
                        if our_status == CallStatus.COMPLETED:
                            await db.advance_scheduled_call(sc_id)
                            logger.info(f"Call completed — advanced scheduled_call {sc_id} to next week")
                        elif sc_attempts >= sc_max:
                            await db.advance_scheduled_call(sc_id)
                            logger.warning(f"Call missed after {sc_max} attempts — advanced scheduled_call {sc_id} to next week")
                        else:
                            await db.update_scheduled_call(sc_id, {"status": "pending"})
                            logger.info(f"Call missed (attempt {sc_attempts}/{sc_max}) — scheduled_call {sc_id} reset for retry")
                except Exception as sc_err:
                    logger.error(f"Error updating scheduled_call after Twilio webhook for user {user_id}: {sc_err}")

            # Send push notification for terminal statuses
            if user_id and our_status in [
                CallStatus.COMPLETED, CallStatus.FAILED,
                CallStatus.NO_ANSWER, CallStatus.BUSY, CallStatus.CANCELED,
            ]:
                push_token = await get_user_push_token(user_id)
                if push_token:
                    if our_status == CallStatus.COMPLETED:
                        ticket_id = await send_push_notification(
                            push_token=push_token,
                            title="Great Work",
                            body="Check-in done. You're on track",
                            data={"notificationType": "call_completed"},
                        )
                    else:
                        ticket_id = await send_push_notification(
                            push_token=push_token,
                            title="We Missed You",
                            body="No worries! Open Praxa to adjust your check-in time",
                            data={"notificationType": "call_missed"},
                        )
                    if ticket_id:
                        schedule_receipt_check(ticket_id, user_id)
        else:
            logger.warning(f"No call log found for SID: {call_sid}")
        
        return JSONResponse({"status": "ok"})
        
    except Exception as e:
        logger.error(f"Error processing Twilio webhook: {e}")
        # Still return 200 to prevent Twilio retries
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/webhook/twilio/inbound-sms")
@limiter.limit("60/minute")
async def twilio_inbound_sms(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio Messaging webhook for inbound SMS replies (CANONICAL / PREFERRED PATH).

    This is the full AI agent path using sms_agent.py + praxa_core tools.
    It is more capable than the legacy Supabase edge function (twilio-inbound-sms),
    which uses simple intent classification only.

    IMPORTANT — SMS routing audit note:
    A legacy `twilio-inbound-sms` Supabase edge function also exists and is ACTIVE.
    Only ONE endpoint can be configured in the Twilio console at a time.
    To use this backend path (recommended):
      1. Set the Twilio Messaging Service webhook URL to this backend's URL:
         {BASE_URL}/webhook/twilio/inbound-sms
      2. Disable or leave the Supabase edge function unconfigured in Twilio.
    The edge function URL would be:
      https://<project-ref>.supabase.co/functions/v1/twilio-inbound-sms

    A tool-using, memory-loaded agent handles the reply, which exceeds Twilio's
    synchronous webhook window. So we ack immediately with an empty TwiML and
    process the message in a BackgroundTask; the agent sends its reply via
    Twilio outbound REST (services/twilio_sms.send_sms).
    """
    from services.reply_processor import lookup_user_by_phone
    from services.twilio_sms import send_sms
    from services.sms_agent import handle_inbound

    _empty = Response(
        content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
    )

    try:
        form = await request.form()
        from_number: str = form.get("From", "")
        body: str = (form.get("Body") or "").strip()

        if not from_number or not body:
            return _empty

        logger.info(f"[InboundSMS] from={from_number} body_len={len(body)}")

        user_id = await lookup_user_by_phone(from_number)
        if not user_id:
            logger.warning(f"[InboundSMS] Unrecognized number: {from_number}")
            background_tasks.add_task(
                send_sms,
                from_number,
                "We couldn't match your number to a Praxa account. Open the app to manage your tasks.",
            )
            return _empty

        background_tasks.add_task(handle_inbound, user_id, body, from_number)
        return _empty

    except Exception as exc:
        logger.error(f"[InboundSMS] Unhandled error: {exc}")
        return _empty


@app.post("/webhook/slack/events")
@limiter.limit("120/minute")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    """Receives Slack events forwarded by the `slack-events` edge function.

    The edge function verifies the Slack signature and forwards the raw event
    JSON with the shared secret. We authenticate the secret, ack immediately,
    and run the Slack agent in a BackgroundTask (it replies in-thread).
    """
    secret = os.getenv("PRAXA_WEBHOOK_SECRET")
    if secret and request.headers.get("X-Praxa-Secret") != secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": True})

    from services.slack_agent import handle_event, enabled
    if not enabled():
        return JSONResponse({"ok": True})

    background_tasks.add_task(handle_event, payload)
    return JSONResponse({"ok": True})


_NYLAS_EMAIL_EVENTS = {"message.created", "email.created", "message.updated"}
_NYLAS_CALENDAR_EVENTS = {"event.created", "event.updated", "event.deleted"}


def _verify_nylas_signature(request: Request, body_bytes: bytes) -> bool:
    """Return True if the Nylas HMAC-SHA256 signature is valid (or secret not configured)."""
    import hashlib
    import hmac as _hmac
    nylas_secret = os.getenv("NYLAS_WEBHOOK_SECRET", "")
    if not nylas_secret:
        return True
    sig = request.headers.get("X-Nylas-Signature", "")
    expected = _hmac.new(nylas_secret.encode(), body_bytes, hashlib.sha256).hexdigest()
    return _hmac.compare_digest(sig, expected)


@app.post("/webhook/nylas/email")
@limiter.limit("300/minute")
async def nylas_email_webhook(request: Request, background_tasks: BackgroundTasks):
    """Nylas v3 unified webhook for email AND calendar events.

    Configure a single Nylas webhook in the dashboard pointing at this URL with
    the following triggers enabled:
      Email:    message.created, message.updated
      Calendar: event.created, event.updated, event.deleted

    Set NYLAS_WEBHOOK_SECRET to the Nylas signing secret to enable HMAC
    verification. Without it the endpoint accepts all POST requests (dev mode).

    Email events: classified via email_classifier.py → email_insights table.
    Calendar events: trigger world-state refresh for affected user.
    """
    body_bytes = await request.body()
    if not _verify_nylas_signature(request, body_bytes):
        raise HTTPException(status_code=403, detail="Invalid Nylas signature")
    try:
        payload = json.loads(body_bytes)
    except Exception:
        return JSONResponse({"ok": True})

    event_type = payload.get("type", "")
    if event_type in _NYLAS_EMAIL_EVENTS:
        background_tasks.add_task(_handle_nylas_email_event, payload)
    elif event_type in _NYLAS_CALENDAR_EVENTS:
        background_tasks.add_task(_handle_nylas_calendar_event, payload)
    else:
        logger.debug(f"[nylas_webhook] Unhandled event type '{event_type}' — ignoring")
    return JSONResponse({"ok": True})


async def _handle_nylas_email_event(payload: dict) -> None:
    """Classify an inbound Nylas email event and notify the user if urgent."""
    try:
        event_type = payload.get("type", "")
        if event_type not in ("message.created", "email.created"):
            return

        data = payload.get("data", {}).get("object", {})
        grant_id = data.get("grant_id") or payload.get("grant_id")
        if not grant_id:
            return

        db = get_supabase_client()
        token_resp = await asyncio.to_thread(
            lambda: db.client.table("nylas_oauth_tokens")
            .select("user_id")
            .eq("grant_id", grant_id)
            .eq("integration_type", "email")
            .maybe_single()
            .execute()
        )
        if not token_resp or not token_resp.data:
            return
        user_id = token_resp.data.get("user_id")
        if not user_id:
            return

        email_id = data.get("id", "")
        thread_id = data.get("thread_id")
        subject = data.get("subject", "(no subject)")
        from_list = data.get("from") or [{}]
        sender_name = from_list[0].get("name") or from_list[0].get("email", "someone")
        sender_email = from_list[0].get("email", "")
        snippet = (data.get("snippet") or "")[:300]
        received_at_raw = data.get("date")
        received_at = (
            datetime.utcfromtimestamp(received_at_raw).isoformat() if received_at_raw else None
        )

        from services.email_classifier import classify_and_store_email_insight
        classification = await classify_and_store_email_insight(
            user_id=user_id,
            email_id=email_id,
            thread_id=thread_id,
            from_email=sender_email,
            from_name=sender_name,
            subject=subject,
            snippet=snippet,
            received_at=received_at,
        )

        if classification and classification.get("insight_type") == "needs_attention":
            from services.notify_service import notify_user
            await notify_user(
                user_id=user_id,
                event_type="urgent_email",
                title=f"Email from {sender_name}",
                body=f"Re: {subject[:80]}",
                data={"route": "/email-mode", "grant_id": grant_id},
            )
            logger.info(f"[nylas_webhook] Classified and notified {user_id} for urgent email from {sender_name}")
        else:
            logger.debug(f"[nylas_webhook] Email from {sender_name} classified as no_action or awaiting_response; no push sent")
    except Exception as e:
        logger.error(f"[nylas_webhook] event handling failed: {e}", exc_info=True)


async def _handle_nylas_calendar_event(payload: dict) -> None:
    """Resolve the user from a Nylas calendar event and refresh their world state."""
    try:
        event_type = payload.get("type", "")
        data = payload.get("data", {}).get("object", {})
        grant_id = data.get("grant_id") or payload.get("grant_id")
        if not grant_id:
            return

        db = get_supabase_client()
        token_resp = await asyncio.to_thread(
            lambda: db.client.table("nylas_oauth_tokens")
            .select("user_id")
            .eq("grant_id", grant_id)
            .eq("integration_type", "calendar")
            .maybe_single()
            .execute()
        )
        if not token_resp or not token_resp.data:
            return
        user_id = token_resp.data.get("user_id")
        if not user_id:
            return

        event_title = data.get("title", "(untitled)")
        logger.info(f"[nylas_calendar] {event_type} for user={user_id}: '{event_title}'")

        try:
            from services.world_state import refresh_world_state
            await refresh_world_state(user_id)
        except ImportError:
            pass

    except Exception as e:
        logger.error(f"[nylas_calendar] event handling failed: {e}", exc_info=True)


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
    
    PRODUCTION OPTIMIZATION:
    - Uses schedule hash to detect actual changes (idempotency)
    - Only cancels/recreates if schedule truly changed
    - Prevents unnecessary database churn (was 83% cancellation rate)
    
    **Requires Authentication**: Supabase JWT token in Authorization header
    
    Args:
        user_id: The UUID of the user
        auth: Authenticated user info from JWT token (dependency)
        
    Returns:
        List of scheduled calls (created or unchanged)
    """
    import hashlib
    
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
        
        # Compute hash of current schedule for idempotency
        schedule_data = {
            "schedule": sorted([
                {"day": item.get("day"), "time": item.get("time")}
                for item in checkin_schedule
            ], key=lambda x: (x["day"], x["time"])),
            "timezone": timezone_str,
            "enabled": checkin_enabled
        }
        current_hash = hashlib.md5(
            json.dumps(schedule_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Get stored hash
        stored_hash = settings.get("checkin_schedule_hash")
        
        # IDEMPOTENCY CHECK: If hash matches, schedule hasn't changed
        if stored_hash == current_hash and checkin_enabled:
            logger.info(f"Schedule unchanged for user {user_id} (hash: {current_hash[:8]}...), skipping sync")
            
            # Return existing pending calls
            existing_calls = db.client.table("scheduled_calls").select("*").eq(
                "user_id", user_id
            ).eq("status", "pending").execute()
            
            return {
                "success": True,
                "scheduled_calls": existing_calls.data or [],
                "count": len(existing_calls.data or []),
                "message": "Schedule unchanged, no sync needed"
            }
        
        if not checkin_enabled or not checkin_schedule:
            # Cancel all pending calls if checkin is disabled or no schedule
            db.client.table("scheduled_calls").update({
                "status": "cancelled",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("user_id", user_id).eq("status", "pending").execute()
            
            hash_update = db.client.table("user_settings").update({
                "checkin_schedule_hash": current_hash,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("user_id", user_id).execute()
            if not hash_update.data:
                logger.error(f"Failed to persist checkin_schedule_hash (disabled) for user {user_id}")

            logger.info(f"Cancelled all pending calls for user {user_id} (checkin disabled or no schedule)")
            return {"success": True, "scheduled_calls": [], "count": 0}
        
        # Schedule changed - cancel pending slots no longer in schedule, then upsert current slots
        logger.info(f"Schedule changed for user {user_id} (old: {stored_hash[:8] if stored_hash else 'none'}... new: {current_hash[:8]}...)")

        # Calculate UTC datetimes for all current schedule slots
        from zoneinfo import ZoneInfo
        from datetime import timedelta as _td
        user_tz = ZoneInfo(timezone_str)
        now_local = datetime.now(user_tz)
        target_slots: list[str] = []
        for entry in checkin_schedule:
            day = entry.get("day")
            time_str = entry.get("time")
            if day is None or not time_str:
                continue
            try:
                hour, minute = map(int, time_str.split(":"))
            except (ValueError, AttributeError):
                continue
            target_weekday = 6 if day == 0 else day - 1
            days_ahead = (target_weekday - now_local.weekday()) % 7
            candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0) + _td(days=days_ahead)
            if candidate <= now_local:
                candidate += _td(days=7)
            target_slots.append(candidate.astimezone(ZoneInfo("UTC")).isoformat())

        # Cancel pending records whose slot is no longer in the schedule
        existing_pending = db.client.table("scheduled_calls").select("id, scheduled_for").eq(
            "user_id", user_id
        ).eq("status", "pending").execute()

        def _normalise_iso(s: str) -> str:
            from datetime import timezone as _tz
            dt = datetime.fromisoformat(s.replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_tz.utc)
            return dt.astimezone(_tz.utc).replace(microsecond=0).isoformat()

        normalised_targets = {_normalise_iso(s) for s in target_slots}
        ids_to_cancel = [
            r["id"] for r in (existing_pending.data or [])
            if _normalise_iso(r["scheduled_for"]) not in normalised_targets
        ]
        if ids_to_cancel:
            db.client.table("scheduled_calls").update({
                "status": "cancelled",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).in_("id", ids_to_cancel).execute()
            logger.info(f"Cancelled {len(ids_to_cancel)} removed slots for user {user_id}")

        # Create (or skip if already active) scheduled calls for current slots
        created_calls = await db.create_all_scheduled_calls(
            user_id=user_id,
            checkin_schedule=checkin_schedule,
            timezone=timezone_str,
            checkin_enabled=checkin_enabled
        )

        # Store new hash and log if it failed
        hash_update = db.client.table("user_settings").update({
            "checkin_schedule_hash": current_hash,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("user_id", user_id).execute()
        if not hash_update.data:
            logger.error(f"Failed to persist checkin_schedule_hash for user {user_id} — idempotency check will not work next sync")

        logger.info(f"✅ Synced schedule for user {user_id}: {len(created_calls)} calls created/confirmed")
        
        return {
            "success": True,
            "scheduled_calls": created_calls,
            "count": len(created_calls),
            "message": "Schedule updated successfully"
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


# ==================== Memory Endpoints ====================

from pydantic import BaseModel as PydanticBase

class SaveSessionMemoryRequest(PydanticBase):
    user_id: str
    surface: str
    transcript: list | str
    summary: str = ""
    duration_seconds: int = 0
    session_id: Optional[str] = None


@app.post("/save-session-memory")
async def save_session_memory(
    request_body: SaveSessionMemoryRequest,
    background_tasks: BackgroundTasks,
    req: Request,
):
    """
    Save a session's memory (transcript + extracted facts) after it ends.
    Called by PraxaPanel on voice disconnect and by praxa-chat edge function.
    """
    try:
        from services.memory_service import extract_and_store_session_memory

        background_tasks.add_task(
            extract_and_store_session_memory,
            user_id=request_body.user_id,
            surface=request_body.surface,
            transcript=request_body.transcript,
            summary=request_body.summary,
            duration=request_body.duration_seconds,
            session_id=request_body.session_id,
        )

        return {"status": "queued", "user_id": request_body.user_id}
    except Exception as e:
        logger.error(f"Error queuing session memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to save session memory")


class ExtractSkillsRequest(PydanticBase):
    user_id: str
    surface: str
    messages: list
    session_id: Optional[str] = None


@app.post("/extract-skills")
async def extract_skills(
    request_body: ExtractSkillsRequest,
    background_tasks: BackgroundTasks,
    req: Request,
):
    """
    Extract behavioral skills from a conversation and store them.
    Called fire-and-forget by praxa-chat edge function after substantial conversations.
    """
    try:
        from services.memory_service import extract_skills_from_session

        background_tasks.add_task(
            extract_skills_from_session,
            user_id=request_body.user_id,
            transcript=request_body.messages,
            surface=request_body.surface,
        )

        return {"status": "queued", "user_id": request_body.user_id}
    except Exception as e:
        logger.error(f"Error queuing skill extraction: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract skills")


class ActionOutcomeRequest(PydanticBase):
    action_id: str
    action_type: str
    user_id: str
    outcome: str
    match_key: Optional[str] = None


@app.post("/action-outcome")
async def action_outcome(request_body: ActionOutcomeRequest, req: Request):
    """Called by mobile after a user discards (or confirms) a pending action.

    Inserts a row into action_approval_log so the autonomy learner can detect
    patterns and eventually propose user_autonomy_rules.
    """
    if request_body.outcome not in ("confirmed", "discarded"):
        raise HTTPException(status_code=422, detail="outcome must be 'confirmed' or 'discarded'")
    try:
        db = get_supabase_client()
        await asyncio.to_thread(
            lambda: db.client.table("action_approval_log").insert({
                "user_id": request_body.user_id,
                "action_type": request_body.action_type,
                "match_key": request_body.match_key,
                "outcome": request_body.outcome,
            }).execute()
        )
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"[action-outcome] Failed to log outcome: {e}")
        raise HTTPException(status_code=500, detail="Failed to log outcome")


class AutonomyRulePatchRequest(PydanticBase):
    mode: str


@app.patch("/autonomy-rules/{rule_id}")
async def patch_autonomy_rule(rule_id: str, request_body: AutonomyRulePatchRequest, req: Request):
    """Update the mode of a user_autonomy_rules row (proposed → auto, or auto → confirm)."""
    if request_body.mode not in ("auto", "confirm", "proposed"):
        raise HTTPException(status_code=422, detail="mode must be 'auto', 'confirm', or 'proposed'")
    try:
        db = get_supabase_client()
        await asyncio.to_thread(
            lambda: db.client.table("user_autonomy_rules")
            .update({"mode": request_body.mode})
            .eq("id", rule_id)
            .execute()
        )
        return {"status": "updated", "rule_id": rule_id, "mode": request_body.mode}
    except Exception as e:
        logger.error(f"[autonomy-rules] patch failed for {rule_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update autonomy rule")


@app.delete("/autonomy-rules/{rule_id}")
async def delete_autonomy_rule(rule_id: str, req: Request):
    """Delete a user_autonomy_rules row (user dismisses a proposed rule)."""
    try:
        db = get_supabase_client()
        await asyncio.to_thread(
            lambda: db.client.table("user_autonomy_rules").delete().eq("id", rule_id).execute()
        )
        return {"status": "deleted", "rule_id": rule_id}
    except Exception as e:
        logger.error(f"[autonomy-rules] delete failed for {rule_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete autonomy rule")


class InternalTriggerCallRequest(PydanticBase):
    user_id: str
    reason: Optional[str] = None


@app.post("/internal/trigger-call")
async def internal_trigger_call(request_body: InternalTriggerCallRequest, req: Request):
    """Internal endpoint for the background agent to trigger outbound calls without a user JWT."""
    secret = req.headers.get("X-Internal-Secret", "")
    if not PRAXA_INTERNAL_SECRET or secret != PRAXA_INTERNAL_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    result = await trigger_call_for_user(request_body.user_id, reason=request_body.reason)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to initiate call")
    return {"status": "initiated", **result}


class GenerateSessionBriefRequest(PydanticBase):
    user_id: str
    session_id: Optional[str] = None
    transcript: list


@app.post("/generate-session-brief")
async def generate_session_brief(request_body: GenerateSessionBriefRequest, background_tasks: BackgroundTasks):
    """Generate a structured post-call brief from a voice session transcript and push it to the user."""
    background_tasks.add_task(_do_generate_session_brief, request_body)
    return {"status": "queued"}


async def _do_generate_session_brief(req: GenerateSessionBriefRequest):
    try:
        from openai import AsyncOpenAI

        if len(req.transcript) < 2:
            return

        transcript_text = "\n".join(
            f"{m['speaker'].title()}: {m['text']}" for m in req.transcript
        )

        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Praxa, an AI executive assistant. "
                        "Given a voice session transcript, produce a structured JSON brief. "
                        'Respond ONLY with: {"key_points": [], "decisions": [], "action_items": [], "follow_ups": []}. '
                        "Each list contains plain-text strings. Be concise."
                    ),
                },
                {"role": "user", "content": f"Transcript:\n{transcript_text}"},
            ],
            max_tokens=600,
            temperature=0.2,
        )

        import json as _json
        brief = _json.loads(resp.choices[0].message.content or "{}")

        db = get_supabase_client()
        row = {
            "user_id": req.user_id,
            "brief": brief,
        }
        if req.session_id:
            row["session_id"] = req.session_id

        result = await asyncio.to_thread(
            lambda: db.client.table("session_briefs").insert(row).execute()
        )

        brief_id = None
        if result.data:
            brief_id = result.data[0].get("id")

        if brief_id:
            push_token = await get_user_push_token(req.user_id)
            if push_token:
                action_items = brief.get("action_items", [])
                summary_line = action_items[0] if action_items else "Your session summary is ready."
                await send_push_notification(
                    push_token=push_token,
                    title="Session Brief Ready",
                    body=summary_line[:120],
                    data={
                        "notificationType": "session_brief",
                        "briefId": brief_id,
                    },
                )

        logger.info(f"[session-brief] Brief generated for user {req.user_id}, id={brief_id}")
    except Exception as e:
        logger.error(f"[session-brief] Failed to generate brief: {e}")


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=ENVIRONMENT == "development"
    )

