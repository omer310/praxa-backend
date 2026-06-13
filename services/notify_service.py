"""Unified notification delivery router.

A single function `notify_user(user_id, event_type, title, body, data)` that:
1. Loads the user's preferences from `user_settings` (notification_preferences JSONB,
   sms_notifications_opt_in, calls_enabled, push_token, phone_number, timezone).
2. Checks quiet hours in the user's local timezone.
3. Sends push, SMS, or both according to their preference for the event type.

All proactive notification code (initiative_loop, briefing, scheduler, webhooks)
calls this instead of directly invoking send_push_notification or send_sms.
"""

import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional

logger = logging.getLogger(__name__)


# Event types the routing table recognises.
EVENT_TYPES = frozenset({
    "urgent_email",
    "task_due",
    "pending_action",
    "follow_up_needed",
    "draft_ready",
    "general",
})

# Default channel when user has no explicit preference for an event type.
_DEFAULT_CHANNEL = "push"


def _to_local_hour(tz_name: str) -> int:
    """Return the current local hour (0–23) in the user's timezone."""
    try:
        tz = ZoneInfo(tz_name)
        return datetime.now(tz).hour
    except (ZoneInfoNotFoundError, Exception):
        return datetime.now(timezone.utc).hour


def _is_quiet(hour: int, quiet_start: int, quiet_end: int) -> bool:
    """Return True if `hour` falls inside the quiet window.

    Handles overnight windows (e.g. 22–7) correctly.
    """
    if quiet_start == quiet_end:
        return False
    if quiet_start < quiet_end:
        return quiet_start <= hour < quiet_end
    # Overnight window
    return hour >= quiet_start or hour < quiet_end


async def _load_user_settings(user_id: str) -> dict:
    """Fetch notification-relevant columns from user_settings."""
    from .supabase_client import get_supabase_client
    try:
        db = get_supabase_client()
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_settings")
            .select(
                "notification_preferences, sms_notifications_opt_in, "
                "push_token, phone_number, timezone, calls_enabled"
            )
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        return resp.data or {}
    except Exception as e:
        logger.warning(f"[notify] Could not load user settings for {user_id}: {e}")
        return {}


def _resolve_channel(settings: dict, event_type: str) -> str:
    """Determine the notification channel ('push', 'sms', 'both', 'none').

    Priority order:
    1. event-type-specific override in notification_preferences JSONB
    2. Top-level smart_delivery rules (push_enabled, sms enabled via opt-in)
    3. Fallback default
    """
    prefs = settings.get("notification_preferences") or {}

    # Per-event-type override: {event_types: {urgent_email: "sms", ...}}
    event_overrides: dict = prefs.get("event_types") or {}
    if event_type in event_overrides:
        return event_overrides[event_type]

    push_ok = prefs.get("push_enabled", True) and bool(settings.get("push_token"))
    sms_ok = bool(settings.get("sms_notifications_opt_in")) and bool(settings.get("phone_number"))

    if push_ok and sms_ok:
        return "both"
    if sms_ok:
        return "sms"
    if push_ok:
        return "push"
    return "none"


def _should_suppress_for_quiet(settings: dict, event_type: str) -> bool:
    """Return True if the current local time is in the user's quiet hours."""
    prefs = settings.get("notification_preferences") or {}
    smart = prefs.get("smart_delivery") or {}

    quiet_start: int = smart.get("quiet_hours_start", 22)
    quiet_end: int = smart.get("quiet_hours_end", 7)

    # Check per-channel suppression for SMS (push may still go through)
    respect_quiet = smart.get("respect_quiet_for_sms_email", True)
    if not respect_quiet:
        return False

    tz_name = settings.get("timezone") or "UTC"
    current_hour = _to_local_hour(tz_name)
    return _is_quiet(current_hour, quiet_start, quiet_end)


async def notify_user(
    user_id: str,
    event_type: str,
    title: str,
    body: str,
    data: Optional[dict] = None,
    *,
    bypass_quiet_hours: bool = False,
) -> dict:
    """Send a notification to a user via the appropriate channel(s).

    Args:
        user_id: Supabase auth user ID.
        event_type: One of the EVENT_TYPES (e.g. 'urgent_email', 'draft_ready').
        title: Short notification title.
        body: Notification body text.
        data: Optional payload dict forwarded to the app on tap.
        bypass_quiet_hours: If True, ignore quiet-hours suppression.

    Returns:
        Dict with keys 'push_sent', 'sms_sent', 'suppressed'.
    """
    from .push_service import send_push_notification, schedule_receipt_check
    from .twilio_sms import send_sms

    result = {"push_sent": False, "sms_sent": False, "suppressed": False}

    settings = await _load_user_settings(user_id)
    if not settings:
        logger.debug(f"[notify] No settings for user {user_id} — skipping")
        return result

    channel = _resolve_channel(settings, event_type)
    if channel == "none":
        logger.debug(f"[notify] Channel=none for user {user_id} event={event_type}")
        return result

    in_quiet = (not bypass_quiet_hours) and _should_suppress_for_quiet(settings, event_type)

    want_push = channel in ("push", "both")
    want_sms = channel in ("sms", "both")

    # Push is not suppressed by quiet hours (silent push or badge only)
    if want_push:
        push_token = settings.get("push_token")
        if push_token:
            ticket_id = await send_push_notification(
                push_token=push_token,
                title=title,
                body=body,
                data={"event_type": event_type, **(data or {})},
            )
            if ticket_id:
                schedule_receipt_check(ticket_id, user_id)
                result["push_sent"] = True

    # SMS is suppressed during quiet hours
    if want_sms:
        if in_quiet:
            result["suppressed"] = True
            logger.info(f"[notify] SMS suppressed (quiet hours) for user {user_id} event={event_type}")
        else:
            phone = settings.get("phone_number")
            if phone:
                sms_body = f"{title}: {body}" if title and body else (title or body)
                sid = await send_sms(phone, sms_body[:1500])
                if sid:
                    result["sms_sent"] = True

    logger.info(
        f"[notify] user={user_id} event={event_type} channel={channel} "
        f"push={result['push_sent']} sms={result['sms_sent']} suppressed={result['suppressed']}"
    )
    return result
