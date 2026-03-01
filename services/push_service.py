"""Expo Push Notification service for delivering push notifications to iOS and Android."""

import asyncio
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"
RECEIPT_CHECK_DELAY_SECONDS = 15 * 60  # 15 minutes


async def send_push_notification(
    push_token: str,
    title: str,
    body: str,
    data: Optional[dict] = None,
    sound: str = "default",
    badge: int = 1,
) -> Optional[str]:
    """
    Send a push notification via the Expo Push API.

    Works for both iOS (APNs) and Android (FCM) — Expo routes automatically.

    Args:
        push_token: The Expo push token stored on the device (ExponentPushToken[...])
        title: Notification title
        body: Notification body text
        data: Optional dict of extra data passed to the app on tap
        sound: 'default' or None
        badge: Badge count for iOS

    Returns:
        Expo ticket ID if sent successfully (use for receipt checking), None otherwise
    """
    if not push_token:
        logger.debug("No push token provided, skipping push notification")
        return None

    if not push_token.startswith("ExponentPushToken["):
        logger.warning(f"Invalid push token format: {push_token[:20]}...")
        return None

    payload = {
        "to": push_token,
        "title": title,
        "body": body,
        "sound": sound,
        "badge": badge,
        "data": data or {},
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                EXPO_PUSH_URL,
                json=payload,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            result = response.json()

            ticket = result.get("data", {})
            if ticket.get("status") == "error":
                logger.error(f"Expo push error: {ticket.get('message')} (details: {ticket.get('details')})")
                return None

            ticket_id = ticket.get("id")
            logger.info(f"Push notification sent: '{title}' → {push_token[:30]}... (ticket: {ticket_id})")
            return ticket_id

    except httpx.HTTPStatusError as e:
        logger.error(f"Expo Push API HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Failed to send push notification: {e}")
        return None


async def check_push_receipt(ticket_id: str, user_id: str) -> None:
    """
    Check an Expo push receipt after a delay and handle delivery errors.

    Expo delivers tickets immediately but actual delivery confirmation takes
    a few minutes. This waits 15 minutes then checks the receipt.
    If the device is no longer registered (uninstalled app, etc.), the token
    is removed from the database so we stop sending to it.

    Args:
        ticket_id: The ticket ID returned from send_push_notification
        user_id: The user's UUID (needed to clear invalid tokens)
    """
    await asyncio.sleep(RECEIPT_CHECK_DELAY_SECONDS)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                EXPO_RECEIPTS_URL,
                json={"ids": [ticket_id]},
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        receipt = data.get("data", {}).get(ticket_id, {})

        if receipt.get("status") == "error":
            error_type = receipt.get("details", {}).get("error", "Unknown")
            logger.warning(f"Push receipt error for user {user_id}: {error_type}")

            if error_type == "DeviceNotRegistered":
                await clear_push_token(user_id)

    except Exception as e:
        logger.error(f"Failed to check push receipt {ticket_id} for user {user_id}: {e}")


def schedule_receipt_check(ticket_id: str, user_id: str) -> None:
    """
    Fire-and-forget: schedule a receipt check 15 minutes after sending a push.
    Safe to call from any async context (FastAPI route or APScheduler job).

    Args:
        ticket_id: The ticket ID from send_push_notification
        user_id: The user's UUID
    """
    try:
        asyncio.create_task(check_push_receipt(ticket_id, user_id))
    except RuntimeError:
        logger.warning(f"Could not schedule receipt check for ticket {ticket_id} — no running event loop")


async def get_user_push_token(user_id: str) -> Optional[str]:
    """
    Fetch the push token for a user from user_settings.

    Args:
        user_id: The user's UUID

    Returns:
        The push token string, or None if not set
    """
    from .supabase_client import get_supabase_client
    try:
        db = get_supabase_client()
        response = db.client.table("user_settings").select("push_token").eq("user_id", user_id).execute()
        if response.data and response.data[0].get("push_token"):
            return response.data[0]["push_token"]
        return None
    except Exception as e:
        logger.error(f"Failed to fetch push token for user {user_id}: {e}")
        return None


async def clear_push_token(user_id: str) -> None:
    """
    Remove the push token for a user from user_settings.
    Called when a receipt confirms the token is no longer valid
    (e.g. app was uninstalled).

    Args:
        user_id: The user's UUID
    """
    from .supabase_client import get_supabase_client
    try:
        db = get_supabase_client()
        db.client.table("user_settings").update({"push_token": None}).eq("user_id", user_id).execute()
        logger.info(f"Cleared invalid push token for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to clear push token for user {user_id}: {e}")
