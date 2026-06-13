"""Outbound SMS via Twilio REST (the `twilio` dependency is already present).

Used by the SMS agent to reply asynchronously (the inbound webhook now acks
immediately) and by proactive notifications. Requires:
  - TWILIO_ACCOUNT_SID
  - TWILIO_AUTH_TOKEN
  - TWILIO_MESSAGING_SERVICE_SID  (preferred)  OR  TWILIO_FROM_NUMBER
"""
import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _client():
    from twilio.rest import Client

    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        raise ValueError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set")
    return Client(sid, token)


def _send_sync(to_number: str, body: str) -> Optional[str]:
    client = _client()
    messaging_service_sid = os.getenv("TWILIO_MESSAGING_SERVICE_SID")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    kwargs = {"to": to_number, "body": body[:1500]}
    if messaging_service_sid:
        kwargs["messaging_service_sid"] = messaging_service_sid
    elif from_number:
        kwargs["from_"] = from_number
    else:
        raise ValueError("Set TWILIO_MESSAGING_SERVICE_SID or TWILIO_FROM_NUMBER")

    msg = client.messages.create(**kwargs)
    return msg.sid


async def send_sms(to_number: str, body: str) -> Optional[str]:
    """Send an SMS. Returns the message SID, or None on failure."""
    if not to_number or not body:
        return None
    try:
        sid = await asyncio.to_thread(_send_sync, to_number, body)
        logger.info(f"[twilio_sms] sent sid={sid} to={to_number[-4:].rjust(len(to_number), '*')}")
        return sid
    except Exception as e:
        logger.error(f"[twilio_sms] send failed: {e}")
        return None
