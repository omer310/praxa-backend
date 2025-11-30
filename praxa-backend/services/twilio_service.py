"""Twilio service for initiating outbound phone calls."""

import os
import logging
from typing import Optional

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

logger = logging.getLogger(__name__)


class TwilioService:
    """Service for making outbound calls via Twilio connected to LiveKit."""

    def __init__(self):
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([account_sid, auth_token, self.from_number]):
            raise ValueError(
                "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER must be set"
            )
        
        self.client = Client(account_sid, auth_token)
        self.livekit_url = os.getenv("LIVEKIT_URL", "").replace("wss://", "").replace("ws://", "")

    async def initiate_call(
        self,
        to_number: str,
        livekit_room_name: str,
        status_callback_url: Optional[str] = None
    ) -> str:
        """
        Initiate an outbound call that connects to a LiveKit room.
        
        This uses Twilio's SIP integration with LiveKit to bridge the phone call
        audio into the LiveKit room where the AI agent is waiting.
        
        Args:
            to_number: The phone number to call (E.164 format: +1234567890)
            livekit_room_name: The LiveKit room name for this call
            status_callback_url: Optional webhook URL for call status updates
            
        Returns:
            The Twilio Call SID
        """
        try:
            # Construct the SIP URI for LiveKit
            # Format: sip:room_name@livekit_host
            livekit_sip_uri = f"sip:{livekit_room_name}@{self.livekit_url}"
            
            # TwiML to connect the call to LiveKit via SIP
            # When answered, it will connect the caller to the LiveKit room
            twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Sip>{livekit_sip_uri}</Sip>
    </Dial>
</Response>'''

            # Create the outbound call
            call_params = {
                "to": to_number,
                "from_": self.from_number,
                "twiml": twiml,
            }
            
            if status_callback_url:
                call_params["status_callback"] = status_callback_url
                call_params["status_callback_event"] = [
                    "initiated", "ringing", "answered", "completed"
                ]
                call_params["status_callback_method"] = "POST"
            
            call = self.client.calls.create(**call_params)
            
            logger.info(f"Initiated call {call.sid} to {to_number}")
            return call.sid
            
        except TwilioRestException as e:
            logger.error(f"Twilio error initiating call: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initiating call: {e}")
            raise

    async def get_call_status(self, call_sid: str) -> dict:
        """
        Get the current status of a call.
        
        Args:
            call_sid: The Twilio Call SID
            
        Returns:
            Dict with call status information
        """
        try:
            call = self.client.calls(call_sid).fetch()
            
            return {
                "sid": call.sid,
                "status": call.status,
                "to": call.to,
                "from_": call.from_,
                "direction": call.direction,
                "duration": call.duration,
                "start_time": call.start_time.isoformat() if call.start_time else None,
                "end_time": call.end_time.isoformat() if call.end_time else None,
            }
        except TwilioRestException as e:
            logger.error(f"Twilio error fetching call status: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching call status: {e}")
            raise

    async def end_call(self, call_sid: str) -> bool:
        """
        End an active call.
        
        Args:
            call_sid: The Twilio Call SID
            
        Returns:
            True if call was ended successfully
        """
        try:
            self.client.calls(call_sid).update(status="completed")
            logger.info(f"Ended call {call_sid}")
            return True
        except TwilioRestException as e:
            logger.error(f"Twilio error ending call: {e}")
            return False
        except Exception as e:
            logger.error(f"Error ending call: {e}")
            return False


# Singleton instance
_service: Optional[TwilioService] = None


def get_twilio_service() -> TwilioService:
    """Get or create the Twilio service singleton."""
    global _service
    if _service is None:
        _service = TwilioService()
    return _service

