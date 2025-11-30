"""
Twilio/LiveKit SIP service for initiating outbound phone calls.

NOTE: For LiveKit Agents, outbound calls are made through LiveKit's SIP infrastructure,
not by calling Twilio directly. You need to:
1. Configure a SIP Trunk in LiveKit Cloud connected to Twilio
2. Use the LiveKit SIP Participant API to dial out

See: https://docs.livekit.io/agents/quickstarts/outbound-calls/
"""

import os
import logging
from typing import Optional

from livekit import api as livekit_api

logger = logging.getLogger(__name__)


class SIPCallService:
    """Service for making outbound calls via LiveKit SIP (connected to Twilio)."""

    def __init__(self):
        self.livekit_url = os.getenv("LIVEKIT_URL", "")
        self.livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        self.livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
        self.sip_trunk_id = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")
        
        if not all([self.livekit_url, self.livekit_api_key, self.livekit_api_secret]):
            raise ValueError(
                "LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set"
            )
        
        self.lk_api = livekit_api.LiveKitAPI(
            self.livekit_url,
            self.livekit_api_key,
            self.livekit_api_secret
        )

    async def create_room(self, room_name: str, user_id: str, call_log_id: str) -> bool:
        """
        Create a LiveKit room for the call.
        
        Args:
            room_name: The name for the room
            user_id: The user ID for metadata
            call_log_id: The call log ID for metadata
            
        Returns:
            True if room created successfully
        """
        try:
            await self.lk_api.room.create_room(
                livekit_api.CreateRoomRequest(
                    name=room_name,
                    empty_timeout=300,  # 5 minutes
                    max_participants=3,
                    metadata=f'{{"user_id": "{user_id}", "call_log_id": "{call_log_id}"}}'
                )
            )
            logger.info(f"Created LiveKit room: {room_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create LiveKit room: {e}")
            return False

    async def dial_phone(
        self,
        room_name: str,
        to_number: str,
        participant_identity: str = "phone-user"
    ) -> Optional[str]:
        """
        Dial a phone number and connect them to a LiveKit room via SIP.
        
        This uses LiveKit's SIP Participant API to initiate an outbound call.
        Requires a SIP trunk configured in LiveKit Cloud.
        
        Args:
            room_name: The LiveKit room to connect the call to
            to_number: The phone number to call (E.164 format: +1234567890)
            participant_identity: Identity for the phone participant
            
        Returns:
            The SIP participant ID if successful, None if failed
        """
        try:
            # Use LiveKit SIP API to dial out
            # This requires LIVEKIT_SIP_TRUNK_ID to be configured
            if not self.sip_trunk_id:
                logger.error("LIVEKIT_SIP_TRUNK_ID not configured - cannot make outbound calls")
                logger.error("Please configure a SIP trunk in LiveKit Cloud: https://cloud.livekit.io")
                return None
            
            response = await self.lk_api.sip.create_sip_participant(
                livekit_api.CreateSIPParticipantRequest(
                    sip_trunk_id=self.sip_trunk_id,
                    sip_call_to=to_number,
                    room_name=room_name,
                    participant_identity=participant_identity,
                    participant_name="Phone User",
                    play_ringtone=True,
                )
            )
            
            logger.info(f"Initiated SIP call to {to_number} in room {room_name}")
            return response.participant_id if response else None
            
        except Exception as e:
            logger.error(f"Failed to dial phone via SIP: {e}")
            raise

    async def close(self):
        """Close the LiveKit API client."""
        await self.lk_api.aclose()


# Keep backward-compatible function names
class TwilioService(SIPCallService):
    """Alias for SIPCallService for backward compatibility."""
    
    async def initiate_call(
        self,
        to_number: str,
        livekit_room_name: str,
        status_callback_url: Optional[str] = None
    ) -> str:
        """
        Backward-compatible method that uses LiveKit SIP instead of Twilio directly.
        
        Args:
            to_number: The phone number to call
            livekit_room_name: The room name
            status_callback_url: Ignored (LiveKit handles callbacks differently)
            
        Returns:
            SIP participant ID (used as call identifier)
        """
        result = await self.dial_phone(
            room_name=livekit_room_name,
            to_number=to_number
        )
        return result or ""


# Singleton instance
_service: Optional[TwilioService] = None


def get_twilio_service() -> TwilioService:
    """Get or create the Twilio/SIP service singleton."""
    global _service
    if _service is None:
        _service = TwilioService()
    return _service
