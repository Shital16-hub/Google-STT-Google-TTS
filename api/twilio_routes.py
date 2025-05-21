# api/twilio_routes.py

"""
Twilio integration routes for voice calls.
"""
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.request_validator import RequestValidator

from core.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter()

def get_settings():
    """Get application settings."""
    return Settings()

class TwilioEvent(BaseModel):
    """Twilio event model."""
    CallSid: str
    AccountSid: str
    From: Optional[str]
    To: Optional[str]
    CallStatus: Optional[str]
    Direction: Optional[str]
    ForwardedFrom: Optional[str]
    CallerName: Optional[str]
    ParentCallSid: Optional[str]

async def validate_twilio_request(
    request: Request,
    settings: Settings = Depends(get_settings)
) -> bool:
    """
    Validate that request came from Twilio.
    
    Args:
        request: FastAPI request
        settings: Application settings
        
    Returns:
        True if request is valid
    """
    validator = RequestValidator(settings.twilio.auth_token)
    
    # Get original Twilio signature
    twilio_signature = request.headers.get("X-Twilio-Signature", "")
    
    # Get full URL
    url = str(request.url)
    
    # Get form data
    form_data = await request.form()
    
    return validator.validate(
        url,
        form_data,
        twilio_signature
    )

@router.post("/incoming")
async def handle_incoming_call(
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """Handle incoming voice calls."""
    # Validate request
    if not await validate_twilio_request(request, settings):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    
    try:
        # Parse form data
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Create WebSocket URL for media stream
        ws_url = f"{settings.base_url}/ws/{call_sid}"
        
        # Set up bi-directional media stream
        connect = Connect()
        stream = Stream(
            name="stream_1",
            url=ws_url,
            track="inbound_track"
        )
        connect.append(stream)
        response.append(connect)
        
        logger.info(f"Initialized call {call_sid} with WebSocket {ws_url}")
        
        return PlainTextResponse(
            content=str(response),
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        
        # Create error response
        error_response = VoiceResponse()
        error_response.say(
            "We're sorry, but we're experiencing technical difficulties. Please try again later.",
            voice="alice"
        )
        error_response.hangup()
        
        return PlainTextResponse(
            content=str(error_response),
            media_type="application/xml"
        )

@router.post("/status")
async def handle_call_status(
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """Handle call status callbacks."""
    # Validate request
    if not await validate_twilio_request(request, settings):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    
    try:
        # Parse form data
        form_data = await request.form()
        event = TwilioEvent(**form_data)
        
        logger.info(f"Call {event.CallSid} status update: {event.CallStatus}")
        
        # Handle different call statuses
        if event.CallStatus == "completed":
            # Clean up resources
            from main import agent_router
            if agent_router:
                agent_router.cleanup_session(event.CallSid)
        
        return Response(status_code=200)
        
    except Exception as e:
        logger.error(f"Error handling call status: {e}")
        return Response(status_code=500)

@router.post("/events")
async def handle_call_events(
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """Handle call events."""
    # Validate request
    if not await validate_twilio_request(request, settings):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    
    try:
        # Parse form data
        form_data = await request.form()
        
        # Log event
        logger.info(f"Call event received: {dict(form_data)}")
        
        return Response(status_code=200)
        
    except Exception as e:
        logger.error(f"Error handling call event: {e}")
        return Response(status_code=500)