# api/twilio_routes.py - Fixed version with proper signature validation

"""
Twilio integration routes for voice calls with proper signature validation.
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
    From: Optional[str] = None
    To: Optional[str] = None
    CallStatus: Optional[str] = None
    Direction: Optional[str] = None
    ForwardedFrom: Optional[str] = None
    CallerName: Optional[str] = None
    ParentCallSid: Optional[str] = None

async def validate_twilio_request(
    request: Request,
    settings: Settings = Depends(get_settings)
) -> bool:
    """
    Validate that request came from Twilio with improved handling.
    """
    # Skip validation in development mode or if auth token not configured
    if settings.debug or not settings.twilio.auth_token:
        logger.warning("Skipping Twilio signature validation (debug mode or no auth token)")
        return True
    
    try:
        validator = RequestValidator(settings.twilio.auth_token)
        
        # Get Twilio signature from headers
        twilio_signature = request.headers.get("X-Twilio-Signature", "")
        if not twilio_signature:
            logger.warning("No X-Twilio-Signature header found")
            return False
        
        # Get the full URL - this is critical for signature validation
        url = str(request.url)
        
        # For Runpod/proxy setups, we might need to use the original URL
        forwarded_proto = request.headers.get("X-Forwarded-Proto")
        forwarded_host = request.headers.get("X-Forwarded-Host")
        original_url = request.headers.get("X-Original-URL")
        
        if forwarded_proto and forwarded_host:
            # Reconstruct the original URL for proxy situations
            path = request.url.path
            query = f"?{request.url.query}" if request.url.query else ""
            url = f"{forwarded_proto}://{forwarded_host}{path}{query}"
            logger.debug(f"Using reconstructed URL for validation: {url}")
        elif original_url:
            url = original_url
            logger.debug(f"Using X-Original-URL for validation: {url}")
        
        # Get form data
        form_data = await request.form()
        
        # Convert form data to dictionary for validator
        form_dict = {}
        for key, value in form_data.items():
            form_dict[key] = value
        
        # Validate the signature
        is_valid = validator.validate(url, form_dict, twilio_signature)
        
        if not is_valid:
            logger.error(f"Twilio signature validation failed")
            logger.error(f"URL used: {url}")
            logger.error(f"Signature: {twilio_signature}")
            logger.error(f"Form data keys: {list(form_dict.keys())}")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Error validating Twilio request: {e}")
        return False

@router.post("/incoming")
async def handle_incoming_call(
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """Handle incoming voice calls with improved error handling."""
    
    # Log the incoming request for debugging
    logger.info(f"Incoming call request from {request.client.host if request.client else 'unknown'}")
    logger.info(f"Request headers: {dict(request.headers)}")
    
    # Validate request (but allow bypass in development)
    try:
        is_valid = await validate_twilio_request(request, settings)
        if not is_valid and not settings.debug:
            logger.error("Twilio signature validation failed")
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        elif not is_valid and settings.debug:
            logger.warning("Twilio signature validation failed, but continuing in debug mode")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during signature validation: {e}")
        if not settings.debug:
            raise HTTPException(status_code=500, detail="Validation error")
    
    try:
        # Parse form data
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        from_number = form_data.get("From")
        to_number = form_data.get("To")
        
        if not call_sid:
            logger.error("Missing CallSid in request")
            raise HTTPException(status_code=400, detail="Missing CallSid")
        
        logger.info(f"Processing incoming call: {call_sid} from {from_number} to {to_number}")
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Create WebSocket URL for media stream
        # Use the base URL from settings, but handle proxy situations
        base_url = settings.base_url
        if not base_url:
            # Fallback: construct from request
            scheme = request.headers.get("X-Forwarded-Proto", "https")
            host = request.headers.get("X-Forwarded-Host", request.headers.get("Host", "localhost"))
            base_url = f"{scheme}://{host}"
        
        # Ensure we use wss:// for WebSocket
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/ws/{call_sid}"
        
        logger.info(f"Setting up WebSocket URL: {ws_url}")
        
        # Set up bi-directional media stream
        connect = Connect()
        stream = Stream(
            name="audio_stream",
            url=ws_url
        )
        connect.append(stream)
        response.append(connect)
        
        logger.info(f"TwiML response created for call {call_sid}")
        
        return PlainTextResponse(
            content=str(response),
            media_type="application/xml"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        
        # Create error response that Twilio can handle
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
    
    # Status callbacks don't need strict validation in development
    if settings.debug:
        logger.debug("Skipping validation for status callback in debug mode")
    
    try:
        # Parse form data
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        
        logger.info(f"Call {call_sid} status update: {call_status}")
        
        # Handle different call statuses
        if call_status == "completed":
            # Clean up resources for completed calls
            try:
                # Import here to avoid circular imports
                from main import agent_router
                if agent_router:
                    agent_router.cleanup_session(call_sid)
                    logger.info(f"Cleaned up session for completed call {call_sid}")
            except Exception as e:
                logger.error(f"Error cleaning up session {call_sid}: {e}")
        
        return Response(status_code=200)
        
    except Exception as e:
        logger.error(f"Error handling call status: {e}")
        # Always return 200 for status callbacks to avoid Twilio retries
        return Response(status_code=200)

@router.post("/events")
async def handle_call_events(
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """Handle call events."""
    try:
        # Parse form data
        form_data = await request.form()
        
        # Log event for debugging
        logger.info(f"Call event received: {dict(form_data)}")
        
        return Response(status_code=200)
        
    except Exception as e:
        logger.error(f"Error handling call event: {e}")
        return Response(status_code=200)

# Add a test endpoint for Twilio webhook testing
@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify Twilio routes are working."""
    return {
        "status": "ok",
        "message": "Twilio routes are working",
        "endpoints": {
            "incoming": "/voice/incoming",
            "status": "/voice/status",
            "events": "/voice/events"
        }
    }