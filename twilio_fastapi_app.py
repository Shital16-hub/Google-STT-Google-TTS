# twilio_fastapi_app.py

"""
Updated FastAPI application with optimized STT and speaking state management
for low-latency voice interactions.
"""
import os
import asyncio
import logging
import json
import time
import threading
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Twilio imports
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from dotenv import load_dotenv

# Import our optimized components
from telephony.simple_websocket_handler import SimpleWebSocketHandler
from telephony.config import HOST, PORT, DEBUG
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from integration.tts_integration import TTSIntegration

# Load environment variables
load_dotenv()

# Configure logging with timestamp and thread ID for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)

# Global instances
voice_ai_pipeline = None
base_url = None

# Track active calls with session management
active_calls = {}
call_sessions = {}  # Track call session metadata

# Initialize event for signaling system initialization
initialization_complete = asyncio.Event()

# Request models
class CallStatusModel(BaseModel):
    CallSid: str
    CallStatus: str
    CallDuration: Optional[str] = "0"
    From: Optional[str] = None
    To: Optional[str] = None

class TwilioIncomingCallModel(BaseModel):
    From: str
    To: str
    CallSid: str

async def initialize_system():
    """Initialize the Voice AI system with optimized settings for low latency."""
    global voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI Agent with low-latency optimizations...")
    
    # Validate required environment variables
    base_url = os.getenv('BASE_URL')
    if not base_url:
        raise ValueError("BASE_URL environment variable must be set")
    
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not google_creds:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set, using default credentials")
    elif not os.path.exists(google_creds):
        logger.error(f"Google credentials file not found: {google_creds}")
        raise FileNotFoundError(f"Credentials file not found: {google_creds}")
    
    logger.info(f"Using BASE_URL: {base_url}")
    
    # Initialize Voice AI Agent with low-latency optimizations
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='gpt-3.5-turbo',
        llm_temperature=0.7,
        credentials_file=google_creds
    )
    await agent.init()
    
    # Initialize TTS
    tts = TTSIntegration(
        voice_name="en-US-Neural2-C",
        voice_gender=None,
        language_code="en-US",
        enable_caching=True,
        credentials_file=google_creds
    )
    await tts.init()
    
    # Create optimized pipeline
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    logger.info("System initialized with low-latency optimizations")
    # Set the initialization event
    initialization_complete.set()

# twilio_fastapi_app.py (continued)

async def shutdown_cleanup():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Voice AI Agent API")
    
    # Clean up active calls
    for call_sid, handler in list(active_calls.items()):
        try:
            await handler._cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up call {call_sid}: {e}")
    
    active_calls.clear()
    # We'll keep call_sessions for logging purposes

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Initialize the system
    try:
        # Start initialization in background
        asyncio.create_task(initialize_system())
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        # Don't raise here to let the server start anyway
        # The health endpoint will report the initialization status
    
    # Yield control back to FastAPI
    yield
    
    # Shutdown: Clean up resources
    await shutdown_cleanup()

# FastAPI app setup with lifespan
app = FastAPI(
    title="Voice AI Agent API",
    description="Voice AI Agent with low-latency optimizations",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Voice AI Agent running with low-latency optimizations",
        "version": "3.0.0",
        "active_calls": len(active_calls),
        "initialized": initialization_complete.is_set()
    }

@app.get("/health")
async def health_check():
    """Detailed health check with conversation metrics."""
    initialized = initialization_complete.is_set()
    
    health_status = {
        "status": "healthy" if initialized else "initializing",
        "timestamp": time.time(),
        "components": {
            "pipeline": voice_ai_pipeline is not None,
            "active_calls": len(active_calls),
            "active_sessions": len(call_sessions)
        }
    }
    
    if voice_ai_pipeline:
        health_status["components"].update({
            "stt": hasattr(voice_ai_pipeline, 'speech_recognizer'),
            "tts": hasattr(voice_ai_pipeline, 'tts_integration'),
            "kb": hasattr(voice_ai_pipeline, 'query_engine')
        })
    
    # Add session health info
    if call_sessions:
        session_ages = [time.time() - session['start_time'] for session in call_sessions.values()]
        health_status["session_metrics"] = {
            "oldest_session_age": max(session_ages),
            "average_session_age": sum(session_ages) / len(session_ages),
            "sessions_over_5min": len([age for age in session_ages if age > 300])
        }
    
    return health_status

@app.post("/voice/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming voice calls with optimized TwiML for low-latency."""
    logger.info("Received incoming call request")
    
    # Wait for initialization to complete with timeout
    try:
        initialization_timeout = 10  # seconds
        await asyncio.wait_for(initialization_complete.wait(), timeout=initialization_timeout)
    except asyncio.TimeoutError:
        logger.error(f"System initialization timed out after {initialization_timeout} seconds")
        return HTMLResponse(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">System is still initializing. Please try again later.</Say>
    <Hangup/>
</Response>''',
            media_type="text/xml"
        )
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        return HTMLResponse(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">System is not initialized. Please try again later.</Say>
    <Hangup/>
</Response>''',
            media_type="text/xml"
        )
    
    # Use form data for Twilio compatibility
    form_data = await request.form()
    from_number = form_data.get('From')
    to_number = form_data.get('To')
    call_sid = form_data.get('CallSid')
    
    logger.info(f"Incoming call - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
    
    # Track call session
    call_sessions[call_sid] = {
        "start_time": time.time(),
        "from_number": from_number,
        "to_number": to_number,
        "status": "initiated"
    }
    
    try:
        # Validate base_url
        if not base_url:
            raise ValueError("BASE_URL not configured")
        
        # Create TwiML response optimized for continuous conversation
        response = VoiceResponse()
        
        # Create WebSocket URL
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        logger.info(f"WebSocket URL: {ws_url}")
        
        # Create Connect with Stream - optimized for bidirectional conversation
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"  # Capture inbound audio for processing
        )
        
        connect.append(stream)
        response.append(connect)
        
        logger.info(f"Generated TwiML for low-latency conversation - Call {call_sid}")
        return HTMLResponse(
            content=str(response),
            media_type="text/xml"
        )
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        # Update session status
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "error"
            call_sessions[call_sid]["error"] = str(e)
        
        return HTMLResponse(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">An error occurred. Please try again later.</Say>
    <Hangup/>
</Response>''',
            media_type="text/xml"
        )

@app.post("/voice/status")
async def handle_status_callback(request: Request, background_tasks: BackgroundTasks):
    """Handle call status callbacks with session tracking."""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    call_status = form_data.get('CallStatus')
    call_duration = form_data.get('CallDuration', '0')
    from_number = form_data.get('From')
    to_number = form_data.get('To')
    
    logger.info(f"Call {call_sid} status: {call_status}, duration: {call_duration}s")
    
    # Update session info
    if call_sid in call_sessions:
        call_sessions[call_sid].update({
            "status": call_status,
            "duration": call_duration,
            "end_time": time.time() if call_status in ['completed', 'failed', 'busy', 'no-answer'] else None
        })
    
    # Clean up completed calls
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        if call_sid in active_calls:
            handler = active_calls[call_sid]
            # Trigger cleanup with session preservation logic in background
            background_tasks.add_task(cleanup_call, call_sid, handler)
        
        # Keep session info for a while for debugging
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "completed"
            # Clean up old sessions (older than 1 hour)
            background_tasks.add_task(cleanup_sessions)
    
    return Response(status_code=204)

async def cleanup_call(call_sid: str, handler):
    """Clean up call resources."""
    try:
        # Trigger cleanup with session preservation logic
        await handler._cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    # Remove from active calls
    if call_sid in active_calls:
        del active_calls[call_sid]
        logger.info(f"Cleaned up call {call_sid}")

async def cleanup_sessions():
    """Clean up old call sessions to prevent memory leaks."""
    current_time = time.time()
    sessions_to_remove = []
    
    for call_sid, session in call_sessions.items():
        # Remove sessions older than 1 hour
        if current_time - session['start_time'] > 3600:
            sessions_to_remove.append(call_sid)
    
    for call_sid in sessions_to_remove:
        del call_sessions[call_sid]
        logger.debug(f"Cleaned up old session: {call_sid}")

@app.websocket("/ws/stream/{call_sid}")
async def handle_media_stream(websocket: WebSocket, call_sid: str):
    """Handle WebSocket media stream with optimized binary data transfer."""
    logger.info(f"WebSocket connection request for call {call_sid}")
    
    # Wait for initialization to complete with timeout
    try:
        initialization_timeout = 5  # seconds
        await asyncio.wait_for(initialization_complete.wait(), timeout=initialization_timeout)
    except asyncio.TimeoutError:
        logger.error(f"System initialization timed out after {initialization_timeout} seconds")
        await websocket.close(code=1013, reason="Service unavailable - still initializing")
        return
    
    if not voice_ai_pipeline:
        logger.error("System not initialized for WebSocket connection")
        await websocket.close(code=1013, reason="Service unavailable - not initialized")
        return
    
    handler = None
    
    try:
        # Accept the WebSocket connection with binary mode
        await websocket.accept()
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Create handler optimized for low-latency conversation
        handler = SimpleWebSocketHandler(call_sid, voice_ai_pipeline)
        active_calls[call_sid] = handler
        
        # Update session status
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "connected"
            call_sessions[call_sid]["ws_connected_time"] = time.time()
        
        # Process incoming messages
        while True:
            try:
                # Check for both text and binary messages with timeout
                message_type = "text"
                try:
                    # Wait for either text or binary with timeout
                    done, pending = await asyncio.wait([
                        asyncio.create_task(websocket.receive_text()),
                        asyncio.create_task(websocket.receive_bytes())
                    ], return_when=asyncio.FIRST_COMPLETED, timeout=30.0)
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                    
                    if not done:
                        # Timeout occurred
                        logger.debug("WebSocket receive timeout, checking connection status")
                        continue
                    
                    # Get the result from the completed task
                    message_task = list(done)[0]
                    try:
                        message = message_task.result()
                        # Determine message type
                        if isinstance(message, bytes):
                            message_type = "binary"
                    except Exception as e:
                        logger.error(f"Error getting message result: {e}")
                        continue
                    
                except asyncio.TimeoutError:
                    # Check if we should still be connected
                    logger.debug("WebSocket receive timeout, checking connection status")
                    continue
                
                if not message:
                    logger.debug("Received empty message, continuing...")
                    continue
                
                # Parse and handle message
                if message_type == "binary":
                    # Handle binary message (client-side optimization)
                    # For now, we don't expect binary from client, but could implement in future
                    logger.debug("Received binary message from client")
                    continue
                else:
                    # Handle text message (standard Twilio format)
                    try:
                        data = json.loads(message)
                        event_type = data.get('event')
                        
                        if event_type == 'connected':
                            logger.info(f"WebSocket connected for call {call_sid}")
                            
                        elif event_type == 'start':
                            stream_sid = data.get('streamSid')
                            logger.info(f"Stream started: {stream_sid}")
                            handler.stream_sid = stream_sid
                            
                            # Start the conversation with optimized welcome
                            await handler.start_conversation(websocket)
                            
                            # Update session
                            if call_sid in call_sessions:
                                call_sessions[call_sid]["stream_started"] = True
                                call_sessions[call_sid]["stream_sid"] = stream_sid
                            
                        elif event_type == 'media':
                            # Handle audio data with optimized processing
                            await handler._handle_audio(data, websocket)
                            
                        elif event_type == 'stop':
                            logger.info(f"Stream stopped for call {call_sid}")
                            
                            # Run cleanup
                            await handler._cleanup()
                            
                            # Update session
                            if call_sid in call_sessions:
                                call_sessions[call_sid]["stream_stopped"] = True
                            break
                            
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket connection closed for call {call_sid}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break
        
    except Exception as e:
        logger.error(f"Error establishing WebSocket: {e}", exc_info=True)
        
    finally:
        # Cleanup resources
        if handler:
            try:
                await handler._cleanup()
            except Exception as e:
                logger.error(f"Error during handler cleanup: {e}")
        
        if call_sid in active_calls:
            del active_calls[call_sid]
        
        # Update session
        if call_sid in call_sessions:
            call_sessions[call_sid]["ws_disconnected_time"] = time.time()
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info(f"WebSocket cleanup complete for call {call_sid}")

@app.get("/stats")
async def get_stats():
    """Get comprehensive statistics including latency metrics."""
    stats = {
        "timestamp": time.time(),
        "system": {
            "initialized": initialization_complete.is_set(),
            "active_calls": len(active_calls),
            "total_sessions": len(call_sessions),
            "base_url": base_url
        },
        "calls": {},
        "sessions": {}
    }
    
    # Add individual call stats
    for call_sid, handler in active_calls.items():
        try:
            stats["calls"][call_sid] = handler.get_stats()
        except Exception as e:
            logger.error(f"Error getting stats for call {call_sid}: {e}")
            stats["calls"][call_sid] = {"error": str(e)}
    
    # Add session information
    for call_sid, session in call_sessions.items():
        stats["sessions"][call_sid] = {
            **session,
            "age": time.time() - session['start_time']
        }
    
    # Add conversation metrics
    if active_calls:
        total_transcriptions = sum(handler.transcriptions for handler in active_calls.values())
        total_responses = sum(handler.responses_sent for handler in active_calls.values())
        stats["conversation_metrics"] = {
            "total_transcriptions": total_transcriptions,
            "total_responses": total_responses,
            "average_transcriptions_per_call": total_transcriptions / len(active_calls),
            "average_responses_per_call": total_responses / len(active_calls)
        }
    
    return stats

@app.get("/config")
async def get_config():
    """Get current configuration."""
    config = {
        "host": HOST,
        "port": PORT,
        "debug": DEBUG,
        "base_url": base_url,
        "google_credentials": os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        "google_project": os.getenv('GOOGLE_CLOUD_PROJECT'),
        "optimizations": {
            "low_latency_stt": True, 
            "speaking_state_management": True,
            "early_result_processing": True
        }
    }
    return config

# Add more specific error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

if __name__ == '__main__':
    print("Starting Voice AI Agent with low-latency optimizations...")
    print(f"Base URL: {os.getenv('BASE_URL', 'Not set')}")
    print(f"Google Credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    
    # Start FastAPI using Uvicorn
    uvicorn.run(
        "twilio_fastapi_app:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info" if DEBUG else "error",
        workers=1  # Keep a single worker for this application due to global state
    )