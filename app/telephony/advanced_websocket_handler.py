"""
Advanced WebSocket Handler for Twilio Voice Integration - Updated Version
========================================================================

Comprehensive WebSocket handler integrated with your multi-agent orchestrator system.
Combines the best of your original handler with the new agent architecture.

Features:
- Real-time audio streaming with Twilio Media Streams
- Integration with MultiAgentOrchestrator and specialized agents
- Enhanced STT/TTS pipeline integration
- Echo detection and prevention
- Session state management
- Performance monitoring and latency optimization
- Error recovery and connection management
"""

import asyncio
import logging
import json
import base64
import time
import uuid
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import audioop

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketState(str, Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    PAUSED = "paused"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class TwilioMediaMessage:
    """Parsed Twilio media stream message"""
    event: str
    sequence_number: Optional[int] = None
    media: Optional[Dict[str, Any]] = None
    start: Optional[Dict[str, Any]] = None
    mark: Optional[Dict[str, Any]] = None
    stop: Optional[Dict[str, Any]] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class StreamingTranscriptionResult:
    """STT transcription result"""
    text: str
    confidence: float
    is_final: bool
    session_id: str


@dataclass
class CallSession:
    """Voice call session tracking"""
    call_sid: str
    session_id: str
    start_time: float
    last_activity: float
    state: WebSocketState
    total_audio_chunks: int = 0
    total_responses: int = 0
    avg_latency_ms: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class AdvancedWebSocketHandler:
    """
    Advanced WebSocket Handler for Twilio Voice Integration
    
    Integrates your original handler's best features with the new multi-agent system.
    """
    
    def __init__(self,
                 call_sid: str,
                 orchestrator,  # MultiAgentOrchestrator
                 state_manager,  # ConversationStateManager
                 target_latency_ms: int = 377):
        """Initialize the advanced WebSocket handler"""
        
        self.call_sid = call_sid
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.target_latency_ms = target_latency_ms
        
        # WebSocket reference
        self.websocket: Optional[WebSocket] = None
        self.stream_sid: Optional[str] = None
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.call_session = CallSession(
            call_sid=call_sid,
            session_id=self.session_id,
            start_time=time.time(),
            last_activity=time.time(),
            state=WebSocketState.CONNECTING
        )
        
        # Enhanced conversation state management (from your original)
        self.conversation_active = True
        self.is_speaking = False
        self.expecting_speech = True
        self.call_ended = False
        
        # Audio processing with flow control
        self.audio_buffer = bytearray()
        self.audio_queue = asyncio.Queue()
        self.is_processing_audio = False
        self.silence_threshold = 500  # ms of silence before processing
        self.last_audio_time = 0
        self.chunk_size = 800  # 100ms at 8kHz
        self.min_chunk_size = 160  # 20ms minimum
        
        # Enhanced session management (from your original)
        self.session_start_time = time.time()
        self.last_transcription_time = time.time()
        self.last_tts_time = None
        
        # Response tracking for echo prevention (from your original)
        self.waiting_for_response = False
        self.last_response_time = time.time()
        self.last_response_text = ""
        
        # Twilio Media Stream configuration
        self.media_config = {
            "sample_rate": 8000,
            "encoding": "mulaw",
            "channels": 1,
            "chunk_size": 160  # 20ms at 8kHz
        }
        
        # Enhanced stats tracking (from your original)
        self.performance_metrics = {
            "audio_chunks_received": 0,
            "audio_chunks_sent": 0,
            "avg_processing_time_ms": 0.0,
            "stt_latency_ms": 0.0,
            "llm_latency_ms": 0.0,
            "tts_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "transcriptions": 0,
            "responses_sent": 0,
            "echo_detections": 0,
            "invalid_transcriptions": 0
        }
        
        # Background tasks
        self.audio_processor_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Get project ID for logging
        self.project_id = self._get_project_id()
        
        logger.info(f"AdvancedWebSocketHandler initialized for call: {call_sid}, Project: {self.project_id}")
    
    def _get_project_id(self) -> str:
        """Get project ID with enhanced error handling (from your original)"""
        # Try environment variable first
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Try to extract from credentials file
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file and os.path.exists(credentials_file):
            try:
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials: {e}")
        
        # Fallback
        return "my-tts-project-458404"
    
    async def handle_websocket_session(self, websocket: WebSocket):
        """Main WebSocket session handler"""
        
        logger.info(f"🔗 Starting WebSocket session for call: {self.call_sid}")
        
        # Store websocket reference
        self.websocket = websocket
        
        try:
            # Update session state
            self.call_session.state = WebSocketState.CONNECTED
            
            # Start background tasks
            self.audio_processor_task = asyncio.create_task(self._audio_processor())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            # Create conversation state
            if self.state_manager:
                await self.state_manager.create_conversation_state(
                    session_id=self.session_id,
                    initial_context={
                        "call_sid": self.call_sid,
                        "media_format": "twilio_stream",
                        "domain": "voice_call",
                        "urgency": "normal"
                    }
                )
            
            # Main message processing loop
            async for message in websocket.iter_text():
                await self._process_twilio_message(websocket, message)
                
        except WebSocketDisconnect:
            logger.info(f"📞 WebSocket disconnected for call: {self.call_sid}")
            self.call_session.state = WebSocketState.DISCONNECTED
            
        except Exception as e:
            logger.error(f"❌ WebSocket error for call {self.call_sid}: {e}", exc_info=True)
            self.call_session.state = WebSocketState.ERROR
            self.call_session.errors.append(str(e))
            
        finally:
            await self._cleanup_session()
    
    async def _process_twilio_message(self, websocket: WebSocket, message: str):
        """Process incoming Twilio Media Stream message"""
        
        try:
            # Parse Twilio message
            twilio_msg = self._parse_twilio_message(message)
            
            if twilio_msg.event == "media":
                await self._handle_media_message(websocket, twilio_msg)
                
            elif twilio_msg.event == "start":
                await self._handle_start_message(websocket, twilio_msg)
                
            elif twilio_msg.event == "stop":
                await self._handle_stop_message(websocket, twilio_msg)
                
            elif twilio_msg.event == "mark":
                await self._handle_mark_message(websocket, twilio_msg)
                
            else:
                logger.debug(f"Unhandled Twilio event: {twilio_msg.event}")
                
        except Exception as e:
            logger.error(f"Error processing Twilio message: {e}")
            self.call_session.errors.append(f"Message processing error: {str(e)}")
    
    def _parse_twilio_message(self, message: str) -> TwilioMediaMessage:
        """Parse incoming Twilio Media Stream message"""
        
        try:
            data = json.loads(message)
            event = data.get("event", "unknown")
            
            parsed_msg = TwilioMediaMessage(
                event=event,
                raw_data=data
            )
            
            if event == "media":
                parsed_msg.sequence_number = data.get("sequenceNumber")
                parsed_msg.media = data.get("media", {})
                
            elif event == "start":
                parsed_msg.start = data.get("start", {})
                
            elif event == "stop":
                parsed_msg.stop = data.get("stop", {})
                
            elif event == "mark":
                parsed_msg.mark = data.get("mark", {})
            
            return parsed_msg
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Twilio message: {e}")
            return TwilioMediaMessage(event="error", raw_data={"error": str(e)})
    
    async def _handle_media_message(self, websocket: WebSocket, msg: TwilioMediaMessage):
        """Handle incoming audio media from Twilio with enhanced flow control"""
        
        if not msg.media or "payload" not in msg.media:
            return
        
        # Skip audio processing if call has ended
        if self.call_ended:
            return
        
        # Skip audio while we're sending a response to prevent echo (from your original)
        if self.waiting_for_response:
            return
        
        try:
            # Decode base64 audio payload (mulaw format)
            audio_data = base64.b64decode(msg.media["payload"])
            
            # Enhanced check: Skip audio if we just sent TTS to prevent immediate echo
            if self.last_tts_time and (time.time() - self.last_tts_time) < 2.0:
                logger.debug("Skipping audio - too close to TTS output")
                return
            
            # Add to audio buffer
            self.audio_buffer.extend(audio_data)
            self.last_audio_time = time.time()
            
            # Update metrics
            self.performance_metrics["audio_chunks_received"] += 1
            self.call_session.total_audio_chunks += 1
            self.call_session.last_activity = time.time()
            
            # Queue for processing if buffer is large enough
            if len(self.audio_buffer) >= self.chunk_size:  # Use configurable chunk size
                await self.audio_queue.put(bytes(self.audio_buffer))
                self.audio_buffer.clear()
                
        except Exception as e:
            logger.error(f"Error handling media message: {e}")
            self.call_session.errors.append(f"Media processing error: {str(e)}")
    
    async def _handle_start_message(self, websocket: WebSocket, msg: TwilioMediaMessage):
        """Handle stream start message from Twilio"""
        
        logger.info(f"🎤 Audio stream started for call: {self.call_sid}")
        
        if msg.start:
            # Update media configuration from Twilio
            self.stream_sid = msg.start.get("streamSid")
            account_sid = msg.start.get("accountSid")
            
            logger.info(f"Stream details - StreamSid: {self.stream_sid}, AccountSid: {account_sid}")
            
            # Update session state
            self.call_session.state = WebSocketState.STREAMING
            
            # Start conversation (from your original approach)
            await self.start_conversation(websocket)
    
    async def _handle_stop_message(self, websocket: WebSocket, msg: TwilioMediaMessage):
        """Handle stream stop message from Twilio"""
        
        logger.info(f"🛑 Audio stream stopped for call: {self.call_sid}")
        self.call_session.state = WebSocketState.DISCONNECTING
        self.call_ended = True
        
        # Process any remaining audio
        if self.audio_buffer:
            await self.audio_queue.put(bytes(self.audio_buffer))
            self.audio_buffer.clear()
    
    async def _handle_mark_message(self, websocket: WebSocket, msg: TwilioMediaMessage):
        """Handle mark message from Twilio (used for timing)"""
        
        if msg.mark:
            mark_name = msg.mark.get("name")
            logger.debug(f"Received mark: {mark_name}")
    
    async def start_conversation(self, websocket: WebSocket):
        """Start conversation with enhanced initialization (from your original)"""
        
        # Send welcome message with delay to ensure connection is stable
        await asyncio.sleep(0.1)
        await self._send_response("Hello! I'm your AI assistant. How can I help you today?")
    
    async def _audio_processor(self):
        """Background task to process audio chunks"""
        
        logger.info("🎵 Audio processor started")
        
        try:
            while self.call_session.state in [WebSocketState.CONNECTED, WebSocketState.STREAMING] and not self.call_ended:
                try:
                    # Wait for audio data with timeout
                    audio_data = await asyncio.wait_for(
                        self.audio_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process the audio chunk
                    await self._process_audio_chunk(audio_data)
                    
                except asyncio.TimeoutError:
                    # Check for silence timeout
                    if (self.last_audio_time > 0 and 
                        time.time() - self.last_audio_time > self.silence_threshold / 1000.0 and
                        self.audio_buffer):
                        
                        # Process remaining audio after silence
                        await self.audio_queue.put(bytes(self.audio_buffer))
                        self.audio_buffer.clear()
                        
                except Exception as e:
                    logger.error(f"Error in audio processor: {e}")
                    
        except Exception as e:
            logger.error(f"Audio processor crashed: {e}")
        finally:
            logger.info("🎵 Audio processor stopped")
    
    async def _process_audio_chunk(self, audio_data: bytes):
        """Process individual audio chunk through the system"""
        
        if self.is_processing_audio or self.call_ended:
            logger.debug("Skipping audio chunk - already processing or call ended")
            return
        
        self.is_processing_audio = True
        processing_start = time.time()
        
        try:
            # Convert mulaw to linear PCM for STT
            pcm_audio = audioop.ulaw2lin(audio_data, 2)  # Convert to 16-bit PCM
            
            # Simulate STT processing (since we're using the orchestrator)
            # In a real implementation, this would use your STT system
            if len(pcm_audio) > 1600:  # Minimum viable audio length
                await self._handle_transcription_simulation(pcm_audio)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self.call_session.errors.append(f"Audio processing error: {str(e)}")
            
        finally:
            self.is_processing_audio = False
    
    async def _handle_transcription_simulation(self, pcm_audio: bytes):
        """Simulate transcription handling (integrate with your STT system here)"""
        
        # This is where you'd integrate with your actual STT system
        # For now, we'll simulate the process
        try:
            # Process through your orchestrator system
            if self.orchestrator:
                result = await self.orchestrator.process_conversation(
                    session_id=self.session_id,
                    input_text="",  # Will be filled by STT
                    context={
                        "audio_data": pcm_audio,
                        "audio_format": "pcm_16_8000",
                        "call_sid": self.call_sid,
                        "input_mode": "voice"
                    }
                )
                
                if result and hasattr(result, 'success') and result.success and hasattr(result, 'response') and result.response:
                    # Validate the transcription before processing (from your original logic)
                    if hasattr(result, 'transcription') and result.transcription:
                        if self._is_valid_transcription(result.transcription, getattr(result, 'confidence', 0.8)):
                            await self._process_final_transcription(result.transcription, result.response)
                        else:
                            self.performance_metrics["invalid_transcriptions"] += 1
                    else:
                        # Direct response without transcription validation
                        await self._send_response(result.response)
                        
        except Exception as e:
            logger.error(f"Error in transcription simulation: {e}")
    
    def _is_valid_transcription(self, transcription: str, confidence: float) -> bool:
        """Enhanced transcription validation with echo detection (from your original)"""
        
        # Basic length check
        if len(transcription) < 2:
            return False
        
        # Confidence threshold (lower for telephony)
        if confidence < 0.3:
            logger.debug(f"Low confidence transcription: {confidence:.2f}")
            return False
        
        # Enhanced echo detection
        if self._is_likely_echo(transcription):
            self.performance_metrics["echo_detections"] += 1
            logger.debug(f"Echo detected: '{transcription}'")
            return False
        
        # Skip common filler words and short responses
        transcription_lower = transcription.lower().strip()
        skip_patterns = [
            'um', 'uh', 'mmm', 'hmm', 'ah', 'er', 'oh',
            'only', 'series', 'okay', 'ok', 'yes', 'no',
            'ready to help', 'what would you like', 'how can i',
            'voice assist', 'features', 'pricing', 'plan'
        ]
        
        if transcription_lower in skip_patterns:
            logger.debug(f"Skipping pattern: '{transcription}'")
            return False
        
        # Check for word-for-word matches with recent response
        if self.last_response_text:
            response_words = set(self.last_response_text.lower().split())
            transcription_words = set(transcription_lower.split())
            
            if len(transcription_words) > 0:
                overlap_ratio = len(response_words & transcription_words) / len(transcription_words)
                if overlap_ratio > 0.8:  # 80% overlap indicates echo
                    logger.debug(f"High word overlap with last response: {overlap_ratio:.2f}")
                    return False
        
        return True
    
    def _is_likely_echo(self, transcription: str) -> bool:
        """Enhanced echo detection using multiple heuristics (from your original)"""
        
        # Check timing - if transcription comes too soon after TTS, likely echo
        if self.last_tts_time and (time.time() - self.last_tts_time) < 3.0:
            return True
        
        # Check against specific system phrases
        system_phrases = [
            "i'm ready to help",
            "what would you like to know",
            "how can i help",
            "ai assistant"
        ]
        
        for phrase in system_phrases:
            if phrase in transcription.lower():
                return True
        
        return False
    
    async def _process_final_transcription(self, transcription: str, response: str = None):
        """Process transcription with enhanced error handling (from your original)"""
        
        # Update state
        self.performance_metrics["transcriptions"] += 1
        self.waiting_for_response = True
        self.last_transcription_time = time.time()
        
        logger.info(f"Processing transcription: '{transcription}'")
        
        try:
            if response:
                # Use provided response
                await self._send_response(response)
            else:
                # Generate response through orchestrator
                if self.orchestrator:
                    result = await self.orchestrator.process_conversation(
                        session_id=self.session_id,
                        input_text=transcription,
                        context={
                            "call_sid": self.call_sid,
                            "input_mode": "voice",
                            "platform": "twilio_websocket"
                        }
                    )
                    
                    if result and hasattr(result, 'success') and result.success and result.response:
                        await self._send_response(result.response)
                    else:
                        await self._send_response("I'm sorry, I couldn't process that request.")
                else:
                    await self._send_response("I understand what you said, but I'm having trouble connecting to my knowledge base.")
                    
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)
            await self._send_response("I'm sorry, I encountered an error processing your request.")
        finally:
            self.waiting_for_response = False
    
    async def _send_response(self, text: str):
        """Send TTS response with enhanced error handling and echo prevention (from your original)"""
        
        if not text.strip() or self.call_ended:
            return
        
        try:
            # Set response state
            self.is_speaking = True
            self.waiting_for_response = True
            self.last_response_time = time.time()
            self.last_response_text = text
            
            logger.info(f"Sending response: '{text}'")
            
            # Use your TTS engine to generate audio
            if (hasattr(self.orchestrator, 'tts_engine') and 
                self.orchestrator.tts_engine and 
                hasattr(self.orchestrator.tts_engine, 'stream_synthesis')):
                
                # Generate audio using your DualStreamingTTSEngine
                audio_chunks = []
                
                async for chunk in self.orchestrator.tts_engine.stream_synthesis(
                    text=text,
                    voice_config={
                        "voice": "en-US-Neural2-C",
                        "sample_rate": 8000,
                        "format": "mulaw"
                    }
                ):
                    audio_chunks.append(chunk)
                
                # Send audio chunks to Twilio
                if audio_chunks:
                    combined_audio = b''.join(audio_chunks)
                    await self._send_audio_chunks(combined_audio)
                    self.performance_metrics["responses_sent"] += 1
                    self.last_tts_time = time.time()
                    
            else:
                logger.warning("TTS engine not available, cannot send voice response")
                
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
        finally:
            # Clear speaking flags
            self.is_speaking = False
            self.waiting_for_response = False
            
            # Small delay to ensure audio playback completes
            await asyncio.sleep(0.8)
            
            logger.debug("Ready for next utterance")
    
    async def _send_audio_chunks(self, audio_data: bytes):
        """Send audio data with proper chunking (from your original)"""
        
        if not self.stream_sid or not self.websocket:
            logger.warning("Cannot send audio: missing stream_sid or websocket")
            return
        
        chunk_size = 400  # 50ms chunks for smooth playback
        total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            
            try:
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": audio_base64}
                }
                
                await self.websocket.send_text(json.dumps(message))
                
                # Dynamic delay based on chunk size
                await asyncio.sleep(0.025)  # 25ms delay
                
            except Exception as e:
                logger.error(f"Error sending audio chunk {chunk_num}/{total_chunks}: {e}")
                break
        
        logger.debug(f"Sent {total_chunks} audio chunks")
        self.performance_metrics["audio_chunks_sent"] += total_chunks
    
    async def _heartbeat_monitor(self):
        """Monitor connection health with heartbeat"""
        
        while (self.call_session.state in [WebSocketState.CONNECTED, WebSocketState.STREAMING] and 
               not self.call_ended):
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check if session is still active
                current_time = time.time()
                if current_time - self.call_session.last_activity > 300:  # 5 minutes
                    logger.warning(f"Call {self.call_sid} inactive for 5+ minutes")
                    
                # Log performance metrics
                if self.call_session.total_audio_chunks > 0:
                    logger.info(f"Call {self.call_sid} metrics: "
                              f"Audio chunks: {self.call_session.total_audio_chunks}, "
                              f"Transcriptions: {self.performance_metrics['transcriptions']}, "
                              f"Responses: {self.performance_metrics['responses_sent']}, "
                              f"Echo detections: {self.performance_metrics['echo_detections']}")
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                break
    
    async def _cleanup_session(self):
        """Enhanced cleanup with proper session management (from your original)"""
        
        logger.info(f"🧹 Cleaning up WebSocket session for call: {self.call_sid}")
        
        try:
            # Set cleanup flags
            self.call_ended = True
            self.conversation_active = False
            
            # Cancel background tasks
            if self.audio_processor_task and not self.audio_processor_task.done():
                self.audio_processor_task.cancel()
                try:
                    await self.audio_processor_task
                except asyncio.CancelledError:
                    pass
                
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # End conversation in state manager
            if self.state_manager:
                try:
                    await self.state_manager.end_conversation(
                        session_id=self.session_id,
                        resolution_status="call_ended"
                    )
                except Exception as e:
                    logger.error(f"Error ending conversation in state manager: {e}")
            
            # Calculate final statistics
            duration = time.time() - self.session_start_time
            
            # Enhanced logging with conversation metrics (from your original)
            logger.info(f"📊 Call {self.call_sid} completed: "
                      f"Duration: {duration:.1f}s, "
                      f"Audio chunks: {self.call_session.total_audio_chunks}, "
                      f"Valid transcriptions: {self.performance_metrics['transcriptions']}, "
                      f"Invalid/Echo: {self.performance_metrics['invalid_transcriptions'] + self.performance_metrics['echo_detections']}, "
                      f"Responses: {self.performance_metrics['responses_sent']}, "
                      f"Echo detections: {self.performance_metrics['echo_detections']}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def cleanup(self):
        """Public cleanup method"""
        await self._cleanup_session()
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics (enhanced from your original)"""
        
        duration = time.time() - self.session_start_time
        
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "session_id": self.session_id,
            "duration_seconds": round(duration, 2),
            "state": self.call_session.state.value,
            "project_id": self.project_id,
            
            # Audio metrics
            "audio_chunks_received": self.performance_metrics["audio_chunks_received"],
            "audio_chunks_sent": self.performance_metrics["audio_chunks_sent"],
            
            # Conversation metrics
            "transcriptions": self.performance_metrics["transcriptions"],
            "invalid_transcriptions": self.performance_metrics["invalid_transcriptions"],
            "echo_detections": self.performance_metrics["echo_detections"],
            "responses_sent": self.performance_metrics["responses_sent"],
            
            # State flags
            "is_speaking": self.is_speaking,
            "conversation_active": self.conversation_active,
            "call_ended": self.call_ended,
            "expecting_speech": self.expecting_speech,
            "waiting_for_response": self.waiting_for_response,
            
            # Timing
            "session_start_time": self.session_start_time,
            "last_transcription_time": self.last_transcription_time,
            "last_audio_time": self.last_audio_time,
            "last_response_time": self.last_response_time,
            "target_latency_ms": self.target_latency_ms,
            
            # Quality metrics (from your original)
            "transcription_rate": round(self.performance_metrics["transcriptions"] / max(duration / 60, 1), 2),  # per minute
            "response_rate": round(self.performance_metrics["responses_sent"] / max(duration / 60, 1), 2),      # per minute
            "echo_rate": round(self.performance_metrics["echo_detections"] / max(self.performance_metrics["audio_chunks_received"], 1) * 100, 2),  # percentage
            
            # Performance metrics
            "avg_processing_time_ms": self.performance_metrics["avg_processing_time_ms"],
            "stt_latency_ms": self.performance_metrics["stt_latency_ms"],
            "llm_latency_ms": self.performance_metrics["llm_latency_ms"],
            "tts_latency_ms": self.performance_metrics["tts_latency_ms"],
            "total_latency_ms": self.performance_metrics["total_latency_ms"],
            
            # Connection status
            "websocket_connected": self.websocket is not None,
            
            # Error tracking
            "error_count": len(self.call_session.errors),
            "errors": self.call_session.errors
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics (alias for compatibility)"""
        return self.get_session_metrics()