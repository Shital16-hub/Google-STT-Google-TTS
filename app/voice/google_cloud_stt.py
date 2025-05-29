"""
Fixed Google Cloud Speech-to-Text v2 implementation with proper session management,
robust timeout handling, and echo detection for telephony applications.
Moved to app/voice/ directory with enhanced error handling.
"""
import logging
import asyncio
import time
import os
import json
import queue
import threading
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float = 0.0
    session_id: str = ""

class GoogleCloudStreamingSTT:
    """
    Google Cloud Speech-to-Text v2 client with robust session management and echo detection.
    Optimized for continuous telephony conversations with proper timeout handling.
    """
    
    # Enhanced constants for better session management
    STREAMING_LIMIT = 240000  # 4 minutes - safely under 5min limit
    CHUNK_TIMEOUT = 10.0      # Reduced timeout for more responsive reconnection
    RECONNECT_DELAY = 0.5     # Faster reconnection
    MAX_SILENCE_TIME = 30.0   # Maximum silence before stopping session
    ECHO_DETECTION_WINDOW = 3.0  # Time window to detect echoes
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = False,
        project_id: Optional[str] = None,
        location: str = "global",
        credentials_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize with robust telephony settings and echo detection."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.location = location
        self.credentials_file = credentials_file
        
        # Initialize Google Cloud components with error handling
        self.client = None
        self.project_id = None
        self.google_available = False
        
        try:
            # Get project ID with better error handling
            self.project_id = self._get_project_id(project_id)
            
            # Initialize client with explicit credentials
            self._initialize_client()
            
            # Create recognizer path
            self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
            
            # Setup configuration with enhanced telephony settings
            self._setup_config()
            
            self.google_available = True
            logger.info(f"✅ Google Cloud STT initialized - Project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"❌ Google Cloud STT initialization failed: {e}")
            logger.info("🔄 STT will operate in fallback mode")
            self.google_available = False
        
        # Enhanced state tracking
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())
        
        # Session management for handling timeouts
        self.stream_start_time = None
        self.reconnection_in_progress = False
        self.current_stream = None
        self.last_response_time = None
        
        # Voice activity and echo detection
        self.last_audio_time = time.time()
        self.speech_detected = False
        self.last_spoken_texts = []  # Track recent spoken text for echo detection
        self.speaking_start_time = None
        
        # Audio processing tracking
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.session_count = 0
        self.timeout_count = 0
        self.consecutive_errors = 0
        
        # Create callback event loop for async operations
        self.callback_loop = None
        self.callback_thread = None
    
    def _get_project_id(self, project_id: Optional[str]) -> str:
        """Get project ID with robust fallback mechanisms."""
        # Try provided project_id first
        if project_id:
            return project_id
            
        # Try environment variable
        env_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if env_project_id:
            return env_project_id
        
        # Try to extract from credentials file
        credentials_file_to_check = self.credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file_to_check and os.path.exists(credentials_file_to_check):
            try:
                with open(credentials_file_to_check, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        logger.info(f"Extracted project ID from credentials: {project_id}")
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials file: {e}")
        
        # If no credentials found, use a default project ID for fallback mode
        logger.warning("No Google Cloud project ID found, using fallback mode")
        return "fallback-project"
    
    def _initialize_client(self):
        """Initialize the Google Cloud Speech client with enhanced error handling."""
        try:
            # Import Google Cloud dependencies with error handling
            try:
                from google.cloud.speech_v2 import SpeechClient
                from google.cloud.speech_v2.types import cloud_speech
                from google.oauth2 import service_account
                from google.protobuf.duration_pb2 import Duration
                
                # Store classes for later use
                self.SpeechClient = SpeechClient
                self.cloud_speech = cloud_speech
                self.Duration = Duration
                
            except ImportError as e:
                logger.error(f"Google Cloud Speech libraries not available: {e}")
                raise
            
            if self.credentials_file and os.path.exists(self.credentials_file):
                # Use service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                self.client = SpeechClient(credentials=credentials)
                logger.info(f"Initialized Speech client with credentials from {self.credentials_file}")
            else:
                # Use default credentials (ADC)
                self.client = SpeechClient()
                logger.info("Initialized Speech client with default credentials")
                
        except Exception as e:
            logger.error(f"Error initializing Speech client: {e}")
            raise
    
    def _setup_config(self):
        """Setup recognition configuration with enhanced telephony optimization."""
        if not self.google_available:
            logger.warning("Google Cloud not available, skipping config setup")
            return
        
        try:
            # Audio encoding configuration
            if self.encoding == "MULAW":
                audio_encoding = self.cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
            else:
                audio_encoding = self.cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
            
            # Enhanced recognition config for telephony
            self.recognition_config = self.cloud_speech.RecognitionConfig(
                explicit_decoding_config=self.cloud_speech.ExplicitDecodingConfig(
                    sample_rate_hertz=self.sample_rate,
                    encoding=audio_encoding,
                    audio_channel_count=self.channels,
                ),
                language_codes=[self.language],
                model="telephony",  # Telephony model for better phone call recognition
                features=self.cloud_speech.RecognitionFeatures(
                    # Enhanced features for better telephony performance
                    enable_automatic_punctuation=True,
                    enable_spoken_punctuation=False,
                    enable_spoken_emojis=False,
                    profanity_filter=False,  # Disable to avoid false positives
                    enable_word_confidence=True,  # Get word-level confidence
                    max_alternatives=1,  # Only get the best alternative
                ),
            )
            
            # Enhanced streaming configuration with voice activity detection
            self.streaming_config = self.cloud_speech.StreamingRecognitionConfig(
                config=self.recognition_config,
                streaming_features=self.cloud_speech.StreamingRecognitionFeatures(
                    interim_results=self.interim_results,
                    enable_voice_activity_events=True,
                    voice_activity_timeout=self.cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                        # More aggressive timeouts for telephony
                        speech_start_timeout=self.Duration(seconds=5),   # Wait 5s for speech to start
                        speech_end_timeout=self.Duration(seconds=1)      # Wait 1s after speech ends
                    ),
                ),
            )
            
            
            
            # Verify configuration was created successfully
            if hasattr(self, 'streaming_config') and self.streaming_config:
                logger.info("✅ STT streaming configuration created successfully")
            else:
                logger.error("❌ Failed to create streaming configuration")
                
        except Exception as e:
            logger.error(f"❌ Error in _setup_config: {e}")
            self.google_available = False
    
    def _create_callback_loop(self):
        """Create a separate event loop for handling async callbacks."""
        def run_callback_loop():
            self.callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.callback_loop)
            try:
                self.callback_loop.run_forever()
            except Exception as e:
                logger.error(f"Error in callback loop: {e}")
            finally:
                self.callback_loop.close()
        
        self.callback_thread = threading.Thread(target=run_callback_loop, daemon=True)
        self.callback_thread.start()
        logger.debug("Started callback event loop thread")
    
    def _stop_callback_loop(self):
        """Stop the callback event loop."""
        if self.callback_loop and not self.callback_loop.is_closed():
            self.callback_loop.call_soon_threadsafe(self.callback_loop.stop)
        if self.callback_thread and self.callback_thread.is_alive():
            self.callback_thread.join(timeout=1.0)
        logger.debug("Stopped callback event loop thread")
    
    def _request_generator(self) -> Iterator:
        """Generate requests with enhanced timeout and error handling."""
        if not self.google_available:
            logger.warning("Google Cloud not available, stopping request generator")
            return
        
        try:
            # CRITICAL FIX: Check if streaming_config exists before using it
            if not hasattr(self, 'streaming_config') or not self.streaming_config:
                logger.error("❌ streaming_config not available, recreating...")
                self._setup_config()  # Try to recreate config
                
                if not hasattr(self, 'streaming_config') or not self.streaming_config:
                    logger.error("❌ Failed to create streaming_config, aborting")
                    return
            
            # Create initial config request on-the-fly
            initial_request = self.cloud_speech.StreamingRecognizeRequest(
                recognizer=self.recognizer_path,
                streaming_config=self.streaming_config,
            )
            
            logger.debug("✅ Sending initial streaming config request")
            yield initial_request
            
            # Track last audio time for timeout detection
            last_audio_sent = time.time()
            
            # Send audio chunks with better flow control
            while not self.stop_event.is_set():
                try:
                    # Get audio chunk with shorter timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                    if chunk is None:
                        break
                    
                    # Check if we need to stop due to session limits
                    if self._should_restart_session():
                        logger.info("Session approaching limits, preparing for restart")
                        break
                    
                    # Send audio and track timing
                    audio_request = self.cloud_speech.StreamingRecognizeRequest(audio=chunk)
                    yield audio_request
                    self.audio_queue.task_done()
                    self.last_audio_time = time.time()
                    last_audio_sent = time.time()
                    
                except queue.Empty:
                    # Check for timeout conditions
                    current_time = time.time()
                    if current_time - last_audio_sent > self.MAX_SILENCE_TIME:
                        logger.info(f"No audio for {self.MAX_SILENCE_TIME}s, stopping session")
                        break
                    continue
                except Exception as e:
                    logger.error(f"Error in request generator audio loop: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Critical error in request generator: {e}")
            return

    
    def _should_restart_session(self) -> bool:
        """Enhanced logic to determine when to restart the session."""
        if not self.stream_start_time:
            return False
        
        elapsed_time = (time.time() - self.stream_start_time) * 1000
        
        # Restart if approaching time limit (with buffer)
        if elapsed_time > self.STREAMING_LIMIT:
            return True
        
        # Restart if we've had multiple consecutive errors
        if self.consecutive_errors > 3:
            logger.info(f"Restarting session due to {self.consecutive_errors} consecutive errors")
            return True
        
        return False
    
    def _run_streaming(self):
        """Run streaming with robust error handling and session management."""
        if not self.google_available:
            logger.warning("Google Cloud STT not available, streaming disabled")
            return
        
        while self.is_streaming and not self.stop_event.is_set():
            try:
                logger.info(f"Starting streaming session: {self.session_id}")
                self.stream_start_time = time.time()
                self.consecutive_errors = 0
                
                # Create streaming call with timeout
                self.current_stream = self.client.streaming_recognize(
                    requests=self._request_generator(),
                    timeout=300  # 5 minute timeout
                )
                
                # Process responses with enhanced error handling
                try:
                    for response in self.current_stream:
                        if self.stop_event.is_set():
                            break
                        
                        self.last_response_time = time.time()
                        self._process_response(response)
                        
                except StopIteration:
                    logger.debug("Stream iteration completed normally")
                except Exception as e:
                    logger.error(f"Error processing stream responses: {e}")
                    self.consecutive_errors += 1
                    raise
                
                # If we reach here and still streaming, session ended normally
                if self.is_streaming and not self.stop_event.is_set():
                    logger.info("Stream ended normally, will restart if needed")
                    time.sleep(self.RECONNECT_DELAY)
                    self._start_new_session()
                    continue
                    
            except Exception as e:
                self.consecutive_errors += 1
                self.timeout_count += 1
                
                # Enhanced error classification
                error_str = str(e).lower()
                if "timeout" in error_str or "409" in error_str:
                    logger.warning(f"Stream timeout (#{self.timeout_count}): {e}")
                elif "cancelled" in error_str:
                    logger.info("Stream cancelled by client")
                    break
                else:
                    logger.error(f"Streaming error (#{self.consecutive_errors}): {e}")
                
                # Restart session if still active and errors aren't too frequent
                if self.is_streaming and not self.stop_event.is_set():
                    if self.consecutive_errors < 5:
                        logger.info("Attempting to recover from streaming error")
                        time.sleep(min(self.RECONNECT_DELAY * self.consecutive_errors, 5.0))
                        self._start_new_session()
                        continue
                    else:
                        logger.error(f"Too many consecutive errors ({self.consecutive_errors}), stopping")
                        break
                else:
                    break
        
        logger.info(f"Streaming thread ended (session: {self.session_id})")
    
    def _start_new_session(self):
        """Start a new session with proper cleanup."""
        old_session_id = self.session_id
        self.session_count += 1
        self.session_id = str(uuid.uuid4())
        
        logger.info(f"Starting new STT session: {self.session_id} (replacing {old_session_id})")
        
        # Clear audio queue to prevent old audio from affecting new session
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
    
    def _process_response(self, response):
        """Process response with echo detection and enhanced logging."""
        if not self.google_available:
            return
        
        # Handle voice activity events
        if hasattr(response, 'speech_event_type') and response.speech_event_type:
            speech_event = response.speech_event_type
            logger.debug(f"Voice activity event: {speech_event}")
            
            if speech_event == self.cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
                self.speech_detected = True
                self.speaking_start_time = time.time()
                logger.debug("Speech activity detected")
            elif speech_event == self.cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
                self.speech_detected = False
                speaking_duration = time.time() - (self.speaking_start_time or time.time())
                logger.debug(f"Speech activity ended (duration: {speaking_duration:.2f}s)")
        
        # Process transcription results with echo detection
        for result in response.results:
            if result.alternatives:
                alternative = result.alternatives[0]
                text = alternative.transcript.strip()
                
                if text and result.is_final:
                    confidence = alternative.confidence
                    
                    # Echo detection
                    if self._is_echo(text):
                        logger.debug(f"Echo detected, ignoring: '{text}'")
                        continue
                    
                    # Create result
                    transcription_result = StreamingTranscriptionResult(
                        text=text,
                        is_final=True,
                        confidence=confidence,
                        session_id=self.session_id,
                    )
                    
                    # Track for echo detection
                    self._track_spoken_text(text)
                    
                    # Handle callbacks
                    if hasattr(self, '_current_callback') and self._current_callback:
                        if self.callback_loop and not self.callback_loop.is_closed():
                            asyncio.run_coroutine_threadsafe(
                                self._current_callback(transcription_result),
                                self.callback_loop
                            )
                    
                    self.successful_transcriptions += 1
                    logger.info(f"Final transcription (session {self.session_id}): '{text}' (conf: {confidence:.2f})")
    
    def _is_echo(self, text: str) -> bool:
        """Detect if the transcribed text is likely an echo of our TTS output."""
        current_time = time.time()
        
        # Check against recently spoken texts
        for spoken_text, timestamp in self.last_spoken_texts:
            if current_time - timestamp > self.ECHO_DETECTION_WINDOW:
                continue
            
            # Simple similarity check (can be enhanced with more sophisticated algorithms)
            text_lower = text.lower()
            spoken_lower = spoken_text.lower()
            
            # Check for exact matches or significant overlaps
            if text_lower == spoken_lower:
                return True
            
            # Check for partial matches (words in common)
            text_words = set(text_lower.split())
            spoken_words = set(spoken_lower.split())
            
            if len(text_words) > 0 and len(spoken_words) > 0:
                overlap_ratio = len(text_words & spoken_words) / len(text_words)
                if overlap_ratio > 0.7:  # 70% word overlap
                    return True
        
        return False
    
    def _track_spoken_text(self, text: str):
        """Track spoken text for echo detection."""
        current_time = time.time()
        
        # Add current text
        self.last_spoken_texts.append((text, current_time))
        
        # Clean old entries (keep only recent ones)
        self.last_spoken_texts = [
            (t, ts) for t, ts in self.last_spoken_texts
            if current_time - ts <= self.ECHO_DETECTION_WINDOW * 2
        ]
    
    def add_tts_text(self, text: str):
        """Add TTS output text for echo detection."""
        self._track_spoken_text(text)
    
    async def start_streaming(self) -> None:
        """Start streaming with enhanced initialization."""
        if self.is_streaming:
            logger.debug("Stream already active, keeping existing session")
            return
        
        if not self.google_available:
            logger.warning("Google Cloud STT not available, cannot start streaming")
            return
        
        # Create callback event loop
        self._create_callback_loop()
        
        self.is_streaming = True
        self.stop_event.clear()
        self.session_id = str(uuid.uuid4())
        self.reconnection_in_progress = False
        self.consecutive_errors = 0
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started streaming session: {self.session_id}")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming with proper cleanup."""
        if not self.is_streaming:
            return "", 0.0
        
        logger.info(f"Stopping streaming session: {self.session_id}")
        
        self.is_streaming = False
        self.stop_event.set()
        
        # Signal end to request generator
        try:
            self.audio_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5.0)
            
            if self.stream_thread.is_alive():
                logger.warning("Stream thread did not finish gracefully")
        
        # Cancel current stream
        if self.current_stream:
            try:
                self.current_stream.cancel()
                logger.debug("Cancelled current stream")
            except Exception as e:
                logger.debug(f"Error cancelling stream: {e}")
        
        # Stop callback loop
        self._stop_callback_loop()
        
        # Calculate session duration
        duration = time.time() - self.stream_start_time if self.stream_start_time else 0.0
        
        logger.info(f"Stopped streaming, duration: {duration:.2f}s, "
                   f"sessions: {self.session_count}, timeouts: {self.timeout_count}")
        
        return "", duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk with enhanced error handling."""
        # Store callback for use in response processing
        self._current_callback = callback
        
        if not self.google_available:
            # Return dummy result for fallback mode
            return StreamingTranscriptionResult(
                text="",
                is_final=False,
                confidence=0.0,
                session_id=self.session_id
            )
        
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        try:
            # Convert to bytes if needed
            if hasattr(audio_chunk, 'tobytes'):
                audio_bytes = audio_chunk.tobytes()
            else:
                audio_bytes = bytes(audio_chunk)
            
            # Skip tiny chunks that might be silence
            if len(audio_bytes) < 40:
                return None
            
            # Add to queue with timeout to prevent blocking
            try:
                # Use a very short timeout to prevent blocking
                self.audio_queue.put(audio_bytes, block=True, timeout=0.1)
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk - may need to increase processing speed")
                return None
            
            return None  # Results come through callbacks
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        current_time = time.time()
        session_duration = current_time - self.stream_start_time if self.stream_start_time else 0
        
        return {
            "session_id": self.session_id,
            "is_streaming": self.is_streaming,
            "google_available": self.google_available,
            "project_id": self.project_id,
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "session_count": self.session_count,
            "timeout_count": self.timeout_count,
            "consecutive_errors": self.consecutive_errors,
            "speech_detected": self.speech_detected,
            "session_duration": session_duration,
            "success_rate": round((self.successful_transcriptions / max(self.total_chunks, 1)) * 100, 2),
            "avg_timeouts_per_session": round(self.timeout_count / max(self.session_count, 1), 2),
            "reconnection_in_progress": self.reconnection_in_progress,
            "queue_size": self.audio_queue.qsize(),
            "last_audio_time": self.last_audio_time,
            "last_response_time": self.last_response_time
        }
    
    async def cleanup(self):
        """Clean up all resources."""
        logger.info(f"Cleaning up STT session: {self.session_id}")
        await self.stop_streaming()
        
        # Clear any remaining audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        logger.info("STT cleanup completed")