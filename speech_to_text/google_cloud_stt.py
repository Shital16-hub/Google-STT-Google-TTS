# speech_to_text/google_cloud_stt.py

"""
Optimized Google Cloud Speech-to-Text v2 implementation with streaming, early response,
and low-latency configuration for real-time voice applications.
"""
import logging
import asyncio
import time
import os
import json
import queue
import threading
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Iterator, Set
from dataclasses import dataclass

# Import Speech-to-Text v2 API
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.oauth2 import service_account

# Import Duration from protobuf directly
from google.protobuf.duration_pb2 import Duration

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float = 0.0
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

class GoogleCloudStreamingSTT:
    """
    Optimized Google Cloud Speech-to-Text v2 client for low-latency applications.
    
    Key optimizations:
    1. Early response processing with smaller chunk sizes
    2. Faster end-of-speech detection with optimized timeouts
    3. Efficient state management to prevent processing during TTS output
    4. Better error recovery and session management
    """
    
    # Optimized constants for low latency
    STREAMING_LIMIT = 240000  # 4 minutes
    CHUNK_SIZE = 400  # 50ms chunks for more responsive processing
    SILENCE_THRESHOLD = 0.3  # Lower threshold for faster silence detection
    MAX_SILENCE_TIME = 0.8  # Reduced from 30s to 0.8s for faster end-of-speech detection
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = True,  # Changed to True for streaming response
        project_id: Optional[str] = None,
        location: str = "global",
        credentials_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize with optimized settings for low latency."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.location = location
        self.credentials_file = credentials_file
        
        # Get project ID with better error handling
        self.project_id = self._get_project_id(project_id)
        
        # Initialize client with explicit credentials
        self._initialize_client()
        
        # Create recognizer path
        self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        
        # Setup configuration with enhanced low-latency settings
        self._setup_config()
        
        # Enhanced state tracking
        self.is_streaming = False
        self.is_speaking = False  # Flag to prevent processing during TTS output
        self.audio_queue = asyncio.Queue(maxsize=50)  # Smaller queue for faster processing
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())
        
        # Session management for handling timeouts
        self.stream_start_time = None
        self.current_stream = None
        self.last_audio_time = time.time()
        self.last_speech_time = time.time()
        
        # Voice activity tracking
        self.speech_detected = False
        self.silence_frames = 0
        self.max_silence_frames = int(self.MAX_SILENCE_TIME * (self.sample_rate / self.CHUNK_SIZE))
        
        # Audio processing tracking
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.consecutive_errors = 0
        
        # Results tracking
        self.pending_results: Set[str] = set()  # Track results we've already seen
        
        # Create callback event loop for async operations
        self.callback_loop = None
        self.callback_thread = None
        self._current_callback = None
        
        logger.info(f"Initialized optimized Speech v2 for low latency - Project: {self.project_id}")
    
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
        
        raise ValueError("Google Cloud project ID is required")
    
    def _initialize_client(self):
        """Initialize the Google Cloud Speech client with enhanced error handling."""
        try:
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
        """Setup recognition configuration optimized for low latency."""
        # Audio encoding configuration
        if self.encoding == "MULAW":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        else:
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        
        # Highly optimized recognition config for telephony and low latency
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=self.sample_rate,
                encoding=audio_encoding,
                audio_channel_count=self.channels,
            ),
            language_codes=[self.language],
            model="telephony",  # Telephony model for better phone call recognition
            features=cloud_speech.RecognitionFeatures(
                # Enhanced features for better performance
                enable_automatic_punctuation=True,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
                profanity_filter=False,
                enable_word_confidence=True,
                max_alternatives=1,
            )
        )
        
        # Adding adaptation in a compatible way - removed the complex adaptation structure
        # because it's causing compatibility issues with the library version
        
        # Configure streaming for lower latency with faster end-of-speech detection
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                enable_voice_activity_events=True,
                voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    # Optimized timeouts for faster response
                    speech_start_timeout=Duration(seconds=1, nanos=0),  # Quicker speech detection (1s)
                    speech_end_timeout=Duration(seconds=0, nanos=500000000),  # End speech detection after 0.5s of silence
                ),
            ),
        )
        
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
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
    
    def _request_generator(self) -> Iterator[cloud_speech.StreamingRecognizeRequest]:
        """Generate requests with enhanced low-latency processing."""
        # Send initial config
        yield self.config_request
        
        # Process audio chunks
        while not self.stop_event.is_set():
            try:
                # Get audio chunk with blocking but short timeout
                try:
                    # Use Queue.Queue instead of asyncio.Queue to avoid coroutine issues
                    if not hasattr(self, '_sync_queue'):
                        # Create a synchronized queue if it doesn't exist
                        self._sync_queue = queue.Queue(maxsize=50)
                    
                    # Check if there's audio in the queue with a short timeout
                    audio_chunk = self._sync_queue.get(timeout=0.05)
                    
                    if audio_chunk is None:
                        break
                    
                    # Skip processing if we're speaking to avoid echo
                    if self.is_speaking:
                        # Mark as done
                        self._sync_queue.task_done()
                        continue
                    
                    # Send audio chunk
                    yield cloud_speech.StreamingRecognizeRequest(audio=audio_chunk)
                    self._sync_queue.task_done()
                    self.last_audio_time = time.time()
                    
                except queue.Empty:
                    # Check if we should stop due to inactivity
                    if time.time() - self.last_audio_time > 10.0:  # 10 second absolute timeout
                        logger.info("No audio for 10s, stopping stream")
                        break
                    continue
                    
            except Exception as e:
                logger.error(f"Error in request generator: {e}")
                # Try to continue despite errors
                time.sleep(0.1)
    
    def _run_streaming(self):
        """Run streaming with optimized error handling for low latency."""
        while self.is_streaming and not self.stop_event.is_set():
            try:
                logger.info(f"Starting optimized low-latency streaming session: {self.session_id}")
                self.stream_start_time = time.time()
                self.consecutive_errors = 0
                self.pending_results.clear()
                
                # Create streaming call with timeout
                self.current_stream = self.client.streaming_recognize(
                    requests=self._request_generator(),
                    timeout=60  # 1 minute timeout (shorter for faster error recovery)
                )
                
                # Process responses with enhanced error handling
                try:
                    for response in self.current_stream:
                        if self.stop_event.is_set():
                            break
                        
                        # Skip processing if we're speaking to avoid echo
                        if self.is_speaking:
                            continue
                            
                        self._process_response(response)
                        
                except StopIteration:
                    logger.debug("Stream iteration completed normally")
                except Exception as e:
                    logger.error(f"Error processing stream responses: {e}")
                    self.consecutive_errors += 1
                    
                # If we reach here and still streaming, restart session for continuous operation
                if self.is_streaming and not self.stop_event.is_set():
                    logger.info("Stream ended, restarting for continuous operation")
                    time.sleep(0.1)  # Very short delay
                    continue
                    
            except Exception as e:
                self.consecutive_errors += 1
                
                # Enhanced error recovery
                if "timeout" in str(e).lower():
                    logger.warning(f"Stream timeout: {e}")
                elif "cancelled" in str(e).lower():
                    logger.info("Stream cancelled by client")
                    break
                else:
                    logger.error(f"Streaming error: {e}")
                
                # Quick recovery for continuous operation
                if self.is_streaming and not self.stop_event.is_set():
                    time.sleep(0.1)  # Very short delay
                    # Only give up after many consecutive errors
                    if self.consecutive_errors > 10:
                        logger.error("Too many consecutive errors, stopping")
                        break
                    continue
                else:
                    break
        
        logger.info(f"Streaming thread ended (session: {self.session_id})")
    
    def _process_response(self, response):
        """Process response with optimized early-result handling."""
        # Handle voice activity events
        if hasattr(response, 'speech_event_type') and response.speech_event_type:
            speech_event = response.speech_event_type
            
            if speech_event == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
                self.speech_detected = True
                self.last_speech_time = time.time()
                self.silence_frames = 0
                logger.debug("Speech activity detected")
            elif speech_event == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
                self.speech_detected = False
                speaking_duration = time.time() - self.last_speech_time
                logger.debug(f"Speech activity ended (duration: {speaking_duration:.2f}s)")
        
        # Process transcription results
        for result in response.results:
            if not result.alternatives:
                continue
                
            alternative = result.alternatives[0]
            text = alternative.transcript.strip()
            
            # Skip empty or very short results
            if not text or len(text) < 2:
                continue
                
            # Skip duplicate results we've already seen
            result_hash = f"{text}_{result.is_final}"
            if result_hash in self.pending_results:
                continue
            self.pending_results.add(result_hash)
            
            # Create transcript result
            confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0.7
            
            transcription_result = StreamingTranscriptionResult(
                text=text,
                is_final=result.is_final,
                confidence=confidence,
                session_id=self.session_id,
                start_time=self.last_speech_time,
                end_time=time.time()
            )
            
            # Skip processing if we're speaking to avoid echo
            if self.is_speaking:
                logger.debug(f"Skipping result while speaking: '{text}'")
                continue
            
            # Always dispatch interim results for early processing
            if hasattr(self, '_current_callback') and self._current_callback:
                if self.callback_loop and not self.callback_loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._current_callback(transcription_result),
                        self.callback_loop
                    )
            
            if result.is_final:
                self.successful_transcriptions += 1
                logger.info(f"Final transcription: '{text}' (conf: {confidence:.2f})")
            elif self.interim_results:
                logger.debug(f"Interim result: '{text}'")
    
    def set_speaking_state(self, is_speaking: bool):
        """Set speaking state to prevent processing during TTS output."""
        self.is_speaking = is_speaking
        logger.debug(f"Speaking state set to: {is_speaking}")
    
    async def start_streaming(self) -> None:
        """Start streaming with optimized initialization."""
        if self.is_streaming:
            logger.debug("Stream already active, keeping existing session")
            return
        
        # Create callback event loop
        self._create_callback_loop()
        
        self.is_streaming = True
        self.stop_event.clear()
        self.session_id = str(uuid.uuid4())
        self.consecutive_errors = 0
        self.is_speaking = False
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                await self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started optimized low-latency streaming session: {self.session_id}")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming with proper cleanup."""
        if not self.is_streaming:
            return "", 0.0
        
        logger.info(f"Stopping streaming session: {self.session_id}")
        
        self.is_streaming = False
        self.stop_event.set()
        
        # Signal end to request generator
        try:
            await self.audio_queue.put(None)
        except:
            pass
        
        # Wait for thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        
        # Cancel current stream
        if self.current_stream:
            try:
                self.current_stream.cancel()
            except Exception as e:
                logger.debug(f"Error cancelling stream: {e}")
        
        # Stop callback loop
        self._stop_callback_loop()
        
        # Calculate session duration
        duration = time.time() - self.stream_start_time if self.stream_start_time else 0.0
        
        logger.info(f"Stopped streaming, duration: {duration:.2f}s")
        
        return "", duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk with optimized handling."""
        # Skip processing if we're speaking to avoid echo
        if self.is_speaking:
            return None
            
        # Store callback for use in response processing
        self._current_callback = callback
        
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        try:
            # Skip tiny chunks
            if len(audio_chunk) < 32:  # Minimal size check
                return None
            
            # Put audio in the synchronized queue instead of asyncio queue
            if not hasattr(self, '_sync_queue'):
                import queue
                self._sync_queue = queue.Queue(maxsize=50)
                
            try:
                # Use non-blocking put with a short timeout
                self._sync_queue.put(audio_chunk, block=True, timeout=0.05)
            except queue.Full:
                # Queue full - we're getting backed up, drop the chunk
                logger.warning("Audio queue full, dropping chunk")
                return None
            
            return None  # Results come through callbacks
        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    async def cleanup(self):
        """Clean up all resources."""
        logger.info(f"Cleaning up STT session: {self.session_id}")
        await self.stop_streaming()
        
        # Clear any remaining audio queue
        while not self.audio_queue.empty():
            try:
                await self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        logger.info("STT cleanup completed")