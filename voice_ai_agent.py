"""
Enhanced Voice AI Agent main class that coordinates all components with improved
speech/noise discrimination and Google Cloud Speech-to-Text integration for telephony applications.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np

# Import the enhanced audio preprocessor
from telephony.audio_preprocessor import AudioPreprocessor

# Google STT imports
from speech_to_text.google_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import GoogleCloudTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """
    Enhanced Voice AI Agent class that integrates the AudioPreprocessor component
    and uses Google Cloud Speech-to-Text for superior speech/noise discrimination.
    """
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        credentials_path: Optional[str] = None,
        llm_temperature: float = 0.0,  # Changed to 0.0 for faster responses
        enable_debug: bool = False,
        **kwargs
    ):
        
        """
        Initialize the Voice AI Agent with minimal speech processing.
        
        Args:
            storage_dir: Directory for persistent storage
            model_name: LLM model name for knowledge base
            credentials_path: Path to Google Cloud credentials JSON
            llm_temperature: LLM temperature for response generation
            enable_debug: Enable detailed debug logging
            **kwargs: Additional parameters for customization
        """
        self.storage_dir = storage_dir
        
        # Ensure model name includes 4-bit quantization for faster inference
        if model_name and not ":q4_" in model_name and not model_name.endswith("-q4_0"):
            model_name = f"{model_name}-q4_0"
        self.model_name = model_name
        
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.llm_temperature = llm_temperature
        self.enable_debug = enable_debug
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        
        # Optimized telephony keywords
        self.stt_keywords = kwargs.get('keywords', [
            'price', 'plan', 'cost', 'subscription', 'service', 'features',
            'support', 'help', 'agent', 'assistant', 'voice'
        ])
        
        # Additional STT parameters
        self.stt_model = kwargs.get('stt_model', 'phone_call')
        
        # Use more responsive speech detection parameters
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            enable_barge_in=True,
            # Use dual threshold approach
            barge_in_threshold=0.045,  # Higher threshold for start
            min_speech_frames_for_barge_in=5,  # Reduced from 6 for faster detection
            barge_in_cooldown_ms=1000,  # Reduced for quicker barge-in
            enable_debug=enable_debug
        )
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        # Initialize preprocessing latency tracking
        self._preprocessing_times = []
        self._max_preprocessing_samples = 100
        
        logger.info(f"Initialized Voice AI Agent with 4-bit quantized model: {self.model_name}")
                
    async def init(self):
        """Initialize all components with enhanced speech processing using Google Cloud."""
        logger.info("Initializing Voice AI Agent components with enhanced speech processing...")
        
        try:
            # Check if Google Cloud credentials are available
            if not self.credentials_path:
                raise ValueError("Google Cloud credentials are required for Speech-to-Text")

            # Initialize speech recognizer with Google Cloud - optimized frame size
            try:
                # Create an optimized Google Cloud client with telephony-specific settings
                self.speech_recognizer = GoogleCloudStreamingSTT(
                    credentials_path=self.credentials_path,
                    language_code=self.stt_language,
                    sample_rate=8000,  # 8kHz for telephony
                    encoding="LINEAR16",
                    channels=1,
                    interim_results=True,  # Enable interim results for responsiveness
                    model=self.stt_model  # Use phone_call model
                )
                logger.info("Successfully initialized Google Cloud Speech-to-Text")
                
                # Set audio processor for the speech recognizer
                if not hasattr(self.speech_recognizer, 'audio_processor'):
                    # Create an audio processor from telephony module if not using integrated one
                    from telephony.audio_processor import AudioProcessor
                    self.speech_recognizer.audio_processor = AudioProcessor()
                
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud Speech-to-Text: {e}")
                raise
            
            # Initialize STT integration with better validation thresholds
            self.stt_integration = STTIntegration(
                speech_recognizer=self.speech_recognizer,
                language=self.stt_language
            )
            
            # Initialize document store and index manager
            doc_store = DocumentStore()
            index_manager = IndexManager(storage_dir=self.storage_dir)
            await index_manager.init()
            
            # Initialize query engine with improved configuration
            self.query_engine = QueryEngine(
                index_manager=index_manager, 
                llm_model_name=self.model_name,
                llm_temperature=self.llm_temperature
            )
            await self.query_engine.init()
            
            # Pre-load models with warm-up queries for better cold-start performance
            try:
                logger.info("Performing warm-up queries...")
                _ = await self.query_engine.query("What's the weather today?")
                logger.info("Warm-up complete")
            except Exception as warm_up_error:
                logger.warning(f"Warm-up error (non-critical): {warm_up_error}")
            
            # Initialize conversation manager with optimized parameters
            self.conversation_manager = ConversationManager(
                query_engine=self.query_engine,
                llm_model_name=self.model_name,
                llm_temperature=self.llm_temperature,
                # Skip greeting for better telephone experience
                skip_greeting=True
            )
            await self.conversation_manager.init()
            
            # Initialize TTS client - use Standard voice for lower latency
            self.tts_client = GoogleCloudTTS(
                credentials_path=self.credentials_path,
                voice_name="en-US-Standard-D"  # Standard voice is 3-4x faster than Neural
            )
            
            logger.info("Voice AI Agent initialization complete with enhanced speech processing")
            
        except Exception as e:
            logger.error(f"Error initializing Voice AI Agent: {e}", exc_info=True)
            raise
    
    def process_audio_with_enhanced_preprocessing(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio with optimized preprocessing for faster processing.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        # Track preprocessing time
        start_time = time.time()
        
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Use the dedicated audio preprocessor
        processed_audio = self.audio_preprocessor.process_audio(audio_data)
        
        # Track preprocessing latency
        preprocessing_time = time.time() - start_time
        self._preprocessing_times.append(preprocessing_time)
        
        # Keep only recent samples
        if len(self._preprocessing_times) > self._max_preprocessing_samples:
            self._preprocessing_times.pop(0)
        
        # Log average preprocessing time periodically
        if len(self._preprocessing_times) % 50 == 0:
            avg_time = sum(self._preprocessing_times) / len(self._preprocessing_times)
            logger.info(f"Average audio preprocessing time: {avg_time*1000:.2f}ms")
        
        return processed_audio
    
    def detect_speech_with_enhanced_processor(self, audio_data: np.ndarray) -> bool:
        """
        Detect speech using the enhanced AudioPreprocessor with dual thresholds.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if speech is detected
        """
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Use the dedicated audio preprocessor's speech detection
        return self.audio_preprocessor.contains_speech(audio_data)
    
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with enhanced speech/noise discrimination and early detection.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not hasattr(self, 'speech_recognizer') or not self.speech_recognizer:
            raise RuntimeError("Voice AI Agent not initialized")
        
        start_time = time.time()
        
        # Process audio for better speech recognition
        if isinstance(audio_data, np.ndarray):
            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Apply enhanced preprocessing
            audio_data = self.process_audio_with_enhanced_preprocessing(audio_data)
            
            # Early detection - check for speech onset after minimal processing
            contains_speech = self.detect_speech_with_enhanced_processor(audio_data)
            if not contains_speech:
                logger.info("No speech detected in audio, skipping processing")
                return {
                    "status": "no_speech",
                    "transcription": "",
                    "error": "No speech detected",
                    "processing_time": time.time() - start_time
                }
        
        # Start processing early after detecting speech onset
        # Use smaller chunk sizes (100ms) for more responsive processing
        
        # Use STT integration for processing
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Add additional validation for noise filtering
        transcription = result.get("transcription", "")
        
        # Filter out noise-only transcriptions
        noise_keywords = ["click", "keyboard", "tongue", "scoff", "static", "*", "(", ")"]
        if any(keyword in transcription.lower() for keyword in noise_keywords):
            logger.info(f"Filtered out noise transcription: '{transcription}'")
            return {
                "status": "filtered_noise",
                "transcription": transcription,
                "filtered": True,
                "processing_time": time.time() - start_time
            }
        
        # Only process valid transcriptions - reduced minimum length for faster responses
        if result.get("is_valid", False) and transcription and len(transcription.split()) >= 2:
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager with asyncio tasks for parallelism
            response_task = asyncio.create_task(self.conversation_manager.handle_user_input(transcription))
            
            # Use asyncio.gather to process other tasks in parallel if needed
            response = await response_task
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Log performance metrics
            logger.info(f"Total processing time: {total_time*1000:.2f}ms")
            
            return {
                "transcription": transcription,
                "response": response.get("response", ""),
                "status": "success",
                "processing_time": total_time
            }
        else:
            logger.info(f"Invalid or too short transcription: '{transcription}'")
            return {
                "status": "invalid_transcription",
                "transcription": transcription,
                "error": "No valid speech detected",
                "processing_time": time.time() - start_time
            }
    
    async def process_streaming_audio(
        self,
        audio_stream,
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process streaming audio with parallel pipeline for lower latency.
        
        Args:
            audio_stream: Async iterator of audio chunks
            result_callback: Callback for streaming results
            
        Returns:
            Final processing stats
        """
        if not hasattr(self, 'speech_recognizer') or not self.speech_recognizer:
            raise RuntimeError("Voice AI Agent not initialized")
            
        # Track stats
        start_time = time.time()
        chunks_processed = 0
        results_count = 0
        
        # Start streaming session
        if hasattr(self.speech_recognizer, 'start_streaming'):
            await self.speech_recognizer.start_streaming()
        
        # Use a buffer for early detection
        audio_buffer = []
        speech_detected = False
        chunks_since_speech_start = 0
        
        try:
            # Process each audio chunk
            async for chunk in audio_stream:
                chunks_processed += 1
                
                # Process audio for better recognition if it's numpy array
                if isinstance(chunk, np.ndarray):
                    # Ensure float32 format
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32)
                    
                    # Apply enhanced preprocessing
                    chunk = self.process_audio_with_enhanced_preprocessing(chunk)
                    
                    # Early detection - track speech onset
                    if not speech_detected and self.detect_speech_with_enhanced_processor(chunk):
                        speech_detected = True
                        logger.debug("Speech detected, starting early processing")
                    
                    # If speech is detected, track chunks since detection
                    if speech_detected:
                        chunks_since_speech_start += 1
                        audio_buffer.append(chunk)
                        
                        # Start processing after minimal detection (100-150ms of speech)
                        if chunks_since_speech_start >= 3:
                            # Process audio buffer in parallel
                            buffer_array = np.concatenate(audio_buffer)
                            
                            # Start processing as a task to avoid blocking
                            process_task = asyncio.create_task(
                                self._process_buffer(buffer_array, result_callback)
                            )
                            
                            # Reset buffer and detection
                            audio_buffer = []
                            speech_detected = False
                            chunks_since_speech_start = 0
                            
                            # Increment results count
                            results_count += 1
                    elif chunks_processed % 5 == 0:
                        # Skip some chunks for efficiency if no speech
                        continue
                
                # Convert to bytes for STT if needed
                if isinstance(chunk, np.ndarray):
                    audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = chunk
                    
                # Process through STT
                if hasattr(self.speech_recognizer, 'process_audio_chunk'):
                    await self.speech_recognizer.process_audio_chunk(audio_bytes)
                
            # Process any remaining audio in buffer
            if audio_buffer:
                buffer_array = np.concatenate(audio_buffer)
                await self._process_buffer(buffer_array, result_callback)
                results_count += 1
            
            # Stop streaming session
            if hasattr(self.speech_recognizer, 'stop_streaming'):
                await self.speech_recognizer.stop_streaming()
            
            # Return stats
            return {
                "status": "complete",
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming audio: {e}")
            
            # Stop streaming session
            if hasattr(self.speech_recognizer, 'stop_streaming'):
                await self.speech_recognizer.stop_streaming()
            
            return {
                "status": "error",
                "error": str(e),
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
            }
    
    async def _process_buffer(self, buffer_array: np.ndarray, result_callback: Optional[Callable]) -> None:
        """Process a buffer of audio with parallel tasks for lower latency."""
        try:
            # Convert to bytes for STT
            audio_bytes = (buffer_array * 32767).astype(np.int16).tobytes()
            
            # Start STT processing
            stt_start = time.time()
            result = await self.stt_integration.transcribe_audio_data(audio_bytes)
            stt_time = time.time() - stt_start
            
            transcription = result.get("transcription", "")
            
            # Only process if valid transcription
            if result.get("is_valid", False) and transcription and len(transcription.split()) >= 2:
                # Start LLM processing as a task
                llm_start = time.time()
                response_task = asyncio.create_task(
                    self.conversation_manager.handle_user_input(transcription)
                )
                
                # Wait for LLM processing
                response = await response_task
                llm_time = time.time() - llm_start
                
                # Log performance breakdown
                logger.info(f"Performance: STT {stt_time*1000:.2f}ms, LLM {llm_time*1000:.2f}ms")
                
                # Format result with timing breakdown
                result_data = {
                    "transcription": transcription,
                    "response": response.get("response", ""),
                    "confidence": result.get("confidence", 1.0),
                    "is_final": True,
                    "stt_time_ms": stt_time * 1000,
                    "llm_time_ms": llm_time * 1000,
                    "total_time_ms": (stt_time + llm_time) * 1000
                }
                
                # Call callback if provided
                if result_callback:
                    await result_callback(result_data)
        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return (self.speech_recognizer is not None and 
                self.conversation_manager is not None and 
                self.query_engine is not None)
                
    async def shutdown(self):
        """Shut down all components properly."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Close STT streaming session if active
        if self.speech_recognizer:
            if hasattr(self.speech_recognizer, 'is_streaming') and getattr(self.speech_recognizer, 'is_streaming', False):
                if hasattr(self.speech_recognizer, 'stop_streaming'):
                    await self.speech_recognizer.stop_streaming()
        
        # Reset conversation if active
        if self.conversation_manager:
            self.conversation_manager.reset()
            
        # Reset audio preprocessor state
        self.audio_preprocessor.reset()