# speech_to_text/stt_integration.py

"""
Speech-to-Text integration module optimized for low latency with minimal processing.
"""
import logging
import time
import os
import json
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

logger = logging.getLogger(__name__)

class STTIntegration:
    """
    Speech-to-Text integration optimized for low latency with zero processing overhead.
    Uses Google Cloud Speech-to-Text v2 API optimally for telephony.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[GoogleCloudStreamingSTT] = None,
        language: str = "en-US"
    ):
        """Initialize the STT integration."""
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        logger.info("STTIntegration initialized for low latency with zero processing overhead")
    
    async def init(self, project_id: Optional[str] = None) -> None:
        """Initialize the STT component if not already initialized."""
        if self.initialized:
            return
        
        # Get project ID with automatic extraction
        final_project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # If not provided, try to extract from credentials file
        if not final_project_id:
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_file and os.path.exists(credentials_file):
                try:
                    with open(credentials_file, 'r') as f:
                        creds_data = json.load(f)
                        final_project_id = creds_data.get('project_id')
                        logger.info(f"STTIntegration: Auto-extracted project ID from credentials: {final_project_id}")
                except Exception as e:
                    logger.error(f"Error reading credentials file: {e}")
        
        if not final_project_id:
            raise ValueError(
                "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or ensure your credentials file contains a project_id field."
            )
            
        try:
            # Create optimized Google Cloud v2 streaming client with low-latency settings
            self.speech_recognizer = GoogleCloudStreamingSTT(
                language=self.language,
                sample_rate=8000,  # Match Twilio exactly
                encoding="MULAW",  # Match Twilio exactly
                channels=1,
                interim_results=True,  # Enable interim results for early processing
                project_id=final_project_id,
                location="global"
            )
            
            self.initialized = True
            logger.info(f"Initialized STT with Google Cloud v2 API (low-latency optimized)")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def cleanup_transcription(self, text: str) -> str:
        """Absolutely minimal cleanup for speed."""
        return text.strip()
    
    def is_valid_transcription(self, text: str, min_words: int = 1) -> bool:
        """Simplified validation for speed."""
        cleaned = text.strip()
        
        if not cleaned:
            return False
        
        # Check if it has at least one word
        words = cleaned.split()
        return len(words) >= min_words
    
    def set_speaking_state(self, is_speaking: bool):
        """Set speaking state on the STT recognizer."""
        if self.speech_recognizer and hasattr(self.speech_recognizer, 'set_speaking_state'):
            self.speech_recognizer.set_speaking_state(is_speaking)
    
    async def transcribe_audio_data(
        self,
        audio_data: Union[bytes, List[float]],
        is_short_audio: bool = False,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Process audio with zero-overhead for maximum speed."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Convert list to bytes if needed (no other processing)
            if isinstance(audio_data, list):
                audio_data = bytes(audio_data)
            
            # Send directly to STT with zero preprocessing
            final_results = []
            
            # Define a callback to collect results
            async def store_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                
                # Call the original callback if provided
                if callback:
                    await callback(result)
            
            # Make sure STT is streaming
            await self.speech_recognizer.start_streaming()
            
            # Process the audio directly
            result = await self.speech_recognizer.process_audio_chunk(audio_data, store_result)
            
            # Get the best result if we have final results
            if final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
                confidence = best_result.confidence
            else:
                # No results yet
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": 0.0,
                    "processing_time": time.time() - start_time,
                    "is_final": False,
                    "is_valid": False
                }
            
            # Minimal validation
            is_valid = self.is_valid_transcription(transcription)
            
            return {
                "transcription": transcription,
                "confidence": confidence,
                "duration": 0.0,
                "processing_time": time.time() - start_time,
                "is_final": True,
                "is_valid": is_valid,
                "api_version": "v2"
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "api_version": "v2"
            }
    
    async def start_streaming(self) -> None:
        """Start a new streaming transcription session."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return
        
        await self.speech_recognizer.start_streaming()
        logger.debug("Started streaming transcription session")
    
    async def process_stream_chunk(
        self,
        audio_chunk: Union[bytes, List[float]],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process a chunk with zero modifications for speed."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return None
        
        # Convert list to bytes if needed
        if isinstance(audio_chunk, list):
            audio_chunk = bytes(audio_chunk)
        
        # Pass directly to STT without any processing
        return await self.speech_recognizer.process_audio_chunk(
            audio_chunk=audio_chunk,
            callback=callback
        )
    
    async def end_streaming(self) -> Tuple[str, float]:
        """End the streaming session."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return "", 0.0
        
        # Stop streaming session
        return await self.speech_recognizer.stop_streaming()