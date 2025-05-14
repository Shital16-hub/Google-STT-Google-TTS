# telephony/websocket/speech_processor.py

"""
Speech processor using optimized Google Cloud STT for telephony.
"""
import logging
import re
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Speech processor optimized for telephony using Google Cloud STT.
    """
    
    def __init__(self, pipeline):
        """Initialize speech processor."""
        self.pipeline = pipeline
        
        logger.info("Initializing SpeechProcessor with Google Cloud STT")
        
        # Configure Google Cloud STT for telephony
        self.speech_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,         # Match Twilio's rate
            encoding="MULAW",         # Match Twilio's encoding
            channels=1,
            interim_results=False,    # Disable for better final results
            enhanced_model=True       # Use enhanced telephony model
        )
        
        # Minimal cleanup patterns
        self.cleanup_patterns = [
            (re.compile(r'\[.*?\]'), ''),  # Remove [inaudible], [music], etc.
            (re.compile(r'\<.*?\>'), ''),  # Remove <noise>, etc.
            (re.compile(r'\s+'), ' '),     # Normalize whitespace
        ]
        
        # Echo detection
        self.echo_history = []
        self.max_echo_history = 5
        
        # Statistics
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.failed_transcriptions = 0
        
        # Initialize speech session
        asyncio.create_task(self._initialize_session())
        
        logger.info("SpeechProcessor initialized")
    
    async def _initialize_session(self):
        """Initialize Google Cloud Speech session."""
        try:
            await self.speech_client.start_streaming()
            logger.info("Google Cloud Speech session started")
        except Exception as e:
            logger.error(f"Error starting speech session: {e}")
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio through Google Cloud STT.
        
        Args:
            audio_data: Audio data in mulaw format
            callback: Optional callback for interim results
            
        Returns:
            Transcribed text or None
        """
        self.audio_chunks_received += 1
        
        try:
            logger.debug(f"Processing audio chunk #{self.audio_chunks_received}, "
                        f"size: {len(audio_data)} bytes")
            
            # Process through Google Cloud STT
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=callback
            )
            
            if result and result.text:
                self.successful_transcriptions += 1
                
                # Apply minimal cleanup
                cleaned_text = self.cleanup_transcription(result.text)
                
                if cleaned_text:
                    confidence = getattr(result, 'confidence', 0.0)
                    logger.info(f"Transcription: '{cleaned_text}' (confidence: {confidence:.2f})")
                    return cleaned_text
                else:
                    logger.debug("Transcription cleaned to empty")
            else:
                self.failed_transcriptions += 1
                logger.debug("No transcription result")
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            self.failed_transcriptions += 1
            return None
    
    def cleanup_transcription(self, text: str) -> str:
        """Apply minimal cleanup to preserve natural speech."""
        if not text:
            return ""
        
        original_text = text
        cleaned = text
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Basic normalization
        cleaned = cleaned.strip()
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        if original_text != cleaned:
            logger.debug(f"Cleaned: '{original_text}' -> '{cleaned}'")
        
        return cleaned
    
    def is_valid_transcription(self, text: str) -> bool:
        """Validate transcription with minimal requirements."""
        cleaned = self.cleanup_transcription(text)
        
        if not cleaned:
            return False
        
        # Must have at least one word
        words = cleaned.split()
        if len(words) < 1:
            return False
        
        # Must be longer than 1 character
        if len(cleaned) < 2:
            return False
        
        return True
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """Detect echo using string matching."""
        if not transcription or not self.echo_history:
            return False
        
        normalized_trans = transcription.lower().strip()
        
        for recent_response in self.echo_history:
            normalized_response = recent_response.lower().strip()
            
            if self._is_echo_match(normalized_trans, normalized_response):
                logger.info(f"Detected echo: '{transcription}'")
                return True
        
        return False
    
    def _is_echo_match(self, transcription: str, response: str) -> bool:
        """Check if transcription matches response."""
        # Exact match
        if transcription == response:
            return True
        
        # Substring match for longer texts
        if len(response) > 15:
            if transcription in response or response in transcription:
                return True
        
        # Word overlap detection
        trans_words = set(transcription.split())
        response_words = set(response.split())
        
        if trans_words and response_words:
            overlap_ratio = len(trans_words & response_words) / max(len(trans_words), len(response_words))
            return overlap_ratio > 0.7
        
        return False
    
    def add_to_echo_history(self, response: str) -> None:
        """Add response to echo history."""
        if response and len(response) > 5:
            self.echo_history.append(response)
            if len(self.echo_history) > self.max_echo_history:
                self.echo_history.pop(0)
            logger.debug(f"Added to echo history: '{response[:30]}...'")
    
    async def stop_speech_session(self) -> None:
        """Stop the speech session."""
        try:
            await self.speech_client.stop_streaming()
            logger.info("Speech session stopped")
        except Exception as e:
            logger.error(f"Error stopping speech session: {e}")
        
        # Log statistics
        logger.info(f"Total chunks: {self.audio_chunks_received}")
        logger.info(f"Successful: {self.successful_transcriptions}")
        logger.info(f"Failed: {self.failed_transcriptions}")
        
        success_rate = (self.successful_transcriptions / max(self.audio_chunks_received, 1)) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "audio_chunks_received": self.audio_chunks_received,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "echo_history_size": len(self.echo_history),
            "success_rate": round((self.successful_transcriptions / max(self.audio_chunks_received, 1)) * 100, 2),
            "language": "en-US",
            "sample_rate": 8000,
            "encoding": "MULAW"
        }