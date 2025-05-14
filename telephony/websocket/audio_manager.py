# telephony/websocket/audio_manager.py

"""
Audio manager for processing Twilio media streams with minimal overhead.
"""
import base64
import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import time

from telephony.audio_processor import AudioProcessor, MulawBufferProcessor

logger = logging.getLogger(__name__)

class AudioManager:
    """
    Audio manager that handles Twilio media with optimized buffering.
    """
    
    def __init__(self):
        """Initialize audio manager."""
        logger.info("Initializing AudioManager")
        
        self.audio_processor = AudioProcessor()
        self.mulaw_buffer = MulawBufferProcessor(min_chunk_size=6400)  # 800ms at 8kHz
        
        # State management
        self.is_speaking = False
        self.last_response_time = time.time()
        
        # Performance tracking
        self.media_events_received = 0
        self.audio_chunks_sent = 0
        self.total_audio_bytes = 0
        
        # Timing optimization
        self.last_processing_time = 0
        self.min_processing_interval = 0.05  # 50ms minimum
        
        logger.info("AudioManager initialized")
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Process incoming Twilio media event."""
        self.media_events_received += 1
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.debug(f"Media event #{self.media_events_received}: No payload")
            return None
        
        try:
            # Decode mulaw audio data
            mulaw_data = base64.b64decode(payload)
            audio_size = len(mulaw_data)
            self.total_audio_bytes += audio_size
            
            # Skip processing if system is speaking
            if self.is_speaking:
                logger.debug("Skipping audio - system is speaking")
                return None
            
            # Check minimum time interval
            current_time = time.time()
            if current_time - self.last_processing_time < self.min_processing_interval:
                # Buffer for later processing
                self.mulaw_buffer.process(mulaw_data)
                return None
            
            # Process with mulaw buffer
            processed_audio = self.mulaw_buffer.process(mulaw_data)
            
            if processed_audio:
                self.last_processing_time = current_time
                self.audio_chunks_sent += 1
                
                logger.info(f"Sending audio chunk #{self.audio_chunks_sent}: "
                           f"{len(processed_audio)} bytes")
                
                return processed_audio
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing media: {e}", exc_info=True)
            return None
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set speaking state with buffer management."""
        if self.is_speaking != speaking:
            logger.info(f"Speaking state changed: {self.is_speaking} -> {speaking}")
            self.is_speaking = speaking
            
            if speaking:
                # Clear buffer size when starting to speak
                buffer_size = self.mulaw_buffer.clear_buffer()
                logger.info(f"Cleared {buffer_size} bytes from buffer")
    
    def clear_buffer(self) -> None:
        """Clear audio buffer."""
        self.mulaw_buffer.clear_buffer()
    
    def update_response_time(self) -> None:
        """Update response time."""
        self.last_response_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audio statistics."""
        current_time = time.time()
        success_rate = (self.audio_chunks_sent / max(self.media_events_received, 1)) * 100
        
        return {
            "media_events_received": self.media_events_received,
            "audio_chunks_sent": self.audio_chunks_sent,
            "total_audio_bytes": self.total_audio_bytes,
            "is_speaking": self.is_speaking,
            "time_since_response": current_time - self.last_response_time,
            "success_rate": round(success_rate, 2),
            "compression_ratio": round(self.audio_chunks_sent / max(self.media_events_received, 1), 3),
            "buffer_stats": self.mulaw_buffer.get_stats()
        }