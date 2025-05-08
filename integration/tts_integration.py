"""
TTS Integration module for Voice AI Agent using Google Cloud TTS.

This module provides classes and functions for integrating text-to-speech
capabilities with the Voice AI Agent system, optimized for low latency.
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

from text_to_speech import GoogleCloudTTS, RealTimeResponseHandler, AudioProcessor

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration for Voice AI Agent using Google Cloud TTS.
    
    Provides an abstraction layer for TTS functionality,
    handling initialization, single-text processing, and streaming capabilities.
    """
    
    def __init__(
        self,
        voice: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice: Voice name to use for Google Cloud TTS
            enable_caching: Whether to enable TTS caching
        """
        self.voice = voice
        self.enable_caching = enable_caching
        self.tts_client = None
        self.tts_handler = None
        self.initialized = False
        
        # Parameters for better conversation flow
        self.add_pause_after_speech = True
        self.pause_duration_ms = 500  # 500ms pause after speech
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Initialize the Google Cloud TTS client
            self.tts_client = GoogleCloudTTS(
                voice_name=self.voice, 
                enable_caching=self.enable_caching
            )
            
            # Initialize the RealTimeResponseHandler
            self.tts_handler = RealTimeResponseHandler(tts_client=self.tts_client)
            
            self.initialized = True
            logger.info(f"Initialized TTS with voice: {self.voice or 'default'}")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech with optimized chunking strategy.
        
        Args:
            text: Text to convert
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Use the optimized TTS client
            audio_data = await self.tts_client.text_to_speech(text)
            
            # Ensure even number of bytes & add pause
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            if self.add_pause_after_speech:
                silence_size = int(16000 * (self.pause_duration_ms / 1000) * 2)
                silence_data = b'\x00' * silence_size
                audio_data = audio_data + silence_data
            
            return audio_data
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            # Return silence as last resort
            silence_size = int(16000 * 0.5 * 2)
            return b'\x00' * silence_size
    
    async def text_to_speech_streaming(
        self, 
        text: str
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech with optimized chunking for low latency.
        
        Args:
            text: Text to convert
            
        Yields:
            Audio data chunks
        """
        if not self.initialized:
            await self.init()
        
        # Use sentence-based chunking for better streaming
        sentences = []
        for sentence in text.replace('!', '.').replace('?', '.').split('.'):
            if sentence.strip():
                sentences.append(sentence.strip() + '.')
        
        # Create a text generator
        async def text_generator():
            for sentence in sentences:
                yield sentence
                
                # Add small delay between sentences for natural flow
                await asyncio.sleep(0.01)
        
        # Use the specialized streaming method
        async for audio_chunk in self.tts_client.synthesize_streaming(text_generator()):
            yield audio_chunk
            
            # Short artificial sleep to avoid flooding the output
            await asyncio.sleep(0.01)
        
        # Add final silence for natural pause
        if self.add_pause_after_speech:
            silence_size = int(16000 * (self.pause_duration_ms / 1000) * 2)
            yield b'\x00' * silence_size
    
    async def process_realtime_text(
        self,
        text_chunks: AsyncIterator[str],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process text chunks in real-time and generate speech.
        
        Args:
            text_chunks: Async iterator of text chunks
            audio_callback: Callback to handle audio data
            
        Returns:
            Statistics about the processing
        """
        if not self.initialized:
            await self.init()
        
        # Start measuring time
        start_time = time.time()
        
        # Reset the TTS handler for this new session
        if self.tts_handler:
            await self.tts_handler.stop()
            self.tts_handler = RealTimeResponseHandler(tts_client=self.tts_client)
        
        # Process each text chunk
        total_chunks = 0
        total_audio_bytes = 0
        
        try:
            async for chunk in text_chunks:
                if not chunk or not chunk.strip():
                    continue
                
                # Process the text chunk as SSML
                audio_data = await self.tts_client.text_to_speech(chunk)
                
                # Track statistics
                total_chunks += 1
                total_audio_bytes += len(audio_data)
                
                # Send audio to callback
                await audio_callback(audio_data)
                
                # Log progress periodically
                if total_chunks % 10 == 0:
                    logger.debug(f"Processed {total_chunks} text chunks")
        
        except Exception as e:
            logger.error(f"Error processing realtime text: {e}")
            return {
                "error": str(e),
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "elapsed_time": time.time() - start_time
            }
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        
        return {
            "total_chunks": total_chunks,
            "total_audio_bytes": total_audio_bytes,
            "elapsed_time": elapsed_time,
            "avg_chunk_size": total_audio_bytes / total_chunks if total_chunks > 0 else 0
        }
    
    async def process_ssml(self, ssml: str) -> bytes:
        """
        Process SSML text and convert to speech.
        
        Args:
            ssml: SSML-formatted text
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Ensure proper SSML format
            if not ssml.startswith('<speak>'):
                ssml = f"<speak>{ssml}</speak>"
                
            audio_data = await self.tts_client.synthesize(ssml, is_ssml=True)
            
            # Ensure even number of bytes
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            # Add a pause at the end if needed
            if self.add_pause_after_speech:
                silence_size = int(16000 * (self.pause_duration_ms / 1000) * 2)
                silence_data = b'\x00' * silence_size
                audio_data = audio_data + silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause after SSML speech")
                
            return audio_data
        except Exception as e:
            logger.error(f"Error in SSML processing: {e}")
            # Return silent audio rather than raising
            silence_size = int(16000 * 0.5 * 2)  # 500ms of silence
            return b'\x00' * silence_size
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.tts_handler:
            try:
                await self.tts_handler.stop()
            except Exception as e:
                logger.error(f"Error during TTS cleanup: {e}")