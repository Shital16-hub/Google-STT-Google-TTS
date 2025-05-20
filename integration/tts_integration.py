# integration/tts_integration.py

"""
Updated TTS integration using Google Cloud TTS with low-latency optimizations.
"""
import logging
import asyncio
import os
import time
import re
from typing import Optional, Dict, Any, List, AsyncIterator

from google.cloud import texttospeech
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Optimized Text-to-Speech integration using Google Cloud TTS.
    Implements streaming, chunking, and caching for faster responses.
    """
    
    def __init__(
        self,
        voice_name: Optional[str] = None,
        voice_gender: Optional[str] = None,
        language_code: Optional[str] = "en-US",
        enable_caching: bool = True,
        credentials_file: Optional[str] = None,
        voice_type: str = "NEURAL2"
    ):
        """
        Initialize the TTS integration with optimized settings.
        
        Args:
            voice_name: Voice name (e.g., "en-US-Neural2-C")
            voice_gender: Voice gender (None for Neural2 voices)
            language_code: Language code (defaults to en-US)
            enable_caching: Whether to enable TTS caching
            credentials_file: Path to Google Cloud credentials file
            voice_type: Voice type (NEURAL2, STANDARD, etc.)
        """
        # Set default voice name if not provided
        if not voice_name:
            voice_name = "en-US-Neural2-C"  # Default Neural2 voice
        
        # Don't set gender for Neural2 voices
        if voice_name and "Neural2" in voice_name:
            voice_gender = None
            
        self.voice_name = voice_name
        self.voice_gender = voice_gender
        self.language_code = language_code or "en-US"
        self.enable_caching = enable_caching
        self.credentials_file = credentials_file
        self.voice_type = voice_type
        self.tts_client = None
        self.initialized = False
        
        # Response chunking settings
        self.sentence_splitter = re.compile(r'(?<=[.!?])\s+')
        self.max_chunk_size = 100  # Maximum characters per chunk
        
        # Cache for common responses
        self.response_cache = {}
        
        # Performance metrics
        self.synthesis_times = []
        self.total_characters = 0
        self.total_synthesis_time = 0
        
        logger.info(f"TTSIntegration initialized with voice: {self.voice_name}")
    
    async def init(self) -> None:
        """Initialize the TTS client."""
        if self.initialized:
            return
            
        try:
            # Initialize Google Cloud TTS client with proper credentials
            if self.credentials_file and os.path.exists(self.credentials_file):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                self.tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
            else:
                self.tts_client = texttospeech.TextToSpeechClient()
            
            # Create audio config optimized for telephony
            self.audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MULAW,
                sample_rate_hertz=8000,
                effects_profile_id=["telephony-class-application"]
            )
            
            # Create voice selection params
            self.voice_params = texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=self.voice_name
            )
            
            # Add gender only if it's set and appropriate
            if self.voice_gender and not "Neural2" in self.voice_name:
                if self.voice_gender.upper() == "MALE":
                    self.voice_params.ssml_gender = texttospeech.SsmlVoiceGender.MALE
                elif self.voice_gender.upper() == "FEMALE":
                    self.voice_params.ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
                elif self.voice_gender.upper() == "NEUTRAL":
                    self.voice_params.ssml_gender = texttospeech.SsmlVoiceGender.NEUTRAL
            
            self.initialized = True
            logger.info(f"TTS client initialized with voice: {self.voice_name}")
            
            # Initialize cache as a separate step *after* marking as initialized
            # to avoid recursion issues
            if self.enable_caching:
                asyncio.create_task(self._initialize_cache())
                
        except Exception as e:
            logger.error(f"Error initializing TTS client: {e}")
            raise
    
    async def _initialize_cache(self):
        """Initialize cache with common phrases for immediate playback."""
        common_phrases = [
            "How can I help you today?",
            "I'm sorry, could you repeat that?",
            "Thank you for your question.",
            "Let me think about that.",
            "Is there anything else you'd like to know?"
        ]
        
        for phrase in common_phrases:
            try:
                # Generate audio and add to cache
                audio = await self._synthesize_internal(phrase)
                self.response_cache[phrase] = audio
                logger.debug(f"Cached phrase: {phrase} ({len(audio)} bytes)")
            except Exception as e:
                logger.error(f"Error caching phrase '{phrase}': {e}")
    
    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech with optimized latency.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        # Check cache for exact matches
        if self.enable_caching and text in self.response_cache:
            logger.debug(f"Cache hit for text: {text[:30]}...")
            return self.response_cache[text]
        
        return await self._synthesize_internal(text)
    
    async def _synthesize_internal(self, text: str) -> bytes:
        """
        Internal synthesis method to avoid recursion issues.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
        """
        start_time = time.time()
        
        try:
            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Generate speech
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice_params,
                audio_config=self.audio_config
            )
            
            audio_content = response.audio_content
            
            # Update performance metrics
            synthesis_time = time.time() - start_time
            self.synthesis_times.append(synthesis_time)
            self.total_characters += len(text)
            self.total_synthesis_time += synthesis_time
            
            # Cache the result if it's not too long
            if self.enable_caching and len(text) < 200 and text not in self.response_cache:
                self.response_cache[text] = audio_content
            
            logger.debug(f"Synthesized {len(text)} chars in {synthesis_time:.2f}s")
            return audio_content
            
        except Exception as e:
            logger.error(f"Error in TTS synthesis: {e}")
            raise
    
    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """
        Stream synthesized speech for faster playback.
        
        This method breaks the text into smaller chunks and streams them
        as they're synthesized for lower perceived latency.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks as they are generated
        """
        if not self.initialized:
            await self.init()
        
        # Break text into sentences for faster response
        if len(text) > self.max_chunk_size:
            chunks = self._split_into_chunks(text)
        else:
            chunks = [text]
        
        # Synthesize and stream each chunk
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            # Check cache first
            if self.enable_caching and chunk in self.response_cache:
                yield self.response_cache[chunk]
                continue
            
            # Synthesize new chunk
            try:
                audio_data = await self._synthesize_internal(chunk)
                yield audio_data
            except Exception as e:
                logger.error(f"Error synthesizing chunk: {e}")
                # Continue with next chunk instead of failing completely
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into optimal chunks for faster synthesis.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # First try to split by sentences
        sentences = self.sentence_splitter.split(text)
        
        # Then ensure each sentence is not too long
        chunks = []
        for sentence in sentences:
            if len(sentence) <= self.max_chunk_size:
                chunks.append(sentence)
            else:
                # Split long sentences by commas or natural pauses
                parts = re.split(r'(?<=[,:])\s+', sentence)
                
                current_chunk = ""
                for part in parts:
                    if len(current_chunk) + len(part) > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = part
                    else:
                        if current_chunk:
                            current_chunk += ", " + part
                        else:
                            current_chunk = part
                
                if current_chunk:
                    chunks.append(current_chunk)
        
        return chunks
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Alias for synthesize method (for compatibility).
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        return await self.synthesize(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for TTS synthesis."""
        stats = {
            "voice_name": self.voice_name,
            "language_code": self.language_code,
            "cached_responses": len(self.response_cache),
            "total_characters": self.total_characters,
            "total_synthesis_time": self.total_synthesis_time
        }
        
        if self.synthesis_times:
            stats.update({
                "avg_synthesis_time": sum(self.synthesis_times) / len(self.synthesis_times),
                "max_synthesis_time": max(self.synthesis_times),
                "min_synthesis_time": min(self.synthesis_times),
                "chars_per_second": self.total_characters / max(self.total_synthesis_time, 0.001)
            })
        
        return stats