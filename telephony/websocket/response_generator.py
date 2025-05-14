# telephony/websocket/response_generator.py

"""
Response generation and TTS handling for telephony.
"""
import logging
import asyncio
import base64
import json
from typing import Optional

from text_to_speech import GoogleCloudTTS
from telephony.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles response generation and text-to-speech conversion."""
    
    def __init__(self, pipeline, ws_handler):
        """Initialize response generator."""
        self.pipeline = pipeline
        self.ws_handler = ws_handler
        self.audio_processor = AudioProcessor()
        self.tts_client = None
        self._initialize_tts()
    
    def _initialize_tts(self) -> None:
        """Initialize Google Cloud TTS client."""
        try:
            import os
            
            # Initialize Google Cloud TTS with telephony optimization
            self.tts_client = GoogleCloudTTS(
                voice_name=os.environ.get("TTS_VOICE_NAME", "en-US-Standard-J"),
                voice_gender=os.environ.get("TTS_VOICE_GENDER", "NEUTRAL"),
                language_code=os.environ.get("TTS_LANGUAGE_CODE", "en-US"),
                container_format="mulaw",  # For Twilio compatibility
                sample_rate=8000,          # Match Twilio's rate
                enable_caching=True
            )
            logger.info(f"Initialized Google Cloud TTS")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud TTS: {e}")
            # Will fall back to pipeline TTS
    
    async def generate_response(self, transcription: str) -> Optional[str]:
        """
        Generate response from knowledge base.
        
        Args:
            transcription: User transcription
            
        Returns:
            Generated response text
        """
        try:
            if hasattr(self.pipeline, 'query_engine'):
                query_result = await self.pipeline.query_engine.query(transcription)
                return query_result.get("response", "")
            return None
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    async def convert_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert
            
        Returns:
            Audio data as bytes (mulaw format)
        """
        try:
            if self.tts_client:
                # Use Google Cloud TTS
                return await self.tts_client.synthesize(text)
            else:
                # Fallback to pipeline TTS
                speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                # Convert to mulaw for Twilio
                return self.audio_processor.convert_to_mulaw_direct(speech_audio)
        except Exception as e:
            logger.error(f"Error in TTS conversion: {e}")
            raise
    
    async def send_text_response(self, text: str, ws) -> None:
        """
        Send text response by converting to speech and sending to WebSocket.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        try:
            # Set speaking state
            self.ws_handler.audio_manager.set_speaking_state(True)
            
            # Convert to speech (already in mulaw format)
            mulaw_audio = await self.convert_to_speech(text)
            
            # Send audio
            await self._send_audio(mulaw_audio, ws)
            
            logger.info(f"Sent text response: '{text}'")
            
            # Update state
            self.ws_handler.audio_manager.set_speaking_state(False)
            self.ws_handler.audio_manager.update_response_time()
            
        except Exception as e:
            logger.error(f"Error sending text response: {e}")
            # Ensure speaking state is reset
            self.ws_handler.audio_manager.set_speaking_state(False)
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """Send audio data to WebSocket."""
        try:
            stream_sid = self.ws_handler.stream_sid
            
            if not stream_sid:
                logger.error("No stream_sid available for sending audio")
                return
            
            # Split into chunks for better streaming
            chunks = self._split_audio_into_chunks(audio_data)
            
            logger.debug(f"Sending {len(chunks)} audio chunks")
            
            for i, chunk in enumerate(chunks):
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                try:
                    ws.send(json.dumps(message))
                    # Small delay between chunks for smooth playback
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.02)
                except Exception as e:
                    logger.error(f"Error sending chunk {i}: {e}")
                    if "Connection closed" in str(e):
                        logger.warning("WebSocket connection closed during audio send")
                        return
            
            logger.debug(f"Successfully sent {len(chunks)} audio chunks")
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
    
    def _split_audio_into_chunks(self, audio_data: bytes) -> list:
        """Split audio into smaller chunks for streaming."""
        chunk_size = 800  # 100ms at 8kHz
        chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
            
        return chunks