# telephony/simple_websocket_handler.py

"""
Optimized WebSocket handler for low-latency voice interactions.
Provides proper speaking/listening state management and streamlined processing.
"""
import json
import asyncio
import logging
import base64
import time
import os
from typing import Dict, Any, Optional

import fastapi

# Use our optimized STT implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from speech_to_text.stt_integration import STTIntegration

# Use the fixed TTS implementation
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class SimpleWebSocketHandler:
    """
    Optimized WebSocket handler for low-latency voice interactions.
    Properly manages speaking/listening states and streamlines processing flow.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with optimized voice interaction support."""
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Get project ID dynamically
        self.project_id = self._get_project_id()
        
        # Use STT integration from pipeline for consistency
        if hasattr(pipeline, 'speech_recognizer') and pipeline.speech_recognizer:
            self.stt_integration = STTIntegration(pipeline.speech_recognizer)
        else:
            # Initialize new STT with optimized settings
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            self.stt_client = GoogleCloudStreamingSTT(
                language="en-US",
                sample_rate=8000,
                encoding="MULAW",
                channels=1,
                interim_results=True,  # Enable interim results for early processing
                project_id=self.project_id,
                location="global",
                credentials_file=credentials_file
            )
            self.stt_integration = STTIntegration(self.stt_client)
        
        # Use TTS integration from pipeline for consistency
        if hasattr(pipeline, 'tts_integration') and pipeline.tts_integration:
            self.tts_client = pipeline.tts_integration
        else:
            # Initialize new TTS
            self.tts_client = GoogleCloudTTS(
                credentials_file=credentials_file,
                voice_name="en-US-Neural2-C",
                voice_gender=None,
                language_code="en-US",
                container_format="mulaw",
                sample_rate=8000,
                enable_caching=True,
                voice_type="NEURAL2"
            )
        
        # Optimized conversation state management
        self.conversation_active = True
        self.is_speaking = False
        self.call_ended = False
        
        # Audio processing with flow control
        self.audio_buffer = bytearray()
        self.chunk_size = 400  # 50ms chunks for faster processing
        
        # Session management
        self.session_start_time = time.time()
        self.last_transcription_time = time.time()
        self.last_audio_time = time.time()
        self.last_tts_time = None
        
        # Response tracking
        self.last_response_time = time.time()
        self.last_response_text = ""
        
        # Stats tracking
        self.audio_received = 0
        self.transcriptions = 0
        self.responses_sent = 0
        self.interim_transcriptions = 0
        
        # Store reference to current websocket
        self._ws = None
        
        logger.info(f"Optimized WebSocket handler initialized - Call: {call_sid}, Project: {self.project_id}")
    
    def _get_project_id(self) -> str:
        """Get project ID with enhanced error handling."""
        # Try environment variable first
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Try to extract from credentials file
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file and os.path.exists(credentials_file):
            try:
                import json
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        logger.info(f"Extracted project ID from credentials: {project_id}")
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials: {e}")
        
        # Fallback
        logger.warning("Using fallback project ID - this should be configured properly")
        return "my-tts-project-458404"
    
    async def _handle_audio(self, data: Dict[str, Any], ws: fastapi.WebSocket):
        """Handle audio with speaking/listening state management."""
        # Skip audio processing if call has ended
        if self.call_ended:
            return
        
        # Skip audio processing if we're speaking to prevent echo
        if self.is_speaking:
            return
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        # Decode audio
        try:
            audio_data = base64.b64decode(payload)
            self.audio_received += 1
            self.last_audio_time = time.time()
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Process audio directly through STT with early response processing
        try:
            await self.stt_integration.process_stream_chunk(
                audio_data, 
                callback=self._handle_transcription_result
            )
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Process transcription results with early-response handling."""
        transcription = result.text.strip()
        confidence = result.confidence
        
        # Track interim results for early processing
        if not result.is_final:
            self.interim_transcriptions += 1
            
            # Process long-enough interim results for faster response
            if len(transcription.split()) >= 4:
                logger.debug(f"Processing substantial interim result: '{transcription}'")
                await self._process_transcription(transcription, is_final=False)
            return
                
        # For final results
        if transcription:
            logger.info(f"Final transcription: '{transcription}' (confidence: {confidence:.2f})")
            self.last_transcription_time = time.time()
            self.transcriptions += 1
            
            # Process the transcription
            await self._process_transcription(transcription, is_final=True)
    
    async def _process_transcription(self, transcription: str, is_final: bool = True):
        """Process transcription with optimized flow."""
        # Skip if we're speaking to prevent processing echoes
        if self.is_speaking:
            return
            
        try:
            # Process through knowledge base
            if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                # Set speaking state to true while generating response
                self.is_speaking = True
                if hasattr(self.stt_integration, 'set_speaking_state'):
                    self.stt_integration.set_speaking_state(True)
                
                # Use streaming query for faster response
                response_text = ""
                
                if is_final:
                    # For final results, use full query
                    result = await self.pipeline.query_engine.query(transcription)
                    response_text = result.get("response", "")
                else:
                    # For interim results, use fast query path
                    # This will be a simplified query to get faster responses
                    try:
                        # Try to get a quick response based on the interim result
                        chunks = []
                        async for chunk in self.pipeline.query_engine.query_with_streaming(transcription):
                            # Collect the chunks
                            chunks.append(chunk)
                            # If we get a done signal, extract the full response
                            if chunk.get("done", False):
                                response_text = chunk.get("full_response", "")
                                break
                    except Exception as e:
                        logger.error(f"Error in streaming query: {e}")
                
                # Send response if we have one
                if response_text:
                    await self._send_response(response_text)
                elif is_final:
                    # Only for final results with no response
                    logger.warning("No response generated from knowledge base")
                    await self._send_response("I'm sorry, I couldn't find an answer to that question.")
                
                # Reset speaking state after sending response
                self.is_speaking = False
                if hasattr(self.stt_integration, 'set_speaking_state'):
                    self.stt_integration.set_speaking_state(False)
            else:
                logger.error("Pipeline or query engine not available")
                if is_final:
                    await self._send_response("I'm sorry, there's an issue with my knowledge base.")
                
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)
            if is_final:
                await self._send_response("I'm sorry, I encountered an error processing your request.")
            
            # Reset speaking state in case of error
            self.is_speaking = False
            if hasattr(self.stt_integration, 'set_speaking_state'):
                self.stt_integration.set_speaking_state(False)
    
    async def _send_response(self, text: str, ws=None):
        """Send TTS response with optimized streaming for lower latency."""
        if not text.strip() or self.call_ended:
            return
        
        # Use stored WebSocket if not provided
        if ws is None:
            ws = getattr(self, '_ws', None)
            if ws is None:
                logger.error("No WebSocket available for sending response")
                return
        
        try:
            # Set speaking state to prevent processing our own output
            self.is_speaking = True
            if hasattr(self.stt_integration, 'set_speaking_state'):
                self.stt_integration.set_speaking_state(True)
                
            self.last_response_time = time.time()
            self.last_response_text = text
            
            logger.info(f"Sending response: '{text}'")
            
            # Stream TTS for faster perception
            try:
                if hasattr(self.tts_client, 'synthesize_streaming'):
                    # Use streaming synthesis for faster response
                    async for audio_chunk in self.tts_client.synthesize_streaming(text):
                        await self._send_audio_chunk(audio_chunk, ws)
                else:
                    # Fall back to standard synthesis
                    audio_data = await self.tts_client.synthesize(text)
                    if audio_data:
                        await self._send_audio_chunks(audio_data, ws)
                        
                self.last_tts_time = time.time()
                self.responses_sent += 1
                logger.info(f"Sent response for: '{text}'")
                
            except Exception as e:
                logger.error(f"Error synthesizing speech: {e}")
        
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
        finally:
            # Reset speaking state after a small delay to prevent echo
            await asyncio.sleep(0.2)
            self.is_speaking = False
            if hasattr(self.stt_integration, 'set_speaking_state'):
                self.stt_integration.set_speaking_state(False)
            
            logger.debug("Ready for next utterance")
    
    async def _send_audio_chunk(self, audio_data: bytes, ws):
        """Send a single audio chunk for more responsive streaming."""
        if not self.stream_sid:
            logger.warning("Cannot send audio: missing stream_sid")
            return
        
        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": audio_base64}
            }
            
            # Handle different WebSocket types
            if isinstance(ws, fastapi.WebSocket):
                await ws.send_text(json.dumps(message))
            else:
                ws.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")
        
    async def start_conversation(self, ws):
        """Start conversation with optimized welcome message."""
        # Store WebSocket reference
        self._ws = ws
        
        # Start STT streaming
        if hasattr(self.stt_integration, 'start_streaming'):
            await self.stt_integration.start_streaming()
        
        # Send welcome message with minimal delay
        await asyncio.sleep(0.05)
        try:
            # Use more direct method call to reduce complexity
            if hasattr(self.tts_client, 'text_to_speech'):
                audio = await self.tts_client.text_to_speech("How can I help you today?")
                if audio:
                    await self._send_audio_chunk(audio, ws)
                self.responses_sent += 1
            else:
                # Try alternative method if text_to_speech isn't available
                audio = await self.tts_client.synthesize("How can I help you today?")
                if audio:
                    await self._send_audio_chunk(audio, ws)
                self.responses_sent += 1
                
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}")
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            self.call_ended = True
            self.conversation_active = False
            self.is_speaking = False
            
            # Clean up STT
            if hasattr(self.stt_integration, 'end_streaming'):
                await self.stt_integration.end_streaming()
            
            # Calculate stats
            duration = time.time() - self.session_start_time
            
            logger.info(f"Session cleanup completed. Stats: "
                       f"Duration: {duration:.2f}s, "
                       f"Audio packets: {self.audio_received}, "
                       f"Transcriptions: {self.transcriptions}, "
                       f"Interim: {self.interim_transcriptions}, "
                       f"Responses: {self.responses_sent}")
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        duration = time.time() - self.session_start_time
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "duration": round(duration, 2),
            "audio_received": self.audio_received,
            "transcriptions": self.transcriptions,
            "interim_transcriptions": self.interim_transcriptions,
            "responses_sent": self.responses_sent,
            "is_speaking": self.is_speaking,
            "conversation_active": self.conversation_active,
            "call_ended": self.call_ended,
            "session_start_time": self.session_start_time,
            "last_transcription_time": self.last_transcription_time,
            "last_audio_time": self.last_audio_time,
            "transcription_rate": round(self.transcriptions / max(duration / 60, 1), 2),
            "response_rate": round(self.responses_sent / max(duration / 60, 1), 2)
        }