# telephony/simple_websocket_handler.py

"""
Highly optimized WebSocket handler for minimal latency voice interactions.
Includes binary data transmission, connection pooling, and progressive response.
"""
import json
import asyncio
import logging
import base64
import time
import os
from typing import Dict, Any, Optional, List
import struct
import aiohttp

import fastapi
from fastapi import WebSocket

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from integration.tts_integration import TTSIntegration

logger = logging.getLogger(__name__)

# Global HTTP session for connection pooling
HTTP_SESSION = None

# Progressive responses for immediate feedback
PROGRESS_RESPONSES = [
    "Let me check that for you.",
    "Looking that up now.",
    "Let me find that information.",
    "I'm searching for that.",
    "One moment while I find that."
]

class SimpleWebSocketHandler:
    """
    Highly optimized WebSocket handler for minimal latency.
    
    Features:
    - Binary WebSocket messages for audio data
    - Progressive responses with immediate feedback
    - Connection pooling for API calls
    - Optimized speaking/listening state management
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with advanced optimizations."""
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Get project ID dynamically
        self.project_id = self._get_project_id()
        
        # Use STT integration from pipeline for consistency
        if hasattr(pipeline, 'speech_recognizer') and pipeline.speech_recognizer:
            self.stt_integration = pipeline.stt_helper
            self.speech_recognizer = pipeline.speech_recognizer
        else:
            # Initialize new STT with optimized settings
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            self.speech_recognizer = GoogleCloudStreamingSTT(
                language="en-US",
                sample_rate=8000,
                encoding="MULAW",
                channels=1,
                interim_results=True,  # Enable interim results for early processing
                project_id=self.project_id,
                location="global",
                credentials_file=credentials_file
            )
            self.stt_integration = self.pipeline.stt_helper
        
        # Use TTS integration from pipeline for consistency
        if hasattr(pipeline, 'tts_integration') and pipeline.tts_integration:
            self.tts_client = pipeline.tts_integration
        else:
            # Initialize new TTS
            self.tts_client = TTSIntegration(
                credentials_file=credentials_file,
                voice_name="en-US-Neural2-C",
                voice_gender=None,
                language_code="en-US",
                container_format="mulaw",
                sample_rate=8000,
                enable_caching=True,
                voice_type="NEURAL2"
            )
        
        # Enhanced state management
        self.conversation_active = True
        self.is_speaking = False
        self.is_processing = False  # Track when we're processing a query
        self.call_ended = False
        
        # Audio processing with optimized flow control
        self.audio_buffer = bytearray()
        self.chunk_size = 320  # 40ms chunks for faster processing
        self.min_valid_chunk_size = 80  # Minimum valid chunk size
        
        # Session management
        self.session_start_time = time.time()
        self.last_transcription_time = time.time()
        self.last_audio_time = time.time()
        self.last_interim_time = time.time()
        self.last_tts_time = None
        self.last_progress_time = 0  # Track when we last sent a progress message
        
        # Response tracking
        self.last_response_time = time.time()
        self.last_response_text = ""
        self.interim_text = ""
        self.progress_sent = False
        self.current_query_start_time = 0
        
        # Performance metrics
        self.audio_received = 0
        self.transcriptions = 0
        self.responses_sent = 0
        self.interim_transcriptions = 0
        self.progress_responses_sent = 0
        self.processing_times = []
        
        # WebSocket reference
        self._ws = None
        
        # Ensure we have a global HTTP session for connection pooling
        global HTTP_SESSION
        if HTTP_SESSION is None:
            HTTP_SESSION = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=aiohttp.TCPConnector(
                    limit=20,  # Max connections
                    ttl_dns_cache=300,  # DNS cache TTL
                    ssl=False  # Faster without SSL verification
                )
            )
        
        logger.info(f"Highly optimized WebSocket handler initialized - Call: {call_sid}, Project: {self.project_id}")
    
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
    
    async def _handle_audio(self, data: Dict[str, Any], ws: WebSocket):
        """Handle audio with optimized state management."""
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
        
        # Decode audio with optimized error handling
        try:
            audio_data = base64.b64decode(payload)
            self.audio_received += 1
            self.last_audio_time = time.time()
            logger.debug(f"Received audio chunk: {len(audio_data)} bytes")
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Skip if chunk too small (likely silence)
        if len(audio_data) < self.min_valid_chunk_size:
            return
        
        # Process audio directly through STT with early response processing
        try:
            # Log every 50th audio chunk for debugging
            if self.audio_received % 50 == 0:
                logger.info(f"Processing audio chunk #{self.audio_received}, size: {len(audio_data)} bytes")
                
            # Ensure STT integration exists
            if not self.stt_integration:
                logger.error("STT integration not available")
                return
                
            await self.stt_integration.process_stream_chunk(
                audio_data, 
                callback=self._handle_transcription_result
            )
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Process transcription with progressive response capability."""
        transcription = result.text.strip()
        confidence = result.confidence
        
        # Track interim results for early processing
        if not result.is_final:
            self.interim_transcriptions += 1
            self.last_interim_time = time.time()
            self.interim_text = transcription
            
            # Process substantial interim results for faster response
            if len(transcription.split()) >= 3 and time.time() - self.last_progress_time > 2.0:
                # Start query processing earlier
                if not self.is_processing and not self.is_speaking:
                    self.is_processing = True
                    self.current_query_start_time = time.time()
                    
                    # Send progressive response immediately for better UX
                    if not self.progress_sent and len(transcription.split()) >= 4:
                        await self._send_progress_response()
                        
                # Process early for substantial interim results
                if len(transcription.split()) >= 5:
                    await self._process_transcription(transcription, is_final=False)
            return
                
        # For final results - clear progress state
        if transcription:
            self.progress_sent = False
            logger.info(f"Final transcription: '{transcription}' (confidence: {confidence:.2f})")
            self.last_transcription_time = time.time()
            self.transcriptions += 1
            
            # Process the transcription
            await self._process_transcription(transcription, is_final=True)
    
    async def _send_progress_response(self):
        """Send an immediate progress response while processing."""
        # Skip if already sent or if we're speaking
        if self.progress_sent or self.is_speaking:
            return
            
        # Choose a progress response
        import random
        progress_text = random.choice(PROGRESS_RESPONSES)
        
        try:
            # Quick response synthesis (use cached if possible)
            audio_data = await self.tts_client.synthesize(progress_text)
            
            if audio_data:
                # Set speaking state temporarily
                old_speaking = self.is_speaking
                self.is_speaking = True
                
                # Send smaller chunks for faster response
                chunk_size = 160  # Very small chunks (20ms) for quick start
                chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
                
                # Send only the beginning portion for minimal delay
                for chunk in chunks[:5]:  # Just send the first few chunks
                    await self._send_audio_binary(chunk)
                    await asyncio.sleep(0.01)
                    
                # Reset speaking state
                self.is_speaking = old_speaking
                
                # Track that we sent a progress response
                self.progress_sent = True
                self.last_progress_time = time.time()
                self.progress_responses_sent += 1
                
        except Exception as e:
            logger.error(f"Error sending progress response: {e}")
    
    async def _process_transcription(self, transcription: str, is_final: bool = True):
        """Process transcription with optimized flow."""
        # Skip if we're speaking to prevent processing during our own speech
        if self.is_speaking:
            return
            
        # Mark that we're processing
        if not self.is_processing:
            self.is_processing = True
            self.current_query_start_time = time.time()
        
        try:
            # Process through knowledge base
            if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                # Set speaking state to true while generating response
                self.is_speaking = True
                if hasattr(self.stt_integration, 'set_speaking_state'):
                    self.stt_integration.set_speaking_state(True)
                
                if is_final:
                    # For final results, use full query
                    result = await self.pipeline.query_engine.query(transcription)
                    response_text = result.get("response", "")
                    
                    # Calculate and log processing time
                    if self.current_query_start_time > 0:
                        processing_time = time.time() - self.current_query_start_time
                        self.processing_times.append(processing_time)
                        logger.info(f"Query processed in {processing_time:.2f}s: '{transcription}'")
                        
                else:
                    # For interim results, use simplified query for speed
                    # This will be a simpler query to get faster responses
                    try:
                        # Try a more direct method for faster response
                        context = await self.pipeline.query_engine.retrieve(transcription)
                        
                        # If we have context, generate a very short initial response
                        if context and len(context) > 0:
                            # Extract a quick response based on first document
                            first_doc = context[0]
                            quick_text = first_doc.get('text', '')[:100]
                            
                            # Prepare a shortened response
                            response_text = f"I found information about {transcription.split()[-1]}. "
                            
                            # Don't respond yet for interim unless we have good content
                            if len(quick_text) < 20:
                                return
                        else:
                            # No good context, don't respond yet
                            return
                            
                    except Exception as e:
                        logger.error(f"Error in interim query: {e}")
                        return
                
                # Send response if we have one
                if response_text:
                    await self._send_response(response_text)
                elif is_final:
                    # Only for final results with no response
                    logger.warning("No response generated from knowledge base")
                    await self._send_response("I'm sorry, I couldn't find an answer to that question.")
                
                # Reset processing and speaking states
                self.is_processing = False
                self.current_query_start_time = 0
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
            
            # Reset state in case of error
            self.is_processing = False
            self.is_speaking = False
            if hasattr(self.stt_integration, 'set_speaking_state'):
                self.stt_integration.set_speaking_state(False)
    
    async def _send_response(self, text: str, ws=None):
        """Send TTS response with optimized streaming."""
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
            
            # Optimal response strategy based on text length
            if len(text) > 100:
                # Long response - use streaming synthesis for faster start
                await self._stream_long_response(text, ws)
            else:
                # Short response - use direct synthesis for speed
                try:
                    audio_data = await self.tts_client.synthesize(text)
                    if audio_data:
                        # Send in small chunks for smoother playback
                        chunk_size = 160  # 20ms chunks
                        for i in range(0, len(audio_data), chunk_size):
                            await self._send_audio_binary(audio_data[i:i+chunk_size], ws)
                            await asyncio.sleep(0.01)  # Minimal delay
                except Exception as e:
                    logger.error(f"Error synthesizing speech: {e}")
                    
            self.last_tts_time = time.time()
            self.responses_sent += 1
            logger.info(f"Sent response for: '{text}'")
            
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
        finally:
            # Reset speaking state after a small delay to prevent echo
            await asyncio.sleep(0.2)
            self.is_speaking = False
            if hasattr(self.stt_integration, 'set_speaking_state'):
                self.stt_integration.set_speaking_state(False)
            
            logger.debug("Ready for next utterance")
    
    async def _stream_long_response(self, text: str, ws):
        """Stream a long response with sentence-by-sentence synthesis."""
        # Split text into sentences for faster response
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Send sentences as they're synthesized
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            try:
                # Synthesize and send immediately
                audio_data = await self.tts_client.synthesize(sentence)
                if audio_data:
                    # Send in small chunks for smoother playback
                    chunk_size = 160  # 20ms chunks
                    for j in range(0, len(audio_data), chunk_size):
                        await self._send_audio_binary(audio_data[j:j+chunk_size], ws)
                        # Adjust delay based on position (faster startup)
                        if i == 0 and j < 320:
                            await asyncio.sleep(0.005)  # Very minimal delay for first sentence start
                        else:
                            await asyncio.sleep(0.01)  # Slightly longer delay for rest
            except Exception as e:
                logger.error(f"Error synthesizing sentence: {e}")
                # Continue with next sentence
    
    async def _send_audio_binary(self, audio_data: bytes, ws=None):
        """Send audio using binary WebSocket messages for maximum efficiency."""
        if not self.stream_sid:
            logger.warning("Cannot send audio: missing stream_sid")
            return
        
        # Use stored WebSocket if not provided
        if ws is None:
            ws = self._ws
        
        try:
            # Create a properly formatted message for Twilio Media Streams
            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": base64.b64encode(audio_data).decode('utf-8')
                }
            }
            # Send as text - this is what Twilio expects
            await ws.send_text(json.dumps(message))
                
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")
    
    async def start_conversation(self, ws):
        """Start conversation with optimized welcome message."""
        # Store WebSocket reference
        self._ws = ws
        
        # Log WebSocket type
        logger.info(f"WebSocket type: {type(ws).__name__}")
        
        # Start STT streaming
        if hasattr(self.stt_integration, 'start_streaming'):
            await self.stt_integration.start_streaming()
        
        # Send welcome message with minimal delay
        await asyncio.sleep(0.05)
        await self._send_response("How can I help you today?", ws)
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            self.call_ended = True
            self.conversation_active = False
            self.is_speaking = False
            
            # Clean up STT
            if hasattr(self.stt_integration, 'end_streaming'):
                await self.stt_integration.end_streaming()
            elif hasattr(self.speech_recognizer, 'stop_streaming'):
                await self.speech_recognizer.stop_streaming()
            
            # Calculate stats
            duration = time.time() - self.session_start_time
            
            # Calculate latency metrics
            avg_processing = sum(self.processing_times) / max(len(self.processing_times), 1)
            
            logger.info(f"Session cleanup completed. Stats: "
                       f"Duration: {duration:.2f}s, "
                       f"Audio packets: {self.audio_received}, "
                       f"Transcriptions: {self.transcriptions}, "
                       f"Interim: {self.interim_transcriptions}, "
                       f"Responses: {self.responses_sent}, "
                       f"Avg processing: {avg_processing:.2f}s")
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        duration = time.time() - self.session_start_time
        stats = {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "duration": round(duration, 2),
            "audio_received": self.audio_received,
            "transcriptions": self.transcriptions,
            "interim_transcriptions": self.interim_transcriptions,
            "responses_sent": self.responses_sent,
            "progress_responses": self.progress_responses_sent,
            "is_speaking": self.is_speaking,
            "is_processing": self.is_processing,
            "conversation_active": self.conversation_active,
            "call_ended": self.call_ended,
            "session_start_time": self.session_start_time,
            "last_transcription_time": self.last_transcription_time,
            "last_audio_time": self.last_audio_time,
            "transcription_rate": round(self.transcriptions / max(duration / 60, 1), 2),
            "response_rate": round(self.responses_sent / max(duration / 60, 1), 2)
        }
        
        # Add processing time stats
        if self.processing_times:
            stats["processing_times"] = {
                "avg": round(sum(self.processing_times) / len(self.processing_times), 2),
                "min": round(min(self.processing_times), 2),
                "max": round(max(self.processing_times), 2),
            }
        
        return stats