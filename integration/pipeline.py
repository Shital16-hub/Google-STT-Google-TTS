# integration/pipeline.py
"""
Optimized pipeline for Voice AI Agent using OpenAI + Pinecone.
Simplified architecture for sub-2-second response times.
"""
import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, Union, List, Callable, Awaitable

import numpy as np

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from integration.tts_integration import TTSIntegration

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    Optimized pipeline for OpenAI + Pinecone with aggressive latency targets:
    - STT: <0.5s
    - Knowledge Base: <1.0s  
    - TTS: <0.5s
    - Total: <2.0s
    """
    
    def __init__(
        self,
        speech_recognizer: GoogleCloudStreamingSTT,
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """Initialize optimized pipeline."""
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Create STT helper
        self.stt_helper = STTIntegration(speech_recognizer)
        
        # Performance tracking
        self.processed_calls = 0
        self.total_latency = 0.0
        self.avg_latency = 0.0
        
        logger.info("Initialized optimized pipeline for OpenAI + Pinecone")
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        output_speech_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio file with optimized pipeline.
        Target: Complete processing under 2 seconds.
        """
        logger.info(f"Processing audio file: {audio_file_path}")
        
        start_time = time.time()
        timings = {}
        
        # Reset conversation manager for clean state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # STAGE 1: Speech-to-Text (Target: <0.5s)
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Load and validate audio
        from speech_to_text.utils.audio_utils import load_audio_file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=8000)
            logger.info(f"Loaded audio: {len(audio)} samples at {sample_rate}Hz")
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return {"error": f"Audio loading error: {e}"}
        
        # Quick transcription
        transcription, duration = await self._transcribe_audio_fast(audio)
        
        # Validate transcription
        if not self._is_valid_transcription(transcription):
            logger.warning(f"Invalid transcription: '{transcription}'")
            return {
                "error": "No valid speech detected",
                "transcription": transcription,
                "timing": {"stt": time.time() - stt_start}
            }
        
        timings["stt"] = time.time() - stt_start
        logger.info(f"STT completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base (Target: <1.0s)
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        try:
            # Use timeout to ensure KB doesn't exceed 1 second
            response_task = asyncio.create_task(
                self._query_knowledge_base_fast(transcription)
            )
            
            # Apply aggressive timeout
            response = await asyncio.wait_for(response_task, timeout=1.0)
            
            if not response:
                response = "I couldn't find an answer to that question."
            
            timings["kb"] = time.time() - kb_start
            logger.info(f"KB query completed in {timings['kb']:.2f}s")
            
        except asyncio.TimeoutError:
            logger.warning("KB query timed out, using fallback response")
            response = "I'm processing your question, please wait a moment."
            timings["kb"] = 1.0
        except Exception as e:
            logger.error(f"KB query error: {e}")
            response = "I encountered an error processing your question."
            timings["kb"] = time.time() - kb_start
        
        # STAGE 3: Text-to-Speech (Target: <0.5s)
        logger.info("STAGE 3: Text-to-Speech")
        tts_start = time.time()
        
        try:
            # Generate speech with timeout
            speech_audio = await asyncio.wait_for(
                self.tts_integration.text_to_speech(response),
                timeout=0.8  # Slightly more generous for TTS
            )
            
            # Save to file if requested
            if output_speech_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_speech_file)), exist_ok=True)
                with open(output_speech_file, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech to {output_speech_file}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s")
            
        except asyncio.TimeoutError:
            logger.warning("TTS timed out")
            speech_audio = b""
            timings["tts"] = 0.8
        except Exception as e:
            logger.error(f"TTS error: {e}")
            speech_audio = b""
            timings["tts"] = time.time() - tts_start
        
        # Calculate total time and update stats
        total_time = time.time() - start_time
        self._update_latency_stats(total_time)
        
        logger.info(f"Pipeline completed in {total_time:.2f}s (avg: {self.avg_latency:.2f}s)")
        
        return {
            "transcription": transcription,
            "response": response,
            "speech_audio_size": len(speech_audio),
            "speech_audio": speech_audio if not output_speech_file else None,
            "timings": timings,
            "total_time": total_time,
            "avg_latency": self.avg_latency
        }
    
    async def process_audio_streaming(
        self,
        audio_data: Union[bytes, np.ndarray],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process audio with streaming response optimized for real-time.
        """
        logger.info("Starting optimized streaming pipeline")
        start_time = time.time()
        
        # Reset conversation
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        try:
            # Convert audio format if needed
            if isinstance(audio_data, bytes):
                audio = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # Fast transcription
            transcription, duration = await self._transcribe_audio_fast(audio)
            
            if not self._is_valid_transcription(transcription):
                return {
                    "error": "No valid speech detected",
                    "transcription": transcription,
                    "total_time": time.time() - start_time
                }
            
            logger.info(f"Transcription: {transcription}")
            transcription_time = time.time() - start_time
            
            # Stream response with immediate TTS
            total_chunks = 0
            total_audio_bytes = 0
            response_start = time.time()
            full_response = ""
            
            # Use streaming query for real-time response
            async for chunk in self.query_engine.query_with_streaming(transcription):
                chunk_text = chunk.get("chunk", "")
                
                if chunk_text:
                    full_response += chunk_text
                    
                    # Immediately convert to speech and send
                    try:
                        audio_data = await self.tts_integration.text_to_speech(chunk_text)
                        await audio_callback(audio_data)
                        
                        total_chunks += 1
                        total_audio_bytes += len(audio_data)
                    except Exception as e:
                        logger.error(f"Error in chunk TTS: {e}")
            
            response_time = time.time() - response_start
            total_time = time.time() - start_time
            
            # Update stats
            self._update_latency_stats(total_time)
            
            return {
                "transcription": transcription,
                "transcription_time": transcription_time,
                "response_time": response_time,
                "total_time": total_time,
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "full_response": full_response,
                "avg_latency": self.avg_latency
            }
            
        except Exception as e:
            logger.error(f"Error in streaming pipeline: {e}")
            return {
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    async def _transcribe_audio_fast(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Fast transcription optimized for minimal latency.
        """
        try:
            # Ensure proper format for Google Cloud STT
            if audio.dtype == np.float32:
                # Convert to mulaw for Twilio compatibility
                import audioop
                pcm_data = (audio * 32767).astype(np.int16).tobytes()
                audio_bytes = audioop.lin2ulaw(pcm_data, 2)
            else:
                audio_bytes = audio.tobytes()
            
            # Quick STT session
            await self.speech_recognizer.start_streaming()
            
            # Process in one chunk for speed
            final_results = []
            
            async def collect_result(result):
                if result.is_final:
                    final_results.append(result)
            
            # Send all audio at once for fastest processing
            await self.speech_recognizer.process_audio_chunk(audio_bytes, collect_result)
            
            # Stop and get result
            transcription, duration = await self.speech_recognizer.stop_streaming()
            
            # Use best final result if available
            if not transcription and final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
            
            # Clean transcription
            transcription = self.stt_helper.cleanup_transcription(transcription)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in fast transcription: {e}")
            return "", len(audio) / 8000
    
    async def _query_knowledge_base_fast(self, query: str) -> str:
        """
        Fast knowledge base query with timeout.
        """
        try:
            # Query with streaming for immediate response
            response_text = ""
            
            async for chunk in self.conversation_manager.generate_streaming_response(query):
                if chunk.get("chunk"):
                    response_text += chunk["chunk"]
                
                # Get partial response if it's taking too long
                if chunk.get("done", False):
                    break
            
            return response_text or "I couldn't find an answer to that."
            
        except Exception as e:
            logger.error(f"Error in fast KB query: {e}")
            return "I encountered an error processing your question."
    
    def _is_valid_transcription(self, transcription: str) -> bool:
        """Quick validation of transcription."""
        return (transcription and 
                len(transcription.strip()) > 2 and
                len(transcription.split()) >= 1)
    
    def _update_latency_stats(self, latency: float):
        """Update rolling latency statistics."""
        self.processed_calls += 1
        self.total_latency += latency
        self.avg_latency = self.total_latency / self.processed_calls
        
        # Log if latency exceeds target
        if latency > 2.0:
            logger.warning(f"Latency exceeded target: {latency:.2f}s")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        return {
            "processed_calls": self.processed_calls,
            "avg_latency": self.avg_latency,
            "total_latency": self.total_latency,
            "target_latency": 2.0,
            "performance_ratio": 2.0 / max(self.avg_latency, 0.001)  # How much faster than target
        }