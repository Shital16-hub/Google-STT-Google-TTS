# integration/pipeline.py

"""
Optimized end-to-end pipeline for Voice AI Agent with low-latency streaming support.
"""
import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

import numpy as np

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager, ConversationState
from knowledge_base.query_engine import QueryEngine

from integration.tts_integration import TTSIntegration

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    End-to-end pipeline optimized for low-latency voice interactions.
    """
    
    def __init__(
        self,
        speech_recognizer: Union[GoogleCloudStreamingSTT, Any],
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """
        Initialize the pipeline with existing components.
        
        Args:
            speech_recognizer: Initialized STT component
            conversation_manager: Initialized conversation manager
            query_engine: Initialized query engine
            tts_integration: Initialized TTS integration
        """
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Create a helper for STT integration
        self.stt_helper = STTIntegration(speech_recognizer)
        
        # Determine if we're using Google Cloud STT
        self.using_google_cloud = isinstance(speech_recognizer, GoogleCloudStreamingSTT)
        logger.info(f"Optimized pipeline initialized with {'Google Cloud' if self.using_google_cloud else 'Other'} STT")
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        output_speech_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline with low-latency optimizations.
        
        Args:
            audio_file_path: Path to the input audio file
            output_speech_file: Path to save the output speech file (optional)
            
        Returns:
            Dictionary with results from each stage
        """
        logger.info(f"Starting optimized pipeline with audio: {audio_file_path}")
        
        # Track timing for each stage
        timings = {}
        start_time = time.time()
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=8000)
            logger.info(f"Loaded audio: {len(audio)} samples, {sample_rate}Hz")
        except Exception as e:
            logger.error(f"Error loading audio file: {e}", exc_info=True)
            return {"error": f"Error loading audio file: {e}"}
        
        # Set initial state for optimal processing
        if hasattr(self.speech_recognizer, 'set_speaking_state'):
            self.speech_recognizer.set_speaking_state(False)
        
        # Process for transcription
        transcription_results = []
        
        # Define callback to collect all results
        async def collect_transcriptions(result):
            transcription_results.append(result)
        
        # Start streaming
        await self.stt_helper.start_streaming()
        
        # Process audio in smaller chunks for faster response
        chunk_size = 1600  # 200ms chunks
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
        
        for chunk in chunks:
            # Convert to bytes for processing
            chunk_bytes = (chunk * 32767).astype(np.int16).tobytes()
            # Process chunk with streaming
            await self.stt_helper.process_stream_chunk(chunk_bytes, collect_transcriptions)
            # Small delay to simulate real-time processing
            await asyncio.sleep(0.01)
        
        # Stop streaming
        final_text, duration = await self.stt_helper.end_streaming()
        
        # Get best transcription result
        if transcription_results:
            # Prioritize final results
            final_results = [r for r in transcription_results if r.is_final]
            if final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
            else:
                # Use longest interim result if no final result
                best_result = max(transcription_results, key=lambda r: len(r.text))
                transcription = best_result.text
        else:
            transcription = final_text
            
        timings["stt"] = time.time() - stt_start
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query with streaming
        logger.info("STAGE 2: Knowledge Base Query with streaming")
        kb_start = time.time()
        
        try:
            # Process query with conversation context
            query_result = await self.conversation_manager.handle_user_input(transcription)
            response = query_result.get("response", "")
            
            if not response:
                return {"error": "No response generated from knowledge base"}
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech with Google Cloud TTS
        logger.info("STAGE 3: Text-to-Speech with Google Cloud TTS")
        tts_start = time.time()
        
        try:
            # Convert response to speech
            speech_audio = await self.tts_integration.synthesize(response)
            
            # Save speech audio if output file specified
            if output_speech_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_speech_file)), exist_ok=True)
                with open(output_speech_file, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {output_speech_file}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s, generated {len(speech_audio)} bytes")
            
        except Exception as e:
            logger.error(f"Error in TTS stage: {e}")
            return {
                "error": f"TTS error: {str(e)}",
                "transcription": transcription,
                "response": response
            }
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"End-to-end pipeline completed in {total_time:.2f}s")
        
        # Compile results
        return {
            "transcription": transcription,
            "response": response,
            "speech_audio_size": len(speech_audio),
            "speech_audio": None if output_speech_file else speech_audio,
            "timings": timings,
            "total_time": total_time
        }
    
    async def process_audio_streaming(
        self,
        audio_data: Union[bytes, np.ndarray],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process audio with low-latency streaming response.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            audio_callback: Callback to handle audio data
            
        Returns:
            Dictionary with stats about the process
        """
        logger.info(f"Starting optimized streaming pipeline with audio: {type(audio_data)}")
        
        # Record start time for tracking
        start_time = time.time()
        
        try:
            # Ensure audio is in the right format
            if isinstance(audio_data, bytes):
                audio = audio_data
            else:
                # Convert numpy array to bytes
                audio = (audio * 32767).astype(np.int16).tobytes()
            
            # Set speaking state to false for optimal processing
            if hasattr(self.speech_recognizer, 'set_speaking_state'):
                self.speech_recognizer.set_speaking_state(False)
            
            # Process audio in smaller chunks
            chunk_size = 800  # 100ms chunks for faster response
            audio_chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
            
            # Collect all transcription results
            transcription_results = []
            
            # Define callback for transcription results
            async def collect_transcriptions(result):
                transcription_results.append(result)
                
                # Process substantial interim results immediately for faster response
                if not result.is_final and len(result.text.split()) >= 4:
                    # Generate a quick partial response for substantial interim results
                    await self._handle_interim_result(result.text, audio_callback)
            
            # Set up streaming session
            await self.stt_helper.start_streaming()
            
            # Process audio chunks with minimal delay
            for chunk in audio_chunks:
                await self.stt_helper.process_stream_chunk(chunk, collect_transcriptions)
                await asyncio.sleep(0.01)  # Minimal delay to simulate real-time
            
            # Stop streaming to get final results
            final_text, duration = await self.stt_helper.end_streaming()
            
            # Get best transcription
            if transcription_results:
                # Prioritize final results
                final_results = [r for r in transcription_results if r.is_final]
                if final_results:
                    best_result = max(final_results, key=lambda r: r.confidence)
                    transcription = best_result.text
                else:
                    # Use longest interim result if no final result
                    best_result = max(transcription_results, key=lambda r: len(r.text))
                    transcription = best_result.text
            else:
                transcription = final_text
                
            # Generate final response for the complete transcription
            if transcription:
                try:
                    # Set speaking state before generating response
                    if hasattr(self.speech_recognizer, 'set_speaking_state'):
                        self.speech_recognizer.set_speaking_state(True)
                    
                    # Generate response through conversation manager
                    result = await self.conversation_manager.handle_user_input(transcription)
                    response_text = result.get("response", "")
                    
                    if response_text:
                        # Send audio response
                        audio_data = await self.tts_integration.synthesize(response_text)
                        await audio_callback(audio_data)
                    
                    # Reset speaking state
                    if hasattr(self.speech_recognizer, 'set_speaking_state'):
                        self.speech_recognizer.set_speaking_state(False)
                        
                except Exception as e:
                    logger.error(f"Error generating final response: {e}")
            
            # Calculate stats
            total_time = time.time() - start_time
            
            return {
                "transcription": transcription,
                "transcription_results": len(transcription_results),
                "final_results": len([r for r in transcription_results if r.is_final]),
                "interim_results": len([r for r in transcription_results if not r.is_final]),
                "processing_time": total_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in streaming pipeline: {e}")
            return {
                "error": f"Streaming error: {str(e)}",
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    async def _handle_interim_result(self, text: str, audio_callback: Callable[[bytes], Awaitable[None]]):
        """Handle substantial interim results for faster responses."""
        if not text or len(text) < 8:
            return
            
        try:
            # Use streaming query for faster response
            response_chunks = []
            
            async for chunk in self.query_engine.query_with_streaming(text):
                if chunk.get("chunk"):
                    response_chunks.append(chunk.get("chunk"))
                
                # Get complete response once done
                if chunk.get("done") and chunk.get("full_response"):
                    response_text = chunk.get("full_response")
                    
                    # Only process if response is substantial
                    if response_text and len(response_text) > 15:
                        logger.info(f"Generated interim response: {response_text[:50]}...")
                        
                        # Convert to speech
                        if hasattr(self.speech_recognizer, 'set_speaking_state'):
                            self.speech_recognizer.set_speaking_state(True)
                            
                        audio_data = await self.tts_integration.synthesize(response_text)
                        await audio_callback(audio_data)
                        
                        # Reset speaking state after a small delay
                        await asyncio.sleep(0.2)
                        if hasattr(self.speech_recognizer, 'set_speaking_state'):
                            self.speech_recognizer.set_speaking_state(False)
                    break
        except Exception as e:
            logger.error(f"Error handling interim result: {e}")