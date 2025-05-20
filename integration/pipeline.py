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
        Process audio with optimized streaming response.
        
        Implements progressive response system with immediate feedback
        and optimized state transitions.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            audio_callback: Callback to handle audio data
            
        Returns:
            Dictionary with stats about the process
        """
        logger.info(f"Starting optimized streaming pipeline with audio: {type(audio_data)}")
        
        # Record start time for tracking
        start_time = time.time()
        processing_start_time = None
        
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
            
            # Introduce progressive response system
            is_processing = False
            has_sent_progress = False
            progress_messages = [
                "Let me check that for you...",
                "Looking that up now...",
                "Finding that information..."
            ]
            
            # Collect all transcription results
            transcription_results = []
            final_response_sent = False
            
            # Define callback for transcription results
            async def collect_transcriptions(result):
                nonlocal is_processing, has_sent_progress, processing_start_time, final_response_sent
                transcription_results.append(result)
                
                # For substantial interim results, send progress response
                if not result.is_final and len(result.text.split()) >= 3 and not has_sent_progress and not is_processing:
                    import random
                    progress_text = random.choice(progress_messages)
                    
                    # Send immediate progress response
                    progress_audio = await self.tts_integration.synthesize(progress_text)
                    await audio_callback(progress_audio)
                    
                    # Mark progress as sent
                    has_sent_progress = True
                    
                    # Start processing in background
                    is_processing = True
                    processing_start_time = time.time()
                    
                    # Process in background
                    asyncio.create_task(self._process_transcription_background(
                        result.text, audio_callback, partial=True
                    ))
                    
                # For final results, process and respond
                elif result.is_final and not final_response_sent:
                    # If we haven't started processing yet, do so now
                    if not is_processing:
                        is_processing = True
                        processing_start_time = time.time()
                        
                        # Process the final transcription
                        await self._process_transcription_background(
                            result.text, audio_callback, partial=False
                        )
                        final_response_sent = True
            
            # Set up streaming session
            await self.stt_helper.start_streaming()
            
            # Process audio in optimized chunks
            chunk_size = 400  # 50ms chunks for faster response
            audio_chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
            
            # Faster initial processing, then slow down for continuous listening
            for i, chunk in enumerate(audio_chunks):
                if i < 10:  # First 500ms - process quickly
                    await self.stt_helper.process_stream_chunk(chunk, collect_transcriptions)
                    await asyncio.sleep(0.01)
                else:  # Later chunks - more normal pacing
                    await self.stt_helper.process_stream_chunk(chunk, collect_transcriptions)
                    await asyncio.sleep(0.03)
            
            # Stop streaming to get final results
            final_text, duration = await self.stt_helper.end_streaming()
            
            # Wait for processing to complete if needed
            if is_processing and processing_start_time:
                # Wait up to 5 seconds for processing to complete
                processing_time = time.time() - processing_start_time
                if processing_time < 5.0:
                    await asyncio.sleep(min(5.0 - processing_time, 2.0))
                    
            # Calculate stats
            total_time = time.time() - start_time
            
            return {
                "transcription_results": len(transcription_results),
                "final_results": len([r for r in transcription_results if r.is_final]),
                "interim_results": len([r for r in transcription_results if not r.is_final]),
                "processing_time": total_time,
                "progress_response_sent": has_sent_progress,
                "success": True
            }
                
        except Exception as e:
            logger.error(f"Error in streaming pipeline: {e}")
            return {
                "error": f"Streaming error: {str(e)}",
                "processing_time": time.time() - start_time,
                "success": False
            }

    async def _process_transcription_background(
        self, 
        text: str, 
        audio_callback: Callable[[bytes], Awaitable[None]], 
        partial: bool = False
    ) -> None:
        """
        Process transcription in background with optimized response generation.
        
        Args:
            text: Transcription text
            audio_callback: Callback for audio output
            partial: Whether this is a partial (interim) transcription
        """
        try:
            # Set speaking state during response generation
            if hasattr(self.speech_recognizer, 'set_speaking_state'):
                self.speech_recognizer.set_speaking_state(True)
            
            # Different processing approaches based on partial flag
            if partial:
                # For partial transcriptions, use simplified approach
                # Just get key information for faster response
                ctx = None
                if hasattr(self.conversation_manager, 'memory'):
                    ctx = self.conversation_manager.memory.get_context(text)
                
                # Generate a simple response based on context
                response_text = None
                
                # Try to extract a relevant response from context
                if ctx:
                    try:
                        # Use a simplified query approach
                        response_text = await self._generate_quick_response(text, ctx)
                    except Exception as e:
                        logger.error(f"Error generating quick response: {e}")
                
                # Only send audio if we have a good response
                if response_text:
                    audio_data = await self.tts_integration.synthesize(response_text)
                    if audio_data:
                        # Send in smaller chunks for smoother playback
                        chunk_size = 320  # 40ms chunks
                        chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
                        
                        # Send a limited number of chunks for partial response
                        for chunk in chunks[:min(len(chunks), 5)]:
                            await audio_callback(chunk)
                            await asyncio.sleep(0.01)
            else:
                # For final transcriptions, use full conversation manager
                result = await self.conversation_manager.handle_user_input(text)
                response_text = result.get("response", "")
                
                if response_text:
                    # Break into sentences for faster response
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', response_text)
                    
                    # Process each sentence
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                            
                        # Synthesize and send each sentence
                        audio_data = await self.tts_integration.synthesize(sentence)
                        if audio_data:
                            await audio_callback(audio_data)
                else:
                    # No response generated
                    fallback = "I'm sorry, I couldn't find an answer to that."
                    audio_data = await self.tts_integration.synthesize(fallback)
                    if audio_data:
                        await audio_callback(audio_data)
                        
        except Exception as e:
            logger.error(f"Error in background processing: {e}")
            
            # Send error message
            try:
                error_msg = "I'm sorry, I encountered an error processing your request."
                error_audio = await self.tts_integration.synthesize(error_msg)
                if error_audio:
                    await audio_callback(error_audio)
            except:
                pass
                
        finally:
            # Reset speaking state
            if hasattr(self.speech_recognizer, 'set_speaking_state'):
                self.speech_recognizer.set_speaking_state(False)

    async def _generate_quick_response(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate a quick response for interim results.
        Uses a simplified approach for speed.
        
        Args:
            query: Query text
            context: Optional context
            
        Returns:
            Quick response text
        """
        # Create a simple system prompt for quick responses
        system_prompt = """You are a helpful voice assistant. Generate a very short (1-2 sentences) 
        preliminary response to the user's query. Be conversational but brief."""
        
        if context:
            system_prompt += f"\n\nHere's some relevant information:\n{context}"
        
        user_prompt = f"Query: {query}\nGenerate a very brief preliminary response (1-2 sentences max):"
        
        try:
            # Use OpenAI directly for speed
            from openai import OpenAI
            client = OpenAI(api_key=self.conversation_manager.config.openai_api_key)
            
            # Use a faster, smaller model for quick responses
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a faster model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,  # Keep it very short
                temperature=0.7
            )
            
            quick_response = response.choices[0].message.content.strip()
            return quick_response
            
        except Exception as e:
            logger.error(f"Error generating quick response: {e}")
            return None

    
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