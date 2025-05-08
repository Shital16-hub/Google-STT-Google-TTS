"""
End-to-end pipeline orchestration for Voice AI Agent.

This module provides high-level functions for running the complete
STT -> Knowledge Base -> TTS pipeline with improved latency and concurrency.
"""
import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

import numpy as np

from speech_to_text.deepgram_stt import DeepgramStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.query_engine import QueryEngine

from integration.tts_integration import TTSIntegration

# Minimum word count for a valid user query
MIN_VALID_WORDS = 2  # Reduced from 3 for more responsiveness

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    End-to-end pipeline orchestration for Voice AI Agent with parallel processing.
    
    Provides a high-level interface for running the complete
    STT -> Knowledge Base -> TTS pipeline with Deepgram STT.
    """
    
    def __init__(
        self,
        speech_recognizer: Union[DeepgramStreamingSTT, Any],
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """
        Initialize the pipeline with existing components.
        
        Args:
            speech_recognizer: Initialized STT component (Deepgram or other)
            conversation_manager: Initialized conversation manager
            query_engine: Initialized query engine
            tts_integration: Initialized TTS integration
        """
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Create a helper for filtering out non-speech transcriptions
        self.stt_helper = STTIntegration(speech_recognizer)
        
        # Determine if we're using Deepgram STT
        self.using_deepgram = isinstance(speech_recognizer, DeepgramStreamingSTT)
        logger.info(f"Pipeline initialized with {'Deepgram' if self.using_deepgram else 'Whisper'} STT")
    
    async def _is_valid_transcription(self, transcription: str) -> bool:
        """
        Check if a transcription is valid and should be processed.
        
        Args:
            transcription: The transcription text
            
        Returns:
            True if the transcription is valid
        """
        # First clean up the transcription
        cleaned_text = self.stt_helper.cleanup_transcription(transcription)
        
        # If it's empty after cleaning, it's not valid
        if not cleaned_text:
            return False
            
        # Check if it has enough words
        words = cleaned_text.split()
        if len(words) < MIN_VALID_WORDS:
            return False
            
        return True
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        output_speech_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline.
        
        Args:
            audio_file_path: Path to the input audio file
            output_speech_file: Path to save the output speech file (optional)
            
        Returns:
            Dictionary with results from each stage
        """
        logger.info(f"Starting end-to-end pipeline with audio: {audio_file_path}")
        
        # Track timing for each stage
        timings = {}
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Log audio file info
        import os
        logger.info(f"Audio file size: {os.path.getsize(audio_file_path)} bytes")
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
            logger.info(f"Loaded audio: {len(audio)} samples, {sample_rate}Hz")
        except Exception as e:
            logger.error(f"Error loading audio file: {e}", exc_info=True)
            return {"error": f"Error loading audio file: {e}"}
        
        # Process for transcription
        logger.info("Transcribing audio...")
        
        # OPTIMIZATION: Process in parallel tasks
        transcription_task = asyncio.create_task(self._transcribe_audio(audio))
        
        # Start preparing other components while transcription is running
        kb_prep_task = asyncio.create_task(self._prepare_kb())
        tts_prep_task = asyncio.create_task(self._prepare_tts())
        
        # Wait for transcription
        transcription, duration = await transcription_task
        
        # Validate transcription
        is_valid = await self._is_valid_transcription(transcription)
        if not is_valid:
            logger.warning(f"Transcription not valid for processing: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
        timings["stt"] = time.time() - stt_start
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        # Wait for KB preparation if needed
        await kb_prep_task
        
        try:
            # Initiate retrieval and query tasks in parallel
            retrieval_task = asyncio.create_task(self.query_engine.retrieve_with_sources(transcription))
            query_task = asyncio.create_task(self.query_engine.query(transcription))
            
            # Get results from both tasks
            retrieval_results, query_result = await asyncio.gather(retrieval_task, query_task)
            
            response = query_result.get("response", "")
            
            if not response:
                return {"error": "No response generated from knowledge base"}
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech
        logger.info("STAGE 3: Text-to-Speech")
        tts_start = time.time()
        
        # Wait for TTS preparation if needed
        await tts_prep_task
        
        try:
            # OPTIMIZATION: Use sentence-based chunking for faster speech synthesis
            speech_audio = bytearray()
            
            # Split response into sentences for faster processing
            if "." in response or "!" in response or "?" in response:
                sentences = []
                for end_marker in [".", "!", "?"]:
                    parts = response.split(end_marker)
                    if len(parts) > 1:
                        for i in range(len(parts)-1):
                            if parts[i].strip():
                                sentences.append(parts[i] + end_marker)
                        # Last part might not have punctuation
                        if parts[-1].strip():
                            sentences.append(parts[-1])
                
                # If no sentences detected, use whole response
                if not sentences:
                    sentences = [response]
                
                # Process each sentence
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                        
                    # Convert to speech
                    sentence_audio = await self.tts_integration.text_to_speech(sentence)
                    speech_audio.extend(sentence_audio)
            else:
                # Process whole response at once
                speech_audio = await self.tts_integration.text_to_speech(response)
            
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
            "speech_audio": None if output_speech_file else bytes(speech_audio),
            "timings": timings,
            "total_time": total_time
        }
    
    async def _prepare_kb(self) -> None:
        """Prepare knowledge base in advance."""
        if hasattr(self.query_engine, "get_stats"):
            await self.query_engine.get_stats()
    
    async def _prepare_tts(self) -> None:
        """Prepare TTS in advance."""
        # Nothing to do currently, but could preload models or warm up the system
        pass
    
    async def process_audio_streaming(
        self,
        audio_data: Union[bytes, np.ndarray],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process audio data with parallel pipeline and streaming response.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            audio_callback: Callback to handle audio data
            
        Returns:
            Dictionary with stats about the process
        """
        logger.info(f"Starting streaming pipeline with audio: {type(audio_data)}")
        
        # Record start time for tracking
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        try:
            # Ensure audio is in the right format
            if isinstance(audio_data, bytes):
                # Convert bytes to numpy array if needed
                audio = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # OPTIMIZATION: Transcribe audio in a separate task for parallel processing
            transcription_task = asyncio.create_task(self._transcribe_audio(audio))
            
            # While transcription is happening, prepare KB and TTS
            kb_prep_task = asyncio.create_task(self._prepare_kb())
            tts_prep_task = asyncio.create_task(self._prepare_tts())
            
            # Wait for transcription
            transcription, duration = await transcription_task
            
            # Validate transcription
            is_valid = await self._is_valid_transcription(transcription)
            if not is_valid:
                logger.warning(f"Transcription not valid for processing: '{transcription}'")
                return {"error": "No valid transcription detected", "transcription": transcription}
                
            logger.info(f"Transcription: {transcription}")
            transcription_time = time.time() - start_time
            
            # Wait for KB preparation
            await kb_prep_task
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {"error": f"Transcription error: {str(e)}"}
        
        # Stream the response with TTS
        try:
            # Start query engine processing
            query_task = asyncio.create_task(self.query_engine.query(transcription))
            
            # Wait for TTS preparation
            await tts_prep_task
            
            # Wait for query to complete
            query_result = await query_task
            response = query_result.get("response", "")
            
            if not response:
                return {"error": "No response generated"}
            
            # OPTIMIZATION: Use sentence-based streaming for faster responses
            total_chunks = 0
            total_audio_bytes = 0
            response_start_time = time.time()
            full_response = ""
            
            # Stream response by sentences if possible
            if "." in response or "!" in response or "?" in response:
                sentences = []
                for end_marker in [".", "!", "?"]:
                    parts = response.split(end_marker)
                    if len(parts) > 1:
                        for i in range(len(parts)-1):
                            if parts[i].strip():
                                sentences.append(parts[i] + end_marker)
                        # Last part might not have punctuation
                        if parts[-1].strip():
                            sentences.append(parts[-1])
                
                # If no sentences detected, use whole response
                if not sentences:
                    sentences = [response]
                
                # Process each sentence
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                        
                    # Add to full response
                    full_response += sentence
                    
                    # Convert to speech
                    audio_data = await self.tts_integration.text_to_speech(sentence)
                    await audio_callback(audio_data)
                    
                    # Update stats
                    total_chunks += 1
                    total_audio_bytes += len(audio_data)
            else:
                # Process whole response at once
                full_response = response
                audio_data = await self.tts_integration.text_to_speech(response)
                await audio_callback(audio_data)
                total_chunks = 1
                total_audio_bytes = len(audio_data)
            
            # Calculate stats
            response_time = time.time() - response_start_time
            total_time = time.time() - start_time
            
            return {
                "transcription": transcription,
                "transcription_time": transcription_time,
                "response_time": response_time,
                "total_time": total_time,
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "full_response": full_response
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            return {
                "error": f"Streaming error: {str(e)}",
                "transcription": transcription,
                "transcription_time": transcription_time
            }
    
    async def process_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray],
        speech_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio data through the complete pipeline.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            speech_output_path: Path to save speech output
            
        Returns:
            Results dictionary
        """
        logger.info(f"Starting pipeline with audio data: {type(audio_data)}")
        
        # Track timing
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # Convert audio data to numpy array if needed
        if isinstance(audio_data, bytes):
            audio = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio = audio_data
        
        # STAGE 1: Speech-to-Text using parallel tasks
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # OPTIMIZATION: Process STT in parallel with KB/TTS preparation
        transcription_task = asyncio.create_task(self._transcribe_audio(audio))
        kb_prep_task = asyncio.create_task(self._prepare_kb())
        tts_prep_task = asyncio.create_task(self._prepare_tts())
        
        # Wait for transcription
        transcription, duration = await transcription_task
        
        # Validate transcription
        is_valid = await self._is_valid_transcription(transcription)
        if not is_valid:
            logger.warning(f"Transcription not valid for processing: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
        timings = {"stt": time.time() - stt_start}
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        # Wait for KB preparation
        await kb_prep_task
        
        try:
            # OPTIMIZATION: Run retrieval and query tasks in parallel 
            retrieval_task = asyncio.create_task(self.query_engine.retrieve_with_sources(transcription))
            query_task = asyncio.create_task(self.query_engine.query(transcription))
            
            # Wait for both tasks to complete
            retrieval_results, query_result = await asyncio.gather(retrieval_task, query_task)
            
            response = query_result.get("response", "")
            
            if not response:
                return {"error": "No response generated from knowledge base"}
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech
        logger.info("STAGE 3: Text-to-Speech")
        tts_start = time.time()
        
        # Wait for TTS preparation
        await tts_prep_task
        
        try:
            # OPTIMIZATION: Process by sentences for streaming-like behavior
            speech_audio = bytearray()
            
            if "." in response or "!" in response or "?" in response:
                sentences = []
                for end_marker in [".", "!", "?"]:
                    parts = response.split(end_marker)
                    if len(parts) > 1:
                        for i in range(len(parts)-1):
                            if parts[i].strip():
                                sentences.append(parts[i] + end_marker)
                        # Last part might not have punctuation
                        if parts[-1].strip():
                            sentences.append(parts[-1])
                
                # If no sentences detected, use whole response
                if not sentences:
                    sentences = [response]
                
                # Process each sentence
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                    
                    logger.debug(f"Converting sentence {i+1}/{len(sentences)} to speech")
                    
                    # Convert to speech
                    sentence_audio = await self.tts_integration.text_to_speech(sentence)
                    speech_audio.extend(sentence_audio)
            else:
                # Process whole response at once
                speech_audio = await self.tts_integration.text_to_speech(response)
            
            # Save speech audio if output file specified
            if speech_output_path:
                os.makedirs(os.path.dirname(os.path.abspath(speech_output_path)), exist_ok=True)
                with open(speech_output_path, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {speech_output_path}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s, generated {len(speech_audio)} bytes")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Compile results
            return {
                "transcription": transcription,
                "response": response,
                "speech_audio_size": len(speech_audio),
                "speech_audio": bytes(speech_audio),
                "timings": timings,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Error in TTS stage: {e}")
            return {
                "error": f"TTS error: {str(e)}",
                "transcription": transcription,
                "response": response
            }
    
    async def _transcribe_audio(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio data using either Deepgram or Whisper with optimized frame size.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
        logger.info(f"Transcribing audio: {len(audio)} samples")
        
        # Check if we're using Deepgram STT
        if self.using_deepgram:
            return await self._transcribe_audio_deepgram(audio)
        else:
            # Fallback to original Whisper approach if not using Deepgram
            return await self._transcribe_audio_whisper(audio)
    
    async def _transcribe_audio_deepgram(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio using Deepgram STT with optimized frame size.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
        try:
            # Convert to 16-bit PCM bytes
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Start a streaming session
            await self.speech_recognizer.start_streaming()
            
            # Track final results
            final_results = []
            
            # Process callback to collect results
            async def collect_result(result):
                if result.is_final:
                    final_results.append(result)
            
            # OPTIMIZATION: Process audio in 100ms chunks (~1600 bytes at 16kHz)
            chunk_size = 1600  # ~100ms at 16kHz
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                result = await self.speech_recognizer.process_audio_chunk(chunk, collect_result)
                
                # Add final results directly
                if result and result.is_final:
                    final_results.append(result)
            
            # Stop streaming
            await self.speech_recognizer.stop_streaming()
            
            # Get best final result based on confidence
            if final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                text = best_result.text
                # Calculate duration
                duration = best_result.end_time - best_result.start_time if best_result.end_time > 0 else len(audio) / 16000
            else:
                text = ""
                duration = len(audio) / 16000
            
            # Clean up the transcription
            transcription = self.stt_helper.cleanup_transcription(text)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in Deepgram transcription: {e}", exc_info=True)
            return "", len(audio) / 16000
    
    async def _transcribe_audio_whisper(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio using Whisper STT (original implementation).
        """
        # Save original VAD setting
        original_vad = self.speech_recognizer.vad_enabled
        
        # Set VAD based on audio length
        is_short_audio = len(audio) < self.speech_recognizer.sample_rate * 1.0  # Less than 1 second
        self.speech_recognizer.vad_enabled = not is_short_audio  # Disable VAD for short audio
        
        transcription = ""
        duration = 0
        
        try:
            # Reset any existing streaming session
            if hasattr(self.speech_recognizer, 'is_streaming') and self.speech_recognizer.is_streaming:
                await self.speech_recognizer.stop_streaming()
            
            # Handle short audio
            min_audio_length = self.speech_recognizer.sample_rate * 1.0  # 1 second
            if len(audio) < min_audio_length:
                # Pad with silence if too short
                logger.info(f"Audio too short ({len(audio)/self.speech_recognizer.sample_rate:.2f}s), padding to {min_audio_length/self.speech_recognizer.sample_rate:.2f}s")
                padding = np.zeros(min_audio_length - len(audio), dtype=np.float32)
                audio = np.concatenate([audio, padding])
            
            # Start a new streaming session
            self.speech_recognizer.start_streaming()
            logger.info("Started streaming session for transcription")
            
            # OPTIMIZATION: Process audio in 100ms chunks for early results
            chunk_size = int(self.speech_recognizer.sample_rate * 0.1)  # 100ms chunks
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                await self.speech_recognizer.process_audio_chunk(chunk)
            
            # Get final transcription
            transcription, duration = await self.speech_recognizer.stop_streaming()
            
            # Clean up transcription
            transcription = self.stt_helper.cleanup_transcription(transcription)
            
            # Check if we got a valid transcription
            if not transcription or transcription.strip() == "" or transcription == "[BLANK_AUDIO]":
                logger.warning("First transcription attempt returned empty result, trying again with higher temperature")
                
                # Try again with different parameters
                self.speech_recognizer.start_streaming()
                self.speech_recognizer.update_parameters(temperature=0.2)  # Try with higher temperature
                
                # Process in chunks again
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    await self.speech_recognizer.process_audio_chunk(chunk)
                
                raw_transcription, duration = await self.speech_recognizer.stop_streaming()
                transcription = self.stt_helper.cleanup_transcription(raw_transcription)
                self.speech_recognizer.update_parameters(temperature=0.0)  # Reset temperature
                
                # If still no result, try one more time with more padding
                if not transcription or transcription.strip() == "" or transcription == "[BLANK_AUDIO]":
                    logger.warning("Second transcription attempt returned empty result, trying with more padding")
                    
                    # Add more padding (2 seconds total)
                    more_padding = np.zeros(self.speech_recognizer.sample_rate * 1.0, dtype=np.float32)
                    padded_audio = np.concatenate([audio, more_padding])
                    
                    self.speech_recognizer.start_streaming()
                    self.speech_recognizer.update_parameters(temperature=0.4)  # Even higher temperature
                    
                    # Process in chunks
                    for i in range(0, len(padded_audio), chunk_size):
                        chunk = padded_audio[i:i+chunk_size]
                        await self.speech_recognizer.process_audio_chunk(chunk)
                    
                    raw_transcription, duration = await self.speech_recognizer.stop_streaming()
                    transcription = self.stt_helper.cleanup_transcription(raw_transcription)
                    self.speech_recognizer.update_parameters(temperature=0.0)  # Reset temperature
        except Exception as e:
            logger.error(f"Error in transcription: {e}", exc_info=True)
            # Try one more time with basic parameters
            try:
                logger.info("Trying transcription one more time after error")
                self.speech_recognizer.start_streaming()
                
                # Process whole audio at once
                await self.speech_recognizer.process_audio_chunk(audio)
                
                raw_transcription, duration = await self.speech_recognizer.stop_streaming()
                transcription = self.stt_helper.cleanup_transcription(raw_transcription)
            except Exception as e2:
                logger.error(f"Second transcription attempt also failed: {e2}", exc_info=True)
        finally:
            # Restore original VAD setting
            self.speech_recognizer.vad_enabled = original_vad
        
        # Log the result
        if transcription:
            logger.info(f"Transcription result: '{transcription}'")
        else:
            logger.warning("No transcription generated")
        
        return transcription, duration
    
    async def process_realtime_stream(
        self,
        audio_chunk_generator: AsyncIterator[np.ndarray],
        audio_output_callback: Callable[[bytes], Awaitable[None]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a real-time audio stream with immediate response.
        
        This method is designed for WebSocket-based streaming where audio chunks
        are continuously arriving and responses should be sent back as soon as possible.
        
        Args:
            audio_chunk_generator: Async generator producing audio chunks
            audio_output_callback: Callback to handle output audio data
            
        Yields:
            Status updates and results
        """
        logger.info("Starting real-time audio stream processing")
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # Track state with improved early detection
        accumulated_audio = bytearray()
        processing = False
        last_transcription = ""
        silence_frames = 0
        max_silence_frames = 5  # Number of silent chunks before processing
        silence_threshold = 0.01
        chunks_since_speech_start = 0
        early_processing_threshold = 3  # Start after ~100-150ms of speech
        is_speech_detected = False
        
        # Timing stats
        start_time = time.time()
        
        try:
            # Initialize the appropriate speech recognizer
            if self.using_deepgram:
                await self.speech_recognizer.start_streaming()
            else:
                self.speech_recognizer.start_streaming()
            
            # Define result collecting callback
            results = []
            
            async def result_callback(result):
                results.append(result)
                
                # For Deepgram, check if this is a final result
                if hasattr(result, 'is_final') and result.is_final:
                    logger.info(f"Final transcription from callback: {result.text}")
            
            # Process incoming audio chunks
            async for audio_chunk in audio_chunk_generator:
                # OPTIMIZATION: Start processing after minimal speech (~100-150ms)
                # Check for speech activity
                if isinstance(audio_chunk, bytes):
                    audio_chunk = np.frombuffer(audio_chunk, dtype=np.float32)
                
                # Calculate energy for speech detection
                energy = np.mean(np.abs(audio_chunk))
                is_silence = energy < silence_threshold
                
                if not is_silence and not is_speech_detected:
                    # Speech might be starting
                    chunks_since_speech_start += 1
                    
                    # Start early processing after 3 chunks (~100-150ms)
                    if chunks_since_speech_start >= early_processing_threshold:
                        is_speech_detected = True
                        logger.info("Early speech detection - starting processing")
                        
                        # Process through STT early
                        if self.using_deepgram:
                            # Convert for Deepgram
                            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                            await self.speech_recognizer.process_audio_chunk(
                                audio_bytes, callback=result_callback
                            )
                        else:
                            await self.speech_recognizer.process_audio_chunk(
                                audio_chunk=audio_chunk,
                                callback=result_callback
                            )
                elif is_silence:
                    # Reset speech detection if we encounter silence
                    chunks_since_speech_start = 0
                    silence_frames += 1
                else:
                    # Continue processing speech
                    silence_frames = 0
                    
                    # Process through appropriate STT
                    if self.using_deepgram:
                        # Convert for Deepgram
                        audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                        # OPTIMIZATION: Use smaller chunks
                        chunk_size = 800  # 100ms at 8kHz
                        
                        for i in range(0, len(audio_bytes), chunk_size):
                            chunk = audio_bytes[i:i+chunk_size]
                            if len(chunk) < chunk_size/2:  # Skip very small final chunks
                                continue
                                
                            await self.speech_recognizer.process_audio_chunk(
                                chunk, callback=result_callback
                            )
                    else:
                        await self.speech_recognizer.process_audio_chunk(
                            audio_chunk=audio_chunk,
                            callback=result_callback
                        )
                
                # If we have enough silence frames, process any accumulated speech
                if silence_frames >= max_silence_frames and not processing and is_speech_detected:
                    # Get transcription from results
                    final_results = [r for r in results if 
                                    getattr(r, 'is_final', True) and 
                                    hasattr(r, 'text') and r.text]
                    
                    # If we have final results, process them
                    if final_results:
                        # Get best result based on confidence
                        if hasattr(final_results[0], 'confidence'):
                            best_result = max(final_results, key=lambda r: getattr(r, 'confidence', 0))
                        else:
                            # Otherwise use the longest text
                            best_result = max(final_results, key=lambda r: len(getattr(r, 'text', '')))
                        
                        # Get transcription
                        transcription = best_result.text if hasattr(best_result, 'text') else ""
                        
                        # Clean up transcription
                        transcription = self.stt_helper.cleanup_transcription(transcription)
                        
                        # Only process if it's different from last transcription
                        if transcription and transcription != last_transcription:
                            # Valid transcription found
                            processing = True
                            
                            # Process through Knowledge Base
                            try:
                                # OPTIMIZATION: Use parallel tasks for knowledge query and TTS preparation
                                query_task = asyncio.create_task(self.query_engine.query(transcription))
                                
                                # Yield status update
                                yield {
                                    "status": "transcribed",
                                    "transcription": transcription
                                }
                                
                                # Get query result
                                query_result = await query_task
                                response = query_result.get("response", "")
                                
                                if response:
                                    # OPTIMIZATION: Stream by sentences for faster response
                                    if "." in response or "!" in response or "?" in response:
                                        sentences = []
                                        for end_marker in [".", "!", "?"]:
                                            parts = response.split(end_marker)
                                            if len(parts) > 1:
                                                for i in range(len(parts)-1):
                                                    if parts[i].strip():
                                                        sentences.append(parts[i] + end_marker)
                                                # Last part might not have punctuation
                                                if parts[-1].strip():
                                                    sentences.append(parts[-1])
                                        
                                        # If no sentences detected, use whole response
                                        if not sentences:
                                            sentences = [response]
                                        
                                        # Process each sentence
                                        for sentence in sentences:
                                            if not sentence.strip():
                                                continue
                                                
                                            # Convert to speech
                                            speech_audio = await self.tts_integration.text_to_speech(sentence)
                                            
                                            # Send through callback
                                            await audio_output_callback(speech_audio)
                                    else:
                                        # Process whole response at once
                                        speech_audio = await self.tts_integration.text_to_speech(response)
                                        
                                        # Send through callback
                                        await audio_output_callback(speech_audio)
                                    
                                    # Yield response
                                    yield {
                                        "status": "response",
                                        "transcription": transcription,
                                        "response": response
                                    }
                                    
                                    # Update last transcription
                                    last_transcription = transcription
                            except Exception as e:
                                logger.error(f"Error processing transcription: {e}", exc_info=True)
                                yield {
                                    "status": "error",
                                    "error": str(e)
                                }
                            finally:
                                processing = False
                                
                    # Reset state for next utterance
                    results = []  # Clear results
                    is_speech_detected = False
                    chunks_since_speech_start = 0
                    
                    # Reset speech recognizer for next utterance
                    try:
                        if self.using_deepgram:
                            await self.speech_recognizer.stop_streaming()
                            await self.speech_recognizer.start_streaming()
                        else:
                            await self.speech_recognizer.stop_streaming()
                            self.speech_recognizer.start_streaming()
                    except Exception as e:
                        logger.error(f"Error resetting speech recognizer: {e}")
            
            # Process any final audio
            if self.using_deepgram:
                await self.speech_recognizer.stop_streaming()
            else:
                final_transcription, _ = await self.speech_recognizer.stop_streaming()
                final_transcription = self.stt_helper.cleanup_transcription(final_transcription)
                
                # If we have a valid final transcription, process it
                if final_transcription and final_transcription != last_transcription:
                    # Generate final response
                    query_result = await self.query_engine.query(final_transcription)
                    final_response = query_result.get("response", "")
                    
                    if final_response:
                        # Convert to speech
                        final_speech = await self.tts_integration.text_to_speech(final_response)
                        
                        # Send through callback
                        await audio_output_callback(final_speech)
                        
                        # Yield final response
                        yield {
                            "status": "final",
                            "transcription": final_transcription,
                            "response": final_response,
                            "audio_size": len(final_speech),
                            "total_time": time.time() - start_time
                        }
            
            # Yield completion
            yield {
                "status": "complete",
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in real-time stream processing: {e}", exc_info=True)
            yield {
                "status": "error",
                "error": str(e),
                "total_time": time.time() - start_time
            }
        finally:
            # Ensure speech recognizer is properly closed
            try:
                if self.using_deepgram:
                    await self.speech_recognizer.stop_streaming()
                else:
                    await self.speech_recognizer.stop_streaming()
            except:
                pass