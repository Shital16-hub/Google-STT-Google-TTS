"""
Enhanced Speech-to-Text System with Multiple Provider Support
Fixed version with proper imports and fallback mechanisms for robust operation.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import the Google Cloud STT from the same directory
from .google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

logger = logging.getLogger(__name__)

class STTProvider(str, Enum):
    """Supported STT providers."""
    GOOGLE_CLOUD_V2 = "google_cloud_v2"
    ASSEMBLYAI = "assemblyai"
    WHISPER = "whisper"
    FALLBACK = "fallback"

class STTStatus(str, Enum):
    """STT system status."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    ERROR = "error"
    FALLBACK_MODE = "fallback_mode"

@dataclass
class STTConfig:
    """Configuration for STT system."""
    primary_provider: STTProvider
    backup_provider: Optional[STTProvider] = None
    language: str = "en-US"
    sample_rate: int = 8000
    encoding: str = "MULAW"
    channels: int = 1
    interim_results: bool = False
    enable_vad: bool = True
    enable_echo_cancellation: bool = True
    target_latency_ms: int = 80
    confidence_threshold: float = 0.7
    auto_fallback: bool = True
    max_retries: int = 3

@dataclass
class STTMetrics:
    """STT performance metrics."""
    total_audio_chunks: int = 0
    successful_transcriptions: int = 0
    failed_transcriptions: int = 0
    average_latency_ms: float = 0.0
    average_confidence: float = 0.0
    provider_switches: int = 0
    echo_detections: int = 0
    vad_activations: int = 0
    session_duration: float = 0.0
    last_update: float = field(default_factory=time.time)

class FallbackSTT:
    """Fallback STT implementation for when external providers fail."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.is_streaming = False
        self.audio_buffer = []
        self.fallback_responses = [
            "I'm having trouble with speech recognition right now.",
            "Could you please repeat that?",
            "I didn't catch that. Please try again.",
            "Speech recognition is currently unavailable.",
            "Please speak clearly and try again."
        ]
        self.response_index = 0
        
    async def start_streaming(self):
        """Start fallback streaming."""
        self.is_streaming = True
        self.session_id = str(uuid.uuid4())
        logger.info("Started fallback STT streaming")
    
    async def stop_streaming(self):
        """Stop fallback streaming."""
        self.is_streaming = False
        logger.info("Stopped fallback STT streaming")
        return "", 0.0
    
    async def process_audio_chunk(self, audio_chunk, callback=None):
        """Process audio chunk in fallback mode."""
        self.audio_buffer.append(audio_chunk)
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Occasionally return a fallback response
        if len(self.audio_buffer) % 50 == 0:  # Every 50 chunks
            response = StreamingTranscriptionResult(
                text=self.fallback_responses[self.response_index % len(self.fallback_responses)],
                is_final=True,
                confidence=0.1,
                session_id=self.session_id
            )
            self.response_index += 1
            
            if callback:
                await callback(response)
            
            return response
        
        return None
    
    def get_stats(self):
        """Get fallback STT stats."""
        return {
            "provider": "fallback",
            "session_id": self.session_id,
            "is_streaming": self.is_streaming,
            "audio_buffer_size": len(self.audio_buffer),
            "responses_generated": self.response_index
        }
    
    async def cleanup(self):
        """Cleanup fallback STT."""
        self.audio_buffer.clear()
        self.is_streaming = False

class VoiceActivityDetector:
    """Simple voice activity detection."""
    
    def __init__(self, threshold: float = 0.01, window_size: int = 10):
        self.threshold = threshold
        self.window_size = window_size
        self.energy_buffer = []
        self.is_speech = False
        self.speech_start_time = None
        
    def detect_speech(self, audio_chunk: Union[bytes, np.ndarray]) -> bool:
        """Detect if audio chunk contains speech."""
        try:
            # Convert audio to numpy array if needed
            if isinstance(audio_chunk, bytes):
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_array = audio_chunk
            
            # Calculate energy
            energy = np.mean(np.abs(audio_array.astype(np.float32)))
            
            # Add to buffer
            self.energy_buffer.append(energy)
            if len(self.energy_buffer) > self.window_size:
                self.energy_buffer.pop(0)
            
            # Determine if speech is present
            avg_energy = np.mean(self.energy_buffer)
            current_speech = avg_energy > self.threshold
            
            # Track speech transitions
            if current_speech and not self.is_speech:
                self.speech_start_time = time.time()
                logger.debug("Speech activity started")
            elif not current_speech and self.is_speech:
                if self.speech_start_time:
                    duration = time.time() - self.speech_start_time
                    logger.debug(f"Speech activity ended (duration: {duration:.2f}s)")
            
            self.is_speech = current_speech
            return self.is_speech
            
        except Exception as e:
            logger.error(f"Error in VAD: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get VAD statistics."""
        return {
            "is_speech": self.is_speech,
            "threshold": self.threshold,
            "current_energy": self.energy_buffer[-1] if self.energy_buffer else 0.0,
            "avg_energy": np.mean(self.energy_buffer) if self.energy_buffer else 0.0,
            "speech_duration": time.time() - self.speech_start_time if self.speech_start_time else 0.0
        }

class EchoCanceller:
    """Simple echo cancellation using recent TTS output."""
    
    def __init__(self, window_seconds: float = 3.0):
        self.window_seconds = window_seconds
        self.recent_tts_outputs = []
        
    def add_tts_output(self, text: str):
        """Add TTS output for echo detection."""
        current_time = time.time()
        self.recent_tts_outputs.append((text.lower(), current_time))
        
        # Clean old entries
        self.recent_tts_outputs = [
            (text, timestamp) for text, timestamp in self.recent_tts_outputs
            if current_time - timestamp <= self.window_seconds
        ]
    
    def is_echo(self, transcribed_text: str) -> bool:
        """Check if transcribed text is likely an echo."""
        if not transcribed_text:
            return False
        
        text_lower = transcribed_text.lower()
        current_time = time.time()
        
        for tts_text, timestamp in self.recent_tts_outputs:
            if current_time - timestamp > self.window_seconds:
                continue
            
            # Simple similarity check
            if text_lower == tts_text:
                return True
            
            # Check for significant word overlap
            text_words = set(text_lower.split())
            tts_words = set(tts_text.split())
            
            if len(text_words) > 0 and len(tts_words) > 0:
                overlap = len(text_words & tts_words)
                overlap_ratio = overlap / len(text_words)
                if overlap_ratio > 0.7:  # 70% overlap
                    return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get echo canceller statistics."""
        return {
            "recent_outputs": len(self.recent_tts_outputs),
            "window_seconds": self.window_seconds,
            "oldest_output_age": (
                time.time() - self.recent_tts_outputs[0][1] 
                if self.recent_tts_outputs else 0.0
            )
        }

class EnhancedSTTSystem:
    """
    Enhanced Speech-to-Text system with multiple provider support,
    voice activity detection, echo cancellation, and robust fallback mechanisms.
    """
    
    def __init__(
        self,
        primary_provider: str = "google_cloud_v2",
        backup_provider: Optional[str] = "assemblyai",
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = False,
        enable_vad: bool = True,
        enable_echo_cancellation: bool = True,
        target_latency_ms: int = 80,
        **kwargs
    ):
        """Initialize enhanced STT system."""
        # Create configuration
        self.config = STTConfig(
            primary_provider=STTProvider(primary_provider),
            backup_provider=STTProvider(backup_provider) if backup_provider else None,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            channels=channels,
            interim_results=interim_results,
            enable_vad=enable_vad,
            enable_echo_cancellation=enable_echo_cancellation,
            target_latency_ms=target_latency_ms,
            **kwargs
        )
        
        # Initialize components
        self.primary_stt = None
        self.backup_stt = None
        self.fallback_stt = FallbackSTT(self.config)
        self.current_provider = STTProvider.FALLBACK
        
        # Optional components
        self.vad = VoiceActivityDetector() if enable_vad else None
        self.echo_canceller = EchoCanceller() if enable_echo_cancellation else None
        
        # System state
        self.status = STTStatus.IDLE
        self.metrics = STTMetrics()
        self.session_id = str(uuid.uuid4())
        self.is_streaming = False
        self.initialization_start_time = time.time()
        
        # Error tracking
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.error_backoff = 1.0
        
        # Callbacks
        self.transcription_callback = None
        
        self.initialized = False
        logger.info(f"Enhanced STT System created with primary: {primary_provider}, backup: {backup_provider}")
    
    async def initialize(self):
        """Initialize STT providers with fallback handling."""
        logger.info("🎤 Initializing Enhanced STT System...")
        
        # Initialize primary provider
        await self._initialize_primary_provider()
        
        # Initialize backup provider if available
        if self.config.backup_provider:
            await self._initialize_backup_provider()
        
        # Always initialize fallback
        self.fallback_stt = FallbackSTT(self.config)
        
        # Set current provider based on what's available
        if self.primary_stt:
            self.current_provider = self.config.primary_provider
            logger.info(f"✅ Primary STT provider active: {self.config.primary_provider}")
        elif self.backup_stt:
            self.current_provider = self.config.backup_provider
            logger.info(f"✅ Backup STT provider active: {self.config.backup_provider}")
        else:
            self.current_provider = STTProvider.FALLBACK
            logger.warning("⚠️ Using fallback STT provider")
        
        # Calculate initialization time
        init_time = (time.time() - self.initialization_start_time) * 1000
        logger.info(f"Enhanced STT System initialized in {init_time:.2f}ms")
        
        self.initialized = True
    
    async def _initialize_primary_provider(self):
        """Initialize primary STT provider."""
        try:
            if self.config.primary_provider == STTProvider.GOOGLE_CLOUD_V2:
                self.primary_stt = GoogleCloudStreamingSTT(
                    language=self.config.language,
                    sample_rate=self.config.sample_rate,
                    encoding=self.config.encoding,
                    channels=self.config.channels,
                    interim_results=self.config.interim_results
                )
                logger.info("✅ Google Cloud STT v2 initialized")
            else:
                logger.warning(f"Provider {self.config.primary_provider} not implemented yet")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize primary STT provider: {e}")
            self.primary_stt = None
    
    async def _initialize_backup_provider(self):
        """Initialize backup STT provider."""
        try:
            if self.config.backup_provider == STTProvider.ASSEMBLYAI:
                # AssemblyAI implementation would go here
                logger.info("AssemblyAI provider not implemented yet")
                self.backup_stt = None
            else:
                logger.info(f"Backup provider {self.config.backup_provider} not implemented yet")
                self.backup_stt = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize backup STT provider: {e}")
            self.backup_stt = None
    
    async def start_streaming(self, callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None):
        """Start streaming transcription."""
        if self.is_streaming:
            logger.debug("STT streaming already active")
            return
        
        self.transcription_callback = callback
        self.session_id = str(uuid.uuid4())
        self.is_streaming = True
        self.status = STTStatus.LISTENING
        self.metrics.session_duration = time.time()
        
        # Start current provider
        current_stt = self._get_current_stt()
        if current_stt:
            try:
                await current_stt.start_streaming()
                logger.info(f"Started STT streaming with {self.current_provider}")
            except Exception as e:
                logger.error(f"Error starting STT streaming: {e}")
                await self._handle_provider_error()
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming transcription."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        self.status = STTStatus.IDLE
        
        # Calculate session duration
        if self.metrics.session_duration:
            session_duration = time.time() - self.metrics.session_duration
            self.metrics.session_duration = session_duration
        else:
            session_duration = 0.0
        
        # Stop current provider
        current_stt = self._get_current_stt()
        if current_stt:
            try:
                result = await current_stt.stop_streaming()
                logger.info(f"Stopped STT streaming (duration: {session_duration:.2f}s)")
                return result
            except Exception as e:
                logger.error(f"Error stopping STT streaming: {e}")
        
        return "", session_duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk through the enhanced STT pipeline."""
        if not self.initialized:
            await self.initialize()
        
        self.metrics.total_audio_chunks += 1
        processing_start = time.time()
        
        try:
            # Voice Activity Detection
            if self.vad:
                has_speech = self.vad.detect_speech(audio_chunk)
                if has_speech:
                    self.metrics.vad_activations += 1
                
                # Skip processing if no speech detected (optional optimization)
                # if not has_speech:
                #     return None
            
            # Process through current STT provider
            current_stt = self._get_current_stt()
            if not current_stt:
                logger.error("No STT provider available")
                return None
            
            # Use provided callback or stored callback
            processing_callback = callback or self.transcription_callback
            if processing_callback:
                enhanced_callback = self._create_enhanced_callback(processing_callback)
            else:
                enhanced_callback = None
            
            # Process audio chunk
            result = await current_stt.process_audio_chunk(audio_chunk, enhanced_callback)
            
            # Update metrics
            processing_time = (time.time() - processing_start) * 1000
            if self.metrics.average_latency_ms == 0:
                self.metrics.average_latency_ms = processing_time
            else:
                total_chunks = self.metrics.total_audio_chunks
                self.metrics.average_latency_ms = (
                    (self.metrics.average_latency_ms * (total_chunks - 1) + processing_time) / total_chunks
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self.metrics.failed_transcriptions += 1
            await self._handle_provider_error()
            return None
    
    def _create_enhanced_callback(self, original_callback):
        """Create enhanced callback with echo cancellation and metrics."""
        async def enhanced_callback(result: StreamingTranscriptionResult):
            try:
                # Echo cancellation
                if self.echo_canceller and result.is_final:
                    if self.echo_canceller.is_echo(result.text):
                        logger.debug(f"Echo detected and filtered: '{result.text}'")
                        self.metrics.echo_detections += 1
                        return  # Don't call original callback for echoes
                
                # Update metrics
                if result.is_final:
                    self.metrics.successful_transcriptions += 1
                    
                    # Update average confidence
                    if self.metrics.average_confidence == 0:
                        self.metrics.average_confidence = result.confidence
                    else:
                        count = self.metrics.successful_transcriptions
                        self.metrics.average_confidence = (
                            (self.metrics.average_confidence * (count - 1) + result.confidence) / count
                        )
                
                # Call original callback
                if original_callback:
                    await original_callback(result)
                    
            except Exception as e:
                logger.error(f"Error in enhanced callback: {e}")
        
        return enhanced_callback
    
    def _get_current_stt(self):
        """Get current active STT provider."""
        if self.current_provider == STTProvider.GOOGLE_CLOUD_V2:
            return self.primary_stt
        elif self.current_provider == self.config.backup_provider:
            return self.backup_stt
        else:
            return self.fallback_stt
    
    async def _handle_provider_error(self):
        """Handle STT provider errors with fallback logic."""
        self.consecutive_errors += 1
        current_time = time.time()
        
        # If too many errors in a short time, switch providers
        if (current_time - self.last_error_time < 30 and self.consecutive_errors >= 3):
            await self._switch_to_backup_provider()
        
        self.last_error_time = current_time
    
    async def _switch_to_backup_provider(self):
        """Switch to backup STT provider."""
        old_provider = self.current_provider
        
        # Stop current provider
        current_stt = self._get_current_stt()
        if current_stt and hasattr(current_stt, 'stop_streaming'):
            try:
                await current_stt.stop_streaming()
            except:
                pass
        
        # Switch provider
        if self.current_provider == self.config.primary_provider and self.backup_stt:
            self.current_provider = self.config.backup_provider
            logger.warning(f"Switched from {old_provider} to {self.current_provider}")
        else:
            self.current_provider = STTProvider.FALLBACK
            logger.warning(f"Switched from {old_provider} to fallback mode")
        
        self.metrics.provider_switches += 1
        self.consecutive_errors = 0
        
        # Restart streaming with new provider
        if self.is_streaming:
            new_stt = self._get_current_stt()
            if new_stt:
                try:
                    await new_stt.start_streaming()
                    logger.info(f"Restarted streaming with {self.current_provider}")
                except Exception as e:
                    logger.error(f"Failed to restart with new provider: {e}")
    
    def add_tts_output(self, text: str):
        """Add TTS output for echo cancellation."""
        if self.echo_canceller:
            self.echo_canceller.add_tts_output(text)
        
        # Also add to current STT provider if it supports it
        current_stt = self._get_current_stt()
        if current_stt and hasattr(current_stt, 'add_tts_text'):
            current_stt.add_tts_text(text)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive STT system statistics."""
        current_stt = self._get_current_stt()
        provider_stats = current_stt.get_stats() if current_stt else {}
        
        stats = {
            "system": {
                "initialized": self.initialized,
                "status": self.status.value,
                "current_provider": self.current_provider.value,
                "is_streaming": self.is_streaming,
                "session_id": self.session_id,
                "consecutive_errors": self.consecutive_errors
            },
            "metrics": {
                "total_audio_chunks": self.metrics.total_audio_chunks,
                "successful_transcriptions": self.metrics.successful_transcriptions,
                "failed_transcriptions": self.metrics.failed_transcriptions,
                "average_latency_ms": round(self.metrics.average_latency_ms, 2),
                "average_confidence": round(self.metrics.average_confidence, 2),
                "provider_switches": self.metrics.provider_switches,
                "echo_detections": self.metrics.echo_detections,
                "vad_activations": self.metrics.vad_activations,
                "session_duration": self.metrics.session_duration,
                "success_rate": round(
                    (self.metrics.successful_transcriptions / max(self.metrics.total_audio_chunks, 1)) * 100, 2
                )
            },
            "provider_stats": provider_stats,
            "components": {}
        }
        
        # Add component stats
        if self.vad:
            stats["components"]["vad"] = self.vad.get_stats()
        
        if self.echo_canceller:
            stats["components"]["echo_canceller"] = self.echo_canceller.get_stats()
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of STT system."""
        health_status = {
            "healthy": True,
            "issues": [],
            "provider_health": {},
            "performance_metrics": {}
        }
        
        # Check provider health
        if self.primary_stt:
            try:
                primary_stats = self.primary_stt.get_stats()
                health_status["provider_health"]["primary"] = {
                    "available": True,
                    "streaming": primary_stats.get("is_streaming", False),
                    "errors": primary_stats.get("consecutive_errors", 0)
                }
            except Exception as e:
                health_status["provider_health"]["primary"] = {
                    "available": False,
                    "error": str(e)
                }
                health_status["issues"].append("Primary STT provider unavailable")
        
        # Check performance metrics
        if self.metrics.average_latency_ms > self.config.target_latency_ms * 2:
            health_status["healthy"] = False
            health_status["issues"].append("High latency detected")
        
        if self.consecutive_errors > 5:
            health_status["healthy"] = False
            health_status["issues"].append("Multiple consecutive errors")
        
        success_rate = (
            self.metrics.successful_transcriptions / max(self.metrics.total_audio_chunks, 1)
        ) * 100
        
        if success_rate < 80:  # Less than 80% success rate
            health_status["healthy"] = False
            health_status["issues"].append("Low transcription success rate")
        
        health_status["performance_metrics"] = {
            "latency_ms": self.metrics.average_latency_ms,
            "target_latency_ms": self.config.target_latency_ms,
            "success_rate": success_rate,
            "error_count": self.consecutive_errors
        }
        
        return health_status
    
    async def cleanup(self):
        """Cleanup all STT resources."""
        logger.info("Cleaning up Enhanced STT System...")
        
        try:
            # Stop streaming if active
            if self.is_streaming:
                await self.stop_streaming()
            
            # Cleanup providers
            if self.primary_stt and hasattr(self.primary_stt, 'cleanup'):
                await self.primary_stt.cleanup()
            
            if self.backup_stt and hasattr(self.backup_stt, 'cleanup'):
                await self.backup_stt.cleanup()
            
            if self.fallback_stt:
                await self.fallback_stt.cleanup()
            
            self.initialized = False
            logger.info("✅ Enhanced STT System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during STT cleanup: {e}")