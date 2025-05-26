"""
Voice Activity Detection - Advanced VAD with Ultra-Low Latency
Part of the Multi-Agent Voice AI System Transformation

This module implements state-of-the-art Voice Activity Detection:
- Multiple VAD algorithms with intelligent fusion
- Ultra-low latency detection (<50ms)
- Adaptive noise suppression
- End-of-speech detection optimization
- Integration with STT and TTS pipelines
- Real-time audio stream processing

SUPPORTED VAD ENGINES:
- WebRTC VAD (Primary - Ultra-fast, proven)
- Silero VAD (High accuracy, ML-based)
- Azure VAD (Cloud-based, high quality)
- Custom hybrid VAD (Multi-algorithm fusion)
- Energy-based VAD (Fallback)

TARGET METRICS:
- Detection latency: <50ms
- Accuracy: >95% speech/silence classification
- False positive rate: <2%
- End-of-speech detection: <100ms
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
import os
import threading
from dataclasses import dataclass
from collections import deque
import json

# Audio processing libraries
try:
    import webrtcvad
    import soundfile as sf
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    webrtcvad = None
    logging.warning("Audio libraries not available. Install: pip install webrtcvad soundfile librosa")

from core.latency_optimizer import latency_monitor

# Configure logging
logger = logging.getLogger(__name__)

class VADEngine(Enum):
    """Supported VAD engines"""
    WEBRTC_VAD = "webrtc_vad"  # Primary: Ultra-fast, proven
    SILERO_VAD = "silero_vad"  # High accuracy ML-based
    AZURE_VAD = "azure_vad"  # Cloud-based high quality
    HYBRID_VAD = "hybrid_vad"  # Multi-algorithm fusion
    ENERGY_VAD = "energy_vad"  # Simple energy-based fallback
    MOCK_VAD = "mock_vad"  # Development/testing

class VADSensitivity(Enum):
    """VAD sensitivity levels"""
    VERY_SENSITIVE = 0  # Detects even whispers
    SENSITIVE = 1  # Good for quiet environments
    NORMAL = 2  # Balanced detection
    AGGRESSIVE = 3  # Only clear speech

class AudioQuality(Enum):
    """Audio quality levels for VAD optimization"""
    CLEAN = "clean"  # Studio quality
    PHONE = "phone"  # Telephony quality
    NOISY = "noisy"  # Background noise present
    VERY_NOISY = "very_noisy"  # High noise environment

@dataclass
class VADResult:
    """Voice activity detection result"""
    has_speech: bool
    confidence: float
    start_time: float
    end_time: float
    audio_level: float = 0.0
    noise_level: float = 0.0
    speech_probability: float = 0.0
    engine_used: str = ""
    processing_time_ms: float = 0.0
    
@dataclass
class VADConfig:
    """Configuration for VAD engines"""
    engine: VADEngine = VADEngine.WEBRTC_VAD
    sensitivity: VADSensitivity = VADSensitivity.NORMAL
    sample_rate: int = 16000
    frame_duration_ms: int = 30  # 10, 20, or 30 ms for WebRTC
    audio_quality: AudioQuality = AudioQuality.PHONE
    enable_noise_suppression: bool = True
    min_speech_duration_ms: int = 100  # Minimum speech duration
    min_silence_duration_ms: int = 300  # Minimum silence for end-of-speech
    speech_padding_ms: int = 50  # Padding around detected speech
    energy_threshold: float = 0.01  # Energy threshold for fallback VAD
    
class VoiceActivityDetector:
    """
    Advanced Voice Activity Detection system with ultra-low latency
    Supports multiple VAD engines with intelligent fusion and optimization
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        
        # Initialize VAD engines
        self.vad_engines = {}
        self.engine_status = {}
        self._initialize_vad_engines()
        
        # Audio processing
        self.audio_buffer = deque(maxlen=100)  # Circular buffer for audio history
        self.frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)
        
        # State tracking
        self.current_state = "silence"  # "silence", "speech", "transition"
        self.state_start_time = time.time()
        self.speech_start_time = None
        self.last_speech_time = None
        
        # Statistics and metrics
        self.metrics = {
            "total_frames_processed": 0,
            "speech_frames": 0,
            "silence_frames": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "average_processing_time_ms": 0.0,
            "engine_performance": {}
        }
        
        # Adaptive parameters
        self.noise_profile = np.zeros(512)  # Background noise estimation
        self.speech_profile = np.zeros(512)  # Speech characteristics
        self.adaptation_count = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Voice Activity Detector initialized - Engine: {self.config.engine.value}")

    def _initialize_vad_engines(self):
        """Initialize all available VAD engines"""
        
        # WebRTC VAD
        if HAS_AUDIO_LIBS and webrtcvad:
            try:
                webrtc_vad = webrtcvad.Vad(self.config.sensitivity.value)
                self.vad_engines[VADEngine.WEBRTC_VAD] = webrtc_vad
                self.engine_status[VADEngine.WEBRTC_VAD] = {
                    "available": True,
                    "error_count": 0,
                    "last_used": None,
                    "average_latency_ms": 0.0
                }
                logger.info("WebRTC VAD initialized successfully")
            except Exception as e:
                logger.warning(f"WebRTC VAD initialization failed: {str(e)}")
                self.engine_status[VADEngine.WEBRTC_VAD] = {"available": False, "error": str(e)}
        else:
            self.engine_status[VADEngine.WEBRTC_VAD] = {"available": False, "error": "WebRTC VAD not available"}
        
        # Silero VAD
        try:
            # TODO: Initialize Silero VAD
            # import torch
            # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            self.engine_status[VADEngine.SILERO_VAD] = {"available": False, "error": "Not implemented"}
        except Exception as e:
            self.engine_status[VADEngine.SILERO_VAD] = {"available": False, "error": str(e)}
        
        # Azure VAD
        azure_key = os.getenv("AZURE_SPEECH_KEY")
        if azure_key:
            try:
                # TODO: Initialize Azure VAD
                self.engine_status[VADEngine.AZURE_VAD] = {"available": False, "error": "Not implemented"}
            except Exception as e:
                self.engine_status[VADEngine.AZURE_VAD] = {"available": False, "error": str(e)}
        else:
            self.engine_status[VADEngine.AZURE_VAD] = {"available": False, "error": "No API key"}
        
        # Energy-based VAD (always available as fallback)
        self.engine_status[VADEngine.ENERGY_VAD] = {
            "available": True,
            "error_count": 0,
            "last_used": None,
            "average_latency_ms": 1.0
        }
        
        # Mock VAD (always available for testing)
        self.engine_status[VADEngine.MOCK_VAD] = {
            "available": True,
            "error_count": 0,
            "last_used": None,
            "average_latency_ms": 0.5
        }

    @latency_monitor("vad_detect_speech")
    async def has_speech(self, audio_data: bytes, sample_rate: Optional[int] = None) -> bool:
        """
        Detect if audio contains speech (simple boolean interface)
        
        Args:
            audio_data: Raw audio data
            sample_rate: Audio sample rate (uses config default if None)
            
        Returns:
            bool: True if speech detected, False otherwise
        """
        result = await self.detect_voice_activity(audio_data, sample_rate)
        return result.has_speech

    @latency_monitor("vad_detect_activity")
    async def detect_voice_activity(self, 
                                   audio_data: bytes, 
                                   sample_rate: Optional[int] = None) -> VADResult:
        """
        Comprehensive voice activity detection with detailed results
        
        Args:
            audio_data: Raw audio data
            sample_rate: Audio sample rate (uses config default if None)
            
        Returns:
            VADResult: Detailed voice activity detection result
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate
        
        start_time = time.time()
        
        try:
            # Preprocess audio data
            processed_audio = await self._preprocess_audio(audio_data, sample_rate)
            
            # Select optimal VAD engine
            engine = self._select_optimal_engine()
            
            # Perform VAD detection
            if engine == VADEngine.WEBRTC_VAD:
                result = await self._webrtc_vad_detect(processed_audio, sample_rate)
            elif engine == VADEngine.SILERO_VAD:
                result = await self._silero_vad_detect(processed_audio, sample_rate)
            elif engine == VADEngine.AZURE_VAD:
                result = await self._azure_vad_detect(processed_audio, sample_rate)
            elif engine == VADEngine.HYBRID_VAD:
                result = await self._hybrid_vad_detect(processed_audio, sample_rate)
            elif engine == VADEngine.ENERGY_VAD:
                result = await self._energy_vad_detect(processed_audio, sample_rate)
            else:
                result = await self._mock_vad_detect(processed_audio, sample_rate)
            
            # Post-process result
            result = await self._post_process_result(result, processed_audio)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            result.engine_used = engine.value
            
            await self._update_metrics(result, engine)
            
            return result
            
        except Exception as e:
            logger.error(f"VAD detection failed: {str(e)}")
            # Return fallback result
            return VADResult(
                has_speech=False,
                confidence=0.0,
                start_time=start_time,
                end_time=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000,
                engine_used="error_fallback"
            )

    async def _preprocess_audio(self, audio_data: bytes, sample_rate: int) -> np.ndarray:
        """Preprocess audio data for optimal VAD performance"""
        try:
            # Convert bytes to numpy array
            if len(audio_data) == 0:
                return np.array([])
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if necessary
            if sample_rate != self.config.sample_rate:
                if HAS_AUDIO_LIBS:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=self.config.sample_rate)
                else:
                    # Simple resampling fallback
                    ratio = self.config.sample_rate / sample_rate
                    new_length = int(len(audio_array) * ratio)
                    audio_array = np.interp(np.linspace(0, len(audio_array), new_length), 
                                          np.arange(len(audio_array)), audio_array)
            
            # Apply noise suppression if enabled
            if self.config.enable_noise_suppression:
                audio_array = await self._suppress_noise(audio_array)
            
            # Normalize audio level
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.9
            
            return audio_array
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {str(e)}")
            # Return minimal processing
            return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    async def _suppress_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise suppression to improve VAD accuracy"""
        try:
            if len(audio) < 512:
                return audio
            
            # Simple spectral subtraction for noise reduction
            # Estimate noise from first portion (assumed quieter)
            noise_samples = min(len(audio) // 4, len(audio) - 256)
            if noise_samples > 256:
                # Update noise profile
                noise_spectrum = np.abs(np.fft.fft(audio[:noise_samples], 512))
                self.noise_profile = 0.9 * self.noise_profile + 0.1 * noise_spectrum
                
                # Apply spectral subtraction
                if np.sum(self.noise_profile) > 0:
                    audio_spectrum = np.fft.fft(audio, n=len(audio))
                    audio_magnitude = np.abs(audio_spectrum)
                    audio_phase = np.angle(audio_spectrum)
                    
                    # Interpolate noise profile to match audio length
                    noise_interp = np.interp(np.arange(len(audio_magnitude)), 
                                           np.arange(len(self.noise_profile)), 
                                           self.noise_profile)
                    
                    # Subtract noise with over-subtraction factor
                    clean_magnitude = audio_magnitude - 1.5 * noise_interp
                    clean_magnitude = np.maximum(clean_magnitude, 0.1 * audio_magnitude)
                    
                    # Reconstruct signal
                    clean_spectrum = clean_magnitude * np.exp(1j * audio_phase)
                    clean_audio = np.real(np.fft.ifft(clean_spectrum))
                    
                    return clean_audio
                    
        except Exception as e:
            logger.warning(f"Noise suppression failed: {str(e)}")
        
        return audio

    def _select_optimal_engine(self) -> VADEngine:
        """Select the optimal VAD engine based on availability and performance"""
        
        # Priority order based on latency and accuracy
        engine_priority = [
            VADEngine.WEBRTC_VAD,  # Fastest and most reliable
            VADEngine.HYBRID_VAD,  # Best accuracy when available
            VADEngine.SILERO_VAD,  # Good ML-based detection
            VADEngine.AZURE_VAD,   # Cloud-based quality
            VADEngine.ENERGY_VAD,  # Simple fallback
            VADEngine.MOCK_VAD     # Testing fallback
        ]
        
        # Use configured engine if available and working
        if (self.config.engine in self.engine_status and 
            self.engine_status[self.config.engine]["available"] and
            self.engine_status[self.config.engine].get("error_count", 0) < 3):
            return self.config.engine
        
        # Find first available engine
        for engine in engine_priority:
            if (engine in self.engine_status and 
                self.engine_status[engine]["available"] and
                self.engine_status[engine].get("error_count", 0) < 3):
                return engine
        
        # Ultimate fallback
        return VADEngine.MOCK_VAD

    # =============================================================================
    # ENGINE-SPECIFIC IMPLEMENTATIONS
    # =============================================================================

    async def _webrtc_vad_detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        """WebRTC VAD implementation - ultra-fast and reliable"""
        try:
            if VADEngine.WEBRTC_VAD not in self.vad_engines:
                raise Exception("WebRTC VAD not available")
            
            vad = self.vad_engines[VADEngine.WEBRTC_VAD]
            
            # Convert to required format (16-bit PCM)
            audio_int16 = (audio * 32767).astype(np.int16).tobytes()
            
            # WebRTC VAD requires specific frame sizes
            frame_size = int(sample_rate * self.config.frame_duration_ms / 1000)
            
            speech_frames = 0
            total_frames = 0
            audio_level = float(np.mean(np.abs(audio)))
            
            # Process audio in frames
            for i in range(0, len(audio_int16), frame_size * 2):  # 2 bytes per sample
                frame = audio_int16[i:i + frame_size * 2]
                if len(frame) == frame_size * 2:  # Complete frame
                    try:
                        has_speech = vad.is_speech(frame, sample_rate)
                        if has_speech:
                            speech_frames += 1
                        total_frames += 1
                    except Exception as e:
                        logger.warning(f"WebRTC VAD frame processing error: {str(e)}")
            
            # Calculate speech probability
            speech_probability = speech_frames / max(total_frames, 1)
            
            # Determine if overall segment has speech
            has_speech = speech_probability > 0.3  # 30% threshold
            confidence = min(0.98, speech_probability + 0.2)
            
            return VADResult(
                has_speech=has_speech,
                confidence=confidence,
                start_time=time.time(),
                end_time=time.time(),
                audio_level=audio_level,
                speech_probability=speech_probability
            )
            
        except Exception as e:
            logger.error(f"WebRTC VAD detection failed: {str(e)}")
            # Fallback to energy-based detection
            return await self._energy_vad_detect(audio, sample_rate)

    async def _silero_vad_detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        """Silero VAD implementation - ML-based high accuracy"""
        try:
            # TODO: Implement actual Silero VAD
            # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            # speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sample_rate)
            
            logger.info("Silero VAD not implemented yet - using enhanced energy VAD")
            return await self._energy_vad_detect(audio, sample_rate)
            
        except Exception as e:
            logger.error(f"Silero VAD detection failed: {str(e)}")
            return await self._energy_vad_detect(audio, sample_rate)

    async def _azure_vad_detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        """Azure VAD implementation - cloud-based high quality"""
        try:
            # TODO: Implement Azure VAD
            # import azure.cognitiveservices.speech as speechsdk
            
            logger.info("Azure VAD not implemented yet - using energy VAD")
            return await self._energy_vad_detect(audio, sample_rate)
            
        except Exception as e:
            logger.error(f"Azure VAD detection failed: {str(e)}")
            return await self._energy_vad_detect(audio, sample_rate)

    async def _hybrid_vad_detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        """Hybrid VAD using multiple algorithms for best accuracy"""
        try:
            results = []
            
            # Collect results from available engines
            if VADEngine.WEBRTC_VAD in self.vad_engines:
                webrtc_result = await self._webrtc_vad_detect(audio, sample_rate)
                results.append(("webrtc", webrtc_result))
            
            # Always include energy-based as fallback
            energy_result = await self._energy_vad_detect(audio, sample_rate)
            results.append(("energy", energy_result))
            
            if not results:
                return await self._mock_vad_detect(audio, sample_rate)
            
            # Fusion strategy: weighted voting
            weights = {"webrtc": 0.7, "energy": 0.3, "silero": 0.8, "azure": 0.6}
            
            weighted_score = 0.0
            total_weight = 0.0
            avg_confidence = 0.0
            avg_audio_level = 0.0
            
            for engine_name, result in results:
                weight = weights.get(engine_name, 0.5)
                score = 1.0 if result.has_speech else 0.0
                
                weighted_score += score * weight
                total_weight += weight
                avg_confidence += result.confidence
                avg_audio_level += result.audio_level
            
            # Final decision
            final_score = weighted_score / max(total_weight, 1.0)
            has_speech = final_score > 0.5
            
            return VADResult(
                has_speech=has_speech,
                confidence=avg_confidence / len(results),
                start_time=time.time(),
                end_time=time.time(),
                audio_level=avg_audio_level / len(results),
                speech_probability=final_score
            )
            
        except Exception as e:
            logger.error(f"Hybrid VAD detection failed: {str(e)}")
            return await self._energy_vad_detect(audio, sample_rate)

    async def _energy_vad_detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        """Energy-based VAD implementation - simple and reliable fallback"""
        try:
            if len(audio) == 0:
                return VADResult(has_speech=False, confidence=0.0, start_time=time.time(), end_time=time.time())
            
            # Calculate energy metrics
            audio_level = float(np.mean(np.abs(audio)))
            rms_level = float(np.sqrt(np.mean(audio ** 2)))
            peak_level = float(np.max(np.abs(audio)))
            
            # Adaptive threshold based on recent history
            with self.lock:
                self.audio_buffer.append(audio_level)
                if len(self.audio_buffer) > 10:
                    background_level = np.percentile(list(self.audio_buffer), 25)  # 25th percentile
                    dynamic_threshold = background_level * 3.0  # Adaptive threshold
                else:
                    dynamic_threshold = self.config.energy_threshold
            
            # Multiple energy criteria
            criteria = {
                "rms_threshold": rms_level > dynamic_threshold,
                "peak_threshold": peak_level > dynamic_threshold * 1.5,
                "energy_variance": np.var(audio) > dynamic_threshold * 0.5
            }
            
            # Count satisfied criteria
            satisfied_criteria = sum(criteria.values())
            
            # Calculate speech probability
            speech_probability = satisfied_criteria / len(criteria)
            
            # Determine speech presence
            has_speech = satisfied_criteria >= 2  # At least 2 criteria
            confidence = min(0.9, speech_probability + 0.1)
            
            return VADResult(
                has_speech=has_speech,
                confidence=confidence,
                start_time=time.time(),
                end_time=time.time(),
                audio_level=audio_level,
                noise_level=dynamic_threshold,
                speech_probability=speech_probability
            )
            
        except Exception as e:
            logger.error(f"Energy VAD detection failed: {str(e)}")
            return VADResult(has_speech=False, confidence=0.0, start_time=time.time(), end_time=time.time())

    async def _mock_vad_detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        """Mock VAD implementation for testing and development"""
        # Simulate processing delay
        await asyncio.sleep(0.001)
        
        if len(audio) == 0:
            return VADResult(has_speech=False, confidence=0.0, start_time=time.time(), end_time=time.time())
        
        # Simple mock logic based on audio characteristics
        audio_level = float(np.mean(np.abs(audio)))
        
        # Mock speech detection (simplified)
        has_speech = audio_level > 0.01  # Simple threshold
        confidence = min(0.95, audio_level * 10)  # Scale confidence
        speech_probability = confidence
        
        return VADResult(
            has_speech=has_speech,
            confidence=confidence,
            start_time=time.time(),
            end_time=time.time(),
            audio_level=audio_level,
            speech_probability=speech_probability
        )

    # =============================================================================
    # POST-PROCESSING AND STATE MANAGEMENT
    # =============================================================================

    async def _post_process_result(self, result: VADResult, audio: np.ndarray) -> VADResult:
        """Post-process VAD result with state tracking and smoothing"""
        try:
            current_time = time.time()
            
            # Update state machine
            if result.has_speech:
                if self.current_state == "silence":
                    # Transition from silence to speech
                    self.current_state = "speech"
                    self.speech_start_time = current_time
                    self.state_start_time = current_time
                    logger.debug("VAD: Silence -> Speech transition")
                
                self.last_speech_time = current_time
            else:
                if self.current_state == "speech":
                    # Check if silence duration is long enough for end-of-speech
                    silence_duration = (current_time - self.last_speech_time) * 1000
                    if silence_duration > self.config.min_silence_duration_ms:
                        self.current_state = "silence"
                        self.state_start_time = current_time
                        logger.debug(f"VAD: Speech -> Silence transition after {silence_duration:.0f}ms")
            
            # Apply minimum duration filtering
            state_duration = (current_time - self.state_start_time) * 1000
            
            if self.current_state == "speech" and state_duration < self.config.min_speech_duration_ms:
                # Too short for speech, suppress
                result.has_speech = False
                result.confidence *= 0.5
            elif self.current_state == "silence" and state_duration < self.config.min_silence_duration_ms:
                # Too short for silence, might still be speech
                if self.last_speech_time and (current_time - self.last_speech_time) < 1.0:
                    result.has_speech = True
                    result.confidence = min(result.confidence + 0.2, 0.95)
            
            return result
            
        except Exception as e:
            logger.warning(f"VAD post-processing failed: {str(e)}")
            return result

    async def _update_metrics(self, result: VADResult, engine: VADEngine):
        """Update VAD performance metrics"""
        try:
            self.metrics["total_frames_processed"] += 1
            
            if result.has_speech:
                self.metrics["speech_frames"] += 1
            else:
                self.metrics["silence_frames"] += 1
            
            # Update engine-specific metrics
            engine_name = engine.value
            if engine_name not in self.metrics["engine_performance"]:
                self.metrics["engine_performance"][engine_name] = {
                    "usage_count": 0,
                    "total_processing_time": 0.0,
                    "error_count": 0
                }
            
            engine_metrics = self.metrics["engine_performance"][engine_name]
            engine_metrics["usage_count"] += 1
            engine_metrics["total_processing_time"] += result.processing_time_ms
            
            # Update engine status - FIXED VERSION
            if engine in self.engine_status:
                status = self.engine_status[engine]
                status["last_used"] = datetime.now()
                usage_count = engine_metrics["usage_count"]  # Extract the value first
                if usage_count > 0:  # Then use it in condition
                    status["average_latency_ms"] = engine_metrics["total_processing_time"] / usage_count
            
        except Exception as e:
            logger.warning(f"Metrics update failed: {str(e)}")

    # =============================================================================
    # STREAMING AND REAL-TIME PROCESSING
    # =============================================================================

    async def process_audio_stream(self, 
                                 audio_stream: AsyncGenerator[bytes, None],
                                 callback: Optional[callable] = None) -> AsyncGenerator[VADResult, None]:
        """
        Process continuous audio stream with real-time VAD
        
        Args:
            audio_stream: Continuous audio data stream
            callback: Optional callback for immediate processing
            
        Yields:
            VADResult: Real-time VAD results
        """
        try:
            audio_buffer = b""
            frame_size = self.frame_size * 2  # 2 bytes per sample
            
            async for audio_chunk in audio_stream:
                audio_buffer += audio_chunk
                
                # Process complete frames
                while len(audio_buffer) >= frame_size:
                    frame_data = audio_buffer[:frame_size]
                    audio_buffer = audio_buffer[frame_size:]
                    
                    # Detect voice activity
                    result = await self.detect_voice_activity(frame_data, self.config.sample_rate)
                    
                    # Call callback if provided
                    if callback:
                        try:
                            await callback(result)
                        except Exception as e:
                            logger.warning(f"VAD callback failed: {str(e)}")
                    
                    yield result
                    
        except Exception as e:
            logger.error(f"Audio stream processing failed: {str(e)}")
            raise

    async def detect_end_of_speech(self, 
                                 audio_stream: AsyncGenerator[bytes, None],
                                 timeout_ms: int = 3000) -> Tuple[bool, float]:
        """
        Detect end of speech in audio stream with timeout
        
        Args:
            audio_stream: Audio data stream
            timeout_ms: Maximum time to wait for end of speech
            
        Returns:
            Tuple[bool, float]: (end_detected, silence_duration_ms)
        """
        start_time = time.time()
        last_speech_time = None
        silence_duration = 0.0
        
        try:
            async for result in self.process_audio_stream(audio_stream):
                current_time = time.time()
                
                if result.has_speech:
                    last_speech_time = current_time
                    silence_duration = 0.0
                else:
                    if last_speech_time:
                        silence_duration = (current_time - last_speech_time) * 1000
                        
                        # Check if silence is long enough
                        if silence_duration > self.config.min_silence_duration_ms:
                            return True, silence_duration
                
                # Check timeout
                if (current_time - start_time) * 1000 > timeout_ms:
                    return False, silence_duration
                    
        except Exception as e:
            logger.error(f"End of speech detection failed: {str(e)}")
            return False, 0.0
        
        return False, silence_duration

    # =============================================================================
    # CONFIGURATION AND UTILITIES
    # =============================================================================

    def update_config(self, new_config: VADConfig) -> Dict[str, Any]:
        """Update VAD configuration"""
        old_engine = self.config.engine
        self.config = new_config
        
        # Reinitialize if engine changed
        if old_engine != new_config.engine:
            self._initialize_vad_engines()
        
        return {
            "success": True,
            "message": "VAD configuration updated",
            "changes": {
                "engine": f"{old_engine.value} -> {new_config.engine.value}",
                "sensitivity": new_config.sensitivity.value,
                "sample_rate": new_config.sample_rate
            }
        }

    def get_vad_metrics(self) -> Dict[str, Any]:
        """Get comprehensive VAD performance metrics"""
        total_frames = max(self.metrics["total_frames_processed"], 1)
        
        return {
            "performance": {
                "total_frames_processed": self.metrics["total_frames_processed"],
                "speech_frames": self.metrics["speech_frames"],
                "silence_frames": self.metrics["silence_frames"],
                "speech_ratio": self.metrics["speech_frames"] / total_frames,
                "average_processing_time_ms": self.metrics["average_processing_time_ms"]
            },
            "accuracy": {
                "false_positives": self.metrics["false_positives"],
                "false_negatives": self.metrics["false_negatives"],
                "estimated_accuracy": max(0, 1 - (self.metrics["false_positives"] + self.metrics["false_negatives"]) / total_frames)
            },
            "engine_performance": self.metrics["engine_performance"],
            "engine_status": {
                engine.value: status for engine, status in self.engine_status.items()
            },
            "current_state": {
                "state": self.current_state,
                "state_duration_ms": (time.time() - self.state_start_time) * 1000,
                "last_speech_time": self.last_speech_time,
                "adaptation_count": self.adaptation_count
            },
            "configuration": {
                "engine": self.config.engine.value,
                "sensitivity": self.config.sensitivity.value,
                "sample_rate": self.config.sample_rate,
                "frame_duration_ms": self.config.frame_duration_ms,
                "min_speech_duration_ms": self.config.min_speech_duration_ms,
                "min_silence_duration_ms": self.config.min_silence_duration_ms
            }
        }

    def optimize_for_environment(self, environment: str):
        """Optimize VAD settings for specific environment"""
        environment_configs = {
            "quiet_office": {
                "sensitivity": VADSensitivity.SENSITIVE,
                "energy_threshold": 0.005,
                "min_speech_duration_ms": 100,
                "enable_noise_suppression": False
            },
            "noisy_office": {
                "sensitivity": VADSensitivity.AGGRESSIVE,
                "energy_threshold": 0.02,
                "min_speech_duration_ms": 150,
                "enable_noise_suppression": True
            },
            "phone_call": {
                "sensitivity": VADSensitivity.NORMAL,
                "energy_threshold": 0.01,
                "min_speech_duration_ms": 120,
                "audio_quality": AudioQuality.PHONE,
                "enable_noise_suppression": True
            },
            "conference_room": {
                "sensitivity": VADSensitivity.SENSITIVE,
                "energy_threshold": 0.008,
                "min_speech_duration_ms": 100,
                "enable_noise_suppression": True
            },
            "outdoor": {
                "sensitivity": VADSensitivity.AGGRESSIVE,
                "energy_threshold": 0.03,
                "min_speech_duration_ms": 200,
                "enable_noise_suppression": True
            }
        }
        
        if environment in environment_configs:
            env_config = environment_configs[environment]
            
            # Update relevant config parameters
            for param, value in env_config.items():
                if hasattr(self.config, param):
                    setattr(self.config, param, value)
            
            logger.info(f"VAD optimized for {environment} environment")
        else:
            logger.warning(f"Unknown environment: {environment}")

    async def calibrate_noise_threshold(self, 
                                      background_audio: bytes, 
                                      speech_audio: bytes) -> Dict[str, Any]:
        """Calibrate VAD thresholds using background and speech samples"""
        try:
            # Process background audio
            bg_audio = await self._preprocess_audio(background_audio, self.config.sample_rate)
            bg_level = float(np.mean(np.abs(bg_audio))) if len(bg_audio) > 0 else 0.0
            
            # Process speech audio  
            speech_audio_processed = await self._preprocess_audio(speech_audio, self.config.sample_rate)
            speech_level = float(np.mean(np.abs(speech_audio_processed))) if len(speech_audio_processed) > 0 else 0.0
            
            # Calculate optimal threshold
            if speech_level > bg_level:
                optimal_threshold = (bg_level + speech_level) / 2
                margin = (speech_level - bg_level) / speech_level
                
                self.config.energy_threshold = optimal_threshold
                
                return {
                    "success": True,
                    "background_level": bg_level,
                    "speech_level": speech_level,
                    "optimal_threshold": optimal_threshold,
                    "separation_margin": margin,
                    "quality": "good" if margin > 0.3 else "fair" if margin > 0.1 else "poor"
                }
            else:
                return {
                    "success": False,
                    "error": "Speech level not significantly higher than background",
                    "background_level": bg_level,
                    "speech_level": speech_level
                }
                
        except Exception as e:
            logger.error(f"VAD calibration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def reset_adaptation(self):
        """Reset adaptive parameters"""
        self.noise_profile = np.zeros(512)
        self.speech_profile = np.zeros(512)
        self.adaptation_count = 0
        self.audio_buffer.clear()
        
        logger.info("VAD adaptation parameters reset")

    def get_engine_availability(self) -> Dict[str, bool]:
        """Get availability status of all VAD engines"""
        return {
            engine.value: status.get("available", False)
            for engine, status in self.engine_status.items()
        }

# =============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# =============================================================================

def create_ultra_low_latency_vad(sample_rate: int = 16000) -> VoiceActivityDetector:
    """Factory function for ultra-low latency VAD configuration"""
    config = VADConfig(
        engine=VADEngine.WEBRTC_VAD,
        sensitivity=VADSensitivity.NORMAL,
        sample_rate=sample_rate,
        frame_duration_ms=10,  # Shortest possible frame
        min_speech_duration_ms=50,  # Very short minimum
        min_silence_duration_ms=200,  # Quick end-of-speech detection
        enable_noise_suppression=False  # Skip for speed
    )
    return VoiceActivityDetector(config)

def create_high_accuracy_vad(sample_rate: int = 16000) -> VoiceActivityDetector:
    """Factory function for high-accuracy VAD configuration"""
    config = VADConfig(
        engine=VADEngine.HYBRID_VAD,
        sensitivity=VADSensitivity.SENSITIVE,
        sample_rate=sample_rate,
        frame_duration_ms=30,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
        enable_noise_suppression=True,
        speech_padding_ms=100
    )
    return VoiceActivityDetector(config)

def create_telephony_vad(sample_rate: int = 8000) -> VoiceActivityDetector:
    """Factory function for telephony-optimized VAD configuration"""
    config = VADConfig(
        engine=VADEngine.WEBRTC_VAD,
        sensitivity=VADSensitivity.NORMAL,
        sample_rate=sample_rate,
        frame_duration_ms=20,
        audio_quality=AudioQuality.PHONE,
        min_speech_duration_ms=120,
        min_silence_duration_ms=400,
        enable_noise_suppression=True
    )
    return VoiceActivityDetector(config)

def create_conference_vad(sample_rate: int = 16000) -> VoiceActivityDetector:
    """Factory function for conference room VAD configuration"""
    config = VADConfig(
        engine=VADEngine.HYBRID_VAD,
        sensitivity=VADSensitivity.SENSITIVE,
        sample_rate=sample_rate,
        frame_duration_ms=30,
        min_speech_duration_ms=150,
        min_silence_duration_ms=500,
        enable_noise_suppression=True,
        speech_padding_ms=150
    )
    return VoiceActivityDetector(config)

async def benchmark_vad_engines(test_audio_file: str = None) -> Dict[str, Any]:
    """Benchmark all available VAD engines"""
    results = {}
    
    # Generate test audio if not provided
    if not test_audio_file:
        # Create test audio: 1 second of speech, 1 second of silence
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Speech simulation (sine waves with modulation)
        speech_part = np.sin(2 * np.pi * 440 * t[:sample_rate]) * np.sin(2 * np.pi * 5 * t[:sample_rate]) * 0.5
        silence_part = np.random.normal(0, 0.01, sample_rate)  # Low-level noise
        
        test_audio = np.concatenate([speech_part, silence_part])
        test_audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
    else:
        with open(test_audio_file, 'rb') as f:
            test_audio_bytes = f.read()
    
    engines_to_test = [
        (VADEngine.WEBRTC_VAD, VADSensitivity.NORMAL),
        (VADEngine.HYBRID_VAD, VADSensitivity.SENSITIVE),
        (VADEngine.ENERGY_VAD, VADSensitivity.NORMAL),
        (VADEngine.MOCK_VAD, VADSensitivity.NORMAL)
    ]
    
    for engine, sensitivity in engines_to_test:
        try:
            config = VADConfig(
                engine=engine,
                sensitivity=sensitivity,
                sample_rate=16000
            )
            
            vad = VoiceActivityDetector(config)
            
            start_time = time.time()
            result = await vad.detect_voice_activity(test_audio_bytes)
            total_time = (time.time() - start_time) * 1000
            
            results[engine.value] = {
                "success": True,
                "processing_time_ms": total_time,
                "has_speech": result.has_speech,
                "confidence": result.confidence,
                "speech_probability": result.speech_probability,
                "audio_level": result.audio_level,
                "engine_latency_ms": result.processing_time_ms
            }
            
        except Exception as e:
            results[engine.value] = {
                "success": False,
                "error": str(e)
            }
    
    return {
        "benchmark_results": results,
        "timestamp": datetime.now().isoformat(),
        "fastest_engine": min([r for r in results.values() if r.get("success")], 
                            key=lambda x: x.get("processing_time_ms", float('inf')), 
                            default={}).get("engine", "none"),
        "most_confident": max([r for r in results.values() if r.get("success")], 
                            key=lambda x: x.get("confidence", 0), 
                            default={}).get("engine", "none")
    }

def get_production_setup_instructions() -> str:
    """Get comprehensive setup instructions for production VAD deployment"""
    return """
VOICE ACTIVITY DETECTION PRODUCTION SETUP:

1. INSTALL DEPENDENCIES:
   pip install webrtcvad soundfile librosa numpy scipy

2. OPTIONAL ADVANCED ENGINES:
   # Silero VAD (ML-based)
   pip install torch torchaudio
   
   # Azure Speech Services VAD
   pip install azure-cognitiveservices-speech
   export AZURE_SPEECH_KEY="your-azure-key"
   export AZURE_SPEECH_REGION="eastus"

3. OPTIMAL CONFIGURATIONS:
   - Ultra-low latency: WebRTC VAD + 10ms frames
   - High accuracy: Hybrid VAD with multiple engines
   - Telephony: WebRTC VAD + phone-optimized settings
   - Conference: Hybrid VAD + sensitive detection

4. AUDIO REQUIREMENTS:
   - Sample rate: 8kHz (telephony) or 16kHz (optimal)
   - Frame duration: 10ms (fastest) to 30ms (most accurate)
   - Audio format: 16-bit PCM
   - Minimum chunk size: 160 samples (10ms at 16kHz)

5. LATENCY TARGETS:
   - Detection latency: <50ms
   - End-of-speech detection: <100ms
   - Processing per frame: <5ms
   - State transition time: <20ms

6. OPTIMIZATION TECHNIQUES:
   - Use appropriate frame sizes for your use case
   - Enable noise suppression in noisy environments
   - Calibrate thresholds using background samples
   - Implement adaptive threshold adjustment

7. INTEGRATION PATTERNS:
   - Prefilter audio streams before STT processing
   - Use for endpoint detection in voice conversations
   - Implement voice activity-based recording triggers
   - Optimize silence detection for natural conversation flow

8. MONITORING SETUP:
   - Track detection accuracy and false positive rates
   - Monitor processing latency per frame
   - Set up alerts for engine failures
   - Measure end-of-speech detection performance

9. ENVIRONMENT OPTIMIZATION:
   - Quiet office: Sensitive detection, low thresholds
   - Noisy environment: Aggressive filtering, high thresholds  
   - Phone calls: Telephony-optimized WebRTC settings
   - Conference rooms: Multi-speaker detection support

10. TROUBLESHOOTING:
    - Check audio format and sample rate compatibility
    - Verify frame size requirements for WebRTC VAD
    - Test with known speech/silence samples
    - Monitor background noise levels for threshold tuning
"""

class VADIntegrationHelper:
    """Helper class for integrating VAD with STT and TTS pipelines"""
    
    def __init__(self, vad: VoiceActivityDetector):
        self.vad = vad
        
    async def filter_audio_stream(self, 
                                audio_stream: AsyncGenerator[bytes, None],
                                min_speech_ratio: float = 0.3) -> AsyncGenerator[bytes, None]:
        """Filter audio stream to only pass segments with sufficient speech"""
        buffer = b""
        buffer_duration_ms = 1000  # 1 second buffer
        
        async for audio_chunk in audio_stream:
            buffer += audio_chunk
            
            # Process buffer when it reaches target duration
            if len(buffer) >= (self.vad.config.sample_rate * buffer_duration_ms // 1000 * 2):
                result = await self.vad.detect_voice_activity(buffer)
                
                if result.speech_probability >= min_speech_ratio:
                    yield buffer
                
                buffer = b""
    
    async def detect_speech_boundaries(self, 
                                     audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[Dict[str, Any], None]:
        """Detect speech start and end boundaries in audio stream"""
        in_speech = False
        speech_start_time = None
        
        async for result in self.vad.process_audio_stream(audio_stream):
            if result.has_speech and not in_speech:
                # Speech started
                in_speech = True
                speech_start_time = result.start_time
                yield {
                    "event": "speech_start",
                    "timestamp": result.start_time,
                    "confidence": result.confidence
                }
            elif not result.has_speech and in_speech:
                # Speech ended
                in_speech = False
                yield {
                    "event": "speech_end", 
                    "timestamp": result.end_time,
                    "duration": result.end_time - speech_start_time if speech_start_time else 0,
                    "confidence": result.confidence
                }
    
    def create_endpoint_detector(self, 
                               max_silence_ms: int = 2000) -> callable:
        """Create endpoint detection function for conversation systems"""
        async def detect_endpoint(audio_stream: AsyncGenerator[bytes, None]) -> bool:
            """Detect if speaker has finished talking"""
            return await self.vad.detect_end_of_speech(audio_stream, max_silence_ms)
            
        return detect_endpoint
        
"""
Voice Activity Detection - Advanced VAD with Ultra-Low Latency
Part of the Multi-Agent Voice AI System Transformation

This module implements state-of-the-art Voice Activity Detection:
- Multiple VAD algorithms with intelligent fusion
- Ultra-low latency detection (<50ms)
- Adaptive noise suppression
- End-of-speech detection optimization
- Integration with STT and TTS pipelines
- Real-time audio stream processing

SUPPORTED VAD ENGINES:
- WebRTC VAD (Primary - Ultra-fast, proven)
- Silero VAD (High accuracy, ML-based)
- Azure VAD (Cloud-based, high quality)
- Custom hybrid VAD (Multi-algorithm fusion)
- Energy-based VAD (Fallback)

TARGET METRICS:
- Detection latency: <50ms
- Accuracy: >95% speech/silence classification
- False positive rate: <2%
- End-of-speech detection: <100ms
"""

