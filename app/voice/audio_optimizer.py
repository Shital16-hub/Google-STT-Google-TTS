"""
Audio Optimizer - Advanced Audio Quality Enhancement System
==========================================================

Comprehensive audio processing system for voice AI applications with real-time
quality enhancement, adaptive filtering, and telephony optimization.
Designed for ultra-low latency processing with professional-grade audio quality.

Features:
- Real-time audio quality enhancement and restoration
- Adaptive noise reduction with spectral subtraction
- Automatic gain control and dynamic range compression
- Audio format optimization for telephony (Twilio/MULAW)
- Quality metrics and monitoring with adaptive processing
- Multi-stage filtering pipeline with bypass options
- Memory-efficient streaming audio processing
- Bandwidth optimization for network transmission
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import collections
import threading
import queue
import struct
import math

# Audio processing libraries
import scipy.signal
import scipy.fft
from scipy.ndimage import gaussian_filter1d

# For audio format conversion
import audioop
import wave
import io

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Audio processing modes for different quality levels"""
    MINIMAL = "minimal"        # Minimal processing for excellent audio
    STANDARD = "standard"      # Standard processing for good audio
    AGGRESSIVE = "aggressive"  # Aggressive processing for poor audio
    ADAPTIVE = "adaptive"      # Adaptive processing based on audio quality


class AudioFormat(Enum):
    """Supported audio formats"""
    PCM_16 = "pcm_16"         # 16-bit PCM
    PCM_8 = "pcm_8"           # 8-bit PCM
    MULAW = "mulaw"           # μ-law encoding (telephony)
    ALAW = "alaw"             # A-law encoding (telephony)
    FLOAT32 = "float32"       # 32-bit float


class QualityMetric(Enum):
    """Audio quality metrics"""
    SNR = "snr"                    # Signal-to-Noise Ratio
    THD = "thd"                    # Total Harmonic Distortion
    SPECTRAL_CENTROID = "spectral_centroid"
    SPECTRAL_ROLLOFF = "spectral_rolloff"
    RMS_ENERGY = "rms_energy"
    ZERO_CROSSING_RATE = "zcr"
    SPECTRAL_FLATNESS = "spectral_flatness"


@dataclass
class AudioQualityMetrics:
    """Comprehensive audio quality metrics"""
    snr_db: float = 0.0
    thd_percent: float = 0.0
    rms_energy: float = 0.0
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_flatness: float = 0.0
    dynamic_range_db: float = 0.0
    noise_floor_db: float = 0.0
    clipping_detected: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingParams:
    """Audio processing parameters"""
    mode: ProcessingMode = ProcessingMode.STANDARD
    noise_reduction_strength: float = 0.5
    gain_control_enabled: bool = True
    dynamic_compression_enabled: bool = True
    equalization_enabled: bool = False
    bandwidth_optimization: bool = True
    target_sample_rate: int = 8000
    target_format: AudioFormat = AudioFormat.MULAW
    quality_threshold: float = 0.7
    adaptive_processing: bool = True


@dataclass
class ProcessingResult:
    """Audio processing result with metrics"""
    processed_audio: bytes
    original_format: AudioFormat
    processed_format: AudioFormat
    quality_metrics: AudioQualityMetrics
    processing_time_ms: float
    enhancement_applied: Dict[str, bool]
    quality_improvement: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpectralProcessor:
    """
    Advanced spectral processing for noise reduction and enhancement
    """
    
    def __init__(self, sample_rate: int = 8000, frame_size: int = 512):
        """Initialize Spectral Processor"""
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = frame_size // 4
        
        # Spectral analysis parameters
        self.window = scipy.signal.windows.hann(frame_size)
        self.noise_profile = None
        self.noise_update_rate = 0.1
        
        # Smoothing parameters
        self.spectral_smoothing = 0.8
        self.temporal_smoothing = 0.9
        self.previous_magnitude = None
        self.previous_phase = None
        
        # Enhancement parameters
        self.spectral_floor = 0.1
        self.over_subtraction_factor = 2.0
        self.spectral_gain_limits = (0.1, 3.0)
        
        logger.info(f"Spectral Processor initialized: {sample_rate}Hz, {frame_size} samples")
    
    def enhance_spectrum(self, audio_data: np.ndarray, 
                        noise_reduction_strength: float = 0.5) -> np.ndarray:
        """
        Enhance audio using spectral processing
        
        Args:
            audio_data: Input audio data
            noise_reduction_strength: Strength of noise reduction (0.0-1.0)
            
        Returns:
            Enhanced audio data
        """
        if len(audio_data) < self.frame_size:
            return audio_data
        
        # Pad audio to frame size
        padded_length = ((len(audio_data) - 1) // self.hop_size + 1) * self.hop_size + self.frame_size
        padded_audio = np.pad(audio_data, (0, padded_length - len(audio_data)), mode='constant')
        
        # Process in overlapping frames
        enhanced_frames = []
        
        for i in range(0, len(padded_audio) - self.frame_size + 1, self.hop_size):
            frame = padded_audio[i:i + self.frame_size]
            enhanced_frame = self._process_frame(frame, noise_reduction_strength)
            enhanced_frames.append(enhanced_frame)
        
        # Overlap-add reconstruction
        enhanced_audio = self._overlap_add_reconstruction(enhanced_frames)
        
        # Trim to original length
        return enhanced_audio[:len(audio_data)]
    
    def _process_frame(self, frame: np.ndarray, noise_strength: float) -> np.ndarray:
        """Process a single audio frame"""
        # Apply window
        windowed_frame = frame * self.window
        
        # FFT
        fft_data = scipy.fft.fft(windowed_frame)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # Update noise profile
        self._update_noise_profile(magnitude)
        
        # Apply spectral enhancement
        enhanced_magnitude = self._spectral_enhancement(magnitude, noise_strength)
        
        # Apply smoothing
        if self.previous_magnitude is not None:
            enhanced_magnitude = (self.spectral_smoothing * self.previous_magnitude + 
                                (1 - self.spectral_smoothing) * enhanced_magnitude)
        
        self.previous_magnitude = enhanced_magnitude.copy()
        
        # Reconstruct signal
        enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_frame = np.real(scipy.fft.ifft(enhanced_fft))
        
        return enhanced_frame
    
    def _update_noise_profile(self, magnitude: np.ndarray):
        """Update noise profile for spectral subtraction"""
        # Simple voice activity detection
        energy = np.sum(magnitude ** 2)
        
        # Update noise profile during low-energy periods
        if energy < np.percentile(magnitude ** 2, 25):
            if self.noise_profile is None:
                self.noise_profile = magnitude.copy()
            else:
                self.noise_profile = ((1 - self.noise_update_rate) * self.noise_profile + 
                                    self.noise_update_rate * magnitude)
    
    def _spectral_enhancement(self, magnitude: np.ndarray, noise_strength: float) -> np.ndarray:
        """Apply spectral enhancement (noise reduction and enhancement)"""
        if self.noise_profile is None:
            return magnitude
        
        # Spectral subtraction
        noise_estimate = self.noise_profile * self.over_subtraction_factor * noise_strength
        enhanced_magnitude = magnitude - noise_estimate
        
        # Apply spectral floor
        enhanced_magnitude = np.maximum(enhanced_magnitude, 
                                      self.spectral_floor * magnitude)
        
        # Spectral enhancement (boost important frequencies)
        frequency_bins = np.arange(len(magnitude))
        # Boost speech frequencies (300-3400 Hz for telephony)
        speech_boost = self._create_speech_boost_filter(frequency_bins)
        enhanced_magnitude *= speech_boost
        
        # Apply gain limits
        gain = enhanced_magnitude / (magnitude + 1e-10)
        gain = np.clip(gain, self.spectral_gain_limits[0], self.spectral_gain_limits[1])
        enhanced_magnitude = magnitude * gain
        
        return enhanced_magnitude
    
    def _create_speech_boost_filter(self, frequency_bins: np.ndarray) -> np.ndarray:
        """Create frequency boost filter for speech enhancement"""
        # Convert bins to frequencies
        freqs = frequency_bins * self.sample_rate / len(frequency_bins)
        
        # Create boost filter for speech frequencies
        boost = np.ones_like(freqs)
        
        # Boost 300-3400 Hz (telephony speech band)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        boost[speech_mask] *= 1.2
        
        # Gentle roll-off outside speech band
        low_freq_mask = freqs < 300
        high_freq_mask = freqs > 3400
        
        boost[low_freq_mask] *= 0.8
        boost[high_freq_mask] *= 0.7
        
        return boost
    
    def _overlap_add_reconstruction(self, frames: List[np.ndarray]) -> np.ndarray:
        """Reconstruct signal using overlap-add method"""
        if not frames:
            return np.array([])
        
        # Calculate output length
        output_length = (len(frames) - 1) * self.hop_size + self.frame_size
        output = np.zeros(output_length)
        
        # Overlap-add frames
        for i, frame in enumerate(frames):
            start = i * self.hop_size
            end = start + self.frame_size
            output[start:end] += frame
        
        return output


class DynamicRangeProcessor:
    """
    Dynamic range processing with compression and expansion
    """
    
    def __init__(self, sample_rate: int = 8000):
        """Initialize Dynamic Range Processor"""
        self.sample_rate = sample_rate
        
        # Compressor parameters
        self.threshold = 0.7        # Compression threshold
        self.ratio = 4.0           # Compression ratio
        self.attack_time = 0.003   # 3ms attack
        self.release_time = 0.1    # 100ms release
        
        # AGC parameters
        self.target_level = 0.5    # Target RMS level
        self.max_gain = 10.0       # Maximum AGC gain
        self.min_gain = 0.1        # Minimum AGC gain
        
        # State variables
        self.envelope = 0.0
        self.gain_reduction = 0.0
        self.agc_gain = 1.0
        self.rms_history = collections.deque(maxlen=int(sample_rate * 0.1))  # 100ms history
        
        # Calculate attack/release coefficients
        self.attack_coeff = np.exp(-1.0 / (sample_rate * self.attack_time))
        self.release_coeff = np.exp(-1.0 / (sample_rate * self.release_time))
        
        logger.info(f"Dynamic Range Processor initialized: {sample_rate}Hz")
    
    def process_dynamics(self, audio_data: np.ndarray, 
                        enable_compression: bool = True,
                        enable_agc: bool = True) -> np.ndarray:
        """
        Process audio dynamics with compression and AGC
        
        Args:
            audio_data: Input audio data
            enable_compression: Enable dynamic range compression
            enable_agc: Enable automatic gain control
            
        Returns:
            Processed audio data
        """
        processed_audio = audio_data.copy()
        
        # Apply compression
        if enable_compression:
            processed_audio = self._apply_compression(processed_audio)
        
        # Apply AGC
        if enable_agc:
            processed_audio = self._apply_agc(processed_audio)
        
        return processed_audio
    
    def _apply_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression"""
        compressed_audio = np.zeros_like(audio_data)
        
        for i, sample in enumerate(audio_data):
            # Calculate envelope
            sample_abs = abs(sample)
            
            if sample_abs > self.envelope:
                # Attack
                self.envelope = self.attack_coeff * self.envelope + (1 - self.attack_coeff) * sample_abs
            else:
                # Release
                self.envelope = self.release_coeff * self.envelope + (1 - self.release_coeff) * sample_abs
            
            # Calculate gain reduction
            if self.envelope > self.threshold:
                # Above threshold - apply compression
                overshoot = self.envelope - self.threshold
                gain_reduction = overshoot * (1 - 1/self.ratio)
                self.gain_reduction = min(self.gain_reduction + gain_reduction, 
                                        1 - 1/self.ratio)
            else:
                # Below threshold - no compression
                self.gain_reduction *= 0.999  # Gradual release
            
            # Apply gain reduction
            gain = 1.0 - self.gain_reduction
            compressed_audio[i] = sample * gain
        
        return compressed_audio
    
    def _apply_agc(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply Automatic Gain Control"""
        # Calculate RMS in chunks
        chunk_size = max(1, len(audio_data) // 10)
        processed_audio = np.zeros_like(audio_data)
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            
            # Calculate RMS
            rms = np.sqrt(np.mean(chunk ** 2))
            self.rms_history.append(rms)
            
            # Calculate average RMS
            avg_rms = np.mean(list(self.rms_history)) if self.rms_history else rms
            
            # Calculate required gain
            if avg_rms > 0:
                required_gain = self.target_level / avg_rms
                required_gain = np.clip(required_gain, self.min_gain, self.max_gain)
                
                # Smooth gain changes
                self.agc_gain = 0.95 * self.agc_gain + 0.05 * required_gain
            
            # Apply gain
            processed_audio[i:i + chunk_size] = chunk * self.agc_gain
        
        return processed_audio


class AudioFormatConverter:
    """
    Audio format conversion and optimization for telephony
    """
    
    def __init__(self):
        """Initialize Audio Format Converter"""
        self.supported_formats = {
            AudioFormat.PCM_16: {'bits': 16, 'encoding': 'linear'},
            AudioFormat.PCM_8: {'bits': 8, 'encoding': 'linear'},
            AudioFormat.MULAW: {'bits': 8, 'encoding': 'mulaw'},
            AudioFormat.ALAW: {'bits': 8, 'encoding': 'alaw'},
            AudioFormat.FLOAT32: {'bits': 32, 'encoding': 'float'}
        }
        
        logger.info("Audio Format Converter initialized")
    
    def convert_format(self, audio_data: Union[bytes, np.ndarray],
                      source_format: AudioFormat,
                      target_format: AudioFormat,
                      sample_rate: int = 8000) -> bytes:
        """
        Convert audio between different formats
        
        Args:
            audio_data: Input audio data
            source_format: Source audio format
            target_format: Target audio format
            sample_rate: Audio sample rate
            
        Returns:
            Converted audio data as bytes
        """
        try:
            # Convert input to numpy array
            if isinstance(audio_data, bytes):
                audio_array = self._bytes_to_array(audio_data, source_format)
            else:
                audio_array = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1] range
            audio_array = self._normalize_audio(audio_array, source_format)
            
            # Convert to target format
            converted_bytes = self._array_to_bytes(audio_array, target_format)
            
            return converted_bytes
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return audio_data if isinstance(audio_data, bytes) else b''
    
    def optimize_for_telephony(self, audio_data: bytes, 
                              source_format: AudioFormat,
                              target_sample_rate: int = 8000) -> bytes:
        """
        Optimize audio for telephony transmission
        
        Args:
            audio_data: Input audio data
            source_format: Source audio format
            target_sample_rate: Target sample rate for telephony
            
        Returns:
            Optimized audio data in MULAW format
        """
        # Convert to numpy array
        audio_array = self._bytes_to_array(audio_data, source_format)
        
        # Apply telephony-specific processing
        optimized_array = self._apply_telephony_optimization(audio_array, target_sample_rate)
        
        # Convert to MULAW for telephony
        optimized_bytes = self._array_to_bytes(optimized_array, AudioFormat.MULAW)
        
        return optimized_bytes
    
    def _bytes_to_array(self, audio_bytes: bytes, format: AudioFormat) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        if format == AudioFormat.PCM_16:
            return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        elif format == AudioFormat.PCM_8:
            return (np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        
        elif format == AudioFormat.MULAW:
            # Convert MULAW to linear PCM using audioop
            linear_bytes = audioop.ulaw2lin(audio_bytes, 2)
            return np.frombuffer(linear_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        elif format == AudioFormat.ALAW:
            # Convert ALAW to linear PCM using audioop
            linear_bytes = audioop.alaw2lin(audio_bytes, 2)
            return np.frombuffer(linear_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        elif format == AudioFormat.FLOAT32:
            return np.frombuffer(audio_bytes, dtype=np.float32)
        
        else:
            raise ValueError(f"Unsupported source format: {format}")
    
    def _array_to_bytes(self, audio_array: np.ndarray, format: AudioFormat) -> bytes:
        """Convert numpy array to audio bytes"""
        # Ensure audio is in [-1, 1] range
        audio_array = np.clip(audio_array, -1.0, 1.0)
        
        if format == AudioFormat.PCM_16:
            return (audio_array * 32767).astype(np.int16).tobytes()
        
        elif format == AudioFormat.PCM_8:
            return ((audio_array + 1.0) * 127.5).astype(np.uint8).tobytes()
        
        elif format == AudioFormat.MULAW:
            # Convert to 16-bit PCM first, then to MULAW
            pcm_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            return audioop.lin2ulaw(pcm_bytes, 2)
        
        elif format == AudioFormat.ALAW:
            # Convert to 16-bit PCM first, then to ALAW
            pcm_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            return audioop.lin2alaw(pcm_bytes, 2)
        
        elif format == AudioFormat.FLOAT32:
            return audio_array.astype(np.float32).tobytes()
        
        else:
            raise ValueError(f"Unsupported target format: {format}")
    
    def _normalize_audio(self, audio_array: np.ndarray, source_format: AudioFormat) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if source_format == AudioFormat.FLOAT32:
            # Already normalized for float32
            return np.clip(audio_array, -1.0, 1.0)
        else:
            # For other formats, assume already normalized by _bytes_to_array
            return audio_array
    
    def _apply_telephony_optimization(self, audio_array: np.ndarray, 
                                    target_sample_rate: int) -> np.ndarray:
        """Apply telephony-specific optimizations"""
        # Apply telephony frequency response (300-3400 Hz bandpass)
        if target_sample_rate > 0:
            nyquist = target_sample_rate / 2
            low_cutoff = 300.0 / nyquist
            high_cutoff = min(3400.0 / nyquist, 0.95)  # Ensure below Nyquist
            
            # Design bandpass filter
            if high_cutoff > low_cutoff:
                b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
                audio_array = scipy.signal.filtfilt(b, a, audio_array)
        
        # Apply pre-emphasis filter (boost high frequencies slightly)
        pre_emphasis = 0.97
        if len(audio_array) > 1:
            audio_array = np.append(audio_array[0], 
                                  audio_array[1:] - pre_emphasis * audio_array[:-1])
        
        return audio_array


class QualityAnalyzer:
    """
    Comprehensive audio quality analysis and metrics
    """
    
    def __init__(self, sample_rate: int = 8000):
        """Initialize Quality Analyzer"""
        self.sample_rate = sample_rate
        self.analysis_history = collections.deque(maxlen=100)
        
        logger.info(f"Quality Analyzer initialized: {sample_rate}Hz")
    
    def analyze_quality(self, audio_data: np.ndarray) -> AudioQualityMetrics:
        """
        Comprehensive audio quality analysis
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            AudioQualityMetrics with comprehensive analysis
        """
        metrics = AudioQualityMetrics()
        
        try:
            # Basic metrics
            metrics.rms_energy = self._calculate_rms_energy(audio_data)
            metrics.dynamic_range_db = self._calculate_dynamic_range(audio_data)
            metrics.zero_crossing_rate = self._calculate_zcr(audio_data)
            metrics.clipping_detected = self._detect_clipping(audio_data)
            
            # Spectral metrics
            if len(audio_data) > 256:  # Minimum length for spectral analysis
                fft_data = scipy.fft.fft(audio_data)
                magnitude = np.abs(fft_data[:len(fft_data)//2])
                freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)[:len(magnitude)]
                
                metrics.spectral_centroid = self._calculate_spectral_centroid(magnitude, freqs)
                metrics.spectral_rolloff = self._calculate_spectral_rolloff(magnitude, freqs)
                metrics.spectral_flatness = self._calculate_spectral_flatness(magnitude)
                metrics.snr_db = self._estimate_snr(audio_data, magnitude)
                metrics.thd_percent = self._calculate_thd(magnitude)
                metrics.noise_floor_db = self._estimate_noise_floor(magnitude)
            
            # Store in history
            self.analysis_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return metrics
    
    def _calculate_rms_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy"""
        return float(np.sqrt(np.mean(audio_data ** 2)))
    
    def _calculate_dynamic_range(self, audio_data: np.ndarray) -> float:
        """Calculate dynamic range in dB"""
        if len(audio_data) == 0:
            return 0.0
        
        max_val = np.max(np.abs(audio_data))
        min_val = np.percentile(np.abs(audio_data), 5)  # 5th percentile as noise floor
        
        if min_val > 0:
            return 20 * np.log10(max_val / min_val)
        return 0.0
    
    def _calculate_zcr(self, audio_data: np.ndarray) -> float:
        """Calculate Zero Crossing Rate"""
        if len(audio_data) <= 1:
            return 0.0
        
        zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
        return float(zero_crossings / (len(audio_data) - 1))
    
    def _detect_clipping(self, audio_data: np.ndarray, threshold: float = 0.99) -> bool:
        """Detect audio clipping"""
        return bool(np.any(np.abs(audio_data) >= threshold))
    
    def _calculate_spectral_centroid(self, magnitude: np.ndarray, freqs: np.ndarray) -> float:
        """Calculate spectral centroid"""
        if np.sum(magnitude) == 0:
            return 0.0
        
        return float(np.sum(freqs * magnitude) / np.sum(magnitude))
    
    def _calculate_spectral_rolloff(self, magnitude: np.ndarray, freqs: np.ndarray, 
                                  threshold: float = 0.85) -> float:
        """Calculate spectral rolloff"""
        total_energy = np.sum(magnitude)
        if total_energy == 0:
            return 0.0
        
        cumulative_energy = np.cumsum(magnitude)
        rolloff_index = np.where(cumulative_energy >= threshold * total_energy)[0]
        
        if len(rolloff_index) > 0:
            return float(freqs[rolloff_index[0]])
        return float(freqs[-1])
    
    def _calculate_spectral_flatness(self, magnitude: np.ndarray) -> float:
        """Calculate spectral flatness (Wiener entropy)"""
        if len(magnitude) == 0 or np.any(magnitude <= 0):
            return 0.0
        
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude)
        
        if arithmetic_mean > 0:
            return float(geometric_mean / arithmetic_mean)
        return 0.0
    
    def _estimate_snr(self, audio_data: np.ndarray, magnitude: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio"""
        # Simple SNR estimation
        signal_power = np.var(audio_data)
        noise_power = np.percentile(magnitude, 10) ** 2  # Estimate noise as 10th percentile
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return float(max(0.0, snr))
        return 0.0
    
    def _calculate_thd(self, magnitude: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion (simplified)"""
        if len(magnitude) < 10:
            return 0.0
        
        # Find fundamental frequency (peak)
        fundamental_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        
        # Calculate harmonic content
        harmonics_power = 0.0
        fundamental_power = magnitude[fundamental_idx] ** 2
        
        # Check first few harmonics
        for harmonic in range(2, 6):  # 2nd to 5th harmonics
            harmonic_idx = fundamental_idx * harmonic
            if harmonic_idx < len(magnitude):
                harmonics_power += magnitude[harmonic_idx] ** 2
        
        if fundamental_power > 0:
            thd = harmonics_power / fundamental_power
            return float(min(100.0, thd * 100))  # Convert to percentage
        return 0.0
    
    def _estimate_noise_floor(self, magnitude: np.ndarray) -> float:
        """Estimate noise floor in dB"""
        if len(magnitude) == 0:
            return -60.0  # Default noise floor
        
        noise_level = np.percentile(magnitude, 10)  # 10th percentile as noise estimate
        if noise_level > 0:
            return float(20 * np.log10(noise_level))
        return -60.0
    
    def get_quality_trend(self) -> Dict[str, float]:
        """Get quality trend analysis"""
        if len(self.analysis_history) < 2:
            return {}
        
        recent_metrics = list(self.analysis_history)[-10:]  # Last 10 measurements
        
        # Calculate trends
        snr_trend = np.mean([m.snr_db for m in recent_metrics])
        energy_trend = np.mean([m.rms_energy for m in recent_metrics])
        quality_stability = np.std([m.snr_db for m in recent_metrics])
        
        return {
            "avg_snr_db": snr_trend,
            "avg_energy": energy_trend,
            "quality_stability": quality_stability,
            "samples_analyzed": len(recent_metrics)
        }


class AudioOptimizer:
    """
    Main Audio Optimizer class - Revolutionary Audio Quality Enhancement System
    
    Provides comprehensive audio processing pipeline with real-time quality enhancement,
    adaptive filtering, and telephony optimization for voice AI applications.
    """
    
    def __init__(self, 
                 sample_rate: int = 8000,
                 processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE,
                 target_format: AudioFormat = AudioFormat.MULAW,
                 enable_real_time: bool = True,
                 **kwargs):
        """Initialize Audio Optimizer"""
        
        self.sample_rate = sample_rate
        self.processing_mode = processing_mode
        self.target_format = target_format
        self.enable_real_time = enable_real_time
        
        # Initialize processing components
        self.spectral_processor = SpectralProcessor(sample_rate)
        self.dynamics_processor = DynamicRangeProcessor(sample_rate)
        self.format_converter = AudioFormatConverter()
        self.quality_analyzer = QualityAnalyzer(sample_rate)
        
        # Processing parameters
        self.processing_params = ProcessingParams(
            mode=processing_mode,
            target_sample_rate=sample_rate,
            target_format=target_format,
            **kwargs
        )
        
        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "avg_processing_time_ms": 0.0,
            "avg_quality_improvement": 0.0,
            "format_conversions": 0,
            "real_time_performance": 0.0
        }
        
        # Adaptive processing state
        self.quality_history = collections.deque(maxlen=50)
        self.adaptive_threshold = 0.7
        
        # Thread pool for async processing
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Audio Optimizer initialized: {sample_rate}Hz, {processing_mode.value} mode")
    
    async def optimize_audio(self, 
                           audio_data: Union[bytes, np.ndarray],
                           source_format: AudioFormat = AudioFormat.PCM_16,
                           processing_params: Optional[ProcessingParams] = None) -> ProcessingResult:
        """
        Main audio optimization method with comprehensive enhancement
        
        Args:
            audio_data: Input audio data
            source_format: Source audio format
            processing_params: Optional processing parameters override
            
        Returns:
            ProcessingResult with enhanced audio and metrics
        """
        processing_start = time.time()
        
        # Use provided params or defaults
        params = processing_params or self.processing_params
        
        try:
            # Convert input to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = self.format_converter._bytes_to_array(audio_data, source_format)
            else:
                audio_array = audio_data.astype(np.float32)
            
            # Initial quality analysis
            initial_metrics = self.quality_analyzer.analyze_quality(audio_array)
            
            # Adaptive processing mode selection
            if params.adaptive_processing:
                params = self._adapt_processing_params(params, initial_metrics)
            
            # Apply audio enhancements
            enhanced_audio = await self._apply_enhancement_pipeline(audio_array, params)
            
            # Final quality analysis
            final_metrics = self.quality_analyzer.analyze_quality(enhanced_audio)
            
            # Format conversion
            if params.target_format != source_format:
                processed_bytes = self.format_converter.convert_format(
                    enhanced_audio, AudioFormat.FLOAT32, params.target_format, self.sample_rate
                )
                self.processing_stats["format_conversions"] += 1
            else:
                processed_bytes = self.format_converter._array_to_bytes(enhanced_audio, params.target_format)
            
            # Calculate quality improvement
            quality_improvement = self._calculate_quality_improvement(initial_metrics, final_metrics)
            
            # Create result
            processing_time = (time.time() - processing_start) * 1000
            
            result = ProcessingResult(
                processed_audio=processed_bytes,
                original_format=source_format,
                processed_format=params.target_format,
                quality_metrics=final_metrics,
                processing_time_ms=processing_time,
                enhancement_applied={
                    "noise_reduction": params.noise_reduction_strength > 0,
                    "gain_control": params.gain_control_enabled,
                    "dynamic_compression": params.dynamic_compression_enabled,
                    "equalization": params.equalization_enabled,
                    "bandwidth_optimization": params.bandwidth_optimization
                },
                quality_improvement=quality_improvement,
                latency_ms=processing_time,
                metadata={
                    "processing_mode": params.mode.value,
                    "sample_rate": self.sample_rate,
                    "adaptive_processing": params.adaptive_processing
                }
            )
            
            # Update statistics
            self._update_processing_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in audio optimization: {e}")
            
            # Return minimal processing result on error
            return ProcessingResult(
                processed_audio=audio_data if isinstance(audio_data, bytes) else b'',
                original_format=source_format,
                processed_format=source_format,
                quality_metrics=AudioQualityMetrics(),
                processing_time_ms=(time.time() - processing_start) * 1000,
                enhancement_applied={},
                quality_improvement=0.0,
                latency_ms=0.0,
                metadata={"error": str(e)}
            )
    
    async def _apply_enhancement_pipeline(self, audio_array: np.ndarray, 
                                        params: ProcessingParams) -> np.ndarray:
        """Apply comprehensive enhancement pipeline"""
        
        enhanced_audio = audio_array.copy()
        
        # Stage 1: Spectral Processing (Noise Reduction & Enhancement)
        if params.noise_reduction_strength > 0:
            enhanced_audio = self.spectral_processor.enhance_spectrum(
                enhanced_audio, params.noise_reduction_strength
            )
        
        # Stage 2: Dynamic Range Processing
        if params.gain_control_enabled or params.dynamic_compression_enabled:
            enhanced_audio = self.dynamics_processor.process_dynamics(
                enhanced_audio,
                enable_compression=params.dynamic_compression_enabled,
                enable_agc=params.gain_control_enabled
            )
        
        # Stage 3: Bandwidth Optimization (if enabled)
        if params.bandwidth_optimization:
            enhanced_audio = self._apply_bandwidth_optimization(enhanced_audio)
        
        # Stage 4: Equalization (if enabled)
        if params.equalization_enabled:
            enhanced_audio = self._apply_equalization(enhanced_audio)
        
        # Final normalization
        enhanced_audio = self._normalize_output(enhanced_audio)
        
        return enhanced_audio
    
    def _adapt_processing_params(self, params: ProcessingParams, 
                               metrics: AudioQualityMetrics) -> ProcessingParams:
        """Adapt processing parameters based on audio quality"""
        
        adapted_params = params
        
        # Determine quality level
        if metrics.snr_db > 20 and metrics.rms_energy > 0.1:
            # Excellent quality - minimal processing
            adapted_params.mode = ProcessingMode.MINIMAL
            adapted_params.noise_reduction_strength = 0.1
        elif metrics.snr_db > 10 and metrics.rms_energy > 0.05:
            # Good quality - standard processing
            adapted_params.mode = ProcessingMode.STANDARD
            adapted_params.noise_reduction_strength = 0.3
        elif metrics.snr_db > 5:
            # Poor quality - aggressive processing
            adapted_params.mode = ProcessingMode.AGGRESSIVE
            adapted_params.noise_reduction_strength = 0.7
        else:
            # Very poor quality - maximum processing
            adapted_params.mode = ProcessingMode.AGGRESSIVE
            adapted_params.noise_reduction_strength = 0.9
            adapted_params.dynamic_compression_enabled = True
            adapted_params.gain_control_enabled = True
        
        # Adapt based on clipping detection
        if metrics.clipping_detected:
            adapted_params.dynamic_compression_enabled = True
            adapted_params.gain_control_enabled = True
        
        return adapted_params
    
    def _apply_bandwidth_optimization(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply bandwidth optimization for network transmission"""
        
        # Apply telephony bandpass filter (300-3400 Hz)
        if self.sample_rate > 0:
            nyquist = self.sample_rate / 2
            low_cutoff = 300.0 / nyquist
            high_cutoff = min(3400.0 / nyquist, 0.95)
            
            if high_cutoff > low_cutoff:
                try:
                    b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
                    audio_array = scipy.signal.filtfilt(b, a, audio_array)
                except Exception as e:
                    logger.debug(f"Bandwidth filtering error: {e}")
        
        return audio_array
    
    def _apply_equalization(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply frequency equalization for voice clarity"""
        
        # Simple 3-band EQ for voice enhancement
        try:
            nyquist = self.sample_rate / 2
            
            # Low band (100-800 Hz) - slight reduction
            low_b, low_a = scipy.signal.butter(2, [100/nyquist, 800/nyquist], btype='band')
            low_band = scipy.signal.filtfilt(low_b, low_a, audio_array) * 0.8
            
            # Mid band (800-2500 Hz) - boost for speech clarity
            mid_b, mid_a = scipy.signal.butter(2, [800/nyquist, 2500/nyquist], btype='band')
            mid_band = scipy.signal.filtfilt(mid_b, mid_a, audio_array) * 1.2
            
            # High band (2500-4000 Hz) - moderate boost for consonants
            high_b, high_a = scipy.signal.butter(2, [2500/nyquist, min(4000/nyquist, 0.95)], btype='band')
            high_band = scipy.signal.filtfilt(high_b, high_a, audio_array) * 1.1
            
            # Combine bands
            equalized = low_band + mid_band + high_band
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(equalized))
            if max_val > 1.0:
                equalized = equalized / max_val
            
            return equalized
            
        except Exception as e:
            logger.debug(f"Equalization error: {e}")
            return audio_array
    
    def _normalize_output(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalize output audio"""
        
        # Peak normalization
        max_val = np.max(np.abs(audio_array))
        if max_val > 0.95:
            audio_array = audio_array * (0.95 / max_val)
        
        # RMS normalization for consistent loudness
        rms = np.sqrt(np.mean(audio_array ** 2))
        target_rms = 0.2  # Target RMS level
        
        if rms > 0:
            rms_gain = target_rms / rms
            rms_gain = min(rms_gain, 3.0)  # Limit gain to 3x
            audio_array = audio_array * rms_gain
        
        return audio_array
    
    def _calculate_quality_improvement(self, initial: AudioQualityMetrics, 
                                     final: AudioQualityMetrics) -> float:
        """Calculate overall quality improvement score"""
        
        # Weight different metrics
        weights = {
            'snr': 0.4,
            'rms': 0.2,
            'dynamic_range': 0.2,
            'spectral_centroid': 0.1,
            'spectral_flatness': 0.1
        }
        
        improvements = {}
        
        # SNR improvement
        snr_improvement = (final.snr_db - initial.snr_db) / max(initial.snr_db, 1.0)
        improvements['snr'] = max(-1.0, min(1.0, snr_improvement))
        
        # RMS energy improvement (toward target level)
        target_rms = 0.2
        initial_rms_error = abs(initial.rms_energy - target_rms)
        final_rms_error = abs(final.rms_energy - target_rms)
        rms_improvement = (initial_rms_error - final_rms_error) / max(initial_rms_error, 0.01)
        improvements['rms'] = max(-1.0, min(1.0, rms_improvement))
        
        # Dynamic range improvement
        dr_improvement = (final.dynamic_range_db - initial.dynamic_range_db) / max(initial.dynamic_range_db, 1.0)
        improvements['dynamic_range'] = max(-1.0, min(1.0, dr_improvement))
        
        # Spectral improvements
        sc_improvement = (final.spectral_centroid - initial.spectral_centroid) / max(initial.spectral_centroid, 1.0)
        improvements['spectral_centroid'] = max(-1.0, min(1.0, sc_improvement * 0.1))  # Small weight
        
        sf_improvement = (final.spectral_flatness - initial.spectral_flatness) / max(initial.spectral_flatness, 0.01)
        improvements['spectral_flatness'] = max(-1.0, min(1.0, sf_improvement * 0.1))  # Small weight
        
        # Calculate weighted average
        total_improvement = sum(improvements[key] * weights[key] for key in weights)
        
        return total_improvement
    
    def _update_processing_stats(self, result: ProcessingResult):
        """Update processing statistics"""
        self.processing_stats["total_processed"] += 1
        total_processed = self.processing_stats["total_processed"]
        
        # Update average processing time
        current_avg = self.processing_stats["avg_processing_time_ms"]
        self.processing_stats["avg_processing_time_ms"] = (
            (current_avg * (total_processed - 1) + result.processing_time_ms) / total_processed
        )
        
        # Update average quality improvement
        current_avg_quality = self.processing_stats["avg_quality_improvement"]
        self.processing_stats["avg_quality_improvement"] = (
            (current_avg_quality * (total_processed - 1) + result.quality_improvement) / total_processed
        )
        
        # Real-time performance tracking
        target_time = 1000.0 / (self.sample_rate / 1024)  # Time for 1024 samples
        real_time_ratio = target_time / result.processing_time_ms
        self.processing_stats["real_time_performance"] = (
            (self.processing_stats["real_time_performance"] * (total_processed - 1) + real_time_ratio) / total_processed
        )
    
    async def optimize_streaming_chunk(self, audio_chunk: bytes,
                                     source_format: AudioFormat = AudioFormat.MULAW,
                                     chunk_id: int = 0) -> ProcessingResult:
        """
        Optimize audio chunk for streaming with minimal latency
        
        Args:
            audio_chunk: Input audio chunk
            source_format: Source audio format
            chunk_id: Chunk identifier for tracking
            
        Returns:
            ProcessingResult with optimized audio chunk
        """
        
        # Use minimal processing for streaming to reduce latency
        streaming_params = ProcessingParams(
            mode=ProcessingMode.MINIMAL,
            noise_reduction_strength=0.2,
            gain_control_enabled=True,
            dynamic_compression_enabled=False,
            equalization_enabled=False,
            bandwidth_optimization=True,
            target_format=self.target_format,
            adaptive_processing=False  # Disable for consistent latency
        )
        
        result = await self.optimize_audio(audio_chunk, source_format, streaming_params)
        result.metadata["chunk_id"] = chunk_id
        result.metadata["streaming_optimized"] = True
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        quality_trend = self.quality_analyzer.get_quality_trend()
        
        return {
            **self.processing_stats,
            "quality_trend": quality_trend,
            "processing_mode": self.processing_mode.value,
            "target_format": self.target_format.value,
            "sample_rate": self.sample_rate,
            "real_time_capable": self.processing_stats["real_time_performance"] > 1.0,
            "components_active": {
                "spectral_processor": True,
                "dynamics_processor": True,
                "format_converter": True,
                "quality_analyzer": True
            }
        }
    
    async def cleanup(self):
        """Clean up resources"""
        self.thread_pool.shutdown(wait=True)
        self.quality_history.clear()
        logger.info("Audio Optimizer cleaned up")


# Utility functions for easy integration

def create_audio_optimizer_for_agent(agent_type: str, **kwargs) -> AudioOptimizer:
    """Create optimized Audio Optimizer for specific agent types"""
    
    agent_configs = {
        "roadside-assistance": {
            "processing_mode": ProcessingMode.AGGRESSIVE,  # Handle noisy emergency calls
            "enable_real_time": True,
            "noise_reduction_strength": 0.8,
            "gain_control_enabled": True,
            "dynamic_compression_enabled": True
        },
        "billing-support": {
            "processing_mode": ProcessingMode.STANDARD,   # Balanced for clear conversations
            "enable_real_time": True,
            "noise_reduction_strength": 0.4,
            "gain_control_enabled": True,
            "dynamic_compression_enabled": False
        },
        "technical-support": {
            "processing_mode": ProcessingMode.ADAPTIVE,   # Adaptive for varied environments
            "enable_real_time": True,
            "noise_reduction_strength": 0.5,
            "equalization_enabled": True,  # Enhanced clarity for technical terms
            "bandwidth_optimization": True
        }
    }
    
    config = agent_configs.get(agent_type, agent_configs["roadside-assistance"])
    config.update(kwargs)
    
    return AudioOptimizer(**config)


def optimize_for_twilio(audio_data: bytes, 
                       source_format: AudioFormat = AudioFormat.PCM_16) -> bytes:
    """Quick utility to optimize audio for Twilio transmission"""
    
    optimizer = AudioOptimizer(
        sample_rate=8000,
        processing_mode=ProcessingMode.MINIMAL,
        target_format=AudioFormat.MULAW
    )
    
    # Synchronous optimization for simple use case
    import asyncio
    
    async def _optimize():
        result = await optimizer.optimize_audio(audio_data, source_format)
        return result.processed_audio
    
    return asyncio.run(_optimize())


# Export main classes and functions
__all__ = [
    'AudioOptimizer',
    'ProcessingResult',
    'AudioQualityMetrics',
    'ProcessingParams',
    'ProcessingMode',
    'AudioFormat',
    'QualityMetric',
    'SpectralProcessor',
    'DynamicRangeProcessor',
    'AudioFormatConverter',
    'QualityAnalyzer',
    'create_audio_optimizer_for_agent',
    'optimize_for_twilio'
]