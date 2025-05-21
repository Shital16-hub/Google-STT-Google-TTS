# utils/audio_utils.py

"""
Audio processing utilities optimized for voice interactions.
"""
import numpy as np
import io
import wave
import logging
from typing import Tuple, Optional, Union, List
from pathlib import Path
import tempfile
import subprocess

logger = logging.getLogger(__name__)

def load_audio_file(
    file_path: Union[str, Path],
    target_sr: int = 8000,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load and process audio file for voice interactions.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        normalize: Whether to normalize audio
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        import soundfile as sf
        audio, orig_sr = sf.read(str(file_path))
    except ImportError:
        logger.warning("soundfile not available, falling back to wave")
        with wave.open(str(file_path), 'rb') as wav:
            # Get audio info
            frames = wav.readframes(wav.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
            orig_sr = wav.getframerate()
            
            # Convert to float32
            audio = audio.astype(np.float32) / 32768.0
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if orig_sr != target_sr:
        audio = resample_audio(audio, orig_sr, target_sr)
    
    # Normalize if requested
    if normalize:
        audio = normalize_audio(audio)
    
    return audio, target_sr

def save_audio_file(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int,
    audio_format: str = "wav"
) -> None:
    """
    Save audio data to file.
    
    Args:
        audio: Audio data
        file_path: Output file path
        sample_rate: Sample rate
        audio_format: Output format
    """
    try:
        import soundfile as sf
        sf.write(str(file_path), audio, sample_rate, format=audio_format)
    except ImportError:
        logger.warning("soundfile not available, falling back to wave")
        # Convert to int16
        audio_int16 = (audio * 32768.0).astype(np.int16)
        
        with wave.open(str(file_path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

def normalize_audio(
    audio: np.ndarray,
    target_level: float = -23.0
) -> np.ndarray:
    """
    Normalize audio with loudness targeting.
    
    Args:
        audio: Audio data
        target_level: Target loudness level in dB
        
    Returns:
        Normalized audio data
    """
    # Convert to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Calculate current RMS level
    rms = np.sqrt(np.mean(audio ** 2))
    current_db = 20 * np.log10(rms) if rms > 0 else -100.0
    
    # Calculate gain needed
    gain_db = target_level - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Apply gain with clipping prevention
    normalized = audio * gain_linear
    if np.max(np.abs(normalized)) > 1.0:
        normalized = normalized / np.max(np.abs(normalized))
    
    return normalized

def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio data.
    
    Args:
        audio: Audio data
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio data
    """
    try:
        import librosa
        return librosa.resample(
            y=audio,
            orig_sr=orig_sr,
            target_sr=target_sr
        )
    except ImportError:
        logger.warning("librosa not available, falling back to scipy")
        from scipy import signal
        # Calculate number of samples for target
        n_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, n_samples)

def convert_audio_format(
    audio_data: bytes,
    from_format: str,
    to_format: str,
    sample_rate: Optional[int] = None
) -> bytes:
    """
    Convert audio between formats.
    
    Args:
        audio_data: Audio data bytes
        from_format: Source format
        to_format: Target format
        sample_rate: Optional sample rate
        
    Returns:
        Converted audio data
    """
    with tempfile.NamedTemporaryFile(suffix=f'.{from_format}', delete=False) as src_file:
        src_path = src_file.name
        src_file.write(audio_data)
    
    with tempfile.NamedTemporaryFile(suffix=f'.{to_format}', delete=False) as dst_file:
        dst_path = dst_file.name
    
    try:
        # Build ffmpeg command
        cmd = ['ffmpeg', '-i', src_path]
        
        if sample_rate:
            cmd.extend(['-ar', str(sample_rate)])
        
        if to_format == 'wav':
            cmd.extend(['-acodec', 'pcm_s16le'])
        elif to_format == 'mulaw':
            cmd.extend(['-acodec', 'pcm_mulaw'])
        
        cmd.extend(['-y', dst_path])
        
        # Run conversion
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Read converted file
        with open(dst_path, 'rb') as f:
            converted_data = f.read()
        
        return converted_data
        
    finally:
        # Clean up temp files
        Path(src_path).unlink()
        Path(dst_path).unlink()

def detect_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_duration: int = 400,  # ms
    sample_rate: int = 8000
) -> List[Tuple[int, int]]:
    """
    Detect silence regions in audio.
    
    Args:
        audio: Audio data
        threshold: Silence threshold
        min_duration: Minimum silence duration (ms)
        sample_rate: Sample rate
        
    Returns:
        List of (start, end) sample indices
    """
    # Calculate energy
    energy = np.abs(audio)
    
    # Find silence regions
    is_silence = energy < threshold
    
    # Convert min_duration to samples
    min_samples = int(min_duration * sample_rate / 1000)
    
    # Find silence regions
    silence_regions = []
    start = None
    
    for i, silent in enumerate(is_silence):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            if i - start >= min_samples:
                silence_regions.append((start, i))
            start = None
    
    # Handle trailing silence
    if start is not None and len(audio) - start >= min_samples:
        silence_regions.append((start, len(audio)))
    
    return silence_regions

def detect_speech_activity(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_duration: int = 200,  # ms
    sample_rate: int = 8000
) -> List[Tuple[int, int]]:
    """
    Detect speech activity regions in audio.
    
    Args:
        audio: Audio data
        threshold: Activity threshold
        min_duration: Minimum activity duration (ms)
        sample_rate: Sample rate
        
    Returns:
        List of (start, end) sample indices
    """
    # Calculate energy
    energy = np.abs(audio)
    
    # Find activity regions
    is_active = energy >= threshold
    
    # Convert min_duration to samples
    min_samples = int(min_duration * sample_rate / 1000)
    
    # Find activity regions
    activity_regions = []
    start = None
    
    for i, active in enumerate(is_active):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if i - start >= min_samples:
                activity_regions.append((start, i))
            start = None
    
    # Handle trailing activity
    if start is not None and len(audio) - start >= min_samples:
        activity_regions.append((start, len(audio)))
    
    return activity_regions

def apply_noise_reduction(
    audio: np.ndarray,
    noise_threshold: float = 0.01
) -> np.ndarray:
    """
    Apply simple noise reduction.
    
    Args:
        audio: Audio data
        noise_threshold: Noise threshold
        
    Returns:
        Noise-reduced audio
    """
    # Simple noise gate
    audio_reduced = audio.copy()
    audio_reduced[np.abs(audio) < noise_threshold] = 0
    
    return audio_reduced

def create_silence(
    duration_ms: int,
    sample_rate: int = 8000
) -> np.ndarray:
    """
    Create silence of specified duration.
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate
        
    Returns:
        Silent audio data
    """
    num_samples = int(duration_ms * sample_rate / 1000)
    return np.zeros(num_samples, dtype=np.float32)

def concatenate_audio(
    audio_segments: List[np.ndarray],
    crossfade_ms: int = 10
) -> np.ndarray:
    """
    Concatenate audio segments with optional crossfade.
    
    Args:
        audio_segments: List of audio segments
        crossfade_ms: Crossfade duration in milliseconds
        
    Returns:
        Concatenated audio
    """
    if not audio_segments:
        return np.array([], dtype=np.float32)
    
    if len(audio_segments) == 1:
        return audio_segments[0]
    
    # Calculate crossfade samples
    crossfade_samples = int(crossfade_ms * 8000 / 1000)  # Assuming 8kHz
    
    # Initialize output
    total_samples = sum(len(seg) for seg in audio_segments)
    total_samples -= crossfade_samples * (len(audio_segments) - 1)
    output = np.zeros(total_samples, dtype=np.float32)
    
    # Copy first segment
    output[:len(audio_segments[0])] = audio_segments[0]
    current_pos = len(audio_segments[0]) - crossfade_samples
    
    # Add remaining segments with crossfade
    for seg in audio_segments[1:]:
        # Create fade curves
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)
        
        # Apply crossfade
        output[current_pos:current_pos + crossfade_samples] *= fade_out
        output[current_pos:current_pos + crossfade_samples] += seg[:crossfade_samples] * fade_in
        
        # Copy rest of segment
        output[current_pos + crossfade_samples:current_pos + len(seg)] = seg[crossfade_samples:]
        current_pos += len(seg) - crossfade_samples
    
    return output