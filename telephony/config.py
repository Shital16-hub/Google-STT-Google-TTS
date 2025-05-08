"""
Enhanced configuration settings for telephony integration with improved noise handling.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Server Configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Audio Configuration - Optimized for 100ms frames
SAMPLE_RATE_TWILIO = 8000  # Twilio's sample rate
SAMPLE_RATE_AI = 16000     # Our AI system's sample rate
CHUNK_SIZE = 160           # 20ms at 8kHz
# Optimized buffer size for speech detection (100ms chunks)
AUDIO_BUFFER_SIZE = 1600   # 100ms buffer at 16kHz, 800 bytes at 8kHz
MAX_BUFFER_SIZE = 16000    # 1 second maximum buffer (reduced from 48000)

# WebSocket Configuration - Optimized buffer sizes
WS_PING_INTERVAL = 10      # Reduced from 20
WS_PING_TIMEOUT = 5        # Reduced from 10
WS_MAX_MESSAGE_SIZE = 524288  # 512KB (reduced from 1MB)

# Enhanced Speech Detection Settings
SILENCE_THRESHOLD = 0.0025   # Increased from 0.0018 for better noise rejection
SILENCE_DURATION = 1.5       # Reduced from 2.0 seconds to improve responsiveness
MAX_CALL_DURATION = 3600     # 1 hour
MAX_PROCESSING_TIME = 2.0    # Reduced from 5.0 seconds for faster processing

# Response Settings
RESPONSE_TIMEOUT = 2.0        # Reduced from 4.0 seconds to meet 2-second target
MIN_TRANSCRIPTION_LENGTH = 2  # Reduced from 4 to process shorter utterances

# Enhanced Noise Filtering Settings
HIGH_PASS_FILTER = 120       # Increased from 100Hz to further reduce low-frequency noise
NOISE_GATE_THRESHOLD = 0.025  # Increased from 0.018
ENABLE_NOISE_FILTERING = True

# Enhanced Barge-in Settings - Dual threshold approach
ENABLE_BARGE_IN = True                   # Enable barge-in functionality
BARGE_IN_THRESHOLD = 0.045               # Higher threshold for detection start
BARGE_IN_LOW_THRESHOLD = 0.025           # Lower threshold for maintaining detection
BARGE_IN_DETECTION_WINDOW = 100          # Reduced from 140ms for faster detection
BARGE_IN_MIN_SPEECH_DURATION = 200       # Reduced from 300ms to improve responsiveness
BARGE_IN_COOLDOWN_MS = 1000              # Reduced from 1500ms for faster barge-in

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Audio Preprocessor Configuration - Speech detection thresholds
PREPROCESSOR_ENABLE_DEBUG = False   # Enable detailed debug logging for audio preprocessor
PREPROCESSOR_WARMUP_FRAMES = 5      # Reduced from 15 for faster startup
PREPROCESSOR_NOISE_FLOOR_MIN = 0.008  # Minimum noise floor level

# STT Optimization Settings
STT_INITIAL_PROMPT = """This is a telephone conversation. 
Focus only on the clearly spoken words and ignore any background noise, static, 
beeps, or line interference. Transcribe only the spoken words."""

STT_NO_CONTEXT = True       # Disable context to prevent false additions in noisy environments
STT_TEMPERATURE = 0.0       # Use greedy decoding for less hallucination
STT_PRESET = "default"      # Use default preset with enhanced noise handling

# WebSocket Buffer Optimization
WS_BUFFER_SIZE = 512  # Optimized buffer size for WebSockets (512-1024 bytes)
WS_BINARY_TYPE = "arraybuffer"  # Use binary transmission for efficiency