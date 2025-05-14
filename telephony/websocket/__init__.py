# telephony/websocket/__init__.py

"""
WebSocket handler components for Twilio media streams.

This module provides modular components for handling Twilio WebSocket connections,
audio processing, speech recognition, and response generation.
"""

from .connection_manager import ConnectionManager
from .audio_manager import AudioManager
from .speech_processor import SpeechProcessor
from .response_generator import ResponseGenerator
from .message_router import MessageRouter

# Optional advanced components
try:
    from .intelligent_audio_manager import IntelligentAudioManager
    from .generalized_speech_processor import GeneralizedSpeechProcessor
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ADVANCED_COMPONENTS_AVAILABLE = False

__all__ = [
    'ConnectionManager',
    'AudioManager',
    'SpeechProcessor',
    'ResponseGenerator',
    'MessageRouter'
]



__version__ = "1.0.0"