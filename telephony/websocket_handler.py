# telephony/websocket_handler.py

"""
Modular WebSocket handler for Twilio media streams with Google Cloud STT integration.
"""
import json
import asyncio
import logging
from typing import Dict, Any, Optional

from telephony.websocket.connection_manager import ConnectionManager
from telephony.websocket.audio_manager import AudioManager
from telephony.websocket.speech_processor import SpeechProcessor
from telephony.websocket.response_generator import ResponseGenerator
from telephony.websocket.message_router import MessageRouter

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """
    Modular WebSocket handler that orchestrates all components for telephony integration.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize WebSocket handler with modular components.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        logger.info(f"Initializing modular WebSocketHandler for call {call_sid}")
        
        # Initialize component managers
        self.connection_manager = ConnectionManager(call_sid)
        self.audio_manager = AudioManager() 
        self.speech_processor = SpeechProcessor(pipeline)
        self.response_generator = ResponseGenerator(pipeline, self)
        self.message_router = MessageRouter(self)
        
        # State tracking
        self.conversation_active = True
        
        logger.info("WebSocketHandler initialized with modular components")
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Handle incoming WebSocket message by routing to appropriate handler.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        await self.message_router.route_message(message, ws)
    
    # Proxy methods for component access
    async def send_text_response(self, text: str, ws) -> None:
        """Send text response through response generator."""
        await self.response_generator.send_text_response(text, ws)
    
    def cleanup_transcription(self, text: str) -> str:
        """Clean up transcription through speech processor."""
        return self.speech_processor.cleanup_transcription(text)
    
    def is_valid_transcription(self, text: str) -> bool:
        """Validate transcription through speech processor."""
        return self.speech_processor.is_valid_transcription(text)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "conversation_active": self.conversation_active,
            "audio_stats": self.audio_manager.get_stats(),
            "speech_stats": self.speech_processor.get_stats(),
            "message_router_stats": self.message_router.get_stats()
        }