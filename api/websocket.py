# api/websocket.py

"""
WebSocket handler for real-time communication.
"""
import logging
import json
import asyncio
from typing import Dict, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
import base64

from core.state_manager import ConversationState
from agents.base_agent import AgentType
from services.analytics import AnalyticsService

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_states: Dict[str, Dict[str, Any]] = {}
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """
        Connect a new WebSocket.
        
        Args:
            session_id: Session identifier
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # Initialize session state
        self.session_states[session_id] = {
            "connected_at": asyncio.get_event_loop().time(),
            "last_message_at": asyncio.get_event_loop().time(),
            "messages_received": 0,
            "messages_sent": 0,
            "current_state": ConversationState.GREETING,
            "current_agent": None
        }
        
        # Start heartbeat
        self.heartbeat_tasks[session_id] = asyncio.create_task(
            self._heartbeat(session_id)
        )
        
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, session_id: str):
        """
        Disconnect a WebSocket.
        
        Args:
            session_id: Session identifier
        """
        # Stop heartbeat
        if session_id in self.heartbeat_tasks:
            self.heartbeat_tasks[session_id].cancel()
            del self.heartbeat_tasks[session_id]
        
        # Remove connection
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        # Store session state for analytics
        if session_id in self.session_states:
            state = self.session_states[session_id]
            state["disconnected_at"] = asyncio.get_event_loop().time()
            state["session_duration"] = (
                state["disconnected_at"] - state["connected_at"]
            )
            
            # Log session stats
            logger.info(
                f"Session {session_id} ended after {state['session_duration']:.1f}s "
                f"with {state['messages_received']} messages received and "
                f"{state['messages_sent']} messages sent"
            )
    
    async def _heartbeat(self, session_id: str):
        """
        Send periodic heartbeats to keep connection alive.
        
        Args:
            session_id: Session identifier
        """
        try:
            while True:
                if session_id in self.active_connections:
                    try:
                        websocket = self.active_connections[session_id]
                        await websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": asyncio.get_event_loop().time()
                        })
                    except Exception as e:
                        logger.error(f"Error sending heartbeat to {session_id}: {e}")
                        break
                    
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat task for {session_id}: {e}")
    
    async def broadcast(self, message: str):
        """
        Broadcast message to all connections.
        
        Args:
            message: Message to broadcast
        """
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")
    
    def get_connection(self, session_id: str) -> Optional[WebSocket]:
        """
        Get WebSocket connection for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            WebSocket connection if found
        """
        return self.active_connections.get(session_id)
    
    def update_session_state(
        self,
        session_id: str,
        state: ConversationState,
        agent_type: Optional[AgentType] = None
    ):
        """
        Update session state.
        
        Args:
            session_id: Session identifier
            state: New conversation state
            agent_type: Optional agent type
        """
        if session_id in self.session_states:
            self.session_states[session_id]["current_state"] = state
            if agent_type:
                self.session_states[session_id]["current_agent"] = agent_type
            self.session_states[session_id]["last_message_at"] = asyncio.get_event_loop().time()

class WebSocketHandler:
    """Handle WebSocket communication for voice AI system."""
    
    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        analytics_service: Optional[AnalyticsService] = None
    ):
        """
        Initialize WebSocket handler.
        
        Args:
            connection_manager: Optional connection manager
            analytics_service: Optional analytics service
        """
        self.manager = connection_manager or ConnectionManager()
        self.analytics = analytics_service or AnalyticsService()
        
        # Track active streams
        self.active_streams: Set[str] = set()
    
    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: str
    ):
        """
        Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            session_id: Session identifier
        """
        await self.manager.connect(session_id, websocket)
        
        try:
            while True:
                # Receive message
                try:
                    message = await websocket.receive()
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for session {session_id}")
                    break
                
                # Update message count
                if session_id in self.manager.session_states:
                    self.manager.session_states[session_id]["messages_received"] += 1
                
                # Process message based on type
                if "text" in message:
                    await self._handle_text_message(session_id, message["text"])
                elif "bytes" in message:
                    await self._handle_binary_message(session_id, message["bytes"])
                
        finally:
            # Clean up
            self.manager.disconnect(session_id)
            if session_id in self.active_streams:
                self.active_streams.remove(session_id)
    
    async def _handle_text_message(self, session_id: str, message: str):
        """
        Handle text message from WebSocket.
        
        Args:
            session_id: Session identifier
            message: Text message
        """
        try:
            # Parse JSON message
            data = json.loads(message)
            event_type = data.get("event")
            
            if event_type == "start":
                # Start of media stream
                stream_sid = data.get("streamSid")
                logger.info(f"Stream started: {stream_sid} for session {session_id}")
                self.active_streams.add(session_id)
                
            elif event_type == "media":
                # Media packet
                if "payload" in data.get("media", {}):
                    await self._handle_media_packet(
                        session_id,
                        data["media"]["payload"]
                    )
                    
            elif event_type == "stop":
                # End of media stream
                logger.info(f"Stream stopped for session {session_id}")
                if session_id in self.active_streams:
                    self.active_streams.remove(session_id)
                    
            elif event_type == "error":
                # Error event
                logger.error(f"Stream error for session {session_id}: {data.get('error')}")
                await self.analytics.track_error(
                    "websocket",
                    "stream_error",
                    {"session_id": session_id, "error": data.get("error")}
                )
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from session {session_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
    
    async def _handle_binary_message(self, session_id: str, data: bytes):
        """
        Handle binary message from WebSocket.
        
        Args:
            session_id: Session identifier
            data: Binary data
        """
        try:
            # Process binary data
            await self._handle_media_packet(session_id, base64.b64encode(data).decode())
            
        except Exception as e:
            logger.error(f"Error handling binary message: {e}")
    
    async def _handle_media_packet(self, session_id: str, payload: str):
        """
        Handle media packet.
        
        Args:
            session_id: Session identifier
            payload: Base64 encoded media data
        """
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            
            # Get agent router from main application
            from main import agent_router
            if not agent_router:
                logger.error("Agent router not initialized")
                return
            
            # Process audio through agent router
            response = await agent_router.route_message(
                session_id=session_id,
                message=audio_data,
                is_audio=True
            )
            
            # Send response
            websocket = self.manager.get_connection(session_id)
            if websocket and response:
                await websocket.send_json(response)
                
                # Update message count
                if session_id in self.manager.session_states:
                    self.manager.session_states[session_id]["messages_sent"] += 1
                
                # Track analytics
                if "response_time" in response:
                    await self.analytics.track_conversation(
                        session_id=session_id,
                        agent_type=response.get("agent_type"),
                        response_time=response["response_time"],
                        state_transition=response.get("state_transition")
                    )
            
        except Exception as e:
            logger.error(f"Error handling media packet: {e}")
            await self.analytics.track_error(
                "websocket",
                "media_processing_error",
                {"session_id": session_id, "error": str(e)}
            )