"""
Advanced Conversation State Manager for Multi-Agent Voice AI System
Provides intelligent state management with Redis persistence and context compression.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ConversationPhase(str, Enum):
    """Conversation phases for intelligent state management."""
    GREETING = "greeting"
    INFORMATION_GATHERING = "information_gathering"
    PROBLEM_SOLVING = "problem_solving"
    TOOL_EXECUTION = "tool_execution"
    CONFIRMATION = "confirmation"
    RESOLUTION = "resolution"
    FOLLOW_UP = "follow_up"
    ESCALATION = "escalation"
    COMPLETED = "completed"

class UrgencyLevel(str, Enum):
    """Urgency levels for prioritized handling."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ConversationMessage:
    """Individual message in a conversation."""
    message_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    agent_id: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserProfile:
    """User profile information."""
    user_id: Optional[str] = None
    phone_number: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[str] = field(default_factory=list)  # Session IDs
    satisfaction_ratings: List[float] = field(default_factory=list)
    last_interaction: Optional[float] = None
    total_interactions: int = 0

@dataclass
class ConversationContext:
    """Rich conversation context for intelligent processing."""
    domain: Optional[str] = None
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    urgency_indicators: List[str] = field(default_factory=list)
    emotional_state: Optional[str] = None
    complexity_score: float = 0.0
    requires_human_handoff: bool = False

@dataclass
class ConversationState:
    """Complete conversation state with intelligent management."""
    session_id: str
    start_time: float
    last_activity: float
    current_phase: ConversationPhase = ConversationPhase.GREETING
    urgency_level: UrgencyLevel = UrgencyLevel.NORMAL
    
    # Messages and history
    message_history: List[ConversationMessage] = field(default_factory=list)
    compressed_history: Optional[str] = None
    
    # User and context
    user_profile: UserProfile = field(default_factory=UserProfile)
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    
    # Agent and processing
    primary_agent_id: Optional[str] = None
    agent_handoffs: List[Dict[str, Any]] = field(default_factory=list)
    active_tools: List[str] = field(default_factory=list)
    
    # State management
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    temporary_data: Dict[str, Any] = field(default_factory=dict)  # For within-session data
    persistent_data: Dict[str, Any] = field(default_factory=dict)  # Cross-session data
    
    # Performance tracking
    total_latency_ms: float = 0.0
    interaction_count: int = 0
    satisfaction_score: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create from dictionary."""
        # Handle nested dataclasses
        if 'user_profile' in data and isinstance(data['user_profile'], dict):
            data['user_profile'] = UserProfile(**data['user_profile'])
        
        if 'conversation_context' in data and isinstance(data['conversation_context'], dict):
            data['conversation_context'] = ConversationContext(**data['conversation_context'])
        
        # Handle message history
        if 'message_history' in data:
            messages = []
            for msg_data in data['message_history']:
                if isinstance(msg_data, dict):
                    messages.append(ConversationMessage(**msg_data))
                else:
                    messages.append(msg_data)
            data['message_history'] = messages
        
        return cls(**data)

class ConversationStateManager:
    """
    Advanced conversation state manager with Redis persistence and intelligent compression.
    Handles stateful conversations across multiple interactions and agent handoffs.
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        enable_persistence: bool = True,
        max_context_length: int = 2048,
        context_compression: str = "intelligent_summarization",
        session_timeout_hours: int = 24,
        cleanup_interval_minutes: int = 60
    ):
        """Initialize the conversation state manager."""
        self.redis_client = redis_client
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.enable_persistence = enable_persistence
        self.max_context_length = max_context_length
        self.context_compression = context_compression
        self.session_timeout = session_timeout_hours * 3600  # Convert to seconds
        self.cleanup_interval = cleanup_interval_minutes * 60
        
        # In-memory cache for active sessions
        self.active_sessions: Dict[str, ConversationState] = {}
        
        # Performance tracking
        self.stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compressions_performed": 0,
            "context_compressions": 0,
            "average_session_duration": 0.0
        }
        
        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self.initialized = False
        
        logger.info("Conversation State Manager initialized")
    
    async def initialize(self):
        """Initialize Redis connection and background tasks."""
        if not self.redis_client and self.enable_persistence:
            try:
                self.redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info(f"✅ Redis connection established: {self.redis_host}:{self.redis_port}")
                
            except Exception as e:
                logger.error(f"❌ Redis connection failed: {e}")
                self.enable_persistence = False
                logger.warning("⚠️ Running without persistence - sessions will be memory-only")
        
        # Start background cleanup task
        if self.cleanup_interval > 0:
            self.cleanup_task = asyncio.create_task(self._background_cleanup())
        
        self.initialized = True
        logger.info("✅ Conversation State Manager initialized")
    
    async def get_conversation_state(self, session_id: str) -> Optional[ConversationState]:
        """Get conversation state from cache or persistent storage."""
        # Check in-memory cache first
        if session_id in self.active_sessions:
            self.stats["cache_hits"] += 1
            state = self.active_sessions[session_id]
            state.last_activity = time.time()
            return state
        
        self.stats["cache_misses"] += 1
        
        # Try to load from Redis if persistence is enabled
        if self.enable_persistence and self.redis_client:
            try:
                state_data = await self.redis_client.get(f"conversation:{session_id}")
                if state_data:
                    state_dict = json.loads(state_data)
                    state = ConversationState.from_dict(state_dict)
                    
                    # Update last activity
                    state.last_activity = time.time()
                    
                    # Cache in memory
                    self.active_sessions[session_id] = state
                    
                    logger.debug(f"Loaded conversation state from Redis: {session_id}")
                    return state
                    
            except Exception as e:
                logger.error(f"Error loading conversation state from Redis: {e}")
        
        # Return None if not found
        return None
    
    async def create_conversation_state(
        self,
        session_id: str,
        user_phone: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> ConversationState:
        """Create a new conversation state."""
        current_time = time.time()
        
        # Create user profile
        user_profile = UserProfile(
            phone_number=user_phone,
            last_interaction=current_time,
            total_interactions=1
        )
        
        # Create conversation context
        conversation_context = ConversationContext()
        if initial_context:
            if 'domain' in initial_context:
                conversation_context.domain = initial_context['domain']
            if 'intent' in initial_context:
                conversation_context.intent = initial_context['intent']
            if 'urgency' in initial_context:
                conversation_context.urgency_indicators = [initial_context['urgency']]
        
        # Create conversation state
        state = ConversationState(
            session_id=session_id,
            start_time=current_time,
            last_activity=current_time,
            user_profile=user_profile,
            conversation_context=conversation_context,
            metadata=initial_context or {}
        )
        
        # Store in cache and persistence
        await self.save_conversation_state(state)
        
        self.stats["total_sessions"] += 1
        self.stats["active_sessions"] = len(self.active_sessions)
        
        logger.info(f"Created new conversation state: {session_id}")
        return state
    
    async def update_conversation(
        self,
        session_id: str,
        user_message: Optional[str] = None,
        assistant_message: Optional[str] = None,
        agent_id: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update conversation with new messages and context."""
        state = await self.get_conversation_state(session_id)
        if not state:
            state = await self.create_conversation_state(session_id)
        
        current_time = time.time()
        
        # Add user message if provided
        if user_message:
            user_msg = ConversationMessage(
                message_id=str(uuid.uuid4()),
                role="user",
                content=user_message,
                timestamp=current_time,
                metadata=metadata or {}
            )
            state.message_history.append(user_msg)
        
        # Add assistant message if provided
        if assistant_message:
            assistant_msg = ConversationMessage(
                message_id=str(uuid.uuid4()),
                role="assistant",
                content=assistant_message,
                timestamp=current_time,
                agent_id=agent_id,
                tools_used=tools_used or [],
                confidence=confidence,
                metadata=metadata or {}
            )
            state.message_history.append(assistant_msg)
        
        # Update agent tracking
        if agent_id:
            if state.primary_agent_id != agent_id:
                # Record agent handoff
                handoff = {
                    "from_agent": state.primary_agent_id,
                    "to_agent": agent_id,
                    "timestamp": current_time,
                    "reason": metadata.get("handoff_reason") if metadata else None
                }
                state.agent_handoffs.append(handoff)
                state.primary_agent_id = agent_id
        
        # Update conversation phase based on content and tools
        await self._update_conversation_phase(state, user_message, assistant_message, tools_used)
        
        # Update urgency level if indicators present
        await self._update_urgency_level(state, user_message)
        
        # Update activity and interaction count
        state.last_activity = current_time
        state.interaction_count += 1
        
        # Apply context compression if needed
        if len(state.message_history) > 10:  # Compress after 10 messages
            await self._compress_context_if_needed(state)
        
        # Save updated state
        await self.save_conversation_state(state)
        
        logger.debug(f"Updated conversation state: {session_id} (phase: {state.current_phase})")
    
    async def save_conversation_state(self, state: ConversationState):
        """Save conversation state to cache and persistent storage."""
        # Update in-memory cache
        self.active_sessions[state.session_id] = state
        
        # Save to Redis if persistence enabled
        if self.enable_persistence and self.redis_client:
            try:
                state_json = json.dumps(state.to_dict(), default=str)
                await self.redis_client.setex(
                    f"conversation:{state.session_id}",
                    self.session_timeout,
                    state_json
                )
                logger.debug(f"Saved conversation state to Redis: {state.session_id}")
                
            except Exception as e:
                logger.error(f"Error saving conversation state to Redis: {e}")
    
    async def update_user_preferences(
        self,
        session_id: str,
        preferences: Dict[str, Any]
    ):
        """Update user preferences for the session."""
        state = await self.get_conversation_state(session_id)
        if state:
            state.user_preferences.update(preferences)
            await self.save_conversation_state(state)
    
    async def set_temporary_data(
        self,
        session_id: str,
        key: str,
        value: Any
    ):
        """Set temporary data for the session (cleared when session ends)."""
        state = await self.get_conversation_state(session_id)
        if state:
            state.temporary_data[key] = value
            await self.save_conversation_state(state)
    
    async def get_temporary_data(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """Get temporary data from the session."""
        state = await self.get_conversation_state(session_id)
        if state:
            return state.temporary_data.get(key, default)
        return default
    
    async def set_persistent_data(
        self,
        session_id: str,
        key: str,
        value: Any
    ):
        """Set persistent data (survives across sessions for the user)."""
        state = await self.get_conversation_state(session_id)
        if state:
            state.persistent_data[key] = value
            await self.save_conversation_state(state)
    
    async def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the conversation state."""
        state = await self.get_conversation_state(session_id)
        if not state:
            return None
        
        return {
            "session_id": session_id,
            "duration_minutes": (time.time() - state.start_time) / 60,
            "interaction_count": state.interaction_count,
            "current_phase": state.current_phase,
            "urgency_level": state.urgency_level,
            "primary_agent": state.primary_agent_id,
            "agent_handoffs": len(state.agent_handoffs),
            "tools_used": len(state.active_tools),
            "message_count": len(state.message_history),
            "satisfaction_score": state.satisfaction_score,
            "context_domain": state.conversation_context.domain,
            "context_intent": state.conversation_context.intent
        }
    
    async def end_conversation(
        self,
        session_id: str,
        satisfaction_score: Optional[float] = None,
        resolution_status: str = "completed"
    ):
        """End a conversation and update final metrics."""
        state = await self.get_conversation_state(session_id)
        if state:
            state.current_phase = ConversationPhase.COMPLETED
            state.satisfaction_score = satisfaction_score
            state.metadata["resolution_status"] = resolution_status
            state.metadata["end_time"] = time.time()
            
            # Update user profile
            if satisfaction_score:
                state.user_profile.satisfaction_ratings.append(satisfaction_score)
            
            # Final save
            await self.save_conversation_state(state)
            
            # Update stats
            duration = time.time() - state.start_time
            current_avg = self.stats["average_session_duration"]
            total_sessions = self.stats["total_sessions"]
            self.stats["average_session_duration"] = (current_avg * (total_sessions - 1) + duration) / total_sessions
            
            logger.info(f"Ended conversation: {session_id} (duration: {duration/60:.1f}min, "
                       f"satisfaction: {satisfaction_score})")
    
    async def _update_conversation_phase(
        self,
        state: ConversationState,
        user_message: Optional[str],
        assistant_message: Optional[str],
        tools_used: Optional[List[str]]
    ):
        """Intelligently update conversation phase based on content."""
        if not user_message and not assistant_message:
            return
        
        # Simple phase detection logic (can be enhanced with ML)
        if tools_used:
            state.current_phase = ConversationPhase.TOOL_EXECUTION
        elif user_message:
            # Analyze user message for phase indicators
            message_lower = user_message.lower()
            
            if any(word in message_lower for word in ["thank", "thanks", "bye", "goodbye"]):
                state.current_phase = ConversationPhase.RESOLUTION
            elif any(word in message_lower for word in ["help", "problem", "issue", "wrong"]):
                state.current_phase = ConversationPhase.PROBLEM_SOLVING
            elif any(word in message_lower for word in ["yes", "confirm", "correct", "that's right"]):
                state.current_phase = ConversationPhase.CONFIRMATION
            elif state.interaction_count <= 2:
                state.current_phase = ConversationPhase.INFORMATION_GATHERING
            else:
                state.current_phase = ConversationPhase.PROBLEM_SOLVING
    
    async def _update_urgency_level(self, state: ConversationState, user_message: Optional[str]):
        """Update urgency level based on message content."""
        if not user_message:
            return
        
        message_lower = user_message.lower()
        
        # Emergency indicators
        if any(word in message_lower for word in ["emergency", "urgent", "help", "stuck", "accident"]):
            if any(word in message_lower for word in ["accident", "crash", "injured"]):
                state.urgency_level = UrgencyLevel.EMERGENCY
            else:
                state.urgency_level = UrgencyLevel.HIGH
        elif any(word in message_lower for word in ["quick", "asap", "soon", "waiting"]):
            state.urgency_level = UrgencyLevel.HIGH
        elif state.urgency_level == UrgencyLevel.NORMAL and state.interaction_count > 5:
            # Escalate if conversation is taking too long
            state.urgency_level = UrgencyLevel.HIGH
    
    async def _compress_context_if_needed(self, state: ConversationState):
        """Compress conversation context to manage memory and token usage."""
        if len(state.message_history) < 15:  # Don't compress until we have enough history
            return
        
        try:
            # Simple compression: keep first 2 and last 5 messages, summarize the middle
            if not state.compressed_history:
                # Summarize the middle messages
                middle_messages = state.message_history[2:-5]
                if middle_messages:
                    summary_content = []
                    for msg in middle_messages:
                        summary_content.append(f"{msg.role}: {msg.content[:100]}...")
                    
                    state.compressed_history = "Previous conversation summary: " + " | ".join(summary_content)
                    
                    # Keep only first 2 and last 5 messages plus summary
                    state.message_history = state.message_history[:2] + state.message_history[-5:]
                    
                    self.stats["compressions_performed"] += 1
                    logger.debug(f"Compressed conversation history for {state.session_id}")
            
        except Exception as e:
            logger.error(f"Error compressing conversation context: {e}")
    
    async def _background_cleanup(self):
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = time.time()
                expired_sessions = []
                
                # Find expired sessions in memory cache
                for session_id, state in self.active_sessions.items():
                    if current_time - state.last_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                    logger.debug(f"Cleaned up expired session: {session_id}")
                
                if expired_sessions:
                    self.stats["active_sessions"] = len(self.active_sessions)
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive state manager statistics."""
        return {
            **self.stats,
            "session_timeout_hours": self.session_timeout / 3600,
            "cleanup_interval_minutes": self.cleanup_interval / 60,
            "persistence_enabled": self.enable_persistence,
            "redis_connected": self.redis_client is not None and self.enable_persistence
        }
    
    async def shutdown(self):
        """Shutdown the state manager and cleanup resources."""
        logger.info("Shutting down Conversation State Manager...")
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Clear memory cache
        self.active_sessions.clear()
        
        self.initialized = False
        logger.info("✅ Conversation State Manager shutdown complete")