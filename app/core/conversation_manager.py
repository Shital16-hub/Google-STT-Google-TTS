"""
Enhanced Conversation Manager with Advanced Memory and Session Management.
Optimized for multi-agent voice conversations with persistent context.
"""
import asyncio
import logging
import time
import json
import base64
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib

from app.core.orchestrator import MultiAgentOrchestrator
from app.config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Enhanced conversation states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    WAITING_FOR_INPUT = "waiting_for_input"
    PROCESSING = "processing"
    AGENT_ROUTING = "agent_routing"
    RETRIEVING_CONTEXT = "retrieving_context"
    GENERATING_RESPONSE = "generating_response"
    SENDING_AUDIO = "sending_audio"
    PAUSED = "paused"
    ESCALATED = "escalated"
    ENDED = "ended"
    ERROR = "error"

class ConversationContext(Enum):
    """Types of conversation context."""
    USER_PROFILE = "user_profile"
    CONVERSATION_HISTORY = "conversation_history"
    CURRENT_INTENT = "current_intent"
    AGENT_CONTEXT = "agent_context"
    BUSINESS_CONTEXT = "business_context"
    TECHNICAL_CONTEXT = "technical_context"

@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    turn_id: str
    timestamp: float
    user_input: str
    agent_response: str
    agent_used: str
    confidence_score: float
    processing_time: float
    context_used: List[Dict[str, Any]] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    user_satisfied: Optional[bool] = None
    escalation_requested: bool = False

@dataclass
class UserProfile:
    """Enhanced user profile with learning capabilities."""
    user_id: str
    phone_number: str
    preferred_agent_types: List[str] = field(default_factory=list)
    interaction_history: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    context_memory: Dict[str, Any] = field(default_factory=dict)
    satisfaction_scores: List[float] = field(default_factory=list)
    escalation_triggers: List[str] = field(default_factory=list)
    
    def update_preferences(self, agent_type: str, satisfaction: float):
        """Update user preferences based on interaction."""
        if agent_type not in self.preferred_agent_types and satisfaction > 0.8:
            self.preferred_agent_types.append(agent_type)
        
        self.satisfaction_scores.append(satisfaction)
        
        # Keep only recent scores (last 50)
        if len(self.satisfaction_scores) > 50:
            self.satisfaction_scores = self.satisfaction_scores[-50:]
    
    @property
    def average_satisfaction(self) -> float:
        """Calculate average satisfaction score."""
        return sum(self.satisfaction_scores) / len(self.satisfaction_scores) if self.satisfaction_scores else 0.0

@dataclass
class ConversationSession:
    """Enhanced conversation session with comprehensive tracking."""
    session_id: str
    call_sid: str
    user_profile: UserProfile
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # State management
    current_state: ConversationState = ConversationState.INITIALIZING
    current_agent: Optional[str] = None
    
    # Conversation tracking
    turns: List[ConversationTurn] = field(default_factory=list)
    context_stack: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Performance tracking
    total_processing_time: float = 0.0
    avg_response_time: float = 0.0
    agent_switches: int = 0
    escalations: int = 0
    
    # Quality tracking
    user_satisfaction_indicators: List[str] = field(default_factory=list)
    conversation_quality_score: float = 0.0
    
    # Memory management
    short_term_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn and update metrics."""
        self.turns.append(turn)
        
        # Update performance metrics
        self.total_processing_time += turn.processing_time
        self.avg_response_time = self.total_processing_time / len(self.turns)
        
        # Track agent switches
        if self.current_agent and self.current_agent != turn.agent_used:
            self.agent_switches += 1
        self.current_agent = turn.agent_used
        
        # Track escalations
        if turn.escalation_requested:
            self.escalations += 1
        
        # Update context stack
        self.context_stack.append({
            "turn_id": turn.turn_id,
            "user_input": turn.user_input,
            "agent_response": turn.agent_response,
            "timestamp": turn.timestamp,
            "agent": turn.agent_used
        })
    
    @property
    def duration(self) -> float:
        """Get session duration."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def turn_count(self) -> int:
        """Get number of conversation turns."""
        return len(self.turns)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of conversation context."""
        return {
            "session_duration": self.duration,
            "turn_count": self.turn_count,
            "current_agent": self.current_agent,
            "agent_switches": self.agent_switches,
            "avg_response_time": self.avg_response_time,
            "recent_topics": [turn["user_input"][:50] for turn in list(self.context_stack)[-3:]],
            "user_satisfaction": self.user_profile.average_satisfaction
        }

class EnhancedConversationManager:
    """
    Advanced conversation manager with intelligent memory, context awareness,
    and multi-agent session coordination.
    
    Features:
    - Persistent user profiles with learning
    - Context-aware conversation routing
    - Advanced memory management (short-term + long-term)
    - Quality scoring and satisfaction tracking
    - Intelligent escalation detection
    - Performance optimization
    """
    
    def __init__(self, orchestrator: MultiAgentOrchestrator, performance_tracker=None):
        """Initialize the enhanced conversation manager."""
        self.orchestrator = orchestrator
        self.performance_tracker = performance_tracker
        
        # Session management
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.user_profiles: Dict[str, UserProfile] = {}  # phone_number -> profile
        
        # Memory systems
        self.conversation_memory = {}  # Long-term conversation storage
        self.context_cache = {}       # Frequently accessed context
        
        # Quality and satisfaction tracking
        self.satisfaction_indicators = {
            "positive": ["thank you", "thanks", "great", "perfect", "excellent", "helpful"],
            "negative": ["frustrated", "unhelpful", "wrong", "terrible", "useless", "bad"],
            "escalation": ["manager", "supervisor", "human", "person", "transfer", "escalate"]
        }
        
        # Performance optimization
        self.response_cache = {}  # Cache for common responses
        self.context_preload_queue = asyncio.Queue()
        
        # Analytics tracking
        self.session_metrics = defaultdict(list)
        
        logger.info("EnhancedConversationManager initialized")
    
    async def init(self):
        """Initialize the conversation manager."""
        logger.info("Initializing enhanced conversation manager...")
        
        # Start background tasks
        asyncio.create_task(self._context_preloader())
        asyncio.create_task(self._quality_analyzer())
        asyncio.create_task(self._memory_consolidator())
        
        logger.info("✅ Enhanced conversation manager ready")
    
    async def create_session(
        self,
        session_id: str,
        call_sid: str,
        phone_number: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        logger.info(f"🆕 Creating session {session_id} for call {call_sid}")
        
        # Get or create user profile
        user_profile = await self._get_or_create_user_profile(phone_number, user_context)
        
        # Create session
        session = ConversationSession(
            session_id=session_id,
            call_sid=call_sid,
            user_profile=user_profile
        )
        
        # Initialize session memory
        await self._initialize_session_memory(session)
        
        # Store session
        self.active_sessions[session_id] = session
        
        # Pre-load likely contexts
        await self._preload_context_for_session(session)
        
        logger.info(f"✅ Session {session_id} created and initialized")
        return session
    
    async def process_audio(
        self,
        session_id: str,
        audio_payload: str
    ) -> Dict[str, Any]:
        """Process audio input through the conversation pipeline."""
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        processing_start_time = time.time()
        
        try:
            # Update session state
            session.current_state = ConversationState.PROCESSING
            
            # Decode audio
            audio_data = base64.b64decode(audio_payload)
            
            # Process through STT
            stt_start = time.time()
            user_input = await self._process_stt(audio_data, session)
            stt_time = time.time() - stt_start
            
            if not user_input or len(user_input.strip()) < 2:
                return {"status": "no_speech_detected"}
            
            logger.info(f"🎤 [{session_id}] User said: '{user_input}'")
            
            # Analyze user sentiment and intent
            sentiment_analysis = await self._analyze_user_sentiment(user_input, session)
            
            # Check for escalation triggers
            escalation_needed = await self._check_escalation_triggers(user_input, session)
            
            # Enhance context with session memory
            enhanced_context = await self._build_enhanced_context(session, user_input)
            
            # Process through orchestrator with enhanced context
            orchestrator_start = time.time()
            response = await self.orchestrator.process_conversation(
                user_input=user_input,
                session_id=session_id,
                call_sid=session.call_sid,
                audio_data=audio_data,
                user_context=enhanced_context
            )
            orchestrator_time = time.time() - orchestrator_start
            
            # Create conversation turn
            turn_id = f"{session_id}_{len(session.turns) + 1}"
            turn = ConversationTurn(
                turn_id=turn_id,
                timestamp=time.time(),
                user_input=user_input,
                agent_response=response.get("response", ""),
                agent_used=response.get("agent_used", "unknown"),
                confidence_score=response.get("confidence", 0.0),
                processing_time=time.time() - processing_start_time,
                context_used=response.get("sources", []),
                tools_called=response.get("tools_called", []),
                escalation_requested=escalation_needed
            )
            
            # Add turn to session
            session.add_turn(turn)
            
            # Update session memory
            await self._update_session_memory(session, turn, sentiment_analysis)
            
            # Update user profile
            await self._update_user_profile(session, turn, sentiment_analysis)
            
            # Track performance metrics
            if self.performance_tracker:
                await self.performance_tracker.track_conversation_turn(
                    session_id=session_id,
                    processing_time=orchestrator_time,
                    stt_time=stt_time,
                    agent_used=turn.agent_used,
                    user_satisfaction=sentiment_analysis.get("satisfaction_score", 0.5)
                )
            
            # Prepare response
            result = {
                "response": response.get("response"),
                "audio_response": response.get("audio"),
                "agent_used": turn.agent_used,
                "confidence": turn.confidence_score,
                "processing_time": turn.processing_time,
                "escalation_needed": escalation_needed,
                "sentiment": sentiment_analysis,
                "sources": response.get("sources", [])
            }
            
            # Update session state
            session.current_state = ConversationState.WAITING_FOR_INPUT
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing audio for session {session_id}: {e}", exc_info=True)
            session.current_state = ConversationState.ERROR
            
            return {
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing your request. Could you please try again?",
                "processing_time": time.time() - processing_start_time
            }
    
    async def _process_stt(self, audio_data: bytes, session: ConversationSession) -> str:
        """Process audio through STT with session context."""
        if not self.orchestrator.stt:
            return ""
        
        try:
            # Use STT with context awareness
            result = await self.orchestrator.stt.transcribe_audio_data(
                audio_data=audio_data,
                context={
                    "session_id": session.session_id,
                    "user_profile": session.user_profile,
                    "conversation_context": session.get_context_summary()
                }
            )
            
            return result.get("transcription", "") if isinstance(result, dict) else ""
            
        except Exception as e:
            logger.error(f"STT processing error: {e}")
            return ""
    
    async def _analyze_user_sentiment(
        self,
        user_input: str,
        session: ConversationSession
    ) -> Dict[str, Any]:
        """Analyze user sentiment and satisfaction indicators."""
        sentiment_analysis = {
            "satisfaction_score": 0.5,  # Neutral baseline
            "sentiment": "neutral",
            "indicators": [],
            "escalation_risk": 0.0
        }
        
        user_input_lower = user_input.lower()
        
        # Check for satisfaction indicators
        positive_count = sum(1 for word in self.satisfaction_indicators["positive"] if word in user_input_lower)
        negative_count = sum(1 for word in self.satisfaction_indicators["negative"] if word in user_input_lower)
        escalation_count = sum(1 for word in self.satisfaction_indicators["escalation"] if word in user_input_lower)
        
        # Calculate satisfaction score
        if positive_count > negative_count:
            sentiment_analysis["satisfaction_score"] = min(1.0, 0.5 + (positive_count * 0.2))
            sentiment_analysis["sentiment"] = "positive"
        elif negative_count > positive_count:
            sentiment_analysis["satisfaction_score"] = max(0.0, 0.5 - (negative_count * 0.2))
            sentiment_analysis["sentiment"] = "negative"
        
        # Calculate escalation risk
        if escalation_count > 0 or negative_count > 2:
            sentiment_analysis["escalation_risk"] = min(1.0, (escalation_count * 0.3) + (negative_count * 0.1))
        
        # Add context from conversation history
        if len(session.turns) > 0:
            recent_satisfaction = [turn.user_satisfied for turn in session.turns[-3:] if turn.user_satisfied is not None]
            if recent_satisfaction:
                avg_recent = sum(recent_satisfaction) / len(recent_satisfaction)
                sentiment_analysis["satisfaction_score"] = (sentiment_analysis["satisfaction_score"] + avg_recent) / 2
        
        return sentiment_analysis
    
    async def _check_escalation_triggers(
        self,
        user_input: str, 
        session: ConversationSession
    ) -> bool:
        """Check if conversation should be escalated."""
        # Direct escalation requests
        escalation_words = self.satisfaction_indicators["escalation"]
        if any(word in user_input.lower() for word in escalation_words):
            return True
        
        # Pattern-based escalation detection
        escalation_patterns = [
            "not helping",
            "doesn't work",
            "still broken",
            "third time",
            "already tried",
            "speak to someone",
            "this is ridiculous"
        ]
        
        if any(pattern in user_input.lower() for pattern in escalation_patterns):
            return True
        
        # Session-based escalation triggers
        if len(session.turns) >= 5:  # Long conversation
            recent_satisfaction = [turn.user_satisfied for turn in session.turns[-3:] if turn.user_satisfied is not None]
            if recent_satisfaction and sum(recent_satisfaction) / len(recent_satisfaction) < 0.3:
                return True
        
        # Agent switching too frequently
        if session.agent_switches >= 3:
            return True
        
        return False
    
    async def _build_enhanced_context(
        self,
        session: ConversationSession,
        current_input: str
    ) -> Dict[str, Any]:
        """Build enhanced context for the orchestrator."""
        context = {
            "user_profile": {
                "preferred_agents": session.user_profile.preferred_agent_types,
                "interaction_history": session.user_profile.interaction_history,
                "satisfaction_history": session.user_profile.satisfaction_scores,
                "average_satisfaction": session.user_profile.average_satisfaction
            },
            "session_context": {
                "duration": session.duration,
                "turn_count": session.turn_count,
                "current_agent": session.current_agent,
                "agent_switches": session.agent_switches,
                "recent_topics": [turn["user_input"] for turn in list(session.context_stack)[-3:]]
            },
            "conversation_history": [
                {
                    "user": turn.user_input,
                    "assistant": turn.agent_response,
                    "agent": turn.agent_used,
                    "timestamp": turn.timestamp
                }
                for turn in session.turns[-5:]  # Last 5 turns
            ],
            "memory_context": {
                "short_term": session.short_term_memory,
                "long_term": session.long_term_memory,
                "user_preferences": session.user_profile.preferences
            }
        }
        
        return context
    
    async def _initialize_session_memory(self, session: ConversationSession):
        """Initialize memory systems for a new session."""
        # Load user's long-term memory
        user_id = session.user_profile.user_id
        if user_id in self.conversation_memory:
            session.long_term_memory = self.conversation_memory[user_id].copy()
        
        # Initialize short-term memory
        session.short_term_memory = {
            "session_start": session.start_time,
            "initial_context": {},
            "recurring_topics": [],
            "user_corrections": []
        }
    
    async def _update_session_memory(
        self,
        session: ConversationSession,
        turn: ConversationTurn,
        sentiment_analysis: Dict[str, Any]
    ):
        """Update session memory with new information."""
        # Update short-term memory
        session.short_term_memory.update({
            "last_user_input": turn.user_input,
            "last_agent_response": turn.agent_response,
            "last_agent_used": turn.agent_used,
            "last_satisfaction": sentiment_analysis.get("satisfaction_score", 0.5),
            "current_topic": await self._extract_topic(turn.user_input)
        })
        
        # Track recurring topics
        topic = await self._extract_topic(turn.user_input)
        if topic:
            recurring_topics = session.short_term_memory.get("recurring_topics", [])
            recurring_topics.append(topic)
            session.short_term_memory["recurring_topics"] = recurring_topics[-10:]  # Keep last 10
        
        # Update long-term memory for important information
        if turn.confidence_score > 0.8 or sentiment_analysis.get("satisfaction_score", 0) > 0.8:
            session.long_term_memory[f"successful_interaction_{len(session.turns)}"] = {
                "user_input": turn.user_input,
                "agent_used": turn.agent_used,
                "tools_used": turn.tools_called,
                "satisfaction": sentiment_analysis.get("satisfaction_score"),
                "timestamp": turn.timestamp
            }
    
    async def _update_user_profile(
        self,
        session: ConversationSession,
        turn: ConversationTurn,
        sentiment_analysis: Dict[str, Any]
    ):
        """Update user profile based on interaction."""
        profile = session.user_profile
        
        # Update preferences based on successful interactions
        satisfaction = sentiment_analysis.get("satisfaction_score", 0.5)
        if satisfaction > 0.7:
            profile.update_preferences(turn.agent_used, satisfaction)
        
        # Update interaction history
        profile.interaction_history[session.session_id] = {
            "call_sid": session.call_sid,
            "duration": session.duration,
            "turns": session.turn_count,
            "primary_agent": session.current_agent,
            "satisfaction": satisfaction,
            "escalated": turn.escalation_requested
        }
        
        # Update context memory
        topic = await self._extract_topic(turn.user_input)
        if topic and satisfaction > 0.6:
            profile.context_memory[topic] = {
                "successful_agent": turn.agent_used,
                "last_interaction": turn.timestamp,
                "success_count": profile.context_memory.get(topic, {}).get("success_count", 0) + 1
            }
    
    async def _extract_topic(self, user_input: str) -> Optional[str]:
        """Extract main topic from user input."""
        # Simple topic extraction - in production, use NLP models
        topics = {
            "roadside": ["tow", "stuck", "breakdown", "flat tire", "accident", "jump start"],
            "billing": ["bill", "payment", "charge", "refund", "account", "invoice"],
            "technical": ["not working", "broken", "error", "problem", "fix", "support"]
        }
        
        user_input_lower = user_input.lower()
        for topic, keywords in topics.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return topic
        
        return None
    
    async def _get_or_create_user_profile(
        self,
        phone_number: Optional[str],
        user_context: Optional[Dict[str, Any]]
    ) -> UserProfile:
        """Get existing user profile or create a new one."""
        if not phone_number:
            # Create anonymous profile
            user_id = f"anonymous_{int(time.time())}"
            return UserProfile(user_id=user_id, phone_number="anonymous")
        
        if phone_number in self.user_profiles:
            profile = self.user_profiles[phone_number]
            # Update with any new context
            if user_context:
                profile.preferences.update(user_context.get("preferences", {}))
            return profile
        
        # Create new profile
        user_id = hashlib.md5(phone_number.encode()).hexdigest()[:12]
        profile = UserProfile(
            user_id=user_id,
            phone_number=phone_number,
            preferences=user_context.get("preferences", {}) if user_context else {}
        )
        
        self.user_profiles[phone_number] = profile
        return profile
    
    async def _preload_context_for_session(self, session: ConversationSession):
        """Pre-load likely context for faster responses."""
        # Pre-load context based on user history
        preferred_agents = session.user_profile.preferred_agent_types
        
        for agent_type in preferred_agents[:2]:  # Top 2 preferred agents
            try:
                # Warm up agent-specific context
                await self.context_preload_queue.put({
                    "session_id": session.session_id,
                    "agent_type": agent_type,
                    "user_profile": session.user_profile
                })
            except Exception as e:
                logger.error(f"Error preloading context: {e}")
    
    async def _context_preloader(self):
        """Background task to preload context."""
        while True:
            try:
                # Get preload request
                request = await self.context_preload_queue.get()
                
                # Preload context (implementation depends on vector store)
                if self.orchestrator.vector_store:
                    await self.orchestrator.vector_store.warm_agent_cache(
                        agent_id=request["agent_type"],
                        user_context=request["user_profile"]
                    )
                
                self.context_preload_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in context preloader: {e}")
                await asyncio.sleep(1)
    
    async def _quality_analyzer(self):
        """Background task to analyze conversation quality."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Analyze recent sessions for quality patterns
                for session in self.active_sessions.values():
                    if len(session.turns) > 0:
                        await self._analyze_session_quality(session)
                
            except Exception as e:
                logger.error(f"Error in quality analyzer: {e}")
    
    async def _analyze_session_quality(self, session: ConversationSession):
        """Analyze quality of a conversation session."""
        if len(session.turns) == 0:
            return
        
        # Calculate quality metrics
        avg_confidence = sum(turn.confidence_score for turn in session.turns) / len(session.turns)
        avg_processing_time = sum(turn.processing_time for turn in session.turns) / len(session.turns)
        
        # User satisfaction indicators
        satisfaction_scores = [turn.user_satisfied for turn in session.turns if turn.user_satisfied is not None]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.5
        
        # Calculate overall quality score
        quality_score = (
            (avg_confidence * 0.3) +
            (avg_satisfaction * 0.4) +
            (min(1.0, 2.0 / avg_processing_time) * 0.2) +  # Faster = better (up to 2s)
            ((1.0 - min(1.0, session.agent_switches / 5.0)) * 0.1)  # Fewer switches = better
        )
        
        session.conversation_quality_score = quality_score
        
        # Log quality insights
        if quality_score < 0.6:
            logger.warning(f"🔍 Low quality session {session.session_id}: score={quality_score:.2f}")
    
    async def _memory_consolidator(self):
        """Background task to consolidate memories."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Consolidate session memories into long-term storage
                for session in list(self.active_sessions.values()):
                    if session.duration > 300:  # Sessions longer than 5 minutes
                        await self._consolidate_session_memory(session)
                
            except Exception as e:
                logger.error(f"Error in memory consolidator: {e}")
    
    async def _consolidate_session_memory(self, session: ConversationSession):
        """Consolidate session memory into long-term storage."""
        user_id = session.user_profile.user_id
        
        # Consolidate important interactions
        important_turns = [
            turn for turn in session.turns
            if turn.confidence_score > 0.8 or turn.user_satisfied and turn.user_satisfied > 0.8
        ]
        
        if important_turns:
            if user_id not in self.conversation_memory:
                self.conversation_memory[user_id] = {}
            
            # Store successful patterns
            for turn in important_turns:
                pattern_key = f"successful_{turn.agent_used}_{await self._extract_topic(turn.user_input)}"
                self.conversation_memory[user_id][pattern_key] = {
                    "user_input_pattern": turn.user_input[:100],  # First 100 chars
                    "successful_response": turn.agent_response[:200],  # First 200 chars
                    "agent_used": turn.agent_used,
                    "tools_used": turn.tools_called,
                    "confidence": turn.confidence_score,
                    "timestamp": turn.timestamp
                }
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get detailed status of a conversation session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "call_sid": session.call_sid,
            "state": session.current_state.value,
            "duration": session.duration,
            "turn_count": session.turn_count,
            "current_agent": session.current_agent,
            "agent_switches": session.agent_switches,
            "avg_response_time": session.avg_response_time,
            "quality_score": session.conversation_quality_score,
            "user_satisfaction": session.user_profile.average_satisfaction,
            "escalations": session.escalations,
            "context_summary": session.get_context_summary()
        }
    
    async def cleanup_session(self, session_id: str):
        """Clean up a completed session."""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Final memory consolidation
        await self._consolidate_session_memory(session)
        
        # Update user profile in persistent storage
        user_id = session.user_profile.user_id
        self.user_profiles[session.user_profile.phone_number] = session.user_profile
        
        # Store session analytics
        self.session_metrics["completed_sessions"].append({
            "session_id": session_id,
            "duration": session.duration,
            "turn_count": session.turn_count,
            "quality_score": session.conversation_quality_score,
            "user_satisfaction": session.user_profile.average_satisfaction,
            "completion_time": time.time()
        })
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"🧹 Cleaned up session {session_id} (duration: {session.duration:.1f}s, turns: {session.turn_count})")
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary across all conversations."""
        active_count = len(self.active_sessions)
        completed_sessions = self.session_metrics.get("completed_sessions", [])
        
        if not completed_sessions:
            return {
                "active_sessions": active_count,
                "total_completed": 0,
                "avg_duration": 0.0,
                "avg_quality": 0.0,
                "user_profiles": len(self.user_profiles)
            }
        
        # Calculate averages
        avg_duration = sum(s["duration"] for s in completed_sessions) / len(completed_sessions)
        avg_quality = sum(s["quality_score"] for s in completed_sessions) / len(completed_sessions)
        avg_satisfaction = sum(s["user_satisfaction"] for s in completed_sessions) / len(completed_sessions)
        
        return {
            "active_sessions": active_count,
            "total_completed": len(completed_sessions),
            "avg_duration": avg_duration,
            "avg_quality": avg_quality,
            "avg_satisfaction": avg_satisfaction,
            "user_profiles": len(self.user_profiles),
            "total_users": len(set(profile.user_id for profile in self.user_profiles.values()))
        }
    
    async def health_check(self) -> bool:
        """Health check for the conversation manager."""
        try:
            # Check if orchestrator is healthy
            if not await self.orchestrator.health_check():
                return False
            
            # Check active sessions are not stuck
            current_time = time.time()
            stuck_sessions = [
                s for s in self.active_sessions.values()
                if current_time - s.start_time > 3600 and len(s.turns) == 0  # No activity for 1 hour
            ]
            
            if len(stuck_sessions) > len(self.active_sessions) * 0.5:  # More than 50% stuck
                return False
            
            # Check memory systems
            if len(self.user_profiles) > 10000:  # Memory leak check
                logger.warning("User profiles growing large, may need cleanup")
            
            return True
            
        except Exception as e:
            logger.error(f"Conversation manager health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the conversation manager gracefully."""
        logger.info("🛑 Shutting down enhanced conversation manager...")
        
        # Complete all active sessions
        for session_id, session in list(self.active_sessions.items()):
            try:
                session.end_time = time.time()
                session.current_state = ConversationState.ENDED
                await self.cleanup_session(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
        
        # Save user profiles (in production, this would go to persistent storage)
        logger.info(f"📊 Saving {len(self.user_profiles)} user profiles...")
        
        # Save conversation analytics
        total_sessions = len(self.session_metrics.get("completed_sessions", []))
        logger.info(f"📈 Processed {total_sessions} total sessions")
        
        logger.info("✅ Enhanced conversation manager shutdown complete")
