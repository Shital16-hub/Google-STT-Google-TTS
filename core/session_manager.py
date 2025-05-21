# core/session_manager.py

"""
Session management for voice interactions.
"""
import logging
import time
from typing import Dict, Any, Optional, List
import uuid

from agents.base_agent import AgentType

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manage conversation sessions and state persistence.
    
    Features:
    1. Session creation and tracking
    2. State persistence
    3. Session cleanup
    4. Analytics tracking
    """
    
    def __init__(self, session_timeout: int = 3600):
        """
        Initialize session manager.
        
        Args:
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = session_timeout
        
        # Session tracking
        self.active_sessions: Dict[str, float] = {}  # session_id -> last_activity
        self.completed_sessions: List[Dict[str, Any]] = []
        
        # Analytics tracking
        self.total_sessions = 0
        self.active_count = 0
        self.completed_count = 0
        self.expired_count = 0
        
        logger.info("Initialized session manager")
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new session.
        
        Args:
            session_id: Optional session identifier
            metadata: Optional session metadata
            
        Returns:
            Session information
        """
        # Generate session ID if not provided
        session_id = session_id or str(uuid.uuid4())
        
        # Create session
        session = {
            "session_id": session_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "metadata": metadata or {},
            "state": {},
            "agent_type": None,
            "service_requirements": [],
            "customer_info": {},
            "conversation_history": []
        }
        
        # Store session
        self.sessions[session_id] = session
        self.active_sessions[session_id] = time.time()
        
        # Update stats
        self.total_sessions += 1
        self.active_count += 1
        
        logger.info(f"Created session {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information if found
        """
        if session_id not in self.sessions:
            return None
            
        # Check for timeout
        session = self.sessions[session_id]
        if self._is_session_expired(session):
            self._expire_session(session_id)
            return None
        
        # Update last activity
        session["last_activity"] = time.time()
        self.active_sessions[session_id] = time.time()
        
        return session
    
    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update session information.
        
        Args:
            session_id: Session identifier
            updates: Information to update
            
        Returns:
            True if session was updated
        """
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        
        # Apply updates
        for key, value in updates.items():
            if key == "state":
                session["state"].update(value)
            elif key == "metadata":
                session["metadata"].update(value)
            elif key == "customer_info":
                session["customer_info"].update(value)
            elif key == "service_requirements":
                if isinstance(value, list):
                    session["service_requirements"].extend(value)
                else:
                    session["service_requirements"].append(value)
            elif key == "conversation_history":
                if isinstance(value, list):
                    session["conversation_history"].extend(value)
                else:
                    session["conversation_history"].append(value)
            else:
                session[key] = value
        
        # Update last activity
        session["last_activity"] = time.time()
        self.active_sessions[session_id] = time.time()
        
        return True
    
    def end_session(
        self,
        session_id: str,
        completion_status: str = "completed"
    ):
        """
        End a session.
        
        Args:
            session_id: Session identifier
            completion_status: Status of completion
        """
        if session_id not in self.sessions:
            return
            
        # Get session
        session = self.sessions[session_id]
        
        # Add completion info
        session["completed_at"] = time.time()
        session["completion_status"] = completion_status
        session["duration"] = session["completed_at"] - session["created_at"]
        
        # Move to completed sessions
        self.completed_sessions.append(session)
        
        # Clean up
        del self.sessions[session_id]
        del self.active_sessions[session_id]
        
        # Update stats
        self.active_count -= 1
        self.completed_count += 1
        
        logger.info(f"Ended session {session_id} with status: {completion_status}")
    
    def _is_session_expired(self, session: Dict[str, Any]) -> bool:
        """Check if session has expired."""
        last_activity = session["last_activity"]
        return (time.time() - last_activity) > self.session_timeout
    
    def _expire_session(self, session_id: str):
        """Handle session expiration."""
        # End session with expired status
        self.end_session(session_id, completion_status="expired")
        
        # Update stats
        self.expired_count += 1
        
        logger.info(f"Session {session_id} expired")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        
        # Find expired sessions
        expired_sessions = [
            session_id for session_id, last_activity in self.active_sessions.items()
            if (current_time - last_activity) > self.session_timeout
        ]
        
        # Clean up each expired session
        for session_id in expired_sessions:
            self._expire_session(session_id)
    
    def get_session_state(
        self,
        session_id: str,
        key: str
    ) -> Optional[Any]:
        """
        Get specific state information from session.
        
        Args:
            session_id: Session identifier
            key: State key to retrieve
            
        Returns:
            State value if found
        """
        session = self.get_session(session_id)
        if not session:
            return None
            
        return session["state"].get(key)
    
    def set_session_state(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> bool:
        """
        Set specific state information in session.
        
        Args:
            session_id: Session identifier
            key: State key to set
            value: State value
            
        Returns:
            True if state was set
        """
        session = self.get_session(session_id)
        if not session:
            return False
            
        session["state"][key] = value
        return True
    
    def set_agent_type(
        self,
        session_id: str,
        agent_type: AgentType
    ) -> bool:
        """
        Set agent type for session.
        
        Args:
            session_id: Session identifier
            agent_type: Type of agent
            
        Returns:
            True if agent type was set
        """
        session = self.get_session(session_id)
        if not session:
            return False
            
        session["agent_type"] = agent_type
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        return {
            "total_sessions": self.total_sessions,
            "active_sessions": self.active_count,
            "completed_sessions": self.completed_count,
            "expired_sessions": self.expired_count,
            "average_duration": self._calculate_average_duration(),
            "completion_rate": self._calculate_completion_rate(),
            "session_types": self._count_session_types()
        }
    
    def _calculate_average_duration(self) -> float:
        """Calculate average session duration."""
        if not self.completed_sessions:
            return 0.0
            
        total_duration = sum(
            session["duration"] for session in self.completed_sessions
        )
        return total_duration / len(self.completed_sessions)
    
    def _calculate_completion_rate(self) -> float:
        """Calculate session completion rate."""
        total = self.completed_count + self.expired_count
        if total == 0:
            return 0.0
            
        return self.completed_count / total
    
    def _count_session_types(self) -> Dict[str, int]:
        """Count sessions by agent type."""
        counts = {}
        
        # Count active sessions
        for session in self.sessions.values():
            agent_type = session.get("agent_type")
            if agent_type:
                agent_type = agent_type.value
                if agent_type not in counts:
                    counts[agent_type] = 0
                counts[agent_type] += 1
        
        # Count completed sessions
        for session in self.completed_sessions:
            agent_type = session.get("agent_type")
            if agent_type:
                agent_type = agent_type.value
                if agent_type not in counts:
                    counts[agent_type] = 0
                counts[agent_type] += 1
        
        return counts