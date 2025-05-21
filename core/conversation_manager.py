# core/conversation_manager.py

"""
Enhanced conversation manager with state tracking and context management.
"""
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING
import uuid
import re

from core.state_manager import StateManager, ConversationState
from core.session_manager import SessionManager

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from knowledge_base.query_engine import QueryEngine
    from prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Enhanced conversation manager for complex dialogs.
    
    Features:
    1. State management
    2. Context tracking
    3. Session handling
    4. Memory management
    5. Response generation
    """
    
    def __init__(
        self,
        query_engine: 'QueryEngine',
        prompt_manager: Optional['PromptManager'] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conversation manager.
        
        Args:
            query_engine: Knowledge base query engine
            prompt_manager: Prompt management system (optional)
            config: Optional configuration
        """
        self.query_engine = query_engine
        self.prompt_manager = prompt_manager
        self.config = config or {}
        
        # Initialize managers
        self.state_manager = StateManager()
        self.session_manager = SessionManager()
        
        # Conversation tracking
        self.conversation_history: List[Dict[str, Any]] = []
        self.collected_info: Dict[str, Any] = {}
        self.confirmed_info: Set[str] = set()
        
        # Context window management
        self.context_window_size = self.config.get("context_window_size", 5)
        self.context_importance = {
            "customer_info": 1.0,
            "service_details": 0.9,
            "location_info": 0.8,
            "previous_responses": 0.7
        }
        
        # Session ID
        self.session_id = str(uuid.uuid4())
        
        # Performance tracking
        self.start_time = time.time()
        self.total_turns = 0
        self.response_times: List[float] = []
        
        logger.info(f"Initialized conversation manager for session {self.session_id}")
    
    async def init(self):
        """Initialize async components if needed."""
        # This method can be used for async init tasks
        # Currently we don't need it but it's here for API consistency
        logger.debug(f"Conversation manager init called for session {self.session_id}")
    
    async def cleanup(self):
        """Clean up any resources if needed."""
        # This is a placeholder for future cleanup needs
        logger.debug(f"Conversation manager cleanup called for session {self.session_id}")
    
    async def process_message(
        self,
        message: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process incoming message with context management.
        
        Args:
            message: User message
            session_id: Optional session identifier
            
        Returns:
            Response with context
        """
        start_time = time.time()
        self.total_turns += 1
        
        try:
            # Get or create session
            session = self.session_manager.get_session(session_id or self.session_id)
            if not session:
                session = self.session_manager.create_session(
                    session_id or self.session_id,
                    metadata={
                        "start_time": time.time(),
                        "source": "voice"
                    }
                )
            
            # Update context window
            context = self._build_context(session, message)
            
            # Get appropriate prompt if prompt manager is available
            prompt = None
            if self.prompt_manager and hasattr(session, 'agent_type'):
                prompt = self.prompt_manager.get_prompt(
                    agent_type=session.get("agent_type"),
                    state=self.state_manager.current_state,
                    collected_info=self.collected_info
                )
            
            # Generate response
            response = await self.query_engine.query(
                text=message,
                context=context,
                prompt=prompt
            )
            
            # Extract any new information
            new_info = self._extract_information(message, response)
            if new_info:
                self.collected_info.update(new_info)
                
                # Update confirmed info if high confidence
                if response.get("confidence", 0) > 0.9:
                    self.confirmed_info.update(new_info.keys())
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": time.time()
            })
            
            if response.get("response"):
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response["response"],
                    "timestamp": time.time()
                })
            
            # Determine next state
            next_state = self._determine_next_state(
                current_state=self.state_manager.current_state,
                collected_info=self.collected_info,
                response=response
            )
            
            # Update state if changed
            if next_state != self.state_manager.current_state:
                self.state_manager.update_state(next_state)
            
            # Track response time
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Prepare final response
            final_response = {
                **response,
                "state": self.state_manager.current_state,
                "collected_info": self.collected_info,
                "confirmed_info": list(self.confirmed_info),
                "response_time": response_time,
                "turn_count": self.total_turns
            }
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing your request.",
                "state": self.state_manager.current_state,
                "requires_handoff": True
            }
    
    def _build_context(
        self,
        session: Dict[str, Any],
        current_message: str
    ) -> str:
        """
        Build context window for response generation.
        
        Args:
            session: Session information
            current_message: Current message
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add current state context
        context_parts.append(f"Current state: {self.state_manager.current_state}")
        
        # Add collected information
        if self.collected_info:
            info_parts = []
            for key, value in self.collected_info.items():
                if key in self.confirmed_info:
                    info_parts.append(f"{key}: {value} (confirmed)")
                else:
                    info_parts.append(f"{key}: {value}")
            context_parts.append("Collected information:\n" + "\n".join(info_parts))
        
        # Add recent conversation history
        recent_history = self.conversation_history[-self.context_window_size:]
        if recent_history:
            history_parts = []
            for turn in recent_history:
                role = "Customer" if turn["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {turn['content']}")
            context_parts.append("Recent conversation:\n" + "\n".join(history_parts))
        
        # Add session-specific context
        if "agent_type" in session:
            context_parts.append(f"Service type: {session['agent_type']}")
        if "service_requirements" in session:
            context_parts.append(
                "Service requirements:\n" + 
                "\n".join(f"- {req}" for req in session["service_requirements"])
            )
        
        return "\n\n".join(context_parts)
    
    def _extract_information(
        self,
        message: str,
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract relevant information from message and response.
        
        Args:
            message: User message
            response: Generated response
            
        Returns:
            Dictionary of extracted information
        """
        # Basic implementation - specialized agents should override this
        extracted_info = {}
        
        # Extract phone numbers
        phone_match = re.search(
            r'(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})',
            message
        )
        if phone_match:
            extracted_info["phone_number"] = "".join(phone_match.groups())
        
        # Extract names (simple implementation)
        name_match = re.search(
            r'(?:my name is|this is) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
            message
        )
        if name_match:
            extracted_info["customer_name"] = name_match.group(1)
        
        return extracted_info
    
    def _determine_next_state(
        self,
        current_state: ConversationState,
        collected_info: Dict[str, Any],
        response: Dict[str, Any]
    ) -> ConversationState:
        """
        Determine next conversation state.
        
        Args:
            current_state: Current state
            collected_info: Collected information
            response: Generated response
            
        Returns:
            Next conversation state
        """
        # Handle explicit state changes from response
        if "next_state" in response:
            return ConversationState(response["next_state"])
        
        # Handle handoff requests
        if response.get("requires_handoff"):
            return ConversationState.HANDOFF
        
        # State transition logic
        if current_state == ConversationState.GREETING:
            # Move to collecting information
            if not collected_info.get("customer_name"):
                return ConversationState.COLLECTING_NAME
            elif not collected_info.get("phone_number"):
                return ConversationState.COLLECTING_PHONE
            else:
                return ConversationState.COLLECTING_LOCATION
                
        elif current_state == ConversationState.COLLECTING_NAME:
            if collected_info.get("customer_name"):
                return ConversationState.COLLECTING_PHONE
                
        elif current_state == ConversationState.COLLECTING_PHONE:
            if collected_info.get("phone_number"):
                return ConversationState.COLLECTING_LOCATION
                
        elif current_state == ConversationState.COLLECTING_LOCATION:
            if collected_info.get("location"):
                return ConversationState.COLLECTING_VEHICLE
                
        elif current_state == ConversationState.COLLECTING_VEHICLE:
            if all(key in collected_info for key in ["vehicle_make", "vehicle_model"]):
                return ConversationState.CONFIRMING_SERVICE
                
        elif current_state == ConversationState.CONFIRMING_SERVICE:
            if response.get("service_confirmed"):
                return ConversationState.PROVIDING_PRICE
        
        # Default to staying in current state
        return current_state
    
    async def get_context(self) -> str:
        """Get current conversation context."""
        return self._build_context(
            session=self.session_manager.get_session(self.session_id) or {},
            current_message=""
        )
    
    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
    
    def reset(self):
        """Reset conversation state."""
        self.state_manager.clear_state()
        self.conversation_history.clear()
        self.collected_info.clear()
        self.confirmed_info.clear()
        self.total_turns = 0
        logger.info(f"Reset conversation state for session {self.session_id}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        current_time = time.time()
        
        stats = {
            "session_id": self.session_id,
            "duration": current_time - self.start_time,
            "total_turns": self.total_turns,
            "current_state": self.state_manager.current_state,
            "collected_info_count": len(self.collected_info),
            "confirmed_info_count": len(self.confirmed_info),
            "conversation_length": len(self.conversation_history)
        }
        
        # Add response time stats
        if self.response_times:
            stats.update({
                "avg_response_time": sum(self.response_times) / len(self.response_times),
                "max_response_time": max(self.response_times),
                "min_response_time": min(self.response_times)
            })
        
        # Add state manager stats
        stats["state_manager"] = self.state_manager.get_stats()
        
        # Add session manager stats
        stats["session_manager"] = self.session_manager.get_stats()
        
        return stats