# agents/base_agent.py

"""
Base agent class providing core functionality for all specialized agents.
"""
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum
import logging
import asyncio
import time

from core.state_manager import ConversationState

# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    from core.conversation_manager import ConversationManager
    from knowledge_base.query_engine import QueryEngine
    from prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class AgentType(str, Enum):
    """Types of available agents."""
    TOWING = "towing"
    TIRE = "tire"
    JUMP_START = "jump_start"
    DISPATCHER = "dispatcher"
    STATUS = "status"

class BaseAgent:
    """Base agent providing common functionality for all agents."""
    
    def __init__(
        self,
        agent_type: AgentType,
        conversation_manager: 'ConversationManager',
        query_engine: 'QueryEngine',
        prompt_manager: 'PromptManager',
        **kwargs
    ):
        """
        Initialize base agent.
        
        Args:
            agent_type: Type of agent
            conversation_manager: Conversation manager instance
            query_engine: Query engine for RAG
            prompt_manager: Prompt manager for response generation
            **kwargs: Additional parameters
        """
        self.agent_type = agent_type
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.prompt_manager = prompt_manager
        
        # Information tracking
        self.collected_info = {}
        self.current_state = ConversationState.GREETING
        self.required_fields = self._get_required_fields()
        
        # Session tracking
        self.session_start = time.time()
        self.last_interaction = time.time()
        self.interaction_count = 0
        
        # Stats tracking
        self.total_queries = 0
        self.successful_responses = 0
        self.handoffs = 0
        
        logger.info(f"Initialized {agent_type} agent")
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for this agent type."""
        # Base required fields, override in specialized agents
        return [
            "customer_name",
            "phone_number",
            "location"
        ]
    
    def _validate_info(self, info: Dict[str, Any]) -> bool:
        """
        Validate collected information.
        
        Args:
            info: Information to validate
            
        Returns:
            True if all required fields are present and valid
        """
        for field in self.required_fields:
            if field not in info or not info[field]:
                return False
        return True
    
    async def handle_message(self, message: str) -> Dict[str, Any]:
        """
        Handle incoming message with appropriate response.
        
        Args:
            message: User message
            
        Returns:
            Response dictionary with agent actions
        """
        self.last_interaction = time.time()
        self.interaction_count += 1
        self.total_queries += 1
        
        try:
            # Update conversation with latest message
            context = await self.conversation_manager.get_context()
            
            # Get appropriate prompt for current state
            prompt = self.prompt_manager.get_prompt(
                self.agent_type,
                self.current_state,
                self.collected_info
            )
            
            # Generate response using RAG
            response = await self.query_engine.query(
                text=message,
                context=context,
                prompt=prompt
            )
            
            # Track successful response
            if response and response.get("response"):
                self.successful_responses += 1
            
            # Extract any new information
            new_info = await self._extract_information(message, response)
            if new_info:
                self.collected_info.update(new_info)
            
            # Check if we should update state
            new_state = self._determine_next_state()
            if new_state != self.current_state:
                logger.info(f"State transition: {self.current_state} -> {new_state}")
                self.current_state = new_state
            
            # Check if we need to hand off
            if self._should_handoff():
                self.handoffs += 1
                return await self._prepare_handoff()
            
            return self._prepare_response(response)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your request. Let me try to help you differently.",
                "error": str(e),
                "requires_handoff": True
            }
    
    async def _extract_information(
        self,
        message: str,
        response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract relevant information from message and response.
        
        Args:
            message: User message
            response: Generated response
            
        Returns:
            Dictionary of extracted information or None
        """
        # Basic implementation - override in specialized agents
        return {}
    
    def _determine_next_state(self) -> ConversationState:
        """Determine the next conversation state based on collected info."""
        # If we have all required info, move to confirmation
        if self._validate_info(self.collected_info):
            return ConversationState.CONFIRMING_SERVICE
            
        # Otherwise, look for the first missing required field
        for field in self.required_fields:
            if field not in self.collected_info or not self.collected_info[field]:
                return getattr(ConversationState, f"COLLECTING_{field.upper()}")
                
        return self.current_state
    
    def _should_handoff(self) -> bool:
        """Determine if we should hand off to a human dispatcher."""
        # Base conditions for handoff, override in specialized agents
        return (
            self.interaction_count >= 10 or  # Too many back-and-forth
            time.time() - self.session_start > 300  # Session too long (5 minutes)
        )
    
    async def _prepare_handoff(self) -> Dict[str, Any]:
        """Prepare handoff to human dispatcher."""
        return {
            "requires_handoff": True,
            "collected_info": self.collected_info,
            "conversation_history": await self.conversation_manager.get_conversation_history(),
            "current_state": self.current_state,
            "session_duration": time.time() - self.session_start,
            "interaction_count": self.interaction_count
        }
    
    def _prepare_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare final response with all necessary information."""
        return {
            "response": response.get("response", ""),
            "requires_handoff": False,
            "collected_info": self.collected_info,
            "current_state": self.current_state,
            "missing_fields": [
                field for field in self.required_fields
                if field not in self.collected_info or not self.collected_info[field]
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_type": self.agent_type,
            "session_duration": time.time() - self.session_start,
            "interaction_count": self.interaction_count,
            "total_queries": self.total_queries,
            "successful_responses": self.successful_responses,
            "handoffs": self.handoffs,
            "current_state": self.current_state,
            "collected_info_fields": list(self.collected_info.keys()),
            "missing_fields": [
                field for field in self.required_fields
                if field not in self.collected_info or not self.collected_info[field]
            ]
        }