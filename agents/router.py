# agents/router.py

"""
Intent-based agent router for directing conversations to specialized agents.
"""
import logging
from typing import Dict, Any, Optional, Type, Union, TYPE_CHECKING
import asyncio

from agents.base_agent import BaseAgent, AgentType
from agents.towing_agent import TowingAgent
from agents.tire_agent import TireAgent
from agents.jump_start_agent import JumpStartAgent
from agents.dispatcher_agent import DispatcherAgent

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from core.conversation_manager import ConversationManager
    from knowledge_base.query_engine import QueryEngine
    from prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class AgentRouter:
    """Routes conversations to appropriate specialized agents based on intent."""
    
    def __init__(
        self,
        conversation_manager: 'ConversationManager',
        query_engine: 'QueryEngine',
        prompt_manager: 'PromptManager'
    ):
        """
        Initialize the agent router.
        
        Args:
            conversation_manager: Conversation manager instance
            query_engine: Query engine for RAG
            prompt_manager: Prompt manager for agents
        """
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.prompt_manager = prompt_manager
        
        # Map of agent types to their classes
        self.agent_classes: Dict[AgentType, Type[BaseAgent]] = {
            AgentType.TOWING: TowingAgent,
            AgentType.TIRE: TireAgent,
            AgentType.JUMP_START: JumpStartAgent,
            AgentType.DISPATCHER: DispatcherAgent
        }
        
        # Cache of active agents
        self.active_agents: Dict[str, BaseAgent] = {}
        
        # Intent matching patterns
        self.intent_patterns = {
            AgentType.TOWING: [
                "tow", "stuck", "broken down", "won't start", "accident",
                "crash", "collision", "flat bed", "tow truck"
            ],
            AgentType.TIRE: [
                "tire", "flat tire", "blown tire", "spare tire", "puncture",
                "wheel", "rim"
            ],
            AgentType.JUMP_START: [
                "jump", "battery", "dead battery", "won't turn over",
                "jump start", "jumper cables"
            ]
        }
        
        logger.info("Agent router initialized with intent matching")
    
    async def determine_intent(self, message: str) -> AgentType:
        """
        Determine intent from message for agent selection.
        
        Args:
            message: User message
            
        Returns:
            Determined agent type
        """
        # Convert to lowercase for matching
        message = message.lower()
        
        # Check each intent pattern
        for agent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in message:
                    logger.info(f"Detected intent: {agent_type} from pattern: {pattern}")
                    return agent_type
        
        # If no clear intent, use RAG to analyze
        try:
            analysis = await self.query_engine.analyze_intent(message)
            if analysis and analysis.get("intent"):
                intent = analysis["intent"]
                if intent in AgentType.__members__:
                    logger.info(f"RAG analysis determined intent: {intent}")
                    return AgentType(intent)
        except Exception as e:
            logger.error(f"Error in RAG intent analysis: {e}")
        
        # Default to dispatcher for human handling
        logger.info("No clear intent detected, defaulting to dispatcher")
        return AgentType.DISPATCHER
    
    def get_agent(
        self,
        session_id: str,
        agent_type: Optional[AgentType] = None
    ) -> BaseAgent:
        """
        Get or create appropriate agent for the session.
        
        Args:
            session_id: Session identifier
            agent_type: Optional explicit agent type
            
        Returns:
            Appropriate agent instance
        """
        # Return existing agent if available
        if session_id in self.active_agents:
            existing_agent = self.active_agents[session_id]
            if not agent_type or existing_agent.agent_type == agent_type:
                return existing_agent
        
        # Create new agent of specified type
        agent_class = self.agent_classes[agent_type or AgentType.DISPATCHER]
        new_agent = agent_class(
            conversation_manager=self.conversation_manager,
            query_engine=self.query_engine,
            prompt_manager=self.prompt_manager
        )
        
        # Store in active agents
        self.active_agents[session_id] = new_agent
        logger.info(f"Created new {agent_type} agent for session {session_id}")
        
        return new_agent
    
    async def route_message(
        self,
        session_id: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route message to appropriate agent and get response.
        
        Args:
            session_id: Session identifier
            message: User message
            **kwargs: Additional parameters
            
        Returns:
            Agent response
        """
        try:
            # Determine intent if no existing agent
            if session_id not in self.active_agents:
                intent = await self.determine_intent(message)
                agent = self.get_agent(session_id, intent)
            else:
                agent = self.active_agents[session_id]
            
            # Process message through agent
            response = await agent.handle_message(message)
            
            # Check if we need to hand off to dispatcher
            if response.get("requires_handoff"):
                # Switch to dispatcher agent
                dispatcher = self.get_agent(session_id, AgentType.DISPATCHER)
                # Handle handoff
                handoff_response = await dispatcher.handle_handoff(
                    previous_agent=agent,
                    collected_info=response.get("collected_info", {}),
                    conversation_history=response.get("conversation_history", [])
                )
                return handoff_response
            
            return response
            
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            return {
                "error": str(e),
                "requires_handoff": True,
                "response": "I apologize, but I'm having trouble processing your request. Let me connect you with a human agent."
            }
    
    def cleanup_session(self, session_id: str):
        """
        Clean up resources for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.active_agents:
            del self.active_agents[session_id]
            logger.info(f"Cleaned up agent for session {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        stats = {
            "active_sessions": len(self.active_agents),
            "agent_types": {}
        }
        
        # Count agent types
        for agent in self.active_agents.values():
            agent_type = agent.agent_type
            if agent_type not in stats["agent_types"]:
                stats["agent_types"][agent_type] = 0
            stats["agent_types"][agent_type] += 1
        
        return stats