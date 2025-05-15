# knowledge_base/conversation_manager.py
"""
Simplified conversation manager optimized for telephony latency.
Removes complex state management for faster response times.
"""
import logging
import time
from typing import Dict, Any, List, Optional, AsyncIterator
from enum import Enum

from knowledge_base.query_engine import QueryEngine

logger = logging.getLogger(__name__)

class ConversationState(str, Enum):
    """Simplified conversation states for telephony."""
    ACTIVE = "active"
    ENDED = "ended"

class ConversationTurn:
    """Simple conversation turn tracking."""
    def __init__(self, query: str, response: str, duration: float = 0.0):
        self.query = query
        self.response = response
        self.duration = duration
        self.timestamp = time.time()

class ConversationManager:
    """
    Simplified conversation manager for minimal latency.
    Removed complex state management and conversation tracking.
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        max_history: int = 6,  # Keep only last 3 exchanges
        **kwargs
    ):
        """Initialize simple conversation manager."""
        self.query_engine = query_engine
        self.max_history = max_history
        self.state = ConversationState.ACTIVE
        self.history: List[ConversationTurn] = []
        self.session_start = time.time()
        
        logger.info("Initialized simplified ConversationManager for telephony")
    
    async def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Handle user input with minimal processing for speed.
        """
        start_time = time.time()
        
        if not self.query_engine:
            return {
                "response": "I'm sorry, the knowledge base is not available right now.",
                "state": self.state,
                "processing_time": time.time() - start_time
            }
        
        try:
            # Get chat history for context (simplified format)
            chat_history = self._get_chat_history()
            
            # Query the knowledge base directly
            result = await self.query_engine.query(user_input)
            response_text = result.get("response", "I couldn't find an answer to that.")
            
            # Record this turn
            turn = ConversationTurn(
                query=user_input,
                response=response_text,
                duration=time.time() - start_time
            )
            self._add_turn(turn)
            
            return {
                "response": response_text,
                "state": self.state,
                "sources": result.get("sources", []),
                "processing_time": time.time() - start_time,
                "retrieval_time": result.get("retrieval_time", 0),
                "llm_time": result.get("llm_time", 0)
            }
            
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            return {
                "response": "I'm sorry, I encountered an error. Could you please try again?",
                "state": self.state,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def generate_streaming_response(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate streaming response for real-time applications.
        """
        start_time = time.time()
        
        if not self.query_engine:
            yield {
                "chunk": "I'm sorry, the knowledge base is not available right now.",
                "done": True,
                "state": self.state,
                "processing_time": time.time() - start_time
            }
            return
        
        try:
            # Get chat history for context
            chat_history = self._get_chat_history()
            
            # Stream response from query engine
            full_response = ""
            async for chunk in self.query_engine.query_with_streaming(
                user_input, 
                chat_history=chat_history
            ):
                if chunk.get("chunk"):
                    full_response += chunk["chunk"]
                
                # Add conversation state to chunk
                chunk["state"] = self.state
                yield chunk
                
                # Record turn when done
                if chunk["done"]:
                    turn = ConversationTurn(
                        query=user_input,
                        response=full_response,
                        duration=time.time() - start_time
                    )
                    self._add_turn(turn)
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield {
                "chunk": "",
                "done": True,
                "full_response": "I'm sorry, I encountered an error. Could you please try again?",
                "state": self.state,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get simplified chat history for context.
        Limited to recent exchanges for speed.
        """
        history = []
        
        # Get last few turns (keep it minimal for speed)
        recent_turns = self.history[-3:] if len(self.history) >= 3 else self.history
        
        for turn in recent_turns:
            history.append({"role": "user", "content": turn.query})
            history.append({"role": "assistant", "content": turn.response})
        
        return history
    
    def _add_turn(self, turn: ConversationTurn):
        """Add turn to history with size management."""
        self.history.append(turn)
        
        # Keep history limited for speed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def reset(self):
        """Reset conversation state."""
        self.state = ConversationState.ACTIVE
        self.history = []
        self.session_start = time.time()
        logger.info("Reset conversation state")
    
    def end_conversation(self):
        """End the conversation."""
        self.state = ConversationState.ENDED
        logger.info("Ended conversation")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return [
            {
                "query": turn.query,
                "response": turn.response,
                "duration": turn.duration,
                "timestamp": turn.timestamp
            }
            for turn in self.history
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "state": self.state,
            "session_duration": time.time() - self.session_start,
            "total_turns": len(self.history),
            "avg_response_time": (
                sum(turn.duration for turn in self.history) / len(self.history)
                if self.history else 0
            ),
            "last_activity": self.history[-1].timestamp if self.history else self.session_start
        }