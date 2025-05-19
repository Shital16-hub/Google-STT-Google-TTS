"""
Enhanced conversation manager using LlamaIndex.
"""
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
from enum import Enum

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore

from knowledge_base.rag_config import rag_config, CONVERSATION_SYSTEM_PROMPT
from knowledge_base.query_engine import QueryEngine

logger = logging.getLogger(__name__)

class ConversationState(str, Enum):
    """Enhanced conversation states for voice interactions."""
    GREETING = "greeting"
    WAITING_FOR_QUERY = "waiting_for_query"
    RETRIEVING = "retrieving"
    GENERATING_RESPONSE = "generating_response"
    CONTINUOUS = "continuous"
    ENDED = "ended"

class ConversationManager:
    """Enhanced conversation manager with LlamaIndex memory integration."""
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        session_id: Optional[str] = None,
        config = None,
        skip_greeting: bool = True
    ):
        """
        Initialize the conversation manager.
        
        Args:
            query_engine: Optional QueryEngine instance
            session_id: Optional session ID
            config: Optional configuration object
            skip_greeting: Whether to skip greeting
        """
        self.config = config or rag_config
        self.query_engine = query_engine
        self.session_id = session_id or f"session_{int(time.time())}"
        self.skip_greeting = skip_greeting
        
        # Initialize conversation state
        self.current_state = ConversationState.CONTINUOUS if skip_greeting else ConversationState.GREETING
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        
        # Session tracking
        self.start_time = time.time()
        self.last_activity = time.time()
        self.turn_count = 0
        
        # Statistics tracking
        self.total_tokens = 0
        self.total_retrieval_time = 0
        self.total_generation_time = 0
        
        # LLM setup
        self.llm = None
        self.streaming_llm = None
        
        self.initialized = False
        
        logger.info(f"ConversationManager initialized for session: {self.session_id}")
    
    async def init(self):
        """Initialize conversation manager asynchronously."""
        if self.initialized:
            return
            
        try:
            # Initialize query engine if not provided
            if not self.query_engine:
                self.query_engine = QueryEngine(config=self.config)
                await self.query_engine.init()
            elif not self.query_engine.initialized:
                await self.query_engine.init()
                
            # Initialize LLM
            self.llm = OpenAI(
                model=self.config.openai_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.openai_api_key
            )
            
            # Initialize streaming LLM if needed
            if self.config.streaming_enabled:
                self.streaming_llm = OpenAI(
                    model=self.config.openai_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.max_tokens,
                    api_key=self.config.openai_api_key,
                    streaming=True
                )
            
            # Add system message to memory
            self._add_system_message()
            
            self.initialized = True
            logger.info(f"ConversationManager initialization complete for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error initializing conversation manager: {e}")
            raise
    
    def _add_system_message(self):
        """Add system message to memory."""
        self.memory.put(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=CONVERSATION_SYSTEM_PROMPT
            )
        )
    
    async def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input in conversation.
        
        Args:
            user_input: User input text
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            await self.init()
            
        # Update state and tracking
        self.last_activity = time.time()
        self.turn_count += 1
        
        logger.info(f"[{self.session_id}] User input: {user_input}")
        
        # Process based on current state
        if self.current_state == ConversationState.GREETING:
            response = await self._handle_greeting(user_input)
        else:
            response = await self._handle_query(user_input)
        
        # Update state
        self.current_state = response.get("state", ConversationState.CONTINUOUS)
        
        return response
    
    async def _handle_greeting(self, user_input: str) -> Dict[str, Any]:
        """
        Handle greeting state.
        
        Args:
            user_input: User input text
            
        Returns:
            Response dictionary
        """
        # If user input is substantial, treat as a query
        if len(user_input.split()) > 2:
            return await self._handle_query(user_input)
        
        # Add user input to memory
        self.memory.put(
            ChatMessage(
                role=MessageRole.USER,
                content=user_input
            )
        )
        
        # Generate brief greeting
        greeting = "Hello! I'm ready to help. What would you like to know?"
        
        # Add assistant response to memory
        self.memory.put(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=greeting
            )
        )
        
        return {
            "response": greeting,
            "state": ConversationState.CONTINUOUS,
            "requires_human": False,
            "context": None
        }
    
    async def _handle_query(self, user_input: str) -> Dict[str, Any]:
        """
        Handle user query.
        
        Args:
            user_input: User input text
            
        Returns:
            Response dictionary
        """
        # Add user input to memory
        self.memory.put(
            ChatMessage(
                role=MessageRole.USER,
                content=user_input
            )
        )
        
        try:
            # Update state
            self.current_state = ConversationState.RETRIEVING
            
            # Retrieve context with timing
            retrieval_start = time.time()
            retrieved_context = await self.query_engine.retrieve(user_input)
            retrieval_time = time.time() - retrieval_start
            self.total_retrieval_time += retrieval_time
            
            # Generate response with timing
            generation_start = time.time()
            query_result = await self.query_engine.query(user_input, context=retrieved_context)
            response_text = query_result.get("response", "")
            generation_time = time.time() - generation_start
            self.total_generation_time += generation_time
            
            # Add assistant response to memory
            if response_text:
                self.memory.put(
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response_text
                    )
                )
            
            # Prepare result
            result = {
                "response": response_text,
                "state": ConversationState.CONTINUOUS,
                "requires_human": False,
                "context": retrieved_context,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "sources": query_result.get("sources", [])
            }
            
            logger.info(f"[{self.session_id}] Generated response: {response_text[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Add error response to memory
            error_response = "I'm sorry, I encountered an error processing your request. Could you please try again?"
            self.memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=error_response
                )
            )
            
            return {
                "response": error_response,
                "state": ConversationState.CONTINUOUS,
                "requires_human": False,
                "error": str(e)
            }
    
    async def generate_streaming_response(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate streaming response for user input.
        
        Args:
            user_input: User input text
            
        Yields:
            Response chunks
        """
        if not self.initialized:
            await self.init()
            
        # Update state and tracking
        self.last_activity = time.time()
        self.turn_count += 1
        
        # Add user input to memory
        self.memory.put(
            ChatMessage(
                role=MessageRole.USER,
                content=user_input
            )
        )
        
        try:
            # Stream through query engine
            full_response = ""
            
            async for chunk in self.query_engine.query_with_streaming(user_input):
                chunk_text = chunk.get("chunk", "")
                full_response += chunk_text
                
                # Yield chunk
                yield {
                    "chunk": chunk_text,
                    "done": chunk.get("done", False),
                    "sources": chunk.get("sources", [])
                }
                
                # If done, send final message
                if chunk.get("done", False):
                    # Add final response to memory
                    final_response = chunk.get("full_response", full_response)
                    self.memory.put(
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=final_response
                        )
                    )
                    
                    # Yield final result
                    yield {
                        "chunk": "",
                        "full_response": final_response,
                        "done": True,
                        "sources": chunk.get("sources", [])
                    }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            
            # Add error response to memory
            error_response = "I'm sorry, I encountered an error processing your request. Could you please try again?"
            self.memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=error_response
                )
            )
            
            # Yield error result
            yield {
                "chunk": error_response,
                "done": True,
                "error": str(e)
            }
    
    def reset(self):
        """Reset conversation state while preserving session."""
        # Reset state
        self.current_state = ConversationState.CONTINUOUS if self.skip_greeting else ConversationState.GREETING
        
        # Reset memory but keep system message
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        self._add_system_message()
        
        # Reset tracking but keep session ID
        self.turn_count = 0
        self.total_tokens = 0
        self.total_retrieval_time = 0
        self.total_generation_time = 0
        
        # Reset start time
        self.start_time = time.time()
        self.last_activity = time.time()
        
        logger.info(f"Reset conversation for session: {self.session_id}")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation messages
        """
        messages = []
        
        for msg in self.memory.get_all():
            if msg.role == MessageRole.SYSTEM:
                continue
                
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "last_activity": self.last_activity,
            "session_duration": time.time() - self.start_time,
            "turn_count": self.turn_count,
            "current_state": self.current_state,
            "total_tokens": self.total_tokens,
            "total_retrieval_time": self.total_retrieval_time,
            "total_generation_time": self.total_generation_time,
            "messages_in_memory": len(self.memory.get_all())
        }