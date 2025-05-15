# integration/kb_integration.py
"""
Knowledge Base integration optimized for OpenAI + Pinecone.
Simplified for minimal latency in telephony applications.
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, List

from knowledge_base.query_engine import QueryEngine
from knowledge_base.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

class KnowledgeBaseIntegration:
    """
    Simplified Knowledge Base integration for OpenAI + Pinecone.
    Optimized for sub-2-second response times.
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        conversation_manager: Optional[ConversationManager] = None,
        max_response_time: float = 2.0  # 2 second max response time
    ):
        """Initialize KB integration with fast components."""
        self.query_engine = query_engine or QueryEngine()
        self.conversation_manager = conversation_manager or ConversationManager(
            query_engine=self.query_engine
        )
        self.max_response_time = max_response_time
        self.initialized = True
        
        logger.info("Initialized KnowledgeBaseIntegration with OpenAI + Pinecone")
    
    async def query(self, text: str, include_context: bool = False) -> Dict[str, Any]:
        """
        Query the knowledge base with timeout protection.
        """
        if not self.initialized:
            return {"error": "Knowledge Base not initialized"}
        
        start_time = time.time()
        
        try:
            # Use asyncio.wait_for to enforce timeout
            result = await asyncio.wait_for(
                self.conversation_manager.handle_user_input(text),
                timeout=self.max_response_time
            )
            
            # Add timing information
            result["total_time"] = time.time() - start_time
            result["cache_hit"] = False
            
            # Include context if requested
            if include_context and "sources" in result:
                # Format context from sources
                context_parts = []
                for i, source in enumerate(result["sources"]):
                    context_parts.append(f"Source {i+1}: {source.get('name', 'Unknown')}")
                result["context"] = "\n".join(context_parts)
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Query timed out after {self.max_response_time}s: '{text}'")
            return {
                "error": f"Response timed out after {self.max_response_time} seconds",
                "query": text,
                "total_time": time.time() - start_time,
                "response": "I'm sorry, that's taking too long to process. Could you rephrase your question?"
            }
        except Exception as e:
            logger.error(f"Error in KB query: {e}")
            return {
                "error": str(e),
                "query": text,
                "total_time": time.time() - start_time,
                "response": "I'm sorry, I encountered an error. Please try again."
            }
    
    async def query_streaming(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Query with streaming response for real-time applications.
        """
        if not self.initialized:
            yield {"error": "Knowledge Base not initialized", "done": True}
            return
        
        try:
            # Stream response with timeout protection
            timeout_task = asyncio.create_task(asyncio.sleep(self.max_response_time))
            
            async for chunk in self.conversation_manager.generate_streaming_response(text):
                # Cancel timeout if we got response
                if not timeout_task.done():
                    timeout_task.cancel()
                
                yield chunk
                
                # Create new timeout for next chunk
                if not chunk.get("done", False):
                    timeout_task = asyncio.create_task(asyncio.sleep(1.0))  # 1s timeout between chunks
                else:
                    break
                    
        except asyncio.TimeoutError:
            logger.warning(f"Streaming query timed out: '{text}'")
            yield {
                "chunk": "",
                "done": True,
                "full_response": "I'm sorry, that's taking too long. Could you rephrase?",
                "error": "timeout"
            }
        except Exception as e:
            logger.error(f"Error in streaming KB query: {e}")
            yield {
                "chunk": "",
                "done": True,
                "full_response": "I'm sorry, I encountered an error. Please try again.",
                "error": str(e)
            }
    
    def reset_conversation(self) -> None:
        """Reset conversation state."""
        if self.conversation_manager:
            self.conversation_manager.reset()
            logger.info("Reset conversation state")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if not self.conversation_manager:
            return []
        return self.conversation_manager.get_history()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get KB integration statistics."""
        stats = {
            "initialized": self.initialized,
            "max_response_time": self.max_response_time
        }
        
        # Add query engine stats
        if self.query_engine:
            qe_stats = await self.query_engine.get_stats()
            stats["query_engine"] = qe_stats
        
        # Add conversation stats
        if self.conversation_manager:
            conv_stats = self.conversation_manager.get_stats()
            stats["conversation"] = conv_stats
        
        return stats
    
    async def add_documents(self, documents) -> List[str]:
        """Add documents to the knowledge base."""
        if not self.query_engine:
            raise RuntimeError("Query engine not initialized")
        
        return await self.query_engine.add_documents(documents)