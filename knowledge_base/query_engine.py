# knowledge_base/query_engine.py
"""
Query engine optimized for minimal latency using OpenAI LLM + Pinecone.
Simplified architecture for telephony applications.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncIterator

from knowledge_base.openai_llm import OpenAILLM
from knowledge_base.pinecone_store import PineconeVectorStore
from knowledge_base.config import get_retrieval_config

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Simplified query engine for minimal latency.
    Combines Pinecone retrieval with OpenAI LLM generation.
    """
    
    def __init__(
        self,
        vector_store: Optional[PineconeVectorStore] = None,
        llm: Optional[OpenAILLM] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize query engine with fast components."""
        self.config = config or get_retrieval_config()
        self.top_k = self.config["top_k"]
        self.min_score = self.config["min_score"]
        
        # Initialize components
        self.vector_store = vector_store or PineconeVectorStore()
        self.llm = llm or OpenAILLM()
        
        logger.info("Initialized QueryEngine with OpenAI + Pinecone")
    
    async def retrieve_with_sources(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents with sources.
        Fast parallel retrieval from Pinecone.
        """
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score
        
        start_time = time.time()
        
        try:
            # Query Pinecone for similar documents
            results = await self.vector_store.query_with_sources(
                query_text=query,
                top_k=top_k,
                min_score=min_score
            )
            
            retrieval_time = time.time() - start_time
            logger.debug(f"Retrieved {len(results['results'])} documents in {retrieval_time:.3f}s")
            
            return {
                **results,
                "retrieval_time": retrieval_time
            }
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return {
                "query": query,
                "results": [],
                "sources": [],
                "total_results": 0,
                "retrieval_time": time.time() - start_time,
                "error": str(e)
            }
    
    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieval results into context for LLM."""
        return self.vector_store.format_context(results)
    
    async def query(self, query_text: str) -> Dict[str, Any]:
        """
        Complete query processing - retrieve and generate response.
        Optimized for minimal latency with parallel operations where possible.
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_results = await self.retrieve_with_sources(query_text)
            context = self.format_retrieved_context(retrieval_results["results"])
            
            # Step 2: Generate response using OpenAI
            llm_start = time.time()
            response = await self.llm.generate_response(
                query=query_text,
                context=context if context else None
            )
            llm_time = time.time() - llm_start
            
            total_time = time.time() - start_time
            
            return {
                "query": query_text,
                "response": response,
                "sources": retrieval_results.get("sources", []),
                "context": context,
                "total_results": len(retrieval_results["results"]),
                "retrieval_time": retrieval_results.get("retrieval_time", 0),
                "llm_time": llm_time,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                "query": query_text,
                "response": "I'm sorry, I couldn't process that request right now.",
                "sources": [],
                "context": "",
                "total_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def query_with_streaming(
        self,
        query_text: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Query with streaming response for real-time applications.
        Optimized for telephony with minimal latency.
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents (non-blocking)
            retrieval_task = asyncio.create_task(
                self.retrieve_with_sources(query_text)
            )
            
            # Get retrieval results
            retrieval_results = await retrieval_task
            context = self.format_retrieved_context(retrieval_results["results"])
            
            # Step 2: Stream response from OpenAI
            async for chunk in self.llm.generate_streaming_response(
                query=query_text,
                context=context if context else None,
                chat_history=chat_history
            ):
                # Add timing and source information
                if chunk["done"]:
                    chunk.update({
                        "sources": retrieval_results.get("sources", []),
                        "total_results": len(retrieval_results["results"]),
                        "retrieval_time": retrieval_results.get("retrieval_time", 0),
                        "total_time": time.time() - start_time
                    })
                
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield {
                "chunk": "",
                "done": True,
                "full_response": "I'm sorry, I couldn't process that request right now.",
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    async def add_documents(self, documents) -> List[str]:
        """Add documents to the vector store."""
        return await self.vector_store.add_documents(documents)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        try:
            pinecone_stats = await self.vector_store.get_stats()
            llm_stats = self.llm.get_stats()
            
            return {
                "vector_store": pinecone_stats,
                "llm": llm_stats,
                "config": {
                    "top_k": self.top_k,
                    "min_score": self.min_score
                }
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}