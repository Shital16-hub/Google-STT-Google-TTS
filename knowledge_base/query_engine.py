"""
Query engine for RAG with latest LlamaIndex.
"""
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator, Tuple

from llama_index.core import QueryBundle, Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base import BaseCallbackHandler

from knowledge_base.rag_config import rag_config, RETRIEVE_SYSTEM_PROMPT
from knowledge_base.index_manager import IndexManager

logger = logging.getLogger(__name__)

class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom streaming handler for real-time response generation."""
    
    def __init__(self):
        """Initialize the streaming handler."""
        super().__init__()
        self.streaming_queue = asyncio.Queue()
        self.final_response = ""
        
    def on_llm_stream(self, chunk: str, **kwargs: Any) -> None:
        """Handle streaming chunks from the LLM."""
        self.final_response += chunk
        if not self.streaming_queue.full():
            # Put in a non-blocking way to avoid deadlocks
            try:
                self.streaming_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass
    
    # Add required methods for BaseCallbackHandler
    def start_trace(self, trace_id: str = None) -> None:
        pass
        
    def end_trace(self, trace_id: str = None) -> None:
        pass
        
    def on_event_start(self, trace_id: Optional[str] = None, parent_id: Optional[str] = None, **kwargs: Any) -> str:
        return ""
        
    def on_event_end(self, event_id: str, **kwargs: Any) -> None:
        pass
                
    async def get_chunks(self) -> AsyncIterator[str]:
        """Get streaming chunks."""
        while True:
            try:
                chunk = await self.streaming_queue.get()
                if chunk is None:  # None is our end signal
                    break
                yield chunk
                self.streaming_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in StreamingResponseHandler: {e}")
                break
                
    def finalize_streaming(self):
        """Signal that streaming is complete."""
        try:
            self.streaming_queue.put_nowait(None)  # End signal
        except asyncio.QueueFull:
            pass

class QueryEngine:
    """Query engine for RAG with streaming support and latest LlamaIndex."""
    
    def __init__(self, index_manager: Optional[IndexManager] = None, config=None):
        """
        Initialize the query engine.
        
        Args:
            index_manager: Optional IndexManager instance
            config: Optional configuration object
        """
        self.config = config or rag_config
        self.index_manager = index_manager
        self.top_k = getattr(self.config, 'retrieval_top_k', getattr(self.config, 'default_retrieve_count', 3))
        self.similarity_threshold = self.config.similarity_threshold
        
        # Component placeholders
        self.retriever = None
        self.query_engine = None
        self.llm = None
        
        self.initialized = False
        
        logger.info(f"QueryEngine initialized with top_k={self.top_k}, "
                   f"similarity_threshold={self.similarity_threshold}")
    
    async def init(self):
        """Initialize the query engine asynchronously."""
        if self.initialized:
            return
            
        # Initialize index manager if not provided
        if not self.index_manager:
            self.index_manager = IndexManager(self.config)
            await self.index_manager.init()
        elif not self.index_manager.initialized:
            await self.index_manager.init()
            
        try:
            # Initialize the OpenAI LLM with streaming support
            self.llm = OpenAI(
                model=self.config.openai_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.openai_api_key,
                streaming=self.config.streaming_enabled
            )
            
            # Set LLM in global settings
            Settings.llm = self.llm
            
            # Initialize the retriever
            self.retriever = VectorIndexRetriever(
                index=self.index_manager.index,
                similarity_top_k=self.top_k
            )
            
            # Initialize the query engine
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=self.retriever,
                llm=self.llm
            )
            
            self.initialized = True
            logger.info("Query engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            raise
    
    async def retrieve(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query_text: Query text
            
        Returns:
            List of retrieved documents
        """
        if not self.initialized:
            await self.init()
            
        start_time = time.time()
        
        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query_text)
            
            # Retrieve nodes
            nodes = self.retriever.retrieve(query_bundle)
            
            # Filter by similarity threshold
            filtered_nodes = [
                node for node in nodes 
                if node.score >= self.similarity_threshold
            ]
            
            # Convert to document dicts
            results = []
            for node in filtered_nodes:
                source = None
                if hasattr(node, 'metadata') and node.metadata:
                    source = node.metadata.get('file_name', 'Unknown')
                
                # Create document dict
                doc = {
                    "id": node.node_id,
                    "text": node.text,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {},
                    "score": node.score,
                    "source": source
                }
                results.append(doc)
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def retrieve_with_sources(self, query_text: str) -> Dict[str, Any]:
        """
        Retrieve relevant documents with sources for a query.
        
        Args:
            query_text: Query text
            
        Returns:
            Dictionary with retrieval results
        """
        results = await self.retrieve(query_text)
        
        return {
            "query": query_text,
            "results": results,
            "total": len(results)
        }
    
    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
            
        context_parts = []
        
        for i, doc in enumerate(results):
            source = doc.get("source", f"Document {i+1}")
            score = doc.get("score", 0)
            text = doc.get("text", "")
            
            # Format each document
            context_parts.append(f"Document {i+1} (Source: {source}, Relevance: {score:.2f}):\n{text}")
        
        # Combine all parts
        return "\n\n".join(context_parts)
    
    async def query(self, query_text: str, context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            query_text: Query text
            context: Optional pre-retrieved context
            
        Returns:
            Query response
        """
        if not self.initialized:
            await self.init()
            
        start_time = time.time()
        
        try:
            # Retrieve relevant documents if not provided
            retrieved_context = context if context is not None else await self.retrieve(query_text)
            
            # Format context
            context_str = self.format_retrieved_context(retrieved_context)
            
            # Create prompt with context
            prompt = RETRIEVE_SYSTEM_PROMPT.format(context=context_str)
            
            # Create query bundle
            query_bundle = QueryBundle(query_str=query_text)
            
            # Set system prompt in the query engine
            # With modern LlamaIndex, system prompts are handled differently
            # We can either use a custom response synthesizer or just rely on the standard query
            response = self.query_engine.query(query_bundle)
            
            # Format response
            result = {
                "query": query_text,
                "response": str(response),
                "sources": retrieved_context,
                "total_time": time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return {
                "query": query_text,
                "response": "I'm sorry, I encountered an error processing your request.",
                "sources": [],
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    async def query_with_streaming(self, query_text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Query with optimized streaming response.
        
        This method retrieves context first, then streams the response generation
        for faster perceived latency.
        
        Args:
            query_text: Query text
            
        Yields:
            Response chunks with early results
        """
        if not self.initialized:
            await self.init()
            
        start_time = time.time()
        
        try:
            # OPTIMIZATION: Use a reduced context window to speed up retrieval
            context_start_time = time.time()
            
            # Retrieve relevant documents first but with reduced count
            quick_retriever = VectorIndexRetriever(
                index=self.index_manager.index,
                similarity_top_k=1  # Just get the best match for speed
            )
            
            # Get quick context for immediate response
            quick_query_bundle = QueryBundle(query_str=query_text)
            quick_nodes = quick_retriever.retrieve(quick_query_bundle)
            quick_context = self._process_retrieved_nodes(quick_nodes)
            
            # Yield an immediate chunk while continuing retrieval
            if quick_context:
                # Extract a quick snippet from the best match
                first_doc = quick_context[0]
                yield {
                    "chunk": f"I'm finding information about {query_text.split()[-1]}...",
                    "done": False
                }
            
            # Continue with full retrieval in background
            full_retriever = self.retriever  # Use standard retriever for full context
            full_query_bundle = QueryBundle(query_str=query_text)
            full_nodes = full_retriever.retrieve(full_query_bundle)
            
            # Filter by similarity threshold
            filtered_nodes = [
                node for node in full_nodes
                if node.score >= self.similarity_threshold
            ]
            
            # Format context for LLM
            context_str = self.format_retrieved_context(filtered_nodes)
            retrieval_time = time.time() - context_start_time
            
            # Create streaming handler
            streaming_handler = StreamingCallbackHandler()
            callback_manager = CallbackManager([streaming_handler])
            
            # Create streaming-enabled LLM
            streaming_llm = OpenAI(
                model=Settings.llm.model_name,
                temperature=Settings.llm.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.openai_api_key,
                streaming=True,
                callback_manager=callback_manager
            )
            
            # Start query in background task
            query_task = asyncio.create_task(
                self._run_streaming_query(
                    full_query_bundle, 
                    streaming_llm,
                    filtered_nodes
                )
            )
            
            # Stream response chunks with early results
            async for chunk in streaming_handler.get_chunks():
                yield {
                    "chunk": chunk,
                    "done": False,
                    "sources": self._get_source_info(filtered_nodes)
                }
            
            # Wait for query to complete
            await query_task
            
            # Send final chunk
            yield {
                "chunk": "",
                "full_response": streaming_handler.final_response,
                "done": True,
                "sources": self._get_source_info(filtered_nodes),
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield {
                "chunk": f"I'm sorry, I encountered an error: {str(e)}",
                "done": True,
                "error": str(e)
            }
    
    async def _run_streaming_query(self, query_bundle, streaming_llm, prompt):
        """Run the streaming query in a background task."""
        try:
            # Create streaming-enabled query engine
            streaming_engine = RetrieverQueryEngine.from_args(
                retriever=self.retriever,
                llm=streaming_llm,
                use_async=True
            )
            
            # Run query
            await streaming_engine.aquery(query_bundle)
            
        except Exception as e:
            logger.error(f"Error in streaming query task: {e}")
        finally:
            # Ensure streaming is finalized
            if hasattr(streaming_llm, 'callback_manager'):
                for handler in streaming_llm.callback_manager.handlers:
                    if isinstance(handler, StreamingCallbackHandler):
                        handler.finalize_streaming()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get query engine statistics.
        
        Returns:
            Dictionary with statistics
        """
        doc_count = 0
        if self.index_manager and self.index_manager.initialized:
            doc_count = await self.index_manager.count_documents()
            
        return {
            "document_count": doc_count,
            "retrieval_top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "openai_model": self.config.openai_model,
            "streaming_enabled": self.config.streaming_enabled,
            "max_tokens": self.config.max_tokens
        }