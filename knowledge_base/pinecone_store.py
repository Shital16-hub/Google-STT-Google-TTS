# knowledge_base/pinecone_store.py
"""
Pinecone vector store implementation optimized for minimal latency.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from openai import AsyncOpenAI

from knowledge_base.config import get_pinecone_config, get_embedding_config
from knowledge_base.schema import Document, DocumentMetadata

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    """
    Pinecone vector store with OpenAI embeddings.
    Optimized for minimum latency retrieval.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Pinecone vector store."""
        self.config = config or get_pinecone_config()
        self.embedding_config = get_embedding_config()
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.config["api_key"])
        
        # Initialize OpenAI client for embeddings
        self.openai_client = AsyncOpenAI(api_key=self.embedding_config["api_key"])
        
        # Index configuration
        self.index_name = self.config["index_name"]
        self.namespace = self.config.get("namespace", "default")
        self.dimension = self.config["dimension"]
        
        # Initialize index (create if not exists)
        self._initialize_index()
        
        logger.info(f"Initialized Pinecone store with index: {self.index_name}")
    
    def _initialize_index(self):
        """Initialize or create Pinecone index."""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create serverless index for better performance
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Choose region closest to your users
                    )
                )
                
                # Wait for index to be ready
                time.sleep(5)
            
            # Get index reference
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            raise
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI's fastest embedding model.
        """
        try:
            # Use the fastest embedding model
            response = await self.openai_client.embeddings.create(
                model=self.embedding_config["model"],
                input=texts,
                encoding_format="float"
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to Pinecone store.
        Optimized for batch operations.
        """
        if not documents:
            return []
        
        try:
            # Extract texts and metadata
            texts = [doc.text for doc in documents]
            
            # Generate embeddings in batch
            embeddings = await self._generate_embeddings(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                vector = {
                    "id": doc.doc_id,
                    "values": embedding,
                    "metadata": {
                        "text": doc.text,
                        "source": doc.metadata.get("source", ""),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        **doc.metadata  # Include all metadata
                    }
                }
                vectors.append(vector)
            
            # Upsert to Pinecone with batching for speed
            batch_size = 100  # Optimal batch size for Pinecone
            doc_ids = []
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=self.namespace)
                doc_ids.extend([v["id"] for v in batch])
            
            logger.info(f"Added {len(doc_ids)} documents to Pinecone")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}")
            return []
    
    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone for similar documents.
        Optimized for minimal latency.
        """
        try:
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query_text])
            
            # Perform query
            namespace = namespace or self.namespace
            
            # Query Pinecone
            start_time = time.time()
            results = self.index.query(
                vector=query_embedding[0],
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
                filter=filter_dict
            )
            query_time = time.time() - start_time
            
            # Process results
            documents = []
            for match in results.matches:
                # Only include results above minimum score
                if match.score >= min_score:
                    doc = {
                        "id": match.id,
                        "text": match.metadata.get("text", ""),
                        "score": float(match.score),
                        "metadata": dict(match.metadata)
                    }
                    documents.append(doc)
            
            logger.debug(f"Pinecone query returned {len(documents)} results in {query_time:.3f}s")
            return documents
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return []
    
    async def query_with_sources(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.5
    ) -> Dict[str, Any]:
        """
        Query with formatted sources for easy consumption.
        """
        results = await self.query(query_text, top_k, min_score)
        
        # Extract unique sources
        sources = []
        seen_sources = set()
        
        for doc in results:
            source = doc["metadata"].get("source", "Unknown")
            if source not in seen_sources:
                sources.append({
                    "name": source,
                    "type": doc["metadata"].get("source_type", "document")
                })
                seen_sources.add(source)
        
        return {
            "query": query_text,
            "results": results,
            "sources": sources,
            "total_results": len(results)
        }
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieval results into context string for LLM.
        Optimized for minimal tokens.
        """
        if not results:
            return ""
        
        context_parts = []
        for i, doc in enumerate(results):
            # Keep context concise for telephony
            text = doc["text"]
            if len(text) > 200:  # Truncate long texts
                text = text[:197] + "..."
            
            source = doc["metadata"].get("source", f"Source {i+1}")
            context_parts.append(f"{source}: {text}")
        
        return "\n\n".join(context_parts)
    
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from Pinecone."""
        try:
            self.index.delete(ids=doc_ids, namespace=self.namespace)
            logger.info(f"Deleted {len(doc_ids)} documents from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "index_name": self.index_name,
                "namespace": self.namespace,
                "dimension": self.dimension,
                "total_vectors": stats.total_vector_count,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
            return {}
    
    async def reset_index(self) -> bool:
        """Reset (clear) the Pinecone index."""
        try:
            # Delete all vectors in the namespace
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info(f"Reset Pinecone index namespace: {self.namespace}")
            return True
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return False