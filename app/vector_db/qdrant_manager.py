"""
Optimized Qdrant Manager for Multi-Agent Voice AI System
Handles vector storage with agent-specific collections and performance optimization.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from sentence_transformers import SentenceTransformer

from app.config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

@dataclass
class QdrantConfig:
    """Qdrant configuration optimized for performance."""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    timeout: float = 10.0
    
    # Performance settings
    vector_dimension: int = 384  # all-MiniLM-L6-v2 embeddings
    max_batch_size: int = 100
    parallel_indexing: bool = True
    
    # Memory optimization
    memmap_threshold: int = 1000
    indexing_threshold: int = 20000
    
    # Collection settings
    collection_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.collection_config is None:
            self.collection_config = {
                "vectors": {
                    "distance": models.Distance.COSINE,
                    "size": self.vector_dimension
                },
                "optimizers_config": models.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=0,
                    max_segment_size=None,
                    memmap_threshold=self.memmap_threshold,
                    indexing_threshold=self.indexing_threshold,
                    flush_interval_sec=5,
                    max_optimization_threads=2
                ),
                "hnsw_config": models.HnswConfig(
                    m=16,  # Number of bi-directional links for every node
                    ef_construct=100,  # Size of the dynamic candidate list for construction
                    full_scan_threshold=10000,  # Switch to full scan for small collections
                    max_indexing_threads=0,  # Use all available threads
                    on_disk=False  # Keep index in RAM for speed
                ),
                "quantization_config": models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True  # Keep quantized vectors in RAM
                    )
                )
            }

class QdrantManager:
    """
    Optimized Qdrant Manager for Multi-Agent Vector Storage
    
    Features:
    - Agent-specific collections for optimal performance
    - Hybrid search with semantic and keyword matching
    - Automatic optimization and maintenance
    - Connection pooling and error recovery
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        """Initialize Qdrant manager with optimized configuration."""
        self.config = config or QdrantConfig()
        self.client = None
        
        # Agent collections mapping
        self.agent_collections = {}
        
        # Embedding model for vector generation
        self.embedding_model = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_searches': 0,
            'total_inserts': 0,
            'avg_search_time': 0.0,
            'avg_insert_time': 0.0,
            'error_count': 0,
            'last_optimization': time.time()
        }
        
        # Connection health
        self.connection_healthy = False
        self.last_health_check = 0
        
        logger.info("QdrantManager initialized with optimized configuration")
    
    async def init(self):
        """Initialize Qdrant client and embedding model."""
        logger.info("🚀 Initializing Qdrant manager...")
        
        try:
            # Initialize Qdrant client with performance optimizations
            if self.config.prefer_grpc:
                self.client = QdrantClient(
                    host=self.config.host,
                    grpc_port=self.config.grpc_port,
                    prefer_grpc=True,
                    timeout=self.config.timeout
                )
            else:
                self.client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout
                )
            
            # Test connection
            await self._test_connection()
            
            # Initialize embedding model
            await self._init_embedding_model()
            
            # Create default collections for each agent
            await self._create_agent_collections()
            
            self.connection_healthy = True
            logger.info("✅ Qdrant manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant manager: {e}")
            raise
    
    async def _test_connection(self):
        """Test Qdrant connection and performance."""
        try:
            # Test basic connectivity
            collections = await asyncio.to_thread(self.client.get_collections)
            logger.info(f"📡 Connected to Qdrant, found {len(collections.collections)} collections")
            
            # Test performance with a small operation
            start_time = time.time()
            await asyncio.to_thread(self.client.get_collections)
            response_time = time.time() - start_time
            
            if response_time > 1.0:
                logger.warning(f"⚠️ Slow Qdrant response time: {response_time:.3f}s")
            else:
                logger.info(f"⚡ Qdrant response time: {response_time*1000:.1f}ms")
                
        except Exception as e:
            logger.error(f"❌ Qdrant connection test failed: {e}")
            raise
    
    async def _init_embedding_model(self):
        """Initialize embedding model for vector generation."""
        try:
            # Use a fast, lightweight model optimized for semantic similarity
            model_name = "all-MiniLM-L6-v2"  # 384 dimensions, fast inference
            
            logger.info(f"🤖 Loading embedding model: {model_name}")
            self.embedding_model = await asyncio.to_thread(
                SentenceTransformer, model_name
            )
            
            # Test embedding generation
            test_text = "Hello world"
            test_embedding = await self._generate_embedding(test_text)
            
            if len(test_embedding) != self.config.vector_dimension:
                raise ValueError(f"Expected {self.config.vector_dimension} dimensions, got {len(test_embedding)}")
            
            logger.info(f"✅ Embedding model loaded ({self.config.vector_dimension}D vectors)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise
    
    async def _create_agent_collections(self):
        """Create optimized collections for each agent."""
        agent_configs = {
            "roadside-assistance": {
                "name": "roadside_assistance",
                "description": "Roadside assistance knowledge base"
            },
            "billing-support": {
                "name": "billing_support", 
                "description": "Billing and payment support knowledge base"
            },
            "technical-support": {
                "name": "technical_support",
                "description": "Technical support and troubleshooting knowledge base"
            }
        }
        
        for agent_id, config in agent_configs.items():
            try:
                collection_name = config["name"]
                
                # Check if collection exists
                collections = await asyncio.to_thread(self.client.get_collections)
                existing_names = [c.name for c in collections.collections]
                
                if collection_name not in existing_names:
                    # Create collection with optimized settings
                    await asyncio.to_thread(
                        self.client.create_collection,
                        collection_name=collection_name,
                        vectors_config=self.config.collection_config["vectors"],
                        optimizers_config=self.config.collection_config["optimizers_config"],
                        hnsw_config=self.config.collection_config["hnsw_config"],
                        quantization_config=self.config.collection_config["quantization_config"]
                    )
                    
                    logger.info(f"✅ Created collection: {collection_name}")
                else:
                    logger.info(f"📂 Collection exists: {collection_name}")
                
                # Store agent collection mapping
                self.agent_collections[agent_id] = collection_name
                
            except Exception as e:
                logger.error(f"❌ Failed to create collection for {agent_id}: {e}")
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        try:
            # Use async execution to avoid blocking
            embedding = await asyncio.to_thread(
                self.embedding_model.encode,
                text,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            return embedding.astype(np.float32)  # Reduce memory usage
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.config.vector_dimension, dtype=np.float32)
    
    async def search(
        self,
        query: str,
        agent_id: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.7,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform optimized vector search with hybrid capabilities.
        
        Args:
            query: Search query text
            agent_id: Agent identifier for collection selection
            top_k: Number of results to return
            hybrid_alpha: Balance between semantic (1.0) and keyword (0.0) search
            score_threshold: Minimum similarity score threshold
        
        Returns:
            List of search results with metadata
        """
        search_start = time.time()
        
        try:
            # Get collection name for agent
            collection_name = self.agent_collections.get(agent_id)
            if not collection_name:
                logger.warning(f"No collection found for agent: {agent_id}")
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Prepare search parameters
            search_params = models.SearchParams(
                hnsw_ef=128,  # Higher ef for better recall
                exact=False   # Use approximate search for speed
            )
            
            # Perform vector search
            search_results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k * 2,  # Get more results for filtering
                search_params=search_params,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Process results
            results = []
            for result in search_results[:top_k]:
                if result.score >= score_threshold:
                    processed_result = {
                        'id': str(result.id),
                        'content': result.payload.get('content', ''),
                        'metadata': result.payload.get('metadata', {}),
                        'score': float(result.score),
                        'search_time': time.time() - search_start
                    }
                    results.append(processed_result)
            
            # Update performance metrics
            search_time = time.time() - search_start
            await self._update_search_metrics(search_time)
            
            logger.debug(f"🔍 Qdrant search: {len(results)} results in {search_time*1000:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"❌ Qdrant search error: {e}")
            self.performance_metrics['error_count'] += 1
            return []
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        agent_id: str,
        batch_size: Optional[int] = None
    ) -> bool:
        """
        Add documents to agent's collection with batch processing.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            agent_id: Agent identifier for collection selection
            batch_size: Batch size for processing (uses config default if None)
        
        Returns:
            Success status
        """
        insert_start = time.time()
        
        try:
            # Get collection name for agent
            collection_name = self.agent_collections.get(agent_id)
            if not collection_name:
                logger.error(f"No collection found for agent: {agent_id}")
                return False
            
            if not documents:
                logger.warning("No documents to add")
                return True
            
            batch_size = batch_size or self.config.max_batch_size
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"📥 Adding {len(documents)} documents to {collection_name} in {total_batches} batches")
            
            # Process documents in batches
            for batch_idx in range(0, len(documents), batch_size):
                batch_docs = documents[batch_idx:batch_idx + batch_size]
                
                # Prepare points for batch insertion
                points = []
                for i, doc in enumerate(batch_docs):
                    # Generate embedding
                    content = doc.get('content', '')
                    if not content:
                        logger.warning(f"Empty content in document {batch_idx + i}")
                        continue
                    
                    embedding = await self._generate_embedding(content)
                    
                    # Create point
                    point_id = doc.get('id') or str(uuid.uuid4())
                    point = models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            'content': content,
                            'metadata': doc.get('metadata', {}),
                            'agent_id': agent_id,
                            'indexed_at': time.time()
                        }
                    )
                    points.append(point)
                
                if points:
                    # Insert batch
                    await asyncio.to_thread(
                        self.client.upsert,
                        collection_name=collection_name,
                        points=points
                    )
                    
                    logger.debug(f"✅ Inserted batch {(batch_idx // batch_size) + 1}/{total_batches}")
            
            # Update performance metrics
            insert_time = time.time() - insert_start
            await self._update_insert_metrics(insert_time, len(documents))
            
            logger.info(f"✅ Added {len(documents)} documents in {insert_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            self.performance_metrics['error_count'] += 1
            return False
    
    async def get_top_documents(
        self,
        agent_id: str,
        limit: int = 100,
        sort_by: str = 'indexed_at'
    ) -> List[Dict[str, Any]]:
        """Get top documents for an agent (for FAISS promotion)."""
        try:
            collection_name = self.agent_collections.get(agent_id)
            if not collection_name:
                return []
            
            # Use scroll to get documents efficiently
            scroll_result = await asyncio.to_thread(
                self.client.scroll,
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=True  # Need vectors for FAISS
            )
            
            documents = []
            for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                doc = {
                    'id': str(point.id),
                    'content': point.payload.get('content', ''),
                    'metadata': point.payload.get('metadata', {}),
                    'vector': point.vector
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting top documents: {e}")
            return []
    
    async def delete_documents(
        self,
        document_ids: List[str],
        agent_id: str
    ) -> bool:
        """Delete specific documents from agent's collection."""
        try:
            collection_name = self.agent_collections.get(agent_id)
            if not collection_name:
                return False
            
            # Delete points by IDs
            await asyncio.to_thread(
                self.client.delete,
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=document_ids
                )
            )
            
            logger.info(f"🗑️ Deleted {len(document_ids)} documents from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    async def optimize_collection(self, agent_id: str) -> bool:
        """Optimize collection for better performance."""
        try:
            collection_name = self.agent_collections.get(agent_id)
            if not collection_name:
                return False
            
            logger.info(f"🔧 Optimizing collection: {collection_name}")
            
            # Update collection configuration for optimization
            await asyncio.to_thread(
                self.client.update_collection,
                collection_name=collection_name,
                optimizer_config=self.config.collection_config["optimizers_config"]
            )
            
            logger.info(f"✅ Collection optimized: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing collection: {e}")
            return False
    
    async def get_collection_info(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed information about agent's collection."""
        try:
            collection_name = self.agent_collections.get(agent_id)
            if not collection_name:
                return {}
            
            # Get collection info
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                collection_name=collection_name
            )
            
            return {
                'name': collection_name,
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'disk_data_size': getattr(collection_info, 'disk_data_size', 0),
                'ram_data_size': getattr(collection_info, 'ram_data_size', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    async def _update_search_metrics(self, search_time: float):
        """Update search performance metrics."""
        self.performance_metrics['total_searches'] += 1
        
        # Update running average
        total = self.performance_metrics['total_searches']
        current_avg = self.performance_metrics['avg_search_time']
        self.performance_metrics['avg_search_time'] = (
            (current_avg * (total - 1) + search_time) / total
        )
    
    async def _update_insert_metrics(self, insert_time: float, doc_count: int):
        """Update insert performance metrics."""
        self.performance_metrics['total_inserts'] += doc_count
        
        # Update running average (per document)
        per_doc_time = insert_time / doc_count
        total = self.performance_metrics['total_inserts']
        current_avg = self.performance_metrics['avg_insert_time']
        
        if total > doc_count:  # Not the first batch
            self.performance_metrics['avg_insert_time'] = (
                (current_avg * (total - doc_count) + per_doc_time * doc_count) / total
            )
        else:
            self.performance_metrics['avg_insert_time'] = per_doc_time
    
    async def health_check(self) -> bool:
        """Comprehensive health check for Qdrant."""
        try:
            current_time = time.time()
            
            # Skip frequent health checks
            if current_time - self.last_health_check < 30:
                return self.connection_healthy
            
            # Test basic connectivity
            collections = await asyncio.to_thread(self.client.get_collections)
            
            # Test search performance
            if self.agent_collections:
                test_agent = list(self.agent_collections.keys())[0]
                test_results = await self.search(
                    query="test health check",
                    agent_id=test_agent,
                    top_k=1
                )
                # Health check passes if we get results or empty list (no error)
                health_status = isinstance(test_results, list)
            else:
                health_status = True  # No collections to test
            
            self.connection_healthy = health_status
            self.last_health_check = current_time
            
            return health_status
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            self.connection_healthy = False
            return False
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'connection_healthy': self.connection_healthy,
            'total_searches': self.performance_metrics['total_searches'],
            'total_inserts': self.performance_metrics['total_inserts'],
            'avg_search_time_ms': self.performance_metrics['avg_search_time'] * 1000,
            'avg_insert_time_ms': self.performance_metrics['avg_insert_time'] * 1000,
            'error_count': self.performance_metrics['error_count'],
            'collections': {},
            'config': {
                'vector_dimension': self.config.vector_dimension,
                'max_batch_size': self.config.max_batch_size,
                'prefer_grpc': self.config.prefer_grpc
            }
        }
        
        # Get collection stats
        for agent_id, collection_name in self.agent_collections.items():
            try:
                collection_info = await self.get_collection_info(agent_id)
                stats['collections'][agent_id] = collection_info
            except Exception as e:
                stats['collections'][agent_id] = {'error': str(e)}
        
        return stats
    
    async def maintenance(self):
        """Perform routine maintenance tasks."""
        logger.info("🔧 Performing Qdrant maintenance...")
        
        try:
            # Optimize all collections
            for agent_id in self.agent_collections.keys():
                await self.optimize_collection(agent_id)
            
            # Update last optimization time
            self.performance_metrics['last_optimization'] = time.time()
            
            logger.info("✅ Maintenance complete")
            
        except Exception as e:
            logger.error(f"Error during maintenance: {e}")
    
    async def shutdown(self):
        """Shutdown Qdrant manager gracefully."""
        logger.info("🛑 Shutting down Qdrant manager...")
        
        try:
            # Perform final optimization
            await self.maintenance()
            
            # Close client connection
            if self.client:
                # Qdrant client doesn't have explicit close method
                # but we can set it to None to release resources
                self.client = None
            
            # Log final statistics
            stats = await self.get_performance_stats()
            logger.info(f"📊 Final Qdrant stats: {stats['total_searches']} searches, "
                       f"{stats['total_inserts']} inserts, "
                       f"{stats['avg_search_time_ms']:.1f}ms avg search time")
            
            logger.info("✅ Qdrant manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Qdrant shutdown: {e}")