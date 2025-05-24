"""
FAISS Hot Tier - In-Memory Vector Search for Ultra-Low Latency
Provides <5ms vector search for frequently accessed agent contexts.
"""
import asyncio
import logging
import time
import pickle
import threading
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from app.config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

@dataclass
class FAISSConfig:
    """FAISS configuration optimized for speed."""
    vector_dimension: int = 384  # Match embedding model
    index_type: str = "IVF"      # IVF for speed, Flat for accuracy
    nlist: int = 100             # Number of clusters for IVF
    nprobe: int = 10             # Number of clusters to search
    
    # Performance settings
    max_vectors_per_agent: int = 1000
    batch_size: int = 32
    similarity_threshold: float = 0.3
    
    # Memory management
    enable_gpu: bool = False      # Use GPU if available
    omp_num_threads: int = 4      # OpenMP threads for CPU
    
    # Hot tier management
    promotion_threshold: int = 5   # Minimum access count for promotion
    max_agents_in_memory: int = 10 # Maximum agent indices in memory
    cleanup_interval: int = 300    # Cleanup interval in seconds

class FAISSAgentIndex:
    """FAISS index for a specific agent with metadata."""
    
    def __init__(self, agent_id: str, config: FAISSConfig):
        self.agent_id = agent_id
        self.config = config
        
        # FAISS index and metadata
        self.index = None
        self.documents = []  # Store document metadata
        self.id_to_index = {}  # Map document IDs to index positions
        
        # Performance tracking
        self.access_count = 0
        self.last_accessed = time.time()
        self.creation_time = time.time()
        self.search_times = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        self._create_index()
    
    def _create_index(self):
        """Create optimized FAISS index based on configuration."""
        try:
            if self.config.index_type == "Flat":
                # Flat index for maximum accuracy (slower)
                self.index = faiss.IndexFlatIP(self.config.vector_dimension)
            elif self.config.index_type == "IVF":
                # IVF index for speed (default)
                quantizer = faiss.IndexFlatIP(self.config.vector_dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, 
                    self.config.vector_dimension, 
                    self.config.nlist
                )
                # Set search parameters
                self.index.nprobe = self.config.nprobe
            else:
                # Fallback to flat index
                self.index = faiss.IndexFlatIP(self.config.vector_dimension)
            
            # Enable GPU if configured and available
            if self.config.enable_gpu and faiss.get_num_gpus() > 0:
                try:
                    gpu_resource = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_resource, 0, self.index)
                    logger.info(f"🚀 FAISS GPU acceleration enabled for {self.agent_id}")
                except Exception as e:
                    logger.warning(f"Failed to enable GPU for {self.agent_id}: {e}")
            
            logger.debug(f"Created FAISS index for {self.agent_id}: {type(self.index).__name__}")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            # Fallback to simple flat index
            self.index = faiss.IndexFlatIP(self.config.vector_dimension)
    
    def add_vectors(self, vectors: np.ndarray, documents: List[Dict[str, Any]]):
        """Add vectors and associated documents to the index."""
        with self.lock:
            try:
                if len(vectors) == 0:
                    return
                
                # Train index if necessary (for IVF indices)
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    if len(vectors) >= self.config.nlist:
                        self.index.train(vectors)
                    else:
                        logger.warning(f"Not enough vectors to train IVF index for {self.agent_id}")
                        return
                
                # Add vectors to index
                start_idx = len(self.documents)
                self.index.add(vectors)
                
                # Update metadata
                for i, doc in enumerate(documents):
                    doc_idx = start_idx + i
                    self.documents.append(doc)
                    doc_id = doc.get('id', f"doc_{doc_idx}")
                    self.id_to_index[doc_id] = doc_idx
                
                logger.debug(f"Added {len(vectors)} vectors to {self.agent_id} FAISS index")
                
            except Exception as e:
                logger.error(f"Error adding vectors to FAISS index: {e}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index for similar vectors."""
        search_start = time.time()
        
        with self.lock:
            try:
                self.access_count += 1
                self.last_accessed = time.time()
                
                if self.index.ntotal == 0:
                    return []
                
                # Ensure query vector is 2D
                if query_vector.ndim == 1:
                    query_vector = query_vector.reshape(1, -1)
                
                # Perform search
                scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
                
                # Process results
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:  # FAISS returns -1 for empty slots
                        continue
                    
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        result = {
                            'id': doc.get('id', f"doc_{idx}"),
                            'content': doc.get('content', ''),
                            'metadata': doc.get('metadata', {}),
                            'score': float(score),
                            'search_time': time.time() - search_start
                        }
                        results.append(result)
                
                # Track performance
                search_time = time.time() - search_start
                self.search_times.append(search_time)
                
                return results
                
            except Exception as e:
                logger.error(f"Error searching FAISS index: {e}")
                return []
    
    @property
    def average_search_time(self) -> float:
        """Get average search time."""
        return sum(self.search_times) / len(self.search_times) if self.search_times else 0.0
    
    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal if self.index else 0
    
    def serialize(self) -> bytes:
        """Serialize the index for persistence."""
        with self.lock:
            try:
                # Serialize FAISS index
                index_bytes = faiss.serialize_index(self.index)
                
                # Create complete state
                state = {
                    'index_bytes': index_bytes,
                    'documents': self.documents,
                    'id_to_index': self.id_to_index,
                    'access_count': self.access_count,
                    'creation_time': self.creation_time
                }
                
                return pickle.dumps(state)
                
            except Exception as e:
                logger.error(f"Error serializing FAISS index: {e}")
                return b''
    
    @classmethod
    def deserialize(cls, agent_id: str, config: FAISSConfig, data: bytes) -> 'FAISSAgentIndex':
        """Deserialize index from bytes."""
        try:
            state = pickle.loads(data)
            
            # Create instance
            instance = cls(agent_id, config)
            
            # Restore FAISS index
            instance.index = faiss.deserialize_index(state['index_bytes'])
            instance.documents = state['documents']
            instance.id_to_index = state['id_to_index']
            instance.access_count = state.get('access_count', 0)
            instance.creation_time = state.get('creation_time', time.time())
            
            return instance
            
        except Exception as e:
            logger.error(f"Error deserializing FAISS index: {e}")
            return cls(agent_id, config)

class FAISSHotTier:
    """
    FAISS Hot Tier for Ultra-Low Latency Vector Search
    
    Features:
    - In-memory FAISS indices for active agents
    - Automatic promotion/demotion based on usage
    - Thread-safe operations with performance optimization
    - GPU acceleration support
    - Memory-efficient management
    """
    
    def __init__(self, config: Optional[FAISSConfig] = None):
        """Initialize FAISS hot tier manager."""
        self.config = config or FAISSConfig()
        
        # Agent indices management
        self.agent_indices: Dict[str, FAISSAgentIndex] = {}
        self.embedding_model = None
        
        # Performance tracking
        self.performance_stats = {
            'total_searches': 0,
            'total_cache_hits': 0,
            'avg_search_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Background maintenance
        self.cleanup_task = None
        self.maintenance_running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Set FAISS threading
        faiss.omp_set_num_threads(self.config.omp_num_threads)
        
        logger.info("FAISSHotTier initialized with optimized configuration")
    
    async def init(self):
        """Initialize FAISS hot tier."""
        logger.info("🚀 Initializing FAISS hot tier...")
        
        try:
            # Initialize embedding model (lightweight for speed)
            await self._init_embedding_model()
            
            # Start background maintenance
            self.cleanup_task = asyncio.create_task(self._maintenance_loop())
            
            # Log GPU availability
            if faiss.get_num_gpus() > 0 and self.config.enable_gpu:
                logger.info(f"🚀 FAISS GPU acceleration available ({faiss.get_num_gpus()} GPUs)")
            else:
                logger.info(f"💻 FAISS using CPU with {self.config.omp_num_threads} threads")
            
            logger.info("✅ FAISS hot tier ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS hot tier: {e}")
            raise
    
    async def _init_embedding_model(self):
        """Initialize lightweight embedding model."""
        try:
            # Use same model as Qdrant for consistency
            model_name = "all-MiniLM-L6-v2"
            
            logger.info(f"🤖 Loading FAISS embedding model: {model_name}")
            self.embedding_model = await asyncio.to_thread(
                SentenceTransformer, model_name
            )
            
            logger.info("✅ FAISS embedding model loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS embedding model: {e}")
            raise
    
    async def search(
        self,
        query: str,
        agent_id: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.7
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search FAISS hot tier for agent-specific vectors.
        
        Args:
            query: Search query text
            agent_id: Agent identifier
            top_k: Number of results to return
            hybrid_alpha: Not used in FAISS, kept for interface compatibility
        
        Returns:
            Search results or None if agent not in hot tier
        """
        search_start = time.time()
        
        try:
            # Check if agent has hot tier index
            if not await self.has_agent_index(agent_id):
                return None
            
            # Generate query embedding
            query_vector = await self._generate_embedding(query)
            
            # Search agent's FAISS index
            with self.lock:
                agent_index = self.agent_indices[agent_id]
                results = agent_index.search(query_vector, top_k)
            
            # Update performance stats
            search_time = time.time() - search_start
            await self._update_performance_stats(search_time, len(results) > 0)
            
            logger.debug(f"⚡ FAISS search: {len(results)} results in {search_time*1000:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"❌ FAISS search error: {e}")
            return None
    
    async def has_agent_index(self, agent_id: str) -> bool:
        """Check if agent has an active FAISS index."""
        with self.lock:
            return agent_id in self.agent_indices
    
    async def create_agent_index(
        self,
        agent_id: str,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Create FAISS index for an agent with initial documents."""
        try:
            logger.info(f"🔥 Creating FAISS hot tier index for {agent_id}")
            
            if not documents:
                logger.warning(f"No documents provided for {agent_id}")
                return False
            
            # Limit documents to prevent memory issues
            limited_docs = documents[:self.config.max_vectors_per_agent]
            
            # Generate embeddings for all documents
            contents = [doc.get('content', '') for doc in limited_docs]
            embeddings = await self._generate_embeddings_batch(contents)
            
            if embeddings is None or len(embeddings) == 0:
                logger.error(f"Failed to generate embeddings for {agent_id}")
                return False
            
            # Create agent index
            with self.lock:
                # Remove old index if exists
                if agent_id in self.agent_indices:
                    del self.agent_indices[agent_id]
                
                # Check memory limits
                if len(self.agent_indices) >= self.config.max_agents_in_memory:
                    await self._evict_least_used_agent()
                
                # Create new index
                agent_index = FAISSAgentIndex(agent_id, self.config)
                agent_index.add_vectors(embeddings, limited_docs)
                
                self.agent_indices[agent_id] = agent_index
            
            logger.info(f"✅ Created FAISS index for {agent_id} with {len(limited_docs)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating FAISS index for {agent_id}: {e}")
            return False
    
    async def add_documents(
        self,
        agent_id: str,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Add documents to existing agent index."""
        try:
            if not await self.has_agent_index(agent_id):
                # Create new index if doesn't exist
                return await self.create_agent_index(agent_id, documents)
            
            # Generate embeddings
            contents = [doc.get('content', '') for doc in documents]
            embeddings = await self._generate_embeddings_batch(contents)
            
            if embeddings is None:
                return False
            
            # Add to existing index
            with self.lock:
                agent_index = self.agent_indices[agent_id]
                
                # Check size limits
                if agent_index.size + len(embeddings) > self.config.max_vectors_per_agent:
                    logger.warning(f"FAISS index for {agent_id} would exceed size limit")
                    # Take only what fits
                    remaining_capacity = self.config.max_vectors_per_agent - agent_index.size
                    if remaining_capacity > 0:
                        embeddings = embeddings[:remaining_capacity]
                        documents = documents[:remaining_capacity]
                    else:
                        return False
                
                agent_index.add_vectors(embeddings, documents)
            
            logger.debug(f"Added {len(documents)} documents to FAISS index for {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS index: {e}")
            return False
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate single embedding vector."""
        try:
            embedding = await asyncio.to_thread(
                self.embedding_model.encode,
                text,
                normalize_embeddings=True
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.config.vector_dimension, dtype=np.float32)
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        try:
            if not texts:
                return None
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            if not valid_texts:
                return None
            
            # Generate embeddings in batches for memory efficiency
            all_embeddings = []
            
            for i in range(0, len(valid_texts), self.config.batch_size):
                batch = valid_texts[i:i + self.config.batch_size]
                
                batch_embeddings = await asyncio.to_thread(
                    self.embedding_model.encode,
                    batch,
                    normalize_embeddings=True,
                    batch_size=len(batch)
                )
                
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            if all_embeddings:
                embeddings = np.vstack(all_embeddings).astype(np.float32)
                return embeddings
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return None
    
    async def _evict_least_used_agent(self):
        """Evict the least recently used agent index."""
        try:
            with self.lock:
                if not self.agent_indices:
                    return
                
                # Find least recently used agent
                lru_agent = min(
                    self.agent_indices.items(),
                    key=lambda x: x[1].last_accessed
                )
                
                agent_id, agent_index = lru_agent
                del self.agent_indices[agent_id]
                
                logger.info(f"🗑️ Evicted FAISS index for {agent_id} (LRU)")
                
        except Exception as e:
            logger.error(f"Error evicting agent index: {e}")
    
    async def _update_performance_stats(self, search_time: float, cache_hit: bool):
        """Update performance statistics."""
        self.performance_stats['total_searches'] += 1
        
        if cache_hit:
            self.performance_stats['total_cache_hits'] += 1
        
        # Update average search time
        total = self.performance_stats['total_searches']
        current_avg = self.performance_stats['avg_search_time']
        self.performance_stats['avg_search_time'] = (
            (current_avg * (total - 1) + search_time) / total
        )
    
    async def _maintenance_loop(self):
        """Background maintenance and cleanup."""
        self.maintenance_running = True
        
        while self.maintenance_running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._perform_maintenance()
            except Exception as e:
                logger.error(f"Error in FAISS maintenance loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_maintenance(self):
        """Perform maintenance tasks."""
        try:
            logger.debug("🔧 Performing FAISS maintenance...")
            
            current_time = time.time()
            agents_to_remove = []
            
            with self.lock:
                # Find inactive agents (not accessed in last hour)
                for agent_id, agent_index in self.agent_indices.items():
                    if (current_time - agent_index.last_accessed) > 3600:  # 1 hour
                        agents_to_remove.append(agent_id)
                
                # Remove inactive agents
                for agent_id in agents_to_remove:
                    del self.agent_indices[agent_id]
                    logger.info(f"🗑️ Removed inactive FAISS index: {agent_id}")
            
            # Update memory usage stats
            await self._update_memory_stats()
            
            if agents_to_remove:
                logger.info(f"🧹 FAISS maintenance: removed {len(agents_to_remove)} inactive indices")
            
        except Exception as e:
            logger.error(f"Error during FAISS maintenance: {e}")
    
    async def _update_memory_stats(self):
        """Update memory usage statistics."""
        try:
            # Estimate memory usage (rough calculation)
            total_vectors = sum(idx.size for idx in self.agent_indices.values())
            vector_memory = total_vectors * self.config.vector_dimension * 4  # 4 bytes per float32
            metadata_memory = total_vectors * 1024  # ~1KB per document metadata (estimate)
            
            total_memory_bytes = vector_memory + metadata_memory
            self.performance_stats['memory_usage_mb'] = total_memory_bytes / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Error updating memory stats: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        await self._update_memory_stats()
        
        with self.lock:
            agent_stats = {}
            for agent_id, agent_index in self.agent_indices.items():
                agent_stats[agent_id] = {
                    'size': agent_index.size,
                    'access_count': agent_index.access_count,
                    'last_accessed': agent_index.last_accessed,
                    'avg_search_time_ms': agent_index.average_search_time * 1000,
                    'age_hours': (time.time() - agent_index.creation_time) / 3600
                }
        
        return {
            'total_searches': self.performance_stats['total_searches'],
            'total_cache_hits': self.performance_stats['total_cache_hits'],
            'cache_hit_rate': (
                self.performance_stats['total_cache_hits'] / 
                max(1, self.performance_stats['total_searches'])
            ),
            'avg_search_time_ms': self.performance_stats['avg_search_time'] * 1000,
            'memory_usage_mb': self.performance_stats['memory_usage_mb'],
            'active_agents': len(self.agent_indices),
            'max_agents': self.config.max_agents_in_memory,
            'agent_stats': agent_stats,
            'config': {
                'vector_dimension': self.config.vector_dimension,
                'max_vectors_per_agent': self.config.max_vectors_per_agent,
                'enable_gpu': self.config.enable_gpu,
                'omp_threads': self.config.omp_num_threads
            }
        }
    
    async def health_check(self) -> bool:
        """Health check for FAISS hot tier."""
        try:
            # Check if embedding model is available
            if not self.embedding_model:
                return False
            
            # Test embedding generation
            test_embedding = await self._generate_embedding("test")
            if len(test_embedding) != self.config.vector_dimension:
                return False
            
            # Check if indices are accessible
            with self.lock:
                for agent_id, agent_index in self.agent_indices.items():
                    if agent_index.index is None:
                        logger.warning(f"FAISS index is None for {agent_id}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"FAISS health check failed: {e}")
            return False
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed information about an agent's FAISS index."""
        with self.lock:
            if agent_id not in self.agent_indices:
                return {"error": "Agent not found in hot tier"}
            
            agent_index = self.agent_indices[agent_id]
            return {
                'agent_id': agent_id,
                'size': agent_index.size,
                'access_count': agent_index.access_count,
                'last_accessed': agent_index.last_accessed,
                'creation_time': agent_index.creation_time,
                'avg_search_time_ms': agent_index.average_search_time * 1000,
                'recent_search_times': list(agent_index.search_times),
                'index_type': type(agent_index.index).__name__
            }
    
    async def remove_agent_index(self, agent_id: str) -> bool:
        """Remove an agent's FAISS index from hot tier."""
        try:
            with self.lock:
                if agent_id in self.agent_indices:
                    del self.agent_indices[agent_id]
                    logger.info(f"🗑️ Removed FAISS index for {agent_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error removing FAISS index: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown FAISS hot tier gracefully."""
        logger.info("🛑 Shutting down FAISS hot tier...")
        
        try:
            # Stop maintenance loop
            self.maintenance_running = False
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clear all indices
            with self.lock:
                agent_count = len(self.agent_indices)
                self.agent_indices.clear()
            
            # Log final statistics
            stats = await self.get_performance_stats()
            logger.info(f"📊 Final FAISS stats: {stats['total_searches']} searches, "
                       f"{stats['cache_hit_rate']*100:.1f}% hit rate, "
                       f"{stats['memory_usage_mb']:.1f}MB memory")
            
            logger.info(f"✅ FAISS hot tier shutdown complete ({agent_count} indices cleared)")
            
        except Exception as e:
            logger.error(f"Error during FAISS shutdown: {e}")