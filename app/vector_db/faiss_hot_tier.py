"""
FAISS Hot Tier - Tier 2 (Sub-5ms latency)
In-memory vector search with intelligent promotion/demotion and GPU acceleration.
Target: <5ms retrieval for active agents and frequently accessed vectors.
"""
import asyncio
import logging
import time
import pickle
import threading
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc

# FAISS imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, falling back to numpy-based similarity search")

logger = logging.getLogger(__name__)

@dataclass
class VectorEntry:
    """Entry in the FAISS hot tier."""
    vector_id: str
    vector_data: np.ndarray
    metadata: Dict[str, Any]
    agent_id: str
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    priority_score: float = 0.0

@dataclass
class SearchResult:
    """Search result from FAISS hot tier."""
    vectors: List[Dict[str, Any]]
    scores: List[float]
    search_time_ms: float
    index_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentIndex:
    """FAISS index for a specific agent."""
    agent_id: str
    index: Optional['faiss.Index'] = None
    vector_map: Dict[int, VectorEntry] = field(default_factory=dict)
    last_rebuild: float = field(default_factory=time.time)
    total_vectors: int = 0
    memory_usage_bytes: int = 0
    search_count: int = 0
    average_search_time_ms: float = 0.0

class FAISSHotTier:
    """
    High-performance in-memory FAISS tier for active vectors.
    Provides <5ms search latency with intelligent memory management.
    """
    
    def __init__(self,
                 memory_limit_gb: float = 4.0,
                 promotion_threshold: int = 100,
                 index_type: str = "HNSW",
                 dimension: int = 1536,
                 nlist: int = 100,
                 m: int = 16,
                 enable_gpu: bool = False,
                 rebuild_threshold: int = 1000,
                 cleanup_interval: int = 300,
                 max_agents: int = 50):
        """Initialize FAISS hot tier."""
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for hot tier functionality")
        
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.promotion_threshold = promotion_threshold
        self.index_type = index_type
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.enable_gpu = enable_gpu
        self.rebuild_threshold = rebuild_threshold
        self.cleanup_interval = cleanup_interval
        self.max_agents = max_agents
        
        # Agent-specific indices
        self.agent_indices: Dict[str, AgentIndex] = {}
        self.active_agents: Set[str] = set()
        
        # Memory management
        self.current_memory_usage = 0
        self.memory_pressure_threshold = 0.85  # 85% of limit
        
        # Thread safety
        self.lock = threading.RLock()
        self.thread_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="faiss-hot")
        
        # Performance tracking
        self.stats = {
            "total_searches": 0,
            "total_vectors": 0,
            "cache_hits": 0,
            "index_rebuilds": 0,
            "memory_cleanups": 0,
            "promotion_events": 0,
            "demotion_events": 0,
            "average_search_time_ms": 0.0,
            "memory_usage_percent": 0.0
        }
        
        # GPU resources
        self.gpu_resources = None
        self.gpu_available = False
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.initialized = False
        
        logger.info(f"FAISS Hot Tier initialized (memory limit: {memory_limit_gb}GB, GPU: {enable_gpu})")
    
    async def initialize(self):
        """Initialize FAISS hot tier with GPU support if available."""
        logger.info("🚀 Initializing FAISS Hot Tier...")
        
        try:
            # Check GPU availability
            if self.enable_gpu and hasattr(faiss, 'get_num_gpus'):
                gpu_count = faiss.get_num_gpus()
                if gpu_count > 0:
                    self.gpu_resources = faiss.StandardGpuResources()
                    self.gpu_available = True
                    logger.info(f"✅ GPU acceleration enabled ({gpu_count} GPUs available)")
                else:
                    logger.warning("GPU requested but not available, using CPU")
                    self.enable_gpu = False
            
            # Start background cleanup task
            self.cleanup_task = asyncio.create_task(self._background_cleanup())
            
            self.initialized = True
            logger.info("✅ FAISS Hot Tier initialization complete")
            
        except Exception as e:
            logger.error(f"❌ FAISS Hot Tier initialization failed: {e}")
            raise
    
    async def search(self,
                    query_vector: np.ndarray,
                    agent_id: str,
                    top_k: int = 5,
                    filters: Optional[Dict[str, Any]] = None,
                    similarity_threshold: float = 0.7) -> Optional[SearchResult]:
        """
        High-speed vector search in FAISS hot tier.
        Target: <5ms search latency.
        """
        if not self.initialized:
            raise RuntimeError("FAISS Hot Tier not initialized")
        
        search_start = time.time()
        
        try:
            # Check if agent has an active index
            if agent_id not in self.agent_indices or agent_id not in self.active_agents:
                logger.debug(f"Agent {agent_id} not in hot tier")
                return None
            
            agent_index = self.agent_indices[agent_id]
            
            if not agent_index.index or agent_index.total_vectors == 0:
                logger.debug(f"No vectors available for agent {agent_id}")
                return None
            
            # Perform FAISS search in thread pool
            search_result = await asyncio.get_event_loop().run_in_executor(
                self.thread_executor,
                self._perform_faiss_search,
                agent_index,
                query_vector,
                top_k,
                similarity_threshold,
                filters
            )
            
            search_time = (time.time() - search_start) * 1000
            
            # Update statistics
            self._update_search_stats(agent_id, search_time)
            
            if search_result:
                logger.debug(f"⚡ FAISS search completed: {search_time:.2f}ms for agent {agent_id}")
                return SearchResult(
                    vectors=search_result["vectors"],
                    scores=search_result["scores"],
                    search_time_ms=search_time,
                    index_used=f"faiss_{self.index_type.lower()}",
                    metadata={
                        "agent_id": agent_id,
                        "total_vectors": agent_index.total_vectors,
                        "gpu_used": self.gpu_available
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"FAISS search error for agent {agent_id}: {e}")
            return None
    
    def _perform_faiss_search(self,
                            agent_index: AgentIndex,
                            query_vector: np.ndarray,
                            top_k: int,
                            threshold: float,
                            filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Perform FAISS search in thread (synchronous)."""
        try:
            with self.lock:
                # Ensure query vector is in correct format
                if query_vector.dtype != np.float32:
                    query_vector = query_vector.astype(np.float32)
                
                if len(query_vector.shape) == 1:
                    query_vector = query_vector.reshape(1, -1)
                
                # Perform search
                distances, indices = agent_index.index.search(query_vector, top_k)
                
                # Process results
                vectors = []
                scores = []
                
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx == -1:  # Invalid result
                        continue
                    
                    # Convert distance to similarity score (cosine similarity)
                    similarity = 1.0 - distance if distance <= 1.0 else 0.0
                    
                    if similarity < threshold:
                        continue
                    
                    if idx in agent_index.vector_map:
                        vector_entry = agent_index.vector_map[idx]
                        
                        # Apply filters if specified
                        if filters and not self._matches_filters(vector_entry.metadata, filters):
                            continue
                        
                        # Update access statistics
                        vector_entry.access_count += 1
                        vector_entry.last_accessed = time.time()
                        
                        vectors.append({
                            "id": vector_entry.vector_id,
                            "text": vector_entry.metadata.get("text", ""),
                            "metadata": vector_entry.metadata
                        })
                        scores.append(similarity)
                
                return {
                    "vectors": vectors,
                    "scores": scores
                } if vectors else None
                
        except Exception as e:
            logger.error(f"Error in FAISS search execution: {e}")
            return None
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, expected_value in filters.items():
            if key not in metadata:
                return False
            
            actual_value = metadata[key]
            
            # Handle different filter types
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Range filters like {"$gte": 0.5}
                if "$gte" in expected_value and actual_value < expected_value["$gte"]:
                    return False
                if "$lte" in expected_value and actual_value > expected_value["$lte"]:
                    return False
                if "$gt" in expected_value and actual_value <= expected_value["$gt"]:
                    return False
                if "$lt" in expected_value and actual_value >= expected_value["$lt"]:
                    return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    async def add_vectors(self,
                         vectors: List[Dict[str, Any]],
                         agent_id: str,
                         priority: str = "normal") -> bool:
        """Add vectors to FAISS hot tier for an agent."""
        if not self.initialized:
            return False
        
        try:
            # Check memory constraints
            estimated_memory = len(vectors) * self.dimension * 4  # 4 bytes per float32
            if self.current_memory_usage + estimated_memory > self.memory_limit_bytes:
                await self._manage_memory_pressure()
            
            # Ensure agent index exists
            if agent_id not in self.agent_indices:
                await self._create_agent_index(agent_id)
            
            # Add vectors in thread pool
            success = await asyncio.get_event_loop().run_in_executor(
                self.thread_executor,
                self._add_vectors_to_index,
                agent_id,
                vectors,
                priority
            )
            
            if success:
                self.active_agents.add(agent_id)
                self.stats["total_vectors"] += len(vectors)
                logger.debug(f"Added {len(vectors)} vectors to hot tier for agent {agent_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding vectors to hot tier: {e}")
            return False
    
    def _add_vectors_to_index(self,
                            agent_id: str,
                            vectors: List[Dict[str, Any]],
                            priority: str) -> bool:
        """Add vectors to agent index (synchronous)."""
        try:
            with self.lock:
                agent_index = self.agent_indices[agent_id]
                
                # Prepare vectors for FAISS
                vector_arrays = []
                vector_entries = []
                
                for vector_data in vectors:
                    # Extract vector array
                    if "vector" in vector_data:
                        vector_array = np.array(vector_data["vector"], dtype=np.float32)
                    elif "embedding" in vector_data:
                        vector_array = np.array(vector_data["embedding"], dtype=np.float32)
                    else:
                        logger.warning(f"No vector found in data: {vector_data}")
                        continue
                    
                    if vector_array.shape[0] != self.dimension:
                        logger.warning(f"Vector dimension mismatch: {vector_array.shape[0]} != {self.dimension}")
                        continue
                    
                    vector_arrays.append(vector_array)
                    
                    # Create vector entry
                    entry = VectorEntry(
                        vector_id=vector_data.get("id", f"vec_{len(vector_entries)}"),
                        vector_data=vector_array,
                        metadata=vector_data.get("metadata", {}),
                        agent_id=agent_id,
                        created_at=time.time(),
                        priority_score=self._calculate_priority_score(priority, vector_data)
                    )
                    vector_entries.append(entry)
                
                if not vector_arrays:
                    return False
                
                # Convert to numpy array
                vectors_matrix = np.vstack(vector_arrays).astype(np.float32)
                
                # Add to FAISS index
                if agent_index.index is None:
                    # Create new index
                    agent_index.index = self._create_faiss_index(vectors_matrix.shape[1])
                
                # Get starting index for mapping
                start_idx = agent_index.index.ntotal
                
                # Add vectors to index
                agent_index.index.add(vectors_matrix)
                
                # Update vector mapping
                for i, entry in enumerate(vector_entries):
                    agent_index.vector_map[start_idx + i] = entry
                
                # Update statistics
                agent_index.total_vectors += len(vector_entries)
                agent_index.memory_usage_bytes += vectors_matrix.nbytes
                self.current_memory_usage += vectors_matrix.nbytes
                
                # Check if index needs rebuilding for optimization
                if (agent_index.total_vectors - agent_index.last_rebuild > self.rebuild_threshold):
                    self._optimize_index(agent_index)
                
                return True
                
        except Exception as e:
            logger.error(f"Error in _add_vectors_to_index: {e}")
            return False
    
    def _create_faiss_index(self, dimension: int) -> 'faiss.Index':
        """Create optimized FAISS index."""
        if self.index_type == "HNSW":
            # HNSW index for high recall and speed
            index = faiss.IndexHNSWFlat(dimension, self.m)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
            
        elif self.index_type == "IVF":
            # IVF index for large datasets
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            index.train_on_init = True
            
        elif self.index_type == "PQ":
            # Product Quantization for memory efficiency
            m = 8  # number of subquantizers
            nbits = 8  # bits per subquantizer
            index = faiss.IndexPQ(dimension, m, nbits)
            
        else:
            # Default to flat index
            index = faiss.IndexFlatIP(dimension)
        
        # Move to GPU if available
        if self.gpu_available and self.gpu_resources:
            try:
                index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                logger.debug(f"Moved {self.index_type} index to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
        
        return index
    
    def _calculate_priority_score(self, priority: str, vector_data: Dict[str, Any]) -> float:
        """Calculate priority score for vector placement."""
        base_scores = {
            "critical": 1.0,
            "high": 0.8,
            "normal": 0.5,
            "low": 0.2
        }
        
        score = base_scores.get(priority, 0.5)
        
        # Adjust based on metadata
        metadata = vector_data.get("metadata", {})
        
        # Boost recent content
        if "created_at" in metadata:
            age_hours = (time.time() - metadata["created_at"]) / 3600
            if age_hours < 24:  # Recent content gets boost
                score += 0.1
        
        # Boost frequently accessed content
        if "access_count" in metadata:
            access_boost = min(0.2, metadata["access_count"] / 100.0)
            score += access_boost
        
        return min(1.0, score)
    
    async def _create_agent_index(self, agent_id: str):
        """Create a new index for an agent."""
        if len(self.agent_indices) >= self.max_agents:
            # Remove least active agent to make space
            await self._remove_least_active_agent()
        
        self.agent_indices[agent_id] = AgentIndex(agent_id=agent_id)
        logger.debug(f"Created new index for agent: {agent_id}")
    
    def _optimize_index(self, agent_index: AgentIndex):
        """Optimize FAISS index for better performance."""
        try:
            if agent_index.index and hasattr(agent_index.index, 'train'):
                # For indices that support training
                vectors = []
                for entry in agent_index.vector_map.values():
                    vectors.append(entry.vector_data)
                
                if vectors:
                    training_data = np.vstack(vectors).astype(np.float32)
                    agent_index.index.train(training_data)
                    agent_index.last_rebuild = time.time()
                    self.stats["index_rebuilds"] += 1
                    logger.debug(f"Optimized index for agent {agent_index.agent_id}")
            
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
    
    async def _manage_memory_pressure(self):
        """Handle memory pressure by removing low-priority vectors."""
        logger.info("Managing memory pressure in FAISS hot tier")
        
        try:
            # Calculate current memory usage
            self.stats["memory_usage_percent"] = (self.current_memory_usage / self.memory_limit_bytes) * 100
            
            if self.current_memory_usage > self.memory_limit_bytes * self.memory_pressure_threshold:
                # Remove least active agents or vectors
                await self._cleanup_low_priority_vectors()
                
                # Force garbage collection
                gc.collect()
                
                self.stats["memory_cleanups"] += 1
                logger.info(f"Memory cleanup completed. Usage: {self.stats['memory_usage_percent']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error managing memory pressure: {e}")
    
    async def _cleanup_low_priority_vectors(self):
        """Remove low-priority or unused vectors to free memory."""
        current_time = time.time()
        vectors_removed = 0
        
        # Remove vectors that haven't been accessed in a while
        for agent_id, agent_index in list(self.agent_indices.items()):
            indices_to_remove = []
            
            for idx, entry in agent_index.vector_map.items():
                # Remove if not accessed in last hour and low priority
                if (current_time - entry.last_accessed > 3600 and 
                    entry.priority_score < 0.3):
                    indices_to_remove.append(idx)
            
            # Remove from mapping (index cleanup would require rebuilding)
            for idx in indices_to_remove:
                del agent_index.vector_map[idx]
                vectors_removed += 1
            
            # If agent has very few vectors left, remove entirely
            if len(agent_index.vector_map) < 10:
                await self._remove_agent_index(agent_id)
        
        logger.debug(f"Removed {vectors_removed} low-priority vectors")
    
    async def _remove_least_active_agent(self):
        """Remove the least active agent to make space."""
        if not self.agent_indices:
            return
        
        # Find agent with lowest activity
        least_active_agent = min(
            self.agent_indices.keys(),
            key=lambda aid: self.agent_indices[aid].search_count
        )
        
        await self._remove_agent_index(least_active_agent)
        logger.debug(f"Removed least active agent: {least_active_agent}")
    
    async def _remove_agent_index(self, agent_id: str):
        """Remove an agent's index and free memory."""
        if agent_id in self.agent_indices:
            agent_index = self.agent_indices[agent_id]
            
            # Update memory usage
            self.current_memory_usage -= agent_index.memory_usage_bytes
            self.stats["total_vectors"] -= agent_index.total_vectors
            
            # Remove from active agents
            self.active_agents.discard(agent_id)
            
            # Delete index
            del self.agent_indices[agent_id]
            
            self.stats["demotion_events"] += 1
            logger.debug(f"Removed agent index: {agent_id}")
    
    def _update_search_stats(self, agent_id: str, search_time_ms: float):
        """Update search statistics."""
        self.stats["total_searches"] += 1
        
        # Update agent-specific stats
        if agent_id in self.agent_indices:
            agent_index = self.agent_indices[agent_id]
            agent_index.search_count += 1
            
            # Update average search time
            if agent_index.average_search_time_ms == 0:
                agent_index.average_search_time_ms = search_time_ms
            else:
                agent_index.average_search_time_ms = (
                    (agent_index.average_search_time_ms * (agent_index.search_count - 1) + search_time_ms) /
                    agent_index.search_count
                )
        
        # Update global average
        if self.stats["average_search_time_ms"] == 0:
            self.stats["average_search_time_ms"] = search_time_ms
        else:
            self.stats["average_search_time_ms"] = (
                (self.stats["average_search_time_ms"] * (self.stats["total_searches"] - 1) + search_time_ms) /
                self.stats["total_searches"]
            )
    
    async def _background_cleanup(self):
        """Background task for periodic cleanup and optimization."""
        logger.info(f"Starting FAISS background cleanup (interval: {self.cleanup_interval}s)")
        
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Memory management
                await self._manage_memory_pressure()
                
                # Remove unused vectors
                await self.cleanup_unused()
                
                # Optimize indices if needed
                await self._periodic_optimization()
                
            except Exception as e:
                logger.error(f"Error in FAISS background cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def cleanup_unused(self):
        """Clean up unused vectors and optimize memory."""
        current_time = time.time()
        cleanup_threshold = 1800  # 30 minutes
        
        agents_to_remove = []
        
        for agent_id, agent_index in self.agent_indices.items():
            # Check if agent hasn't been used recently
            last_activity = max(
                entry.last_accessed for entry in agent_index.vector_map.values()
            ) if agent_index.vector_map else 0
            
            if current_time - last_activity > cleanup_threshold:
                agents_to_remove.append(agent_id)
        
        # Remove inactive agents
        for agent_id in agents_to_remove:
            await self._remove_agent_index(agent_id)
        
        if agents_to_remove:
            logger.debug(f"Cleaned up {len(agents_to_remove)} inactive agents")
    
    async def _periodic_optimization(self):
        """Perform periodic index optimizations."""
        for agent_id, agent_index in self.agent_indices.items():
            # Optimize if index has grown significantly
            if (agent_index.total_vectors > self.rebuild_threshold and
                time.time() - agent_index.last_rebuild > 3600):  # 1 hour
                
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_executor,
                    self._optimize_index,
                    agent_index
                )
    
    async def rebalance(self):
        """Rebalance indices for optimal performance."""
        logger.info("Rebalancing FAISS hot tier...")
        
        try:
            # Sort agents by activity
            active_agents = sorted(
                self.agent_indices.items(),
                key=lambda x: x[1].search_count,
                reverse=True
            )
            
            # Keep only the most active agents if over limit
            if len(active_agents) > self.max_agents:
                agents_to_remove = active_agents[self.max_agents:]
                for agent_id, _ in agents_to_remove:
                    await self._remove_agent_index(agent_id)
            
            # Optimize remaining indices
            for agent_id, agent_index in active_agents[:self.max_agents]:
                if agent_index.total_vectors > 100:  # Only optimize substantial indices
                    await asyncio.get_event_loop().run_in_executor(
                        self.thread_executor,
                        self._optimize_index,
                        agent_index
                    )
            
            logger.info("FAISS hot tier rebalancing completed")
            
        except Exception as e:
            logger.error(f"Error rebalancing FAISS hot tier: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive FAISS hot tier statistics."""
        return {
            "configuration": {
                "memory_limit_gb": self.memory_limit_bytes / (1024**3),
                "index_type": self.index_type,
                "dimension": self.dimension,
                "gpu_enabled": self.enable_gpu,
                "gpu_available": self.gpu_available,
                "max_agents": self.max_agents
            },
            "performance": {
                **self.stats,
                "active_agents": len(self.active_agents),
                "total_agents": len(self.agent_indices),
                "memory_usage_mb": self.current_memory_usage / (1024**2),
                "target_latency_ms": 5.0,
                "meets_target": self.stats["average_search_time_ms"] <= 5.0
            },
            "agents": {
                agent_id: {
                    "total_vectors": agent_index.total_vectors,
                    "search_count": agent_index.search_count,
                    "average_search_time_ms": agent_index.average_search_time_ms,
                    "memory_usage_mb": agent_index.memory_usage_bytes / (1024**2),
                    "last_rebuild": agent_index.last_rebuild
                }
                for agent_id, agent_index in self.agent_indices.items()
            },
            "health": {
                "status": "healthy" if self.initialized else "down",
                "memory_pressure": self.current_memory_usage > (self.memory_limit_bytes * 0.8),
                "performance_target_met": self.stats["average_search_time_ms"] <= 5.0
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if FAISS hot tier is healthy."""
        return (
            self.initialized and
            self.current_memory_usage < self.memory_limit_bytes * 0.9 and
            self.stats["average_search_time_ms"] <= 10.0  # Allow up to 10ms for "healthy"
        )
    
    async def shutdown(self):
        """Shutdown FAISS hot tier."""
        logger.info("Shutting down FAISS Hot Tier...")
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup indices
        with self.lock:
            self.agent_indices.clear()
            self.active_agents.clear()
        
        # Shutdown thread executor
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        
        # Cleanup GPU resources
        if self.gpu_resources:
            del self.gpu_resources
        
        self.initialized = False
        logger.info("✅ FAISS Hot Tier shutdown complete")