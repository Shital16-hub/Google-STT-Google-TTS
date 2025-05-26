"""
Redis Vector Cache - Tier 1 (Sub-millisecond latency)
Ultra-fast vector caching with intelligent expiration and compression.
Target: <1ms retrieval for hot vectors and frequent queries.
"""
import asyncio
import logging
import time
import json
import pickle
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import zlib
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.commands.search.field import VectorField, NumericField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Redis cache entry with metadata."""
    vector_data: Dict[str, Any]
    scores: List[float]
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    compressed: bool = False
    namespace: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "vector_data": json.dumps(self.vector_data),
            "scores": json.dumps(self.scores),
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "compressed": self.compressed,
            "namespace": self.namespace
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            vector_data=json.loads(data["vector_data"]),
            scores=json.loads(data["scores"]),
            created_at=float(data["created_at"]),
            access_count=int(data.get("access_count", 0)),
            last_accessed=float(data.get("last_accessed", time.time())),
            compressed=bool(data.get("compressed", False)),
            namespace=data.get("namespace", "")
        )

@dataclass
class SearchResult:
    """Search result from Redis cache."""
    vectors: List[Dict[str, Any]]
    scores: List[float]
    cache_hit: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class RedisVectorCache:
    """
    Ultra-fast Redis vector cache for sub-millisecond retrieval.
    Optimized for high-frequency vector searches with intelligent caching.
    """
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 cache_size: int = 50000,
                 ttl_seconds: int = 3600,
                 compression_enabled: bool = True,
                 compression_threshold: int = 1024,
                 enable_vector_search: bool = True,
                 max_connections: int = 20):
        """Initialize Redis vector cache."""
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold
        self.enable_vector_search = enable_vector_search
        self.max_connections = max_connections
        
        # Redis clients
        self.client: Optional[redis.Redis] = None
        self.connection_pool: Optional[redis.ConnectionPool] = None
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "compression_saves": 0,
            "total_queries": 0,
            "average_latency_ms": 0.0,
            "memory_usage_bytes": 0
        }
        
        # Performance tracking
        self.latency_history: List[float] = []
        self.max_latency_history = 1000
        
        # Cache key prefixes
        self.VECTOR_KEY_PREFIX = "vec:"
        self.QUERY_KEY_PREFIX = "query:"
        self.META_KEY_PREFIX = "meta:"
        self.INDEX_NAME = "vector_index"
        
        self.initialized = False
        logger.info(f"Redis Vector Cache initialized for {host}:{port}")
    
    async def initialize(self):
        """Initialize Redis connection and vector search capabilities."""
        logger.info("🚀 Initializing Redis Vector Cache...")
        
        try:
            # Create connection pool for optimal performance
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=1.0,
                socket_connect_timeout=2.0,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Create Redis client
            self.client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False  # We'll handle encoding manually for binary data
            )
            
            # Test connection
            await self.client.ping()
            logger.info(f"✅ Redis connection established: {self.host}:{self.port}")
            
            # Setup vector search index if enabled
            if self.enable_vector_search:
                await self._setup_vector_search_index()
            
            # Configure memory optimization
            await self._configure_memory_optimization()
            
            self.initialized = True
            logger.info("✅ Redis Vector Cache initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            raise
    
    async def _setup_vector_search_index(self):
        """Setup RediSearch vector index for semantic search."""
        try:
            # Check if index already exists
            try:
                await self.client.ft(self.INDEX_NAME).info()
                logger.info(f"Vector search index '{self.INDEX_NAME}' already exists")
                return
            except:
                # Index doesn't exist, create it
                pass
            
            # Define schema for vector search
            schema = [
                VectorField(
                    "vector",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 1536,  # OpenAI embedding dimension
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 10000,
                        "M": 16,
                        "EF_CONSTRUCTION": 200
                    }
                ),
                TextField("namespace"),
                TextField("agent_id"),
                NumericField("score"),
                NumericField("created_at"),
                NumericField("access_count")
            ]
            
            # Create index
            await self.client.ft(self.INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.VECTOR_KEY_PREFIX],
                    index_type=IndexType.HASH
                )
            )
            
            logger.info(f"✅ Created vector search index: {self.INDEX_NAME}")
            
        except Exception as e:
            logger.warning(f"Failed to setup vector search index: {e}")
            # Disable vector search if setup fails
            self.enable_vector_search = False
    
    async def _configure_memory_optimization(self):
        """Configure Redis for optimal memory usage."""
        try:
            # Set memory policy for cache behavior
            await self.client.config_set("maxmemory-policy", "allkeys-lru")
            
            # Configure hash compression
            await self.client.config_set("hash-max-ziplist-entries", "1000")
            await self.client.config_set("hash-max-ziplist-value", "1024")
            
            logger.debug("Configured Redis memory optimization")
            
        except Exception as e:
            logger.warning(f"Failed to configure memory optimization: {e}")
    
    async def search(self,
                    query_vector: np.ndarray,
                    namespace: str,
                    top_k: int = 5,
                    filters: Optional[Dict[str, Any]] = None,
                    similarity_threshold: float = 0.7) -> Optional[SearchResult]:
        """
        Ultra-fast vector search in Redis cache.
        Target: <1ms retrieval time.
        """
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        search_start = time.time()
        
        try:
            # Generate cache key for this query
            query_key = self._generate_query_key(query_vector, namespace, top_k, filters)
            cache_key = f"{self.QUERY_KEY_PREFIX}{query_key}"
            
            # Try to get cached result first
            cached_result = await self._get_cached_query_result(cache_key)
            if cached_result:
                search_time = (time.time() - search_start) * 1000
                self._update_stats("hit", search_time)
                
                logger.debug(f"🚀 Redis cache hit: {search_time:.3f}ms for {namespace}")
                return cached_result
            
            # If vector search is enabled, try semantic search
            if self.enable_vector_search:
                result = await self._vector_search(query_vector, namespace, top_k, similarity_threshold)
                if result:
                    # Cache the result for future queries
                    await self._cache_query_result(cache_key, result)
                    
                    search_time = (time.time() - search_start) * 1000
                    self._update_stats("hit", search_time)
                    return result
            
            # No results found
            search_time = (time.time() - search_start) * 1000
            self._update_stats("miss", search_time)
            return None
            
        except Exception as e:
            logger.error(f"Redis search error: {e}")
            search_time = (time.time() - search_start) * 1000
            self._update_stats("miss", search_time)
            return None
    
    async def _vector_search(self,
                           query_vector: np.ndarray,
                           namespace: str,
                           top_k: int,
                           similarity_threshold: float) -> Optional[SearchResult]:
        """Perform vector similarity search using RediSearch."""
        try:
            # Convert numpy array to bytes for Redis
            vector_bytes = query_vector.astype(np.float32).tobytes()
            
            # Build search query
            query = (
                Query(f"@namespace:{namespace}")
                .return_fields("vector", "scores", "metadata")
                .sort_by("score", asc=False)
                .limit_offset(0, top_k)
                .dialect(2)
            )
            
            # Add vector similarity search
            query = query.add_param("query_vector", vector_bytes)
            query_str = f"*=>[KNN {top_k} @vector $query_vector AS score]"
            
            # Execute search
            results = await self.client.ft(self.INDEX_NAME).search(query_str, query)
            
            if not results.docs:
                return None
            
            # Process results
            vectors = []
            scores = []
            
            for doc in results.docs:
                # Reconstruct vector data
                vector_data = json.loads(doc.get("metadata", "{}"))
                score = float(doc.get("score", 0.0))
                
                if score >= similarity_threshold:
                    vectors.append(vector_data)
                    scores.append(score)
            
            if not vectors:
                return None
            
            return SearchResult(
                vectors=vectors,
                scores=scores,
                cache_hit=True,
                metadata={"search_type": "vector_similarity"}
            )
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return None
    
    async def store(self,
                   query_vector: np.ndarray,
                   result: 'SearchResult',
                   namespace: str,
                   ttl: Optional[int] = None) -> bool:
        """
        Store search result in Redis cache with compression.
        Optimized for ultra-fast retrieval.
        """
        if not self.initialized:
            return False
        
        try:
            store_start = time.time()
            ttl = ttl or self.ttl_seconds
            
            # Generate cache key
            query_key = self._generate_query_key(query_vector, namespace, len(result.vectors))
            cache_key = f"{self.QUERY_KEY_PREFIX}{query_key}"
            
            # Create cache entry
            entry = CacheEntry(
                vector_data={"vectors": result.vectors},
                scores=result.scores,
                created_at=time.time(),
                namespace=namespace
            )
            
            # Serialize entry
            entry_data = entry.to_dict()
            
            # Apply compression if enabled and data is large enough
            if self.compression_enabled:
                serialized = json.dumps(entry_data).encode('utf-8')
                if len(serialized) > self.compression_threshold:
                    compressed_data = zlib.compress(serialized)
                    if len(compressed_data) < len(serialized):
                        entry_data["compressed"] = True
                        data_to_store = compressed_data
                        self.stats["compression_saves"] += len(serialized) - len(compressed_data)
                    else:
                        data_to_store = serialized
                else:
                    data_to_store = serialized
            else:
                data_to_store = json.dumps(entry_data).encode('utf-8')
            
            # Store in Redis with expiration
            pipe = self.client.pipeline()
            await pipe.setex(cache_key, ttl, data_to_store)
            
            # Store individual vectors for vector search if enabled
            if self.enable_vector_search:
                for i, (vector_data, score) in enumerate(zip(result.vectors, result.scores)):
                    vector_key = f"{self.VECTOR_KEY_PREFIX}{namespace}:{i}:{int(time.time())}"
                    
                    # Store vector with metadata
                    vector_entry = {
                        "vector": query_vector.astype(np.float32).tobytes(),
                        "namespace": namespace,
                        "score": score,
                        "metadata": json.dumps(vector_data),
                        "created_at": time.time(),
                        "access_count": 0
                    }
                    
                    await pipe.hset(vector_key, mapping=vector_entry)
                    await pipe.expire(vector_key, ttl)
            
            # Execute pipeline
            await pipe.execute()
            
            store_time = (time.time() - store_start) * 1000
            self.stats["sets"] += 1
            
            logger.debug(f"Stored cache entry in {store_time:.3f}ms for {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in Redis cache: {e}")
            return False
    
    async def _get_cached_query_result(self, cache_key: str) -> Optional[SearchResult]:
        """Get cached query result from Redis."""
        try:
            data = await self.client.get(cache_key)
            if not data:
                return None
            
            # Try to decompress if needed
            try:
                # First try to load as JSON (uncompressed)
                entry_data = json.loads(data.decode('utf-8'))
            except:
                # Try decompression
                try:
                    decompressed = zlib.decompress(data)
                    entry_data = json.loads(decompressed.decode('utf-8'))
                except:
                    logger.warning(f"Failed to deserialize cached data for key: {cache_key}")
                    return None
            
            # Create cache entry
            entry = CacheEntry.from_dict(entry_data)
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Update in Redis (fire and forget)
            asyncio.create_task(self._update_access_stats(cache_key, entry))
            
            return SearchResult(
                vectors=entry.vector_data.get("vectors", []),
                scores=entry.scores,
                cache_hit=True,
                metadata={"cache_access_count": entry.access_count}
            )
            
        except Exception as e:
            logger.error(f"Error retrieving cached result: {e}")
            return None
    
    async def _cache_query_result(self, cache_key: str, result: SearchResult):
        """Cache a query result for future use."""
        try:
            entry = CacheEntry(
                vector_data={"vectors": result.vectors},
                scores=result.scores,
                created_at=time.time(),
                namespace=""  # Will be set by caller
            )
            
            # Serialize and store
            entry_data = json.dumps(entry.to_dict()).encode('utf-8')
            
            # Apply compression if beneficial
            if self.compression_enabled and len(entry_data) > self.compression_threshold:
                compressed = zlib.compress(entry_data)
                if len(compressed) < len(entry_data):
                    entry_data = compressed
            
            await self.client.setex(cache_key, self.ttl_seconds, entry_data)
            
        except Exception as e:
            logger.error(f"Error caching query result: {e}")
    
    async def _update_access_stats(self, cache_key: str, entry: CacheEntry):
        """Update access statistics for a cache entry."""
        try:
            # Update the entry in Redis with new stats
            updated_data = json.dumps(entry.to_dict()).encode('utf-8')
            await self.client.setex(cache_key, self.ttl_seconds, updated_data)
        except Exception as e:
            logger.debug(f"Failed to update access stats: {e}")
    
    def _generate_query_key(self,
                          query_vector: np.ndarray,
                          namespace: str,
                          top_k: int = 5,
                          filters: Optional[Dict[str, Any]] = None) -> str:
        """Generate a unique cache key for a query."""
        # Create hash from query parameters
        key_components = [
            namespace,
            str(top_k),
            hashlib.md5(query_vector.tobytes()).hexdigest()[:16]
        ]
        
        if filters:
            filter_str = json.dumps(filters, sort_keys=True)
            key_components.append(hashlib.md5(filter_str.encode()).hexdigest()[:8])
        
        return ":".join(key_components)
    
    def _update_stats(self, operation: str, latency_ms: float):
        """Update cache statistics."""
        self.stats["total_queries"] += 1
        
        if operation == "hit":
            self.stats["hits"] += 1
        elif operation == "miss":
            self.stats["misses"] += 1
        
        # Update latency tracking
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.max_latency_history:
            self.latency_history.pop(0)
        
        # Update average latency
        if self.latency_history:
            self.stats["average_latency_ms"] = sum(self.latency_history) / len(self.latency_history)
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information and statistics."""
        try:
            # Get Redis memory info
            memory_info = await self.client.memory_usage_bytes()
            
            # Get key count
            key_count = await self.client.dbsize()
            
            # Calculate hit rate
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate percentiles
            p95_latency = 0.0
            p99_latency = 0.0
            if len(self.latency_history) >= 20:
                sorted_latencies = sorted(self.latency_history)
                p95_idx = int(0.95 * len(sorted_latencies))
                p99_idx = int(0.99 * len(sorted_latencies))
                p95_latency = sorted_latencies[p95_idx]
                p99_latency = sorted_latencies[p99_idx]
            
            return {
                "connection_info": {
                    "host": self.host,
                    "port": self.port,
                    "db": self.db,
                    "connected": self.initialized
                },
                "cache_config": {
                    "cache_size": self.cache_size,
                    "ttl_seconds": self.ttl_seconds,
                    "compression_enabled": self.compression_enabled,
                    "vector_search_enabled": self.enable_vector_search
                },
                "performance_stats": {
                    "total_queries": self.stats["total_queries"],
                    "cache_hits": self.stats["hits"],
                    "cache_misses": self.stats["misses"],
                    "hit_rate_percent": hit_rate,
                    "average_latency_ms": self.stats["average_latency_ms"],
                    "p95_latency_ms": p95_latency,
                    "p99_latency_ms": p99_latency,
                    "sets": self.stats["sets"],
                    "evictions": self.stats["evictions"]
                },
                "memory_info": {
                    "memory_usage_bytes": memory_info if memory_info else 0,
                    "key_count": key_count,
                    "compression_saves_bytes": self.stats["compression_saves"]
                },
                "health": {
                    "status": "healthy" if self.initialized else "down",
                    "target_latency_ms": 1.0,
                    "meets_target": self.stats["average_latency_ms"] <= 1.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    async def cleanup_expired(self):
        """Clean up expired cache entries."""
        try:
            # Redis handles TTL automatically, but we can clean up metadata
            current_time = time.time()
            
            # Get all query keys
            query_keys = await self.client.keys(f"{self.QUERY_KEY_PREFIX}*")
            
            expired_count = 0
            for key in query_keys:
                ttl = await self.client.ttl(key)
                if ttl == -2:  # Key doesn't exist
                    expired_count += 1
            
            if expired_count > 0:
                self.stats["evictions"] += expired_count
                logger.debug(f"Cleaned up {expired_count} expired cache entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
    
    async def clear_namespace(self, namespace: str):
        """Clear all cached entries for a specific namespace."""
        try:
            # Find keys with this namespace
            pattern = f"{self.QUERY_KEY_PREFIX}*{namespace}*"
            keys = await self.client.keys(pattern)
            
            if keys:
                await self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries for namespace: {namespace}")
            
        except Exception as e:
            logger.error(f"Error clearing namespace {namespace}: {e}")
    
    async def invalidate_cache(self):
        """Invalidate entire cache."""
        try:
            await self.client.flushdb()
            
            # Reset statistics
            self.stats = {key: 0 for key in self.stats}
            self.latency_history.clear()
            
            logger.info("Cache invalidated successfully")
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    def is_healthy(self) -> bool:
        """Check if Redis cache is healthy."""
        return (
            self.initialized and
            self.client is not None and
            self.stats["average_latency_ms"] <= 2.0  # Allow up to 2ms for "healthy"
        )
    
    async def shutdown(self):
        """Shutdown Redis cache connection."""
        logger.info("Shutting down Redis Vector Cache...")
        
        try:
            if self.client:
                await self.client.close()
            
            if self.connection_pool:
                await self.connection_pool.disconnect()
            
            self.initialized = False
            logger.info("✅ Redis Vector Cache shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Redis shutdown: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()