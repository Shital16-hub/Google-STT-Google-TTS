"""
Redis Cache - Sub-Millisecond Response Layer
Provides <1ms response times for exact query matches and frequent results.
"""
import asyncio
import logging
import time
import json
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

@dataclass
class RedisConfig:
    """Redis configuration optimized for ultra-low latency."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    # Connection pooling for performance
    max_connections: int = 20
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = None
    
    # Performance settings
    decode_responses: bool = False  # Keep as bytes for performance
    encoding: str = "utf-8"
    connection_timeout: float = 5.0
    socket_timeout: float = 1.0
    
    # Cache settings
    default_ttl: int = 3600      # 1 hour default TTL
    max_key_length: int = 200    # Prevent extremely long keys
    compression_threshold: int = 1024  # Compress data > 1KB
    
    # Memory optimization
    max_memory_usage_mb: int = 512  # Maximum Redis memory usage
    eviction_policy: str = "allkeys-lru"  # LRU eviction when memory full
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            self.socket_keepalive_options = {
                1: 1,   # TCP_KEEPIDLE
                2: 3,   # TCP_KEEPINTVL  
                3: 5    # TCP_KEEPCNT
            }

class RedisCache:
    """
    Ultra-Fast Redis Cache for Vector Search Results
    
    Features:
    - Sub-millisecond response times for exact matches
    - Intelligent key generation and management
    - Automatic compression for large payloads
    - Connection pooling and health monitoring
    - Memory usage optimization with LRU eviction
    - Performance metrics and monitoring
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """Initialize Redis cache with optimized configuration."""
        self.config = config or RedisConfig()
        
        # Redis connection
        self.redis_client = None
        self.connection_pool = None
        
        # Performance tracking
        self.performance_stats = {
            'total_gets': 0,
            'total_sets': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_get_time': 0.0,
            'avg_set_time': 0.0,
            'compression_saves': 0,
            'total_compressed_bytes': 0
        }
        
        # Health monitoring
        self.connection_healthy = False
        self.last_health_check = 0
        
        # Key prefixes for organization
        self.key_prefixes = {
            'search_results': 'sr:',
            'agent_cache': 'ac:',
            'query_metadata': 'qm:',
            'performance': 'perf:'
        }
        
        logger.info("RedisCache initialized with optimized configuration")
    
    async def init(self):
        """Initialize Redis connection and configure for optimal performance."""
        logger.info("🚀 Initializing Redis cache...")
        
        try:
            # Create optimized connection pool
            self.connection_pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                connection_class=redis.Connection,
                decode_responses=self.config.decode_responses,
                encoding=self.config.encoding
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connection_timeout
            )
            
            # Test connection and configure Redis
            await self._test_and_configure_redis()
            
            self.connection_healthy = True
            logger.info("✅ Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis cache: {e}")
            raise
    
    async def _test_and_configure_redis(self):
        """Test connection and configure Redis for optimal performance."""
        try:
            # Test basic connectivity
            start_time = time.time()
            await self.redis_client.ping()
            ping_time = time.time() - start_time
            
            logger.info(f"📡 Redis ping: {ping_time*1000:.1f}ms")
            
            if ping_time > 0.01:  # 10ms
                logger.warning(f"⚠️ High Redis latency: {ping_time*1000:.1f}ms")
            
            # Configure Redis for performance
            config_commands = [
                ('maxmemory', f'{self.config.max_memory_usage_mb}mb'),
                ('maxmemory-policy', self.config.eviction_policy),
                ('tcp-keepalive', '60'),
                ('timeout', '300')
            ]
            
            for key, value in config_commands:
                try:
                    await self.redis_client.config_set(key, value)
                    logger.debug(f"Set Redis config: {key}={value}")
                except Exception as e:
                    logger.warning(f"Could not set Redis config {key}: {e}")
            
            # Get Redis info
            info = await self.redis_client.info()
            logger.info(f"📊 Redis version: {info.get('redis_version', 'unknown')}")
            logger.info(f"💾 Redis memory usage: {info.get('used_memory_human', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Redis configuration failed: {e}")
            raise
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate optimized cache key."""
        # Create deterministic key from arguments
        key_parts = [str(arg) for arg in args if arg is not None]
        key_string = "|".join(key_parts)
        
        # Hash if key would be too long
        if len(key_string) > self.config.max_key_length - len(prefix):
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}{key_hash}"
        
        return f"{prefix}{key_string}"
    
    def _compress_data(self, data: bytes) -> Tuple[bytes, bool]:
        """Compress data if it exceeds threshold."""
        if len(data) > self.config.compression_threshold:
            try:
                import gzip
                compressed = gzip.compress(data)
                if len(compressed) < len(data) * 0.8:  # Only if 20%+ savings
                    self.performance_stats['compression_saves'] += 1
                    self.performance_stats['total_compressed_bytes'] += len(data) - len(compressed)
                    return compressed, True
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
        
        return data, False
    
    def _decompress_data(self, data: bytes, is_compressed: bool) -> bytes:
        """Decompress data if it was compressed."""
        if is_compressed:
            try:
                import gzip
                return gzip.decompress(data)
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                return data
        
        return data
    
    async def get_search_results(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results for a query hash."""
        get_start = time.time()
        
        try:
            self.performance_stats['total_gets'] += 1
            
            # Generate cache key
            cache_key = self._generate_key(self.key_prefixes['search_results'], query_hash)
            
            # Get from Redis
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data is None:
                self.performance_stats['cache_misses'] += 1
                return None
            
            # Check if data is compressed
            compression_flag_key = f"{cache_key}:compressed"
            is_compressed = await self.redis_client.exists(compression_flag_key)
            
            # Decompress if necessary
            if is_compressed:
                cached_data = self._decompress_data(cached_data, True)
            
            # Deserialize results
            results = pickle.loads(cached_data)
            
            # Update performance stats
            get_time = time.time() - get_start
            self._update_get_stats(get_time, True)
            
            self.performance_stats['cache_hits'] += 1
            logger.debug(f"⚡ Redis cache hit: {len(results)} results in {get_time*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Redis get error: {e}")
            self.performance_stats['cache_misses'] += 1
            return None
    
    async def set_search_results(
        self,
        query_hash: str,
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache search results with optional TTL."""
        set_start = time.time()
        
        try:
            self.performance_stats['total_sets'] += 1
            
            if not results:
                return True  # Don't cache empty results
            
            # Generate cache key
            cache_key = self._generate_key(self.key_prefixes['search_results'], query_hash)
            
            # Serialize results
            serialized_data = pickle.dumps(results)
            
            # Compress if beneficial
            final_data, is_compressed = self._compress_data(serialized_data)
            
            # Set TTL
            ttl = ttl or self.config.default_ttl
            
            # Store in Redis with pipeline for atomic operation
            pipe = self.redis_client.pipeline()
            pipe.setex(cache_key, ttl, final_data)
            
            # Store compression flag if compressed
            if is_compressed:
                compression_flag_key = f"{cache_key}:compressed"
                pipe.setex(compression_flag_key, ttl, b"1")
            
            await pipe.execute()
            
            # Update performance stats
            set_time = time.time() - set_start
            self._update_set_stats(set_time)
            
            logger.debug(f"💾 Redis cache set: {len(results)} results in {set_time*1000:.1f}ms "
                        f"({'compressed' if is_compressed else 'uncompressed'})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis set error: {e}")
            return False
    
    async def get_agent_cache(self, agent_id: str, cache_type: str) -> Optional[Any]:
        """Get agent-specific cached data."""
        try:
            cache_key = self._generate_key(self.key_prefixes['agent_cache'], agent_id, cache_type)
            
            cached_data = await self.redis_client.get(cache_key)
            if cached_data is None:
                return None
            
            # Check compression
            compression_flag_key = f"{cache_key}:compressed"
            is_compressed = await self.redis_client.exists(compression_flag_key)
            
            if is_compressed:
                cached_data = self._decompress_data(cached_data, True)
            
            return pickle.loads(cached_data)
            
        except Exception as e:
            logger.error(f"Error getting agent cache: {e}")
            return None
    
    async def set_agent_cache(
        self,
        agent_id: str,
        cache_type: str,
        data: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set agent-specific cached data."""
        try:
            cache_key = self._generate_key(self.key_prefixes['agent_cache'], agent_id, cache_type)
            
            # Serialize data
            serialized_data = pickle.dumps(data)
            
            # Compress if beneficial
            final_data, is_compressed = self._compress_data(serialized_data)
            
            # Set TTL
            ttl = ttl or self.config.default_ttl
            
            # Store with pipeline
            pipe = self.redis_client.pipeline()
            pipe.setex(cache_key, ttl, final_data)
            
            if is_compressed:
                compression_flag_key = f"{cache_key}:compressed"
                pipe.setex(compression_flag_key, ttl, b"1")
            
            await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting agent cache: {e}")
            return False
    
    async def invalidate_agent_cache(self, agent_id: str) -> bool:
        """Invalidate all cache entries for an agent."""
        try:
            # Find all keys for this agent
            pattern = self._generate_key(self.key_prefixes['agent_cache'], agent_id, "*")
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Also get compression flags
                compression_keys = []
                for key in keys:
                    compression_key = f"{key.decode()}:compressed"
                    compression_keys.append(compression_key)
                
                # Delete all keys
                all_keys = keys + [k.encode() for k in compression_keys]
                await self.redis_client.delete(*all_keys)
                
                logger.info(f"🗑️ Invalidated {len(keys)} cache entries for agent {agent_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating agent cache: {e}")
            return False
    
    async def warm_cache_for_queries(
        self,
        agent_id: str,
        common_queries: List[str],
        results_per_query: List[List[Dict[str, Any]]]
    ) -> int:
        """Pre-warm cache with common queries and their results."""
        try:
            warmed_count = 0
            
            for query, results in zip(common_queries, results_per_query):
                # Generate query hash (simplified)
                query_hash = hashlib.md5(f"{query}|{agent_id}".encode()).hexdigest()
                
                success = await self.set_search_results(query_hash, results)
                if success:
                    warmed_count += 1
            
            logger.info(f"🔥 Warmed Redis cache with {warmed_count} queries for {agent_id}")
            return warmed_count
            
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            return 0
    
    async def get_query_metadata(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a query (access count, last accessed, etc.)."""
        try:
            cache_key = self._generate_key(self.key_prefixes['query_metadata'], query_hash)
            
            cached_data = await self.redis_client.get(cache_key)
            if cached_data is None:
                return None
            
            return json.loads(cached_data.decode())
            
        except Exception as e:
            logger.error(f"Error getting query metadata: {e}")
            return None
    
    async def update_query_metadata(
        self,
        query_hash: str,
        access_count: int = 1,
        agent_id: Optional[str] = None
    ) -> bool:
        """Update metadata for a query."""
        try:
            cache_key = self._generate_key(self.key_prefixes['query_metadata'], query_hash)
            
            # Get existing metadata
            existing = await self.get_query_metadata(query_hash)
            
            if existing:
                metadata = existing
                metadata['access_count'] = metadata.get('access_count', 0) + access_count
                metadata['last_accessed'] = time.time()
            else:
                metadata = {
                    'query_hash': query_hash,
                    'access_count': access_count,
                    'first_accessed': time.time(),
                    'last_accessed': time.time(),
                    'agent_id': agent_id
                }
            
            # Store updated metadata
            serialized = json.dumps(metadata).encode()
            await self.redis_client.setex(cache_key, self.config.default_ttl, serialized)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating query metadata: {e}")
            return False
    
    async def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular queries based on access counts."""
        try:
            # Find all query metadata keys
            pattern = f"{self.key_prefixes['query_metadata']}*"
            keys = await self.redis_client.keys(pattern)
            
            if not keys:
                return []
            
            # Get all metadata
            pipe = self.redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            
            results = await pipe.execute()
            
            # Parse and sort by access count
            queries = []
            for result in results:
                if result:
                    try:
                        metadata = json.loads(result.decode())
                        queries.append(metadata)
                    except Exception:
                        continue
            
            # Sort by access count and return top queries
            queries.sort(key=lambda x: x.get('access_count', 0), reverse=True)
            return queries[:limit]
            
        except Exception as e:
            logger.error(f"Error getting popular queries: {e}")
            return []
    
    def _update_get_stats(self, get_time: float, cache_hit: bool):
        """Update GET operation statistics."""
        total = self.performance_stats['total_gets']
        current_avg = self.performance_stats['avg_get_time']
        
        self.performance_stats['avg_get_time'] = (
            (current_avg * (total - 1) + get_time) / total
        )
    
    def _update_set_stats(self, set_time: float):
        """Update SET operation statistics."""
        total = self.performance_stats['total_sets']
        current_avg = self.performance_stats['avg_set_time']
        
        self.performance_stats['avg_set_time'] = (
            (current_avg * (total - 1) + set_time) / total
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            # Get Redis info
            redis_info = await self.redis_client.info()
            
            # Calculate hit rate
            total_requests = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
            hit_rate = (
                self.performance_stats['cache_hits'] / max(1, total_requests)
            )
            
            return {
                'connection_healthy': self.connection_healthy,
                'total_gets': self.performance_stats['total_gets'],
                'total_sets': self.performance_stats['total_sets'],
                'cache_hits': self.performance_stats['cache_hits'],
                'cache_misses': self.performance_stats['cache_misses'],
                'hit_rate': hit_rate,
                'avg_get_time_ms': self.performance_stats['avg_get_time'] * 1000,
                'avg_set_time_ms': self.performance_stats['avg_set_time'] * 1000,
                'compression_saves': self.performance_stats['compression_saves'],
                'total_compressed_bytes': self.performance_stats['total_compressed_bytes'],
                'redis_info': {
                    'version': redis_info.get('redis_version', 'unknown'),
                    'used_memory': redis_info.get('used_memory_human', 'unknown'),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0)
                },
                'config': {
                    'host': self.config.host,
                    'port': self.config.port,
                    'max_connections': self.config.max_connections,
                    'default_ttl': self.config.default_ttl,
                    'compression_threshold': self.config.compression_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {
                'error': str(e),
                'connection_healthy': self.connection_healthy
            }
    
    async def health_check(self) -> bool:
        """Comprehensive health check for Redis cache."""
        try:
            current_time = time.time()
            
            # Skip frequent health checks
            if current_time - self.last_health_check < 30:
                return self.connection_healthy
            
            # Test basic connectivity
            start_time = time.time()
            await self.redis_client.ping()
            ping_time = time.time() - start_time
            
            # Test set/get operations
            test_key = "health_check_test"
            test_value = f"test_{current_time}"
            
            await self.redis_client.setex(test_key, 10, test_value)
            retrieved_value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            # Verify operation
            operation_successful = (
                retrieved_value is not None and 
                retrieved_value.decode() == test_value
            )
            
            self.connection_healthy = operation_successful and ping_time < 0.1
            self.last_health_check = current_time
            
            if ping_time > 0.05:  # 50ms
                logger.warning(f"⚠️ High Redis latency: {ping_time*1000:.1f}ms")
            
            return self.connection_healthy
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            self.connection_healthy = False
            return False
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired keys and return count."""
        try:
            # Get keys that might be expired
            all_prefixes = list(self.key_prefixes.values())
            
            total_cleaned = 0
            for prefix in all_prefixes:
                pattern = f"{prefix}*"
                keys = await self.redis_client.keys(pattern)
                
                if keys:
                    # Check TTL for each key and clean up those close to expiring
                    pipe = self.redis_client.pipeline()
                    for key in keys:
                        pipe.ttl(key)
                    
                    ttls = await pipe.execute()
                    
                    # Remove keys with very low TTL (< 60 seconds)
                    keys_to_delete = []
                    for key, ttl in zip(keys, ttls):
                        if 0 < ttl < 60:  # TTL between 0 and 60 seconds
                            keys_to_delete.append(key)
                    
                    if keys_to_delete:
                        await self.redis_client.delete(*keys_to_delete)
                        total_cleaned += len(keys_to_delete)
            
            if total_cleaned > 0:
                logger.info(f"🧹 Redis cleanup: removed {total_cleaned} expired keys")
            
            return total_cleaned
            
        except Exception as e:
            logger.error(f"Error during Redis cleanup: {e}")
            return 0
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        try:
            info = await self.redis_client.info('memory')
            
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'used_memory_rss': info.get('used_memory_rss', 0),
                'used_memory_peak': info.get('used_memory_peak', 0),
                'used_memory_peak_human': info.get('used_memory_peak_human', '0B'),
                'maxmemory': info.get('maxmemory', 0),
                'maxmemory_human': info.get('maxmemory_human', '0B'),
                'mem_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown Redis cache gracefully."""
        logger.info("🛑 Shutting down Redis cache...")
        
        try:
            # Perform final cleanup
            cleaned_keys = await self.cleanup_expired_keys()
            
            # Get final statistics
            stats = await self.get_performance_stats()
            
            # Close connection pool
            if self.connection_pool:
                await self.connection_pool.disconnect()
            
            # Log final statistics
            logger.info(f"📊 Final Redis stats: {stats.get('total_gets', 0)} gets, "
                       f"{stats.get('total_sets', 0)} sets, "
                       f"{stats.get('hit_rate', 0)*100:.1f}% hit rate")
            
            logger.info("✅ Redis cache shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Redis shutdown: {e}")