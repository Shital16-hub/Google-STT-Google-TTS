"""
Hybrid Vector Store - 3-Tier Architecture for Ultra-Low Latency
Orchestrates Redis Cache (Tier 1) -> FAISS Hot Tier (Tier 2) -> Qdrant (Tier 3)
Optimized for <5ms vector retrieval with intelligent promotion/demotion.
"""
import asyncio
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from app.vector_db.redis_cache import RedisCache
from app.vector_db.faiss_hot_tier import FAISSHotTier
from app.vector_db.qdrant_manager import QdrantManager
from app.config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

class SearchTier(Enum):
    """Vector search tiers by performance."""
    REDIS_CACHE = "redis_cache"      # <1ms - Exact query matches
    FAISS_HOT = "faiss_hot"          # <5ms - Active agent contexts
    QDRANT_COLD = "qdrant_cold"      # <50ms - Full vector search

@dataclass
class SearchResult:
    """Enhanced search result with tier tracking."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    tier_used: SearchTier
    search_time: float
    agent_id: Optional[str] = None

@dataclass
class QueryMetrics:
    """Track query performance and patterns."""
    query_hash: str
    agent_id: str
    query_count: int = 0
    avg_search_time: float = 0.0
    tier_hits: Dict[SearchTier, int] = field(default_factory=lambda: defaultdict(int))
    last_accessed: float = field(default_factory=time.time)
    promotion_score: float = 0.0

class HybridVectorStore:
    """
    3-Tier Hybrid Vector Store for Ultra-Low Latency Search
    
    Architecture:
    - Tier 1 (Redis): <1ms for exact query matches and frequent results
    - Tier 2 (FAISS): <5ms for hot agent contexts and popular queries  
    - Tier 3 (Qdrant): <50ms for comprehensive vector search
    
    Features:
    - Intelligent query routing and result caching
    - Automatic promotion/demotion based on usage patterns
    - Agent-specific optimization and context warming
    - Real-time performance monitoring and optimization
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        faiss_hot_tier: Optional[FAISSHotTier] = None,
        qdrant_manager: Optional[QdrantManager] = None
    ):
        """Initialize the hybrid vector store."""
        self.redis_cache = redis_cache
        self.faiss_hot_tier = faiss_hot_tier
        self.qdrant_manager = qdrant_manager
        
        # Performance tracking
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.tier_performance = {
            SearchTier.REDIS_CACHE: deque(maxlen=1000),
            SearchTier.FAISS_HOT: deque(maxlen=1000),
            SearchTier.QDRANT_COLD: deque(maxlen=1000)
        }
        
        # Agent activity tracking for optimization
        self.agent_activity = defaultdict(lambda: {
            'query_count': 0,
            'last_active': time.time(),
            'avg_query_time': 0.0,
            'cache_hit_rate': 0.0
        })
        
        # Configuration
        self.config = {
            'redis_ttl': 3600,  # 1 hour cache TTL
            'faiss_promotion_threshold': 10,  # Queries per hour to promote to FAISS
            'faiss_max_collections': 5,  # Max FAISS collections per agent
            'qdrant_timeout': 0.1,  # 100ms timeout for Qdrant
            'cache_warming_enabled': True,
            'auto_optimization_interval': 300  # 5 minutes
        }
        
        # Background optimization
        self.optimization_task = None
        self.performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'faiss_hits': 0,
            'qdrant_hits': 0,
            'avg_response_time': 0.0
        }
        
        logger.info("HybridVectorStore initialized with 3-tier architecture")
    
    async def init(self):
        """Initialize all vector store tiers."""
        logger.info("🚀 Initializing hybrid vector store...")
        
        # Initialize each tier
        if not self.redis_cache:
            self.redis_cache = RedisCache()
        await self.redis_cache.init()
        
        if not self.faiss_hot_tier:
            self.faiss_hot_tier = FAISSHotTier()
        await self.faiss_hot_tier.init()
        
        if not self.qdrant_manager:
            self.qdrant_manager = QdrantManager()
        await self.qdrant_manager.init()
        
        # Start background optimization
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("✅ Hybrid vector store ready - 3 tiers operational")
    
    async def hybrid_search(
        self,
        query: str,
        agent_id: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.7,
        force_tier: Optional[SearchTier] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search across all tiers with intelligent routing.
        
        Args:
            query: Search query text
            agent_id: Agent identifier for context
            top_k: Number of results to return
            hybrid_alpha: Balance between semantic (1.0) and keyword (0.0) search
            force_tier: Force search in specific tier (for testing)
            
        Returns:
            List of search results with tier information
        """
        search_start = time.time()
        self.performance_stats['total_queries'] += 1
        
        # Generate query hash for caching and metrics
        query_hash = self._generate_query_hash(query, agent_id, top_k)
        
        # Update query metrics
        await self._update_query_metrics(query_hash, agent_id)
        
        try:
            # Tier 1: Redis Cache - Check for exact matches
            if not force_tier or force_tier == SearchTier.REDIS_CACHE:
                cache_results = await self._search_redis_cache(query_hash, top_k)
                if cache_results:
                    search_time = time.time() - search_start
                    self._record_tier_performance(SearchTier.REDIS_CACHE, search_time)
                    self.performance_stats['cache_hits'] += 1
                    
                    logger.debug(f"⚡ Redis cache hit: {len(cache_results)} results in {search_time*1000:.1f}ms")
                    return cache_results
            
            # Tier 2: FAISS Hot Tier - Check active agent contexts
            if not force_tier or force_tier == SearchTier.FAISS_HOT:
                faiss_results = await self._search_faiss_hot_tier(query, agent_id, top_k, hybrid_alpha)
                if faiss_results:
                    search_time = time.time() - search_start
                    self._record_tier_performance(SearchTier.FAISS_HOT, search_time)
                    self.performance_stats['faiss_hits'] += 1
                    
                    # Cache results in Redis for future queries
                    await self._cache_results_in_redis(query_hash, faiss_results)
                    
                    logger.debug(f"🔥 FAISS hot tier hit: {len(faiss_results)} results in {search_time*1000:.1f}ms")
                    return faiss_results
            
            # Tier 3: Qdrant - Full vector search
            if not force_tier or force_tier == SearchTier.QDRANT_COLD:
                qdrant_results = await self._search_qdrant(query, agent_id, top_k, hybrid_alpha)
                search_time = time.time() - search_start
                self._record_tier_performance(SearchTier.QDRANT_COLD, search_time)
                self.performance_stats['qdrant_hits'] += 1
                
                if qdrant_results:
                    # Cache results and consider promotion to hot tier
                    await self._cache_results_in_redis(query_hash, qdrant_results)
                    await self._consider_faiss_promotion(query_hash, agent_id, qdrant_results)
                    
                    logger.debug(f"❄️ Qdrant search: {len(qdrant_results)} results in {search_time*1000:.1f}ms")
                    return qdrant_results
            
            # No results found
            logger.warning(f"No results found for query: {query[:50]}...")
            return []
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid search: {e}", exc_info=True)
            return []
        finally:
            # Update performance stats
            total_time = time.time() - search_start
            self._update_performance_stats(total_time)
            
            # Update agent activity
            self._update_agent_activity(agent_id, total_time)
    
    async def _search_redis_cache(self, query_hash: str, top_k: int) -> Optional[List[SearchResult]]:
        """Search Redis cache for exact query matches."""
        try:
            cached_results = await self.redis_cache.get_search_results(query_hash)
            if cached_results:
                # Convert cached data to SearchResult objects
                results = []
                for result_data in cached_results[:top_k]:
                    result = SearchResult(
                        id=result_data['id'],
                        content=result_data['content'],
                        metadata=result_data['metadata'],
                        score=result_data['score'],
                        tier_used=SearchTier.REDIS_CACHE,
                        search_time=0.001,  # ~1ms for cache access
                        agent_id=result_data.get('agent_id')
                    )
                    results.append(result)
                return results
            return None
        except Exception as e:
            logger.error(f"Redis cache search error: {e}")
            return None
    
    async def _search_faiss_hot_tier(
        self,
        query: str,
        agent_id: str,
        top_k: int,
        hybrid_alpha: float
    ) -> Optional[List[SearchResult]]:
        """Search FAISS hot tier for active agent contexts."""
        try:
            # Check if agent has active FAISS index
            if not await self.faiss_hot_tier.has_agent_index(agent_id):
                return None
            
            # Perform FAISS search
            faiss_results = await self.faiss_hot_tier.search(
                query=query,
                agent_id=agent_id,
                top_k=top_k,
                hybrid_alpha=hybrid_alpha
            )
            
            if faiss_results:
                # Convert to SearchResult objects
                results = []
                for result_data in faiss_results:
                    result = SearchResult(
                        id=result_data['id'],
                        content=result_data['content'],
                        metadata=result_data['metadata'],
                        score=result_data['score'],
                        tier_used=SearchTier.FAISS_HOT,
                        search_time=result_data.get('search_time', 0.005),
                        agent_id=agent_id
                    )
                    results.append(result)
                return results
            
            return None
        except Exception as e:
            logger.error(f"FAISS hot tier search error: {e}")
            return None
    
    async def _search_qdrant(
        self,
        query: str,
        agent_id: str,
        top_k: int,
        hybrid_alpha: float
    ) -> List[SearchResult]:
        """Search Qdrant for comprehensive vector search."""
        try:
            # Perform Qdrant search with timeout
            qdrant_results = await asyncio.wait_for(
                self.qdrant_manager.search(
                    query=query,
                    agent_id=agent_id,
                    top_k=top_k,
                    hybrid_alpha=hybrid_alpha
                ),
                timeout=self.config['qdrant_timeout']
            )
            
            # Convert to SearchResult objects
            results = []
            for result_data in qdrant_results:
                result = SearchResult(
                    id=result_data['id'],
                    content=result_data['content'],
                    metadata=result_data['metadata'],
                    score=result_data['score'],
                    tier_used=SearchTier.QDRANT_COLD,
                    search_time=result_data.get('search_time', 0.05),
                    agent_id=agent_id
                )
                results.append(result)
            
            return results
            
        except asyncio.TimeoutError:
            logger.warning(f"Qdrant search timeout for agent {agent_id}")
            return []
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []
    
    async def _cache_results_in_redis(self, query_hash: str, results: List[SearchResult]):
        """Cache search results in Redis for future queries."""
        try:
            # Convert SearchResult objects to serializable format
            cache_data = []
            for result in results:
                cache_data.append({
                    'id': result.id,
                    'content': result.content,
                    'metadata': result.metadata,
                    'score': result.score,
                    'agent_id': result.agent_id
                })
            
            await self.redis_cache.set_search_results(
                query_hash=query_hash,
                results=cache_data,
                ttl=self.config['redis_ttl']
            )
        except Exception as e:
            logger.error(f"Error caching results in Redis: {e}")
    
    async def _consider_faiss_promotion(
        self,
        query_hash: str,
        agent_id: str,
        results: List[SearchResult]
    ):
        """Consider promoting query results to FAISS hot tier."""
        try:
            # Check if query meets promotion criteria
            if query_hash not in self.query_metrics:
                return
            
            metrics = self.query_metrics[query_hash]
            
            # Promote if query is accessed frequently
            if metrics.query_count >= self.config['faiss_promotion_threshold']:
                logger.info(f"🔥 Promoting query to FAISS hot tier: {query_hash[:16]}")
                
                # Add to FAISS hot tier
                await self.faiss_hot_tier.add_documents(
                    agent_id=agent_id,
                    documents=[{
                        'id': result.id,
                        'content': result.content,
                        'metadata': result.metadata
                    } for result in results]
                )
                
                # Update promotion score
                metrics.promotion_score += 1.0
                
        except Exception as e:
            logger.error(f"Error considering FAISS promotion: {e}")
    
    def _generate_query_hash(self, query: str, agent_id: str, top_k: int) -> str:
        """Generate consistent hash for query caching."""
        query_string = f"{query.lower().strip()}|{agent_id}|{top_k}"
        return hashlib.md5(query_string.encode()).hexdigest()
    
    async def _update_query_metrics(self, query_hash: str, agent_id: str):
        """Update metrics for query tracking."""
        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                agent_id=agent_id
            )
        
        metrics = self.query_metrics[query_hash]
        metrics.query_count += 1
        metrics.last_accessed = time.time()
    
    def _record_tier_performance(self, tier: SearchTier, search_time: float):
        """Record performance metrics for each tier."""
        self.tier_performance[tier].append({
            'timestamp': time.time(),
            'search_time': search_time
        })
    
    def _update_performance_stats(self, total_time: float):
        """Update overall performance statistics."""
        # Update running average
        total_queries = self.performance_stats['total_queries']
        current_avg = self.performance_stats['avg_response_time']
        
        self.performance_stats['avg_response_time'] = (
            (current_avg * (total_queries - 1) + total_time) / total_queries
        )
    
    def _update_agent_activity(self, agent_id: str, query_time: float):
        """Update agent activity metrics."""
        activity = self.agent_activity[agent_id]
        activity['query_count'] += 1
        activity['last_active'] = time.time()
        
        # Update running average
        current_avg = activity['avg_query_time']
        query_count = activity['query_count']
        activity['avg_query_time'] = (
            (current_avg * (query_count - 1) + query_time) / query_count
        )
    
    # Document Management
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        agent_id: str,
        auto_promote: bool = True
    ) -> bool:
        """Add documents to the vector store (primarily to Qdrant)."""
        try:
            # Add to Qdrant (primary storage)
            success = await self.qdrant_manager.add_documents(documents, agent_id)
            
            if success and auto_promote:
                # Consider adding high-priority documents to FAISS hot tier
                await self._auto_promote_documents(documents, agent_id)
            
            return success
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def _auto_promote_documents(self, documents: List[Dict[str, Any]], agent_id: str):
        """Automatically promote important documents to hot tier."""
        try:
            # Promote based on document importance (metadata flags, etc.)
            important_docs = [
                doc for doc in documents
                if doc.get('metadata', {}).get('priority') == 'high' or
                doc.get('metadata', {}).get('frequently_accessed', False)
            ]
            
            if important_docs:
                await self.faiss_hot_tier.add_documents(agent_id, important_docs)
                logger.info(f"🔥 Auto-promoted {len(important_docs)} documents to hot tier")
                
        except Exception as e:
            logger.error(f"Error auto-promoting documents: {e}")
    
    # Agent-specific optimization
    async def warm_agent_cache(self, agent_id: str, user_context: Optional[Dict[str, Any]] = None):
        """Pre-warm caches for an agent based on likely queries."""
        try:
            logger.info(f"🔥 Warming cache for agent: {agent_id}")
            
            # Get common queries for this agent
            common_queries = await self._get_common_queries_for_agent(agent_id)
            
            # Pre-execute searches to warm caches
            for query in common_queries:
                try:
                    await self.hybrid_search(
                        query=query,
                        agent_id=agent_id,
                        top_k=3
                    )
                except Exception as e:
                    logger.error(f"Error warming cache for query '{query}': {e}")
            
            logger.info(f"✅ Cache warming complete for {agent_id}")
            
        except Exception as e:
            logger.error(f"Error warming agent cache: {e}")
    
    async def _get_common_queries_for_agent(self, agent_id: str) -> List[str]:
        """Get common queries for an agent based on historical data."""
        # This would typically come from analytics/historical data
        agent_queries = {
            "roadside-assistance": [
                "I need a tow truck",
                "My car broke down",
                "I have a flat tire",
                "I need roadside help"
            ],
            "billing-support": [
                "Check my bill",
                "Payment problems",
                "Billing question",
                "Account balance"
            ],
            "technical-support": [
                "Not working properly",
                "Technical issue",
                "Need help with setup",
                "System problem"
            ]
        }
        
        return agent_queries.get(agent_id, [])
    
    # Performance optimization
    async def optimize_performance(self):
        """Optimize vector store performance based on usage patterns."""
        try:
            logger.info("🔧 Optimizing vector store performance...")
            
            # Analyze query patterns
            await self._analyze_query_patterns()
            
            # Optimize tier distribution
            await self._optimize_tier_distribution()
            
            # Clean up old cache entries
            await self._cleanup_old_entries()
            
            logger.info("✅ Performance optimization complete")
            
        except Exception as e:
            logger.error(f"Error during performance optimization: {e}")
    
    async def _analyze_query_patterns(self):
        """Analyze query patterns for optimization opportunities."""
        current_time = time.time()
        
        # Find frequently accessed queries
        frequent_queries = [
            (hash_key, metrics) for hash_key, metrics in self.query_metrics.items()
            if metrics.query_count >= 5 and (current_time - metrics.last_accessed) < 3600
        ]
        
        # Promote to FAISS if not already there
        for query_hash, metrics in frequent_queries:
            if metrics.promotion_score == 0:  # Not yet promoted
                await self._consider_faiss_promotion(query_hash, metrics.agent_id, [])
    
    async def _optimize_tier_distribution(self):
        """Optimize document distribution across tiers."""
        # Get agent activity stats
        active_agents = [
            agent_id for agent_id, activity in self.agent_activity.items()
            if activity['query_count'] > 10 and (time.time() - activity['last_active']) < 1800
        ]
        
        # Ensure active agents have FAISS indices
        for agent_id in active_agents:
            if not await self.faiss_hot_tier.has_agent_index(agent_id):
                await self._promote_agent_to_faiss(agent_id)
    
    async def _promote_agent_to_faiss(self, agent_id: str):
        """Promote an agent's most relevant documents to FAISS."""
        try:
            # Get top documents for this agent from Qdrant
            top_docs = await self.qdrant_manager.get_top_documents(agent_id, limit=100)
            
            if top_docs:
                await self.faiss_hot_tier.create_agent_index(agent_id, top_docs)
                logger.info(f"🔥 Promoted agent {agent_id} to FAISS hot tier")
                
        except Exception as e:
            logger.error(f"Error promoting agent to FAISS: {e}")
    
    async def _cleanup_old_entries(self):
        """Clean up old cache entries and metrics."""
        current_time = time.time()
        
        # Clean up old query metrics (older than 24 hours)
        old_queries = [
            hash_key for hash_key, metrics in self.query_metrics.items()
            if (current_time - metrics.last_accessed) > 86400
        ]
        
        for query_hash in old_queries:
            del self.query_metrics[query_hash]
        
        if old_queries:
            logger.info(f"🧹 Cleaned up {len(old_queries)} old query metrics")
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        while True:
            try:
                await asyncio.sleep(self.config['auto_optimization_interval'])
                await self.optimize_performance()
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    # Analytics and monitoring
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "total_queries": self.performance_stats['total_queries'],
            "avg_response_time": self.performance_stats['avg_response_time'],
            "cache_hit_rate": 0.0,
            "tier_distribution": {
                "redis_hits": self.performance_stats['cache_hits'],
                "faiss_hits": self.performance_stats['faiss_hits'],
                "qdrant_hits": self.performance_stats['qdrant_hits']
            },
            "active_agents": len([
                a for a in self.agent_activity.values()
                if (time.time() - a['last_active']) < 1800
            ]),
            "total_query_metrics": len(self.query_metrics),
            "tier_performance": {}
        }
        
        # Calculate cache hit rate
        total_hits = sum(stats["tier_distribution"].values())
        if total_hits > 0:
            stats["cache_hit_rate"] = (
                stats["tier_distribution"]["redis_hits"] + 
                stats["tier_distribution"]["faiss_hits"]
            ) / total_hits
        
        # Calculate tier performance averages
        for tier, measurements in self.tier_performance.items():
            if measurements:
                avg_time = sum(m['search_time'] for m in measurements) / len(measurements)
                stats["tier_performance"][tier.value] = {
                    "avg_time_ms": avg_time * 1000,
                    "measurement_count": len(measurements)
                }
        
        return stats
    
    async def health_check(self) -> bool:
        """Comprehensive health check for all tiers."""
        try:
            # Check each tier
            redis_healthy = await self.redis_cache.health_check()
            faiss_healthy = await self.faiss_hot_tier.health_check()
            qdrant_healthy = await self.qdrant_manager.health_check()
            
            # At least Qdrant must be healthy for basic functionality
            if not qdrant_healthy:
                return False
            
            # Log tier health status
            logger.info(f"Vector store health: Redis={redis_healthy}, FAISS={faiss_healthy}, Qdrant={qdrant_healthy}")
            
            return True
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all vector store components."""
        logger.info("🛑 Shutting down hybrid vector store...")
        
        # Cancel optimization task
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all tiers
        if self.redis_cache:
            await self.redis_cache.shutdown()
        
        if self.faiss_hot_tier:
            await self.faiss_hot_tier.shutdown()
        
        if self.qdrant_manager:
            await self.qdrant_manager.shutdown()
        
        # Log final statistics
        stats = await self.get_performance_stats()
        logger.info(f"📊 Final stats: {stats['total_queries']} queries, "
                   f"{stats['cache_hit_rate']*100:.1f}% cache hit rate")
        
        logger.info("✅ Hybrid vector store shutdown complete")