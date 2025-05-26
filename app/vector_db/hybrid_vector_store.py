"""
Hybrid Vector Store - 3-Tier Architecture for Ultra-Low Latency (<5ms)
Implements Redis Cache + FAISS Hot Tier + Qdrant Cold Storage
"""
import asyncio
import time
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

# Import tier components
from app.vector_db.redis_cache import RedisVectorCache
from app.vector_db.faiss_hot_tier import FAISSHotTier
from app.vector_db.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Enhanced vector search result with metadata"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_tier: str = "unknown"  # redis, faiss, or qdrant
    retrieval_time_ms: float = 0.0
    agent_id: Optional[str] = None

@dataclass
class TierPerformanceStats:
    """Performance statistics for each tier"""
    tier_name: str
    total_queries: int = 0
    cache_hits: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_accessed: float = 0.0
    hit_rate: float = 0.0

class HybridVectorStore:
    """
    3-Tier Hybrid Vector Store for ultra-low latency retrieval:
    - Tier 1: Redis Cache (<1ms) - Frequent queries and routing decisions
    - Tier 2: FAISS Hot Tier (<5ms) - Active agent knowledge bases in memory
    - Tier 3: Qdrant Cold Storage (<50ms) - Complete knowledge base with metadata
    """
    
    # Performance targets from transformation plan
    REDIS_TARGET_LATENCY = 1.0      # <1ms for cache hits
    FAISS_TARGET_LATENCY = 5.0      # <5ms for hot data
    QDRANT_TARGET_LATENCY = 50.0    # <50ms for cold storage
    
    # Promotion/demotion thresholds
    HOT_TIER_PROMOTION_THRESHOLD = 100    # queries per hour
    CACHE_PROMOTION_THRESHOLD = 50        # queries per hour
    HOT_TIER_SIZE_LIMIT = 10             # max agents in hot tier
    CACHE_TTL = 300                      # 5 minutes cache TTL
    
    def __init__(self):
        # Tier components
        self.redis_cache: Optional[RedisVectorCache] = None
        self.faiss_hot_tier: Optional[FAISSHotTier] = None
        self.qdrant_manager: Optional[QdrantManager] = None
        
        # Agent activity tracking for intelligent promotion/demotion
        self.agent_activity: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "query_count": 0,
            "last_accessed": time.time(),
            "queries_per_hour": 0.0,
            "in_hot_tier": False,
            "promotion_score": 0.0
        })
        
        # Performance tracking
        self.tier_stats: Dict[str, TierPerformanceStats] = {
            "redis": TierPerformanceStats("redis"),
            "faiss": TierPerformanceStats("faiss"),
            "qdrant": TierPerformanceStats("qdrant")
        }
        
        # Query routing intelligence
        self.query_patterns: Dict[str, Dict[str, Any]] = {}
        self.routing_cache: Dict[str, str] = {}  # query_hash -> best_tier
        
        self.initialized = False
        
        logger.info("🎯 Hybrid Vector Store initialized with 3-tier architecture")
    
    async def init(self):
        """Initialize all tier components"""
        logger.info("🔄 Initializing Hybrid Vector Store...")
        
        try:
            # Initialize Redis Cache (Tier 1)
            self.redis_cache = RedisVectorCache()
            await self.redis_cache.init()
            
            # Initialize FAISS Hot Tier (Tier 2)
            self.faiss_hot_tier = FAISSHotTier(
                max_agents=self.HOT_TIER_SIZE_LIMIT,
                dimension=1536  # OpenAI embedding dimension
            )
            await self.faiss_hot_tier.init()
            
            # Initialize Qdrant Manager (Tier 3)
            self.qdrant_manager = QdrantManager()
            await self.qdrant_manager.init()
            
            # Start background optimization tasks
            asyncio.create_task(self._tier_optimization_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            
            self.initialized = True
            logger.info("✅ Hybrid Vector Store initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Hybrid Vector Store initialization failed: {e}")
            raise
    
    async def query_agent(
        self, 
        agent_id: str, 
        query: Union[str, np.ndarray], 
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Query agent knowledge base with intelligent tier routing
        
        Args:
            agent_id: Agent identifier
            query: Query text or embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results with tier information
        """
        if not self.initialized:
            await self.init()
        
        query_start_time = time.time()
        
        # Generate query hash for caching and routing
        query_hash = self._generate_query_hash(agent_id, query, top_k)
        
        # Update agent activity
        await self._update_agent_activity(agent_id)
        
        # Tier 1: Check Redis Cache first
        cache_results = await self._query_redis_cache(query_hash, agent_id)
        if cache_results:
            await self._update_tier_stats("redis", time.time() - query_start_time, hit=True)
            logger.debug(f"🎯 Cache hit for agent {agent_id}")
            return cache_results
        
        # Tier 2: Check FAISS Hot Tier
        if await self.faiss_hot_tier.has_agent(agent_id):
            faiss_start = time.time()
            faiss_results = await self.faiss_hot_tier.query_agent(
                agent_id, query, top_k, similarity_threshold
            )
            
            if faiss_results:
                faiss_latency = (time.time() - faiss_start) * 1000
                await self._update_tier_stats("faiss", faiss_latency / 1000, hit=True)
                
                # Cache successful results
                await self._cache_results(query_hash, faiss_results)
                
                logger.debug(f"🔥 FAISS hit for agent {agent_id} ({faiss_latency:.1f}ms)")
                return faiss_results
        
        # Tier 3: Query Qdrant Cold Storage
        qdrant_start = time.time()
        qdrant_results = await self.qdrant_manager.query_agent(
            agent_id, query, top_k, similarity_threshold
        )
        
        qdrant_latency = (time.time() - qdrant_start) * 1000
        await self._update_tier_stats("qdrant", qdrant_latency / 1000, hit=True)
        
        # Cache results for future queries
        if qdrant_results:
            await self._cache_results(query_hash, qdrant_results)
            
            # Consider promoting agent to hot tier
            await self._consider_hot_tier_promotion(agent_id)
        
        logger.debug(f"❄️ Qdrant query for agent {agent_id} ({qdrant_latency:.1f}ms)")
        
        # Track total query time
        total_latency = (time.time() - query_start_time) * 1000
        
        # Log performance warnings
        if total_latency > self.FAISS_TARGET_LATENCY:
            logger.warning(f"⚠️ Query latency exceeded target: {total_latency:.1f}ms > {self.FAISS_TARGET_LATENCY:.1f}ms")
        
        return qdrant_results or []
    
    async def _query_redis_cache(
        self, 
        query_hash: str, 
        agent_id: str
    ) -> Optional[List[VectorSearchResult]]:
        """Query Redis cache for cached results"""
        try:
            cache_start = time.time()
            cached_data = await self.redis_cache.get_query_result(query_hash)
            
            if cached_data:
                # Convert cached data back to VectorSearchResult objects
                results = []
                for item in cached_data:
                    result = VectorSearchResult(
                        id=item["id"],
                        text=item["text"],
                        score=item["score"],
                        metadata=item.get("metadata", {}),
                        source_tier="redis",
                        retrieval_time_ms=(time.time() - cache_start) * 1000,
                        agent_id=agent_id
                    )
                    results.append(result)
                
                return results
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Redis cache query error: {e}")
            return None
    
    async def _cache_results(self, query_hash: str, results: List[VectorSearchResult]):
        """Cache results in Redis for future queries"""
        try:
            # Convert results to cacheable format
            cache_data = []
            for result in results:
                cache_data.append({
                    "id": result.id,
                    "text": result.text,
                    "score": result.score,
                    "metadata": result.metadata,
                    "agent_id": result.agent_id
                })
            
            await self.redis_cache.set_query_result(
                query_hash, 
                cache_data, 
                ttl=self.CACHE_TTL
            )
            
        except Exception as e:
            logger.error(f"❌ Result caching error: {e}")
    
    def _generate_query_hash(
        self, 
        agent_id: str, 
        query: Union[str, np.ndarray], 
        top_k: int
    ) -> str:
        """Generate a hash for query caching"""
        # Convert query to string representation
        if isinstance(query, np.ndarray):
            query_str = str(query.tolist()[:10])  # Use first 10 elements for hash
        else:
            query_str = str(query)
        
        # Create hash from agent_id, query, and parameters
        hash_input = f"{agent_id}:{query_str}:{top_k}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def _update_agent_activity(self, agent_id: str):
        """Update agent activity tracking for intelligent promotion"""
        current_time = time.time()
        activity = self.agent_activity[agent_id]
        
        # Update query count and timestamp
        activity["query_count"] += 1
        time_diff = current_time - activity["last_accessed"]
        activity["last_accessed"] = current_time
        
        # Calculate queries per hour (exponential moving average)
        if time_diff > 0:
            queries_per_second = 1.0 / time_diff
            queries_per_hour = queries_per_second * 3600
            
            # Exponential moving average
            alpha = 0.1
            activity["queries_per_hour"] = (
                alpha * queries_per_hour + 
                (1 - alpha) * activity["queries_per_hour"]
            )
        
        # Calculate promotion score
        activity["promotion_score"] = self._calculate_promotion_score(activity)
    
    def _calculate_promotion_score(self, activity: Dict[str, Any]) -> float:
        """Calculate promotion score for an agent"""
        # Factors: query frequency, recency, total queries
        frequency_score = min(activity["queries_per_hour"] / self.HOT_TIER_PROMOTION_THRESHOLD, 1.0)
        
        # Recency score (higher for recent activity)
        time_since_access = time.time() - activity["last_accessed"]
        recency_score = max(0, 1.0 - (time_since_access / 3600))  # Decay over 1 hour
        
        # Total query score
        query_score = min(activity["query_count"] / 100, 1.0)  # Normalize to 100 queries
        
        # Weighted combination
        promotion_score = (
            0.5 * frequency_score +
            0.3 * recency_score +
            0.2 * query_score
        )
        
        return promotion_score
    
    async def _consider_hot_tier_promotion(self, agent_id: str):
        """Consider promoting agent to hot tier based on activity"""
        activity = self.agent_activity[agent_id]
        
        # Check if already in hot tier
        if activity["in_hot_tier"]:
            return
        
        # Check promotion criteria
        should_promote = (
            activity["queries_per_hour"] >= self.HOT_TIER_PROMOTION_THRESHOLD or
            activity["promotion_score"] >= 0.7
        )
        
        if should_promote:
            success = await self.faiss_hot_tier.promote_agent(agent_id)
            if success:
                activity["in_hot_tier"] = True
                logger.info(f"🔥 Promoted agent {agent_id} to hot tier (score: {activity['promotion_score']:.2f})")
    
    async def _update_tier_stats(self, tier: str, latency_seconds: float, hit: bool = True):
        """Update performance statistics for a tier"""
        stats = self.tier_stats[tier]
        stats.total_queries += 1
        stats.last_accessed = time.time()
        
        if hit:
            stats.cache_hits += 1
        
        # Update latency (exponential moving average)
        latency_ms = latency_seconds * 1000
        if stats.average_latency_ms == 0:
            stats.average_latency_ms = latency_ms
        else:
            alpha = 0.1
            stats.average_latency_ms = (
                alpha * latency_ms + 
                (1 - alpha) * stats.average_latency_ms
            )
        
        # Update hit rate
        stats.hit_rate = (stats.cache_hits / stats.total_queries) * 100
        
        # Track P95 latency (simplified approximation)
        stats.p95_latency_ms = max(stats.p95_latency_ms, latency_ms)
    
    async def create_agent_collection(
        self, 
        agent_id: str, 
        knowledge_sources: List[Dict[str, Any]]
    ):
        """Create vector collection for a new agent"""
        logger.info(f"📚 Creating collection for agent {agent_id}")
        
        try:
            # Create collection in Qdrant (cold storage)
            await self.qdrant_manager.create_agent_collection(agent_id, knowledge_sources)
            
            # Initialize agent activity tracking
            self.agent_activity[agent_id] = {
                "query_count": 0,
                "last_accessed": time.time(),
                "queries_per_hour": 0.0,
                "in_hot_tier": False,
                "promotion_score": 0.0,
                "created_at": time.time()
            }
            
            logger.info(f"✅ Collection created for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"❌ Error creating collection for agent {agent_id}: {e}")
            raise
    
    async def add_documents_to_agent(
        self, 
        agent_id: str, 
        documents: List[Dict[str, Any]]
    ):
        """Add documents to an agent's knowledge base"""
        logger.info(f"📄 Adding {len(documents)} documents to agent {agent_id}")
        
        try:
            # Add to Qdrant cold storage
            await self.qdrant_manager.add_documents_to_agent(agent_id, documents)
            
            # If agent is in hot tier, update FAISS index
            if self.agent_activity[agent_id]["in_hot_tier"]:
                await self.faiss_hot_tier.update_agent_data(agent_id, documents)
            
            # Invalidate related cache entries
            await self._invalidate_agent_cache(agent_id)
            
            logger.info(f"✅ Documents added to agent {agent_id}")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to agent {agent_id}: {e}")
            raise
    
    async def _invalidate_agent_cache(self, agent_id: str):
        """Invalidate cache entries for an agent"""
        try:
            await self.redis_cache.invalidate_agent_cache(agent_id)
        except Exception as e:
            logger.error(f"❌ Cache invalidation error for agent {agent_id}: {e}")
    
    async def _tier_optimization_loop(self):
        """Background task for tier optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._optimize_tier_distribution()
            except Exception as e:
                logger.error(f"❌ Tier optimization error: {e}")
    
    async def _optimize_tier_distribution(self):
        """Optimize agent distribution across tiers"""
        logger.debug("🔧 Optimizing tier distribution...")
        
        # Get current hot tier agents
        hot_tier_agents = await self.faiss_hot_tier.get_active_agents()
        
        # Evaluate promotion candidates
        promotion_candidates = []
        for agent_id, activity in self.agent_activity.items():
            if not activity["in_hot_tier"] and activity["promotion_score"] >= 0.6:
                promotion_candidates.append((agent_id, activity["promotion_score"]))
        
        # Sort by promotion score
        promotion_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Evaluate demotion candidates
        demotion_candidates = []
        for agent_id in hot_tier_agents:
            activity = self.agent_activity[agent_id]
            if activity["promotion_score"] < 0.3:  # Low activity
                demotion_candidates.append((agent_id, activity["promotion_score"]))
        
        # Perform promotions/demotions
        changes_made = 0
        
        # Promote high-activity agents
        for agent_id, score in promotion_candidates[:3]:  # Top 3 candidates
            if len(hot_tier_agents) < self.HOT_TIER_SIZE_LIMIT:
                success = await self.faiss_hot_tier.promote_agent(agent_id)
                if success:
                    self.agent_activity[agent_id]["in_hot_tier"] = True
                    hot_tier_agents.append(agent_id)
                    changes_made += 1
                    logger.info(f"🔥 Promoted agent {agent_id} to hot tier (score: {score:.2f})")
        
        # Demote low-activity agents
        for agent_id, score in demotion_candidates:
            if len(hot_tier_agents) > 3:  # Keep minimum 3 agents in hot tier
                success = await self.faiss_hot_tier.demote_agent(agent_id)
                if success:
                    self.agent_activity[agent_id]["in_hot_tier"] = False
                    hot_tier_agents.remove(agent_id)
                    changes_made += 1
                    logger.info(f"❄️ Demoted agent {agent_id} from hot tier (score: {score:.2f})")
        
        if changes_made > 0:
            logger.info(f"🔧 Tier optimization complete: {changes_made} changes made")
    
    async def _performance_monitoring_loop(self):
        """Background task for performance monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                await self._monitor_performance()
            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")
    
    async def _monitor_performance(self):
        """Monitor tier performance and alert on issues"""
        # Check each tier against targets
        performance_issues = []
        
        for tier_name, stats in self.tier_stats.items():
            target_latency = {
                "redis": self.REDIS_TARGET_LATENCY,
                "faiss": self.FAISS_TARGET_LATENCY,
                "qdrant": self.QDRANT_TARGET_LATENCY
            }[tier_name]
            
            if stats.average_latency_ms > target_latency:
                performance_issues.append({
                    "tier": tier_name,
                    "issue": "latency_exceeded",
                    "actual": stats.average_latency_ms,
                    "target": target_latency,
                    "severity": "high" if stats.average_latency_ms > target_latency * 2 else "medium"
                })
        
        # Log performance issues
        for issue in performance_issues:
            logger.warning(
                f"⚠️ Performance issue in {issue['tier']} tier: "
                f"{issue['actual']:.1f}ms > {issue['target']:.1f}ms target"
            )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            "timestamp": time.time(),
            "tier_performance": {},
            "agent_activity": {},
            "system_health": {}
        }
        
        # Tier performance
        for tier_name, tier_stats in self.tier_stats.items():
            stats["tier_performance"][tier_name] = {
                "total_queries": tier_stats.total_queries,
                "cache_hits": tier_stats.cache_hits,
                "hit_rate": tier_stats.hit_rate,
                "average_latency_ms": round(tier_stats.average_latency_ms, 2),
                "p95_latency_ms": round(tier_stats.p95_latency_ms, 2),
                "target_latency_ms": {
                    "redis": self.REDIS_TARGET_LATENCY,
                    "faiss": self.FAISS_TARGET_LATENCY,
                    "qdrant": self.QDRANT_TARGET_LATENCY
                }[tier_name],
                "performance_grade": self._calculate_performance_grade(tier_name, tier_stats)
            }
        
        # Agent activity summary
        hot_tier_agents = await self.faiss_hot_tier.get_active_agents() if self.faiss_hot_tier else []
        
        stats["agent_activity"] = {
            "total_agents": len(self.agent_activity),
            "hot_tier_agents": len(hot_tier_agents),
            "hot_tier_utilization": (len(hot_tier_agents) / self.HOT_TIER_SIZE_LIMIT) * 100,
            "most_active_agents": sorted(
                [
                    {
                        "agent_id": agent_id,
                        "queries_per_hour": activity["queries_per_hour"],
                        "promotion_score": activity["promotion_score"],
                        "in_hot_tier": activity["in_hot_tier"]
                    }
                    for agent_id, activity in self.agent_activity.items()
                ],
                key=lambda x: x["queries_per_hour"],
                reverse=True
            )[:10]
        }
        
        # System health
        overall_latency = sum(
            tier_stats.average_latency_ms 
            for tier_stats in self.tier_stats.values()
        ) / len(self.tier_stats)
        
        stats["system_health"] = {
            "overall_latency_ms": round(overall_latency, 2),
            "target_compliance": self._calculate_target_compliance(),
            "system_grade": self._calculate_system_grade(),
            "recommendations": self._generate_optimization_recommendations()
        }
        
        return stats
    
    def _calculate_performance_grade(self, tier_name: str, stats: TierPerformanceStats) -> str:
        """Calculate performance grade for a tier"""
        target_latency = {
            "redis": self.REDIS_TARGET_LATENCY,
            "faiss": self.FAISS_TARGET_LATENCY,
            "qdrant": self.QDRANT_TARGET_LATENCY
        }[tier_name]
        
        if stats.average_latency_ms <= target_latency:
            return "A"
        elif stats.average_latency_ms <= target_latency * 1.5:
            return "B"
        elif stats.average_latency_ms <= target_latency * 2:
            return "C"
        else:
            return "D"
    
    def _calculate_target_compliance(self) -> float:
        """Calculate overall target compliance percentage"""
        compliant_tiers = 0
        total_tiers = len(self.tier_stats)
        
        for tier_name, stats in self.tier_stats.items():
            target_latency = {
                "redis": self.REDIS_TARGET_LATENCY,
                "faiss": self.FAISS_TARGET_LATENCY,
                "qdrant": self.QDRANT_TARGET_LATENCY
            }[tier_name]
            
            if stats.average_latency_ms <= target_latency:
                compliant_tiers += 1
        
        return (compliant_tiers / total_tiers) * 100 if total_tiers > 0 else 100
    
    def _calculate_system_grade(self) -> str:
        """Calculate overall system performance grade"""
        compliance = self._calculate_target_compliance()
        
        if compliance >= 90:
            return "A"
        elif compliance >= 75:
            return "B"
        elif compliance >= 60:
            return "C"
        else:
            return "D"
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check Redis performance
        redis_stats = self.tier_stats["redis"]
        if redis_stats.hit_rate < 80:
            recommendations.append("Increase Redis cache TTL or cache more query patterns")
        
        # Check FAISS performance
        faiss_stats = self.tier_stats["faiss"]
        if faiss_stats.average_latency_ms > self.FAISS_TARGET_LATENCY:
            recommendations.append("Optimize FAISS index parameters or reduce hot tier size")
        
        # Check Qdrant performance
        qdrant_stats = self.tier_stats["qdrant"]
        if qdrant_stats.average_latency_ms > self.QDRANT_TARGET_LATENCY:
            recommendations.append("Optimize Qdrant configuration or consider index sharding")
        
        # Check tier distribution
        hot_tier_agents = sum(1 for activity in self.agent_activity.values() if activity["in_hot_tier"])
        if hot_tier_agents < 3:
            recommendations.append("Consider promoting more agents to hot tier for better performance")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup hybrid vector store resources"""
        logger.info("🧹 Cleaning up Hybrid Vector Store...")
        
        try:
            # Cleanup tier components
            if self.redis_cache:
                await self.redis_cache.cleanup()
            
            if self.faiss_hot_tier:
                await self.faiss_hot_tier.cleanup()
            
            if self.qdrant_manager:
                await self.qdrant_manager.cleanup()
            
            logger.info("✅ Hybrid Vector Store cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    # Public API methods for external use
    
    async def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for a specific agent"""
        if agent_id not in self.agent_activity:
            return {"error": "Agent not found"}
        
        activity = self.agent_activity[agent_id]
        
        return {
            "agent_id": agent_id,
            "query_count": activity["query_count"],
            "queries_per_hour": round(activity["queries_per_hour"], 2),
            "last_accessed": activity["last_accessed"],
            "in_hot_tier": activity["in_hot_tier"],
            "promotion_score": round(activity["promotion_score"], 3),
            "tier_location": "faiss" if activity["in_hot_tier"] else "qdrant",
            "performance_impact": self._calculate_agent_performance_impact(agent_id)
        }
    
    def _calculate_agent_performance_impact(self, agent_id: str) -> Dict[str, float]:
        """Calculate performance impact of agent placement"""
        activity = self.agent_activity[agent_id]
        
        if activity["in_hot_tier"]:
            avg_latency = self.tier_stats["faiss"].average_latency_ms
        else:
            avg_latency = self.tier_stats["qdrant"].average_latency_ms
        
        # Calculate potential improvement if moved to better tier
        potential_improvement = max(0, avg_latency - self.FAISS_TARGET_LATENCY)
        
        return {
            "current_avg_latency_ms": avg_latency,
            "potential_improvement_ms": potential_improvement,
            "queries_affected_per_hour": activity["queries_per_hour"]
        }
    
    async def force_agent_promotion(self, agent_id: str) -> bool:
        """Force promotion of agent to hot tier"""
        if agent_id not in self.agent_activity:
            return False
        
        success = await self.faiss_hot_tier.promote_agent(agent_id)
        if success:
            self.agent_activity[agent_id]["in_hot_tier"] = True
            logger.info(f"🔥 Force promoted agent {agent_id} to hot tier")
        
        return success
    
    async def force_agent_demotion(self, agent_id: str) -> bool:
        """Force demotion of agent from hot tier"""
        if agent_id not in self.agent_activity or not self.agent_activity[agent_id]["in_hot_tier"]:
            return False
        
        success = await self.faiss_hot_tier.demote_agent(agent_id)
        if success:
            self.agent_activity[agent_id]["in_hot_tier"] = False
            logger.info(f"❄️ Force demoted agent {agent_id} from hot tier")
        
        return success