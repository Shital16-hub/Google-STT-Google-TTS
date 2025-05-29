"""
Hybrid Vector System with Robust Fallback Mechanisms
Fixed version with proper error handling and graceful degradation for RunPod deployment.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result from vector search operation."""
    vectors: List[Dict[str, Any]]
    search_time_ms: float
    tier_used: str
    total_results: int
    confidence_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class InMemoryVectorCache:
    """In-memory fallback for Redis cache."""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[np.ndarray]:
        """Get vector from memory cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    async def set(self, key: str, vector: np.ndarray, ttl: int = 3600):
        """Set vector in memory cache with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = vector
        self.access_times[key] = time.time()
    
    async def delete(self, key: str):
        """Delete vector from memory cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "type": "in_memory",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

class RedisCacheWrapper:
    """Redis cache wrapper with fallback to in-memory."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.fallback_cache = InMemoryVectorCache(config.get("cache_size", 10000))
        self.use_fallback = config.get("fallback_to_memory", True)
        self.redis_available = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
    async def initialize(self):
        """Initialize Redis connection with fallback."""
        if self.config.get("host") == ":memory:":
            logger.info("Using in-memory cache mode")
            self.use_fallback = True
            self.redis_available = False
            return
        
        try:
            import redis.asyncio as redis
            
            # Create Redis client
            self.client = redis.Redis(
                host=self.config.get("host", "127.0.0.1"),
                port=self.config.get("port", 6379),
                decode_responses=False,
                socket_timeout=self.config.get("timeout", 5),
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            self.redis_available = True
            logger.info("✅ Redis cache initialized successfully")
            
        except Exception as e:
            self.connection_attempts += 1
            logger.warning(f"❌ Redis connection failed (attempt {self.connection_attempts}): {e}")
            
            if self.use_fallback:
                logger.info("🔄 Falling back to in-memory cache")
                self.redis_available = False
            else:
                raise
    
    async def get(self, key: str) -> Optional[np.ndarray]:
        """Get vector with fallback support."""
        # Try Redis first if available
        if self.redis_available and self.client:
            try:
                data = await self.client.get(key)
                if data:
                    return np.frombuffer(data, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                await self._handle_redis_error()
        
        # Fallback to in-memory cache
        return await self.fallback_cache.get(key)
    
    async def set(self, key: str, vector: np.ndarray, ttl: int = 3600):
        """Set vector with fallback support."""
        # Try Redis first if available
        if self.redis_available and self.client:
            try:
                data = vector.tobytes()
                await self.client.setex(key, ttl, data)
                return
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                await self._handle_redis_error()
        
        # Fallback to in-memory cache
        await self.fallback_cache.set(key, vector, ttl)
    
    async def delete(self, key: str):
        """Delete vector with fallback support."""
        # Try Redis first if available
        if self.redis_available and self.client:
            try:
                await self.client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                await self._handle_redis_error()
        
        # Also delete from fallback cache
        await self.fallback_cache.delete(key)
    
    async def _handle_redis_error(self):
        """Handle Redis errors and potentially switch to fallback."""
        if self.connection_attempts >= self.max_connection_attempts:
            logger.warning("⚠️ Max Redis connection attempts reached, switching to fallback mode")
            self.redis_available = False
        else:
            self.connection_attempts += 1
            # Try to reconnect
            try:
                if self.client:
                    await self.client.ping()
                    self.redis_available = True
                    self.connection_attempts = 0
            except:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.fallback_cache.get_stats()
        stats.update({
            "redis_available": self.redis_available,
            "connection_attempts": self.connection_attempts,
            "mode": "redis" if self.redis_available else "in_memory_fallback"
        })
        return stats

class InMemoryFAISS:
    """In-memory FAISS implementation with numpy fallback."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vectors = {}
        self.index_built = False
        self.dimension = None
        
    async def initialize(self):
        """Initialize in-memory FAISS."""
        logger.info("✅ In-memory FAISS initialized")
        
    async def add_vectors(self, vectors: Dict[str, np.ndarray]):
        """Add vectors to in-memory index."""
        for vector_id, vector in vectors.items():
            if self.dimension is None:
                self.dimension = len(vector)
            
            self.vectors[vector_id] = vector
        
        self.index_built = True
        logger.debug(f"Added {len(vectors)} vectors to in-memory FAISS")
    
    async def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search vectors using cosine similarity."""
        if not self.vectors:
            return []
        
        # Calculate cosine similarity with all vectors
        similarities = []
        for vector_id, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((vector_id, similarity, vector))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for vector_id, similarity, vector in similarities[:top_k]:
            results.append({
                "id": vector_id,
                "score": similarity,
                "vector": vector,
                "metadata": {"source": "in_memory_faiss"}
            })
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS statistics."""
        return {
            "type": "in_memory",
            "vector_count": len(self.vectors),
            "dimension": self.dimension,
            "index_built": self.index_built
        }

class InMemoryQdrant:
    """In-memory Qdrant implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collections = {}
        
    async def initialize(self):
        """Initialize in-memory Qdrant."""
        logger.info("✅ In-memory Qdrant initialized")
    
    async def create_collection(self, collection_name: str, vector_size: int, distance: str = "cosine"):
        """Create a collection."""
        self.collections[collection_name] = {
            "vectors": {},
            "vector_size": vector_size,
            "distance": distance
        }
        logger.debug(f"Created collection: {collection_name}")
    
    async def upsert_vectors(self, collection_name: str, vectors: List[Dict[str, Any]]):
        """Upsert vectors to collection."""
        if collection_name not in self.collections:
            # Auto-create collection
            if vectors:
                vector_size = len(vectors[0]["vector"])
                await self.create_collection(collection_name, vector_size)
        
        collection = self.collections[collection_name]
        for vector_data in vectors:
            vector_id = vector_data["id"]
            collection["vectors"][vector_id] = vector_data
        
        logger.debug(f"Upserted {len(vectors)} vectors to {collection_name}")
    
    async def search(self, collection_name: str, query_vector: np.ndarray, 
                    top_k: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search vectors in collection."""
        if collection_name not in self.collections:
            return []
        
        collection = self.collections[collection_name]
        similarities = []
        
        for vector_id, vector_data in collection["vectors"].items():
            vector = np.array(vector_data["vector"])
            similarity = self._cosine_similarity(query_vector, vector)
            
            if similarity >= score_threshold:
                similarities.append({
                    "id": vector_id,
                    "score": similarity,
                    "payload": vector_data.get("payload", {}),
                    "vector": vector
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant statistics."""
        total_vectors = sum(len(collection["vectors"]) for collection in self.collections.values())
        return {
            "type": "in_memory",
            "collections": len(self.collections),
            "total_vectors": total_vectors,
            "collection_details": {
                name: {
                    "vector_count": len(collection["vectors"]),
                    "vector_size": collection["vector_size"]
                }
                for name, collection in self.collections.items()
            }
        }

class QdrantWrapper:
    """Qdrant wrapper with fallback to in-memory."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.fallback_qdrant = InMemoryQdrant(config)
        self.use_fallback = config.get("fallback_to_memory", True)
        self.qdrant_available = False
        
    async def initialize(self):
        """Initialize Qdrant connection with fallback."""
        if self.config.get("host") == ":memory:":
            logger.info("Using in-memory Qdrant mode")
            self.use_fallback = True
            self.qdrant_available = False
            await self.fallback_qdrant.initialize()
            return
        
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.http.exceptions import UnexpectedResponse
            
            # Create Qdrant client
            self.client = AsyncQdrantClient(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6333),
                grpc_port=self.config.get("grpc_port", 6334),
                prefer_grpc=self.config.get("prefer_grpc", False),
                timeout=self.config.get("timeout", 10.0)
            )
            
            # Test connection
            collections = await self.client.get_collections()
            self.qdrant_available = True
            logger.info("✅ Qdrant initialized successfully")
            
        except Exception as e:
            logger.warning(f"❌ Qdrant connection failed: {e}")
            
            if self.use_fallback:
                logger.info("🔄 Falling back to in-memory Qdrant")
                self.qdrant_available = False
                await self.fallback_qdrant.initialize()
            else:
                raise
    
    async def create_collection(self, collection_name: str, vector_size: int, distance: str = "cosine"):
        """Create collection with fallback."""
        if self.qdrant_available and self.client:
            try:
                from qdrant_client.http.models import Distance, VectorParams
                
                distance_map = {
                    "cosine": Distance.COSINE,
                    "euclidean": Distance.EUCLID,
                    "dot": Distance.DOT
                }
                
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance_map.get(distance, Distance.COSINE)
                    )
                )
                return
            except Exception as e:
                logger.warning(f"Qdrant create_collection error: {e}")
                if not self.use_fallback:
                    raise
        
        # Fallback to in-memory
        await self.fallback_qdrant.create_collection(collection_name, vector_size, distance)
    
    async def upsert_vectors(self, collection_name: str, vectors: List[Dict[str, Any]]):
        """Upsert vectors with fallback."""
        if self.qdrant_available and self.client:
            try:
                from qdrant_client.http.models import PointStruct
                
                points = []
                for vector_data in vectors:
                    point = PointStruct(
                        id=vector_data["id"],
                        vector=vector_data["vector"].tolist() if isinstance(vector_data["vector"], np.ndarray) else vector_data["vector"],
                        payload=vector_data.get("payload", {})
                    )
                    points.append(point)
                
                await self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                return
            except Exception as e:
                logger.warning(f"Qdrant upsert error: {e}")
                if not self.use_fallback:
                    raise
        
        # Fallback to in-memory
        await self.fallback_qdrant.upsert_vectors(collection_name, vectors)
    
    async def search(self, collection_name: str, query_vector: np.ndarray, 
                    top_k: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search with fallback."""
        if self.qdrant_available and self.client:
            try:
                results = await self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                    limit=top_k,
                    score_threshold=score_threshold
                )
                
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload,
                        "vector": np.array(result.vector) if result.vector else None
                    })
                
                return formatted_results
            except Exception as e:
                logger.warning(f"Qdrant search error: {e}")
                if not self.use_fallback:
                    raise
        
        # Fallback to in-memory
        return await self.fallback_qdrant.search(collection_name, query_vector, top_k, score_threshold)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant statistics."""
        stats = self.fallback_qdrant.get_stats()
        stats.update({
            "qdrant_available": self.qdrant_available,
            "mode": "qdrant" if self.qdrant_available else "in_memory_fallback"
        })
        return stats

class HybridVectorSystem:
    """
    Hybrid 3-tier vector system with robust fallback mechanisms.
    Fixed version for RunPod deployment with graceful degradation.
    """
    
    def __init__(
        self,
        redis_config: Dict[str, Any],
        faiss_config: Dict[str, Any],
        qdrant_config: Dict[str, Any]
    ):
        """Initialize hybrid vector system with fallback support."""
        self.redis_config = redis_config
        self.faiss_config = faiss_config
        self.qdrant_config = qdrant_config
        
        # Initialize components
        self.redis_cache = RedisCacheWrapper(redis_config)
        self.faiss_hot_tier = InMemoryFAISS(faiss_config)
        self.qdrant_manager = QdrantWrapper(qdrant_config)
        
        # System state
        self.initialized = False
        self.performance_stats = {
            "searches_performed": 0,
            "cache_hits": 0,
            "faiss_hits": 0,
            "qdrant_hits": 0,
            "average_search_time_ms": 0.0,
            "error_count": 0
        }
        
        logger.info("HybridVectorSystem created with fallback support")

    async def analyze_query_intent(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze query intent using keyword patterns and context.
        This method was missing and causing the orchestration error.
        """
        try:
            # Initialize analysis result
            analysis = {
                "intent": "general",
                "urgency": "normal", 
                "complexity": 0.5,
                "entities": [],
                "keywords": [],
                "requires_tools": False,
                "confidence": 0.8
            }
            
            query_lower = query.lower()
            
            # Extract keywords
            import re
            words = re.findall(r'\b\w+\b', query_lower)
            analysis["keywords"] = words
            
            # Determine intent based on keywords
            roadside_keywords = ["tow", "stuck", "breakdown", "accident", "emergency", "help", "car", "vehicle"]
            billing_keywords = ["bill", "payment", "charge", "refund", "invoice", "money", "cost"]
            technical_keywords = ["error", "not working", "bug", "install", "setup", "login", "problem"]
            
            roadside_score = sum(1 for word in roadside_keywords if word in query_lower)
            billing_score = sum(1 for word in billing_keywords if word in query_lower) 
            technical_score = sum(1 for word in technical_keywords if word in query_lower)
            
            # Determine primary intent
            if roadside_score > billing_score and roadside_score > technical_score:
                analysis["intent"] = "roadside_assistance"
                analysis["requires_tools"] = True
            elif billing_score > roadside_score and billing_score > technical_score:
                analysis["intent"] = "billing_support"
                analysis["requires_tools"] = False
            elif technical_score > 0:
                analysis["intent"] = "technical_support"
                analysis["requires_tools"] = True
            
            # Determine urgency
            urgency_keywords = ["urgent", "emergency", "asap", "critical", "help", "stuck"]
            if any(keyword in query_lower for keyword in urgency_keywords):
                analysis["urgency"] = "high"
                if "emergency" in query_lower or "accident" in query_lower:
                    analysis["urgency"] = "critical"
            
            # Simple entity extraction
            entities = []
            # Look for phone numbers
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = re.findall(phone_pattern, query)
            for phone in phones:
                entities.append({"type": "phone", "value": phone})
            
            # Look for locations (simple pattern)
            location_words = ["street", "avenue", "road", "highway", "parking", "lot", "address"]
            for word in location_words:
                if word in query_lower:
                    entities.append({"type": "location_indicator", "value": word})
            
            analysis["entities"] = entities
            
            # Calculate complexity
            word_count = len(words)
            entity_count = len(entities)
            analysis["complexity"] = min(1.0, (word_count / 20.0) + (entity_count / 5.0))
            
            # Update confidence based on matches
            if roadside_score > 0 or billing_score > 0 or technical_score > 0:
                analysis["confidence"] = 0.9
            
            logger.debug(f"Query intent analysis: {analysis['intent']} (confidence: {analysis['confidence']})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_query_intent: {e}")
            # Return default analysis
            return {
                "intent": "general",
                "urgency": "normal",
                "complexity": 0.5,
                "entities": [],
                "keywords": query.split(),
                "requires_tools": False,
                "confidence": 0.5
            }
    
    async def get_agent_context(
        self,
        agent_id: str,
        query: str,
        base_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get agent-specific context enrichment.
        This method was missing and causing the orchestration error.
        """
        try:
            enriched_context = base_context.copy()
            
            # Add agent-specific context based on agent type
            if "roadside" in agent_id.lower():
                enriched_context.update({
                    "domain": "emergency_services",
                    "service_type": "roadside_assistance",
                    "urgency_handling": "immediate",
                    "tool_categories": ["dispatch", "location", "emergency"],
                    "response_style": "professional_urgent"
                })
                
            elif "billing" in agent_id.lower():
                enriched_context.update({
                    "domain": "financial_services", 
                    "service_type": "billing_support",
                    "urgency_handling": "standard",
                    "tool_categories": ["payment", "refund", "account"],
                    "response_style": "empathetic_helpful"
                })
                
            elif "technical" in agent_id.lower():
                enriched_context.update({
                    "domain": "technical_support",
                    "service_type": "troubleshooting", 
                    "urgency_handling": "methodical",
                    "tool_categories": ["diagnostic", "setup", "troubleshoot"],
                    "response_style": "patient_instructional"
                })
            else:
                # Default context for unknown agents
                enriched_context.update({
                    "domain": "general_support",
                    "service_type": "general_assistance", 
                    "urgency_handling": "standard",
                    "tool_categories": ["general"],
                    "response_style": "professional"
                })
            
            # Add query-specific context
            enriched_context["processed_query"] = query
            enriched_context["agent_specialization"] = agent_id
            enriched_context["context_timestamp"] = time.time()
            
            logger.debug(f"Enriched context for agent {agent_id}: {enriched_context.get('service_type')}")
            return enriched_context
            
        except Exception as e:
            logger.error(f"Error in get_agent_context: {e}")
            # Return base context if enrichment fails
            return base_context
    
    async def initialize(self):
        """Initialize all tiers with error handling."""
        logger.info("🚀 Initializing Hybrid Vector System...")
        
        initialization_errors = []
        
        # Initialize Redis cache (Tier 1)
        try:
            await self.redis_cache.initialize()
            logger.info("✅ Tier 1 (Redis Cache) initialized")
        except Exception as e:
            error_msg = f"Tier 1 (Redis Cache) initialization failed: {e}"
            logger.error(error_msg)
            initialization_errors.append(error_msg)
        
        # Initialize FAISS hot tier (Tier 2)
        try:
            await self.faiss_hot_tier.initialize()
            logger.info("✅ Tier 2 (FAISS Hot Tier) initialized")
        except Exception as e:
            error_msg = f"Tier 2 (FAISS Hot Tier) initialization failed: {e}"
            logger.error(error_msg)
            initialization_errors.append(error_msg)
        
        # Initialize Qdrant manager (Tier 3)
        try:
            await self.qdrant_manager.initialize()
            logger.info("✅ Tier 3 (Qdrant Manager) initialized")
        except Exception as e:
            error_msg = f"Tier 3 (Qdrant Manager) initialization failed: {e}"
            logger.error(error_msg)
            initialization_errors.append(error_msg)
        
        # Mark as initialized even if some tiers failed
        self.initialized = True
        
        if initialization_errors:
            logger.warning(f"⚠️ System initialized with {len(initialization_errors)} errors:")
            for error in initialization_errors:
                logger.warning(f"  • {error}")
            logger.info("🔄 System will continue with available tiers")
        else:
            logger.info("✅ All tiers initialized successfully")
    
    async def hybrid_search(
        self,
        query_vector: np.ndarray,
        agent_id: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[SearchResult]:
        """
        Perform hybrid search across all available tiers with fallback.
        """
        if not self.initialized:
            await self.initialize()
        
        search_start = time.time()
        
        try:
            self.performance_stats["searches_performed"] += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(query_vector, agent_id, top_k, filters)
            
            # Tier 1: Redis Cache
            cached_result = await self._search_redis_cache(cache_key)
            if cached_result:
                search_time = (time.time() - search_start) * 1000
                self.performance_stats["cache_hits"] += 1
                self._update_average_search_time(search_time)
                
                return SearchResult(
                    vectors=cached_result,
                    search_time_ms=search_time,
                    tier_used="redis_cache",
                    total_results=len(cached_result)
                )
            
            # Tier 2: FAISS Hot Tier
            faiss_results = await self._search_faiss_hot_tier(query_vector, top_k, similarity_threshold)
            if faiss_results:
                search_time = (time.time() - search_start) * 1000
                self.performance_stats["faiss_hits"] += 1
                self._update_average_search_time(search_time)
                
                # Cache results for future use
                await self._cache_search_results(cache_key, faiss_results)
                
                return SearchResult(
                    vectors=faiss_results,
                    search_time_ms=search_time,
                    tier_used="faiss_hot_tier",
                    total_results=len(faiss_results)
                )
            
            # Tier 3: Qdrant Manager
            qdrant_results = await self._search_qdrant(query_vector, agent_id, top_k, similarity_threshold, filters)
            if qdrant_results:
                search_time = (time.time() - search_start) * 1000
                self.performance_stats["qdrant_hits"] += 1
                self._update_average_search_time(search_time)
                
                # Cache results and promote to hot tier
                await self._cache_search_results(cache_key, qdrant_results)
                await self._promote_to_hot_tier(qdrant_results)
                
                return SearchResult(
                    vectors=qdrant_results,
                    search_time_ms=search_time,
                    tier_used="qdrant_manager",
                    total_results=len(qdrant_results)
                )
            
            # No results found
            search_time = (time.time() - search_start) * 1000
            self._update_average_search_time(search_time)
            
            return SearchResult(
                vectors=[],
                search_time_ms=search_time,
                tier_used="none",
                total_results=0
            )
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            search_time = (time.time() - search_start) * 1000
            logger.error(f"❌ Hybrid search error: {e}")
            
            return SearchResult(
                vectors=[],
                search_time_ms=search_time,
                tier_used="error",
                total_results=0,
                metadata={"error": str(e)}
            )
    
    async def _search_redis_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Search Redis cache tier."""
        try:
            cached_data = await self.redis_cache.get(cache_key)
            if cached_data is not None:
                # Deserialize cached results
                return self._deserialize_search_results(cached_data)
        except Exception as e:
            logger.debug(f"Redis cache search error: {e}")
        
        return None
    
    async def _search_faiss_hot_tier(
        self,
        query_vector: np.ndarray,
        top_k: int,
        similarity_threshold: float
    ) -> Optional[List[Dict[str, Any]]]:
        """Search FAISS hot tier."""
        try:
            results = await self.faiss_hot_tier.search(query_vector, top_k)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results
                if result.get("score", 0.0) >= similarity_threshold
            ]
            
            return filtered_results if filtered_results else None
        except Exception as e:
            logger.debug(f"FAISS hot tier search error: {e}")
        
        return None
    
    async def _search_qdrant(
        self,
        query_vector: np.ndarray,
        agent_id: Optional[str],
        top_k: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Search Qdrant tier."""
        try:
            # Determine collection name
            collection_name = f"agent_{agent_id}" if agent_id else "default_collection"
            
            # Ensure collection exists
            try:
                await self.qdrant_manager.create_collection(
                    collection_name, 
                    vector_size=len(query_vector)
                )
            except:
                pass  # Collection might already exist
            
            results = await self.qdrant_manager.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=similarity_threshold
            )
            
            return results if results else None
        except Exception as e:
            logger.debug(f"Qdrant search error: {e}")
        
        return None
    
    async def _cache_search_results(self, cache_key: str, results: List[Dict[str, Any]]):
        """Cache search results in Redis."""
        try:
            serialized_results = self._serialize_search_results(results)
            await self.redis_cache.set(cache_key, serialized_results, ttl=3600)
        except Exception as e:
            logger.debug(f"Result caching error: {e}")
    
    async def _promote_to_hot_tier(self, results: List[Dict[str, Any]]):
        """Promote frequently accessed vectors to FAISS hot tier."""
        try:
            vectors_to_add = {}
            for result in results:
                if "vector" in result and "id" in result:
                    vectors_to_add[result["id"]] = np.array(result["vector"])
            
            if vectors_to_add:
                await self.faiss_hot_tier.add_vectors(vectors_to_add)
        except Exception as e:
            logger.debug(f"Hot tier promotion error: {e}")
    
    def _generate_cache_key(
        self,
        query_vector: np.ndarray,
        agent_id: Optional[str],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for search parameters."""
        import hashlib
        
        # Create hash from query vector
        vector_hash = hashlib.md5(query_vector.tobytes()).hexdigest()[:8]
        
        # Include other parameters
        key_parts = [
            vector_hash,
            agent_id or "default",
            str(top_k),
            str(sorted(filters.items())) if filters else "no_filters"
        ]
        
        return ":".join(key_parts)
    
    def _serialize_search_results(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """Serialize search results for caching."""
        # Simple serialization - in production, use more sophisticated method
        import pickle
        serialized = pickle.dumps(results)
        return np.frombuffer(serialized, dtype=np.uint8)
    
    def _deserialize_search_results(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Deserialize cached search results."""
        import pickle
        return pickle.loads(data.tobytes())
    
    def _update_average_search_time(self, search_time_ms: float):
        """Update average search time statistics."""
        current_avg = self.performance_stats["average_search_time_ms"]
        total_searches = self.performance_stats["searches_performed"]
        
        if total_searches == 1:
            self.performance_stats["average_search_time_ms"] = search_time_ms
        else:
            new_avg = ((current_avg * (total_searches - 1)) + search_time_ms) / total_searches
            self.performance_stats["average_search_time_ms"] = new_avg
    
    async def add_vectors(
        self,
        vectors: List[Dict[str, Any]],
        agent_id: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """Add vectors to the system."""
        try:
            target_collection = collection_name or (f"agent_{agent_id}" if agent_id else "default_collection")
            
            # Add to Qdrant (persistent storage)
            await self.qdrant_manager.upsert_vectors(target_collection, vectors)
            
            # Optionally add to hot tier for immediate access
            hot_vectors = {}
            for vector_data in vectors:
                if "vector" in vector_data and "id" in vector_data:
                    hot_vectors[vector_data["id"]] = np.array(vector_data["vector"])
            
            if hot_vectors:
                await self.faiss_hot_tier.add_vectors(hot_vectors)
            
            logger.debug(f"Added {len(vectors)} vectors to {target_collection}")
            
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "system": {
                "initialized": self.initialized,
                "total_searches": self.performance_stats["searches_performed"],
                "average_search_time_ms": self.performance_stats["average_search_time_ms"],
                "error_count": self.performance_stats["error_count"]
            },
            "tier_performance": {
                "cache_hits": self.performance_stats["cache_hits"],
                "faiss_hits": self.performance_stats["faiss_hits"],
                "qdrant_hits": self.performance_stats["qdrant_hits"]
            },
            "tier_stats": {}
        }
        
        # Add individual tier stats
        try:
            stats["tier_stats"]["redis_cache"] = self.redis_cache.get_stats()
        except:
            stats["tier_stats"]["redis_cache"] = {"error": "stats unavailable"}
        
        try:
            stats["tier_stats"]["faiss_hot_tier"] = self.faiss_hot_tier.get_stats()
        except:
            stats["tier_stats"]["faiss_hot_tier"] = {"error": "stats unavailable"}
        
        try:
            stats["tier_stats"]["qdrant_manager"] = self.qdrant_manager.get_stats()
        except:
            stats["tier_stats"]["qdrant_manager"] = {"error": "stats unavailable"}
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all tiers."""
        health_status = {
            "overall_status": "healthy",
            "tiers": {}
        }
        
        # Check Redis cache
        try:
            redis_stats = self.redis_cache.get_stats()
            health_status["tiers"]["redis_cache"] = {
                "status": "healthy" if redis_stats.get("redis_available", False) else "degraded",
                "mode": redis_stats.get("mode", "unknown")
            }
        except Exception as e:
            health_status["tiers"]["redis_cache"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check FAISS hot tier
        try:
            faiss_stats = self.faiss_hot_tier.get_stats()
            health_status["tiers"]["faiss_hot_tier"] = {
                "status": "healthy",
                "vector_count": faiss_stats.get("vector_count", 0)
            }
        except Exception as e:
            health_status["tiers"]["faiss_hot_tier"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check Qdrant manager
        try:
            qdrant_stats = self.qdrant_manager.get_stats()
            health_status["tiers"]["qdrant_manager"] = {
                "status": "healthy" if qdrant_stats.get("qdrant_available", False) else "degraded",
                "mode": qdrant_stats.get("mode", "unknown")
            }
        except Exception as e:
            health_status["tiers"]["qdrant_manager"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Determine overall status
        tier_statuses = [tier["status"] for tier in health_status["tiers"].values()]
        if "error" in tier_statuses:
            health_status["overall_status"] = "degraded"
        elif "degraded" in tier_statuses:
            health_status["overall_status"] = "degraded"
        
        return health_status
    
    async def shutdown(self):
        """Shutdown the hybrid vector system."""
        logger.info("Shutting down Hybrid Vector System...")
        
        try:
            # Close Redis connection
            if hasattr(self.redis_cache, 'client') and self.redis_cache.client:
                await self.redis_cache.client.close()
            
            # Close Qdrant connection
            if hasattr(self.qdrant_manager, 'client') and self.qdrant_manager.client:
                await self.qdrant_manager.client.close()
            
            self.initialized = False
            logger.info("✅ Hybrid Vector System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")