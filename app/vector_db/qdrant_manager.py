"""
Qdrant Manager - RunPod Optimized Version (FIXED)
===================================================

COMPLETE REPLACEMENT for app/vector_db/qdrant_manager.py
This fixes Qdrant connectivity issues and adds robust fallback mechanisms for RunPod deployment.
"""
import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result from Qdrant."""
    vectors: List[Dict[str, Any]]
    scores: List[float]
    search_time_ms: float
    collection_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollectionInfo:
    """Information about a Qdrant collection."""
    name: str
    vector_count: int
    config: Dict[str, Any]
    created_at: float
    last_updated: float = field(default_factory=time.time)
    search_count: int = 0
    average_search_time_ms: float = 0.0
    optimization_level: str = "standard"

class QdrantManager:
    """
    RunPod-optimized Qdrant manager with robust fallback mechanisms.
    Designed for cloud deployment with automatic degradation and recovery.
    """
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6333,
                 grpc_port: int = 6334,
                 prefer_grpc: bool = False,
                 timeout: float = 15.0,
                 api_key: Optional[str] = None,
                 vector_dimension: int = 1536,
                 distance_metric: str = "cosine",
                 enable_quantization: bool = True,
                 optimization_level: str = "runpod_optimized",
                 max_connections: int = 50,
                 enable_replication: bool = False,
                 fallback_to_memory: bool = True):
        """Initialize Qdrant manager with RunPod optimizations."""
        
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = False  # Force HTTP for better RunPod compatibility
        self.timeout = timeout
        self.api_key = api_key
        self.vector_dimension = vector_dimension
        self.distance_metric = distance_metric
        self.enable_quantization = enable_quantization
        self.optimization_level = optimization_level
        self.max_connections = max_connections
        self.enable_replication = enable_replication
        self.fallback_to_memory = fallback_to_memory
        
        # Clients
        self.client = None
        self.sync_client = None
        
        # In-memory fallback system
        self.memory_collections = {}
        self.using_memory_fallback = False
        
        # Connection state
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.last_connection_attempt = 0
        self.connection_retry_interval = 60  # seconds
        
        # Collection tracking
        self.collections: Dict[str, CollectionInfo] = {}
        
        # Performance tracking
        self.stats = {
            "total_searches": 0,
            "total_inserts": 0,
            "total_updates": 0,
            "total_deletes": 0,
            "average_search_time_ms": 0.0,
            "average_insert_time_ms": 0.0,
            "cache_hits": 0,
            "collection_count": 0,
            "total_vectors": 0,
            "optimization_runs": 0,
            "memory_fallback_usage": 0,
            "connection_failures": 0
        }
        
        # RunPod-optimized configuration presets
        self.optimization_configs = {
            "runpod_optimized": {
                "hnsw_m": 16,
                "hnsw_ef_construct": 128,
                "hnsw_max_indexing_threads": 2,  # Conservative for cloud
                "segments": 1,
                "memmap_threshold": 20000,
                "flush_interval": 3,
                "prefer_memory": True
            },
            "cloud_efficient": {
                "hnsw_m": 8,
                "hnsw_ef_construct": 64,
                "hnsw_max_indexing_threads": 1,
                "segments": 1,
                "memmap_threshold": 10000,
                "flush_interval": 5,
                "prefer_memory": True
            },
            "high_performance": {
                "hnsw_m": 32,
                "hnsw_ef_construct": 256,
                "hnsw_max_indexing_threads": 4,
                "segments": 2,
                "memmap_threshold": 50000,
                "flush_interval": 1,
                "prefer_memory": False
            }
        }
        
        self.initialized = False
        logger.info(f"Qdrant Manager initialized for RunPod deployment: {host}:{port}")
    
    async def initialize(self):
        """Initialize Qdrant with enhanced error handling for RunPod."""
        logger.info("🚀 Initializing Qdrant Manager for RunPod...")
        
        # Check if we should use in-memory mode from the start
        if self.host == ":memory:" or self.host == "memory":
            logger.info("🧠 Using in-memory mode by configuration")
            self.using_memory_fallback = True
            self.initialized = True
            return
        
        # Try to connect to Qdrant
        connection_success = await self._attempt_qdrant_connection()
        
        if not connection_success and self.fallback_to_memory:
            logger.info("🔄 Falling back to in-memory vector storage")
            self.using_memory_fallback = True
        
        self.initialized = True
        
        if self.using_memory_fallback:
            logger.info("✅ Qdrant Manager initialized with in-memory fallback")
        else:
            logger.info("✅ Qdrant Manager initialized with external Qdrant")
    
    async def _attempt_qdrant_connection(self) -> bool:
        """Attempt to connect to Qdrant with robust error handling."""
        if time.time() - self.last_connection_attempt < self.connection_retry_interval:
            return False
        
        self.last_connection_attempt = time.time()
        self.connection_attempts += 1
        
        try:
            # Import Qdrant client
            try:
                from qdrant_client import QdrantClient, AsyncQdrantClient
                from qdrant_client.http import models
                from qdrant_client.http.models import Distance, VectorParams
            except ImportError as e:
                logger.error(f"❌ Qdrant client not available: {e}")
                return False
            
            # Create async client with conservative settings
            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout,
                api_key=self.api_key,
                prefer_grpc=False,  # Use HTTP for better compatibility
                https=False  # RunPod typically uses HTTP
            )
            
            # Test connection with timeout
            try:
                collections = await asyncio.wait_for(
                    self.client.get_collections(), 
                    timeout=self.timeout
                )
                logger.info(f"✅ Qdrant connection successful: {len(collections.collections)} collections found")
                
                # Load existing collections
                await self._load_existing_collections()
                
                # Reset connection attempt counter on success
                self.connection_attempts = 0
                return True
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Qdrant connection timeout after {self.timeout}s")
                return False
            except Exception as e:
                logger.warning(f"⚠️ Qdrant connection test failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Qdrant client initialization failed: {e}")
            self.stats["connection_failures"] += 1
            return False
    
    async def _load_existing_collections(self):
        """Load information about existing collections."""
        if not self.client:
            return
        
        try:
            collections_response = await self.client.get_collections()
            
            for collection in collections_response.collections:
                try:
                    collection_info = await self.client.get_collection(collection.name)
                    
                    self.collections[collection.name] = CollectionInfo(
                        name=collection.name,
                        vector_count=collection_info.points_count or 0,
                        config=collection_info.config.dict() if collection_info.config else {},
                        created_at=time.time(),
                        optimization_level=self.optimization_level
                    )
                    
                    self.stats["total_vectors"] += collection_info.points_count or 0
                    
                except Exception as e:
                    logger.warning(f"Could not load collection info for {collection.name}: {e}")
            
            self.stats["collection_count"] = len(self.collections)
            logger.info(f"Loaded {len(self.collections)} existing collections")
            
        except Exception as e:
            logger.error(f"Error loading existing collections: {e}")
    
    async def create_optimized_collection(self,
                                        collection_name: str,
                                        agent_config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a collection with fallback support."""
        
        # If using memory fallback, create in-memory collection
        if self.using_memory_fallback:
            return await self._create_memory_collection(collection_name)
        
        # Try to create in Qdrant
        try:
            if not self.client:
                if not await self._attempt_qdrant_connection():
                    return await self._create_memory_collection(collection_name)
            
            # Check if collection already exists
            if collection_name in self.collections:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Get optimization config
            opt_config = self.optimization_configs.get(
                self.optimization_level, 
                self.optimization_configs["cloud_efficient"]
            )
            
            # Import required models
            from qdrant_client.http.models import (
                Distance, VectorParams, OptimizersConfig, HnswConfig,
                BinaryQuantizationConfig, BinaryQuantization
            )
            
            # Create collection with RunPod-optimized configuration
            create_start = time.time()
            
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE if self.distance_metric == "cosine" else Distance.DOT,
                    on_disk=not opt_config["prefer_memory"]  # Keep in memory for RunPod
                ),
                optimizers_config=OptimizersConfig(
                    default_segment_number=opt_config["segments"],
                    max_segment_size=10000,  # Smaller for cloud
                    memmap_threshold=opt_config["memmap_threshold"],
                    indexing_threshold=5000,  # Lower threshold for cloud
                    flush_interval_sec=opt_config["flush_interval"],
                    max_optimization_threads=opt_config["hnsw_max_indexing_threads"]
                ),
                hnsw_config=HnswConfig(
                    m=opt_config["hnsw_m"],
                    ef_construct=opt_config["hnsw_ef_construct"],
                    full_scan_threshold=5000,  # Lower for cloud efficiency
                    max_indexing_threads=opt_config["hnsw_max_indexing_threads"]
                ),
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ) if self.enable_quantization else None
            )
            
            create_time = (time.time() - create_start) * 1000
            
            # Track collection
            self.collections[collection_name] = CollectionInfo(
                name=collection_name,
                vector_count=0,
                config=opt_config,
                created_at=time.time(),
                optimization_level=self.optimization_level
            )
            
            self.stats["collection_count"] += 1
            
            logger.info(f"✅ Created Qdrant collection '{collection_name}' in {create_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create Qdrant collection {collection_name}: {e}")
            
            # Fallback to memory collection
            if self.fallback_to_memory:
                logger.info(f"🔄 Creating memory fallback collection: {collection_name}")
                return await self._create_memory_collection(collection_name)
            
            return False
    
    async def _create_memory_collection(self, collection_name: str) -> bool:
        """Create an in-memory collection."""
        try:
            self.memory_collections[collection_name] = {
                "vectors": {},
                "metadata": {},
                "created_at": time.time(),
                "vector_count": 0
            }
            
            self.collections[collection_name] = CollectionInfo(
                name=collection_name,
                vector_count=0,
                config={"type": "memory"},
                created_at=time.time(),
                optimization_level="memory"
            )
            
            self.stats["collection_count"] += 1
            logger.info(f"✅ Created in-memory collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create memory collection {collection_name}: {e}")
            return False
    
    async def add_vectors(self,
                         vectors: List[Dict[str, Any]],
                         collection_name: str,
                         batch_size: int = 50) -> Dict[str, Any]:  # Smaller batch for cloud
        """Add vectors with fallback support."""
        
        if not self.initialized:
            await self.initialize()
        
        insert_start = time.time()
        
        # Ensure collection exists
        if collection_name not in self.collections:
            await self.create_optimized_collection(collection_name)
        
        # Use memory fallback if needed
        if self.using_memory_fallback or collection_name in self.memory_collections:
            return await self._add_vectors_to_memory(vectors, collection_name)
        
        # Try Qdrant insertion
        try:
            if not self.client:
                if not await self._attempt_qdrant_connection():
                    return await self._add_vectors_to_memory(vectors, collection_name)
            
            # Prepare points for insertion
            points = []
            for vector_data in vectors:
                # Extract vector
                if "vector" in vector_data:
                    vector = vector_data["vector"]
                elif "embedding" in vector_data:
                    vector = vector_data["embedding"]
                else:
                    logger.warning(f"No vector found in data: {vector_data}")
                    continue
                
                # Ensure vector is the right type
                if isinstance(vector, list):
                    vector = np.array(vector, dtype=np.float32)
                elif isinstance(vector, np.ndarray):
                    vector = vector.astype(np.float32)
                
                # Import required models
                from qdrant_client.http.models import PointStruct
                
                # Create point
                point = PointStruct(
                    id=vector_data.get("id", str(uuid.uuid4())),
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    payload=vector_data.get("metadata", {})
                )
                points.append(point)
            
            if not points:
                return {"success": False, "error": "No valid vectors found"}
            
            # Insert in batches
            inserted_count = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                try:
                    from qdrant_client.http.models import UpdateStatus
                    
                    operation_info = await self.client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True
                    )
                    
                    if operation_info.status == UpdateStatus.COMPLETED:
                        inserted_count += len(batch)
                    else:
                        logger.warning(f"Batch insertion not completed: {operation_info.status}")
                        
                except Exception as e:
                    logger.error(f"Batch insertion failed: {e}")
                    # Try memory fallback for this batch
                    if self.fallback_to_memory:
                        return await self._add_vectors_to_memory(vectors, collection_name)
                    raise
            
            insert_time = (time.time() - insert_start) * 1000
            
            # Update statistics
            self.stats["total_inserts"] += inserted_count
            self.stats["total_vectors"] += inserted_count
            
            if self.stats["average_insert_time_ms"] == 0:
                self.stats["average_insert_time_ms"] = insert_time
            else:
                self.stats["average_insert_time_ms"] = (
                    (self.stats["average_insert_time_ms"] * (self.stats["total_inserts"] - inserted_count) + insert_time) /
                    self.stats["total_inserts"]
                )
            
            # Update collection info
            if collection_name in self.collections:
                self.collections[collection_name].vector_count += inserted_count
                self.collections[collection_name].last_updated = time.time()
            
            logger.info(f"✅ Inserted {inserted_count} vectors into Qdrant '{collection_name}' in {insert_time:.2f}ms")
            
            return {
                "success": True,
                "vectors_inserted": inserted_count,
                "insert_time_ms": insert_time,
                "collection": collection_name,
                "method": "qdrant"
            }
            
        except Exception as e:
            logger.error(f"❌ Error inserting vectors into Qdrant {collection_name}: {e}")
            
            # Fallback to memory
            if self.fallback_to_memory:
                logger.info("🔄 Falling back to memory insertion")
                return await self._add_vectors_to_memory(vectors, collection_name)
            
            return {"success": False, "error": str(e)}
    
    async def _add_vectors_to_memory(self, vectors: List[Dict[str, Any]], collection_name: str) -> Dict[str, Any]:
        """Add vectors to in-memory storage."""
        insert_start = time.time()
        
        try:
            # Ensure memory collection exists
            if collection_name not in self.memory_collections:
                await self._create_memory_collection(collection_name)
            
            collection = self.memory_collections[collection_name]
            inserted_count = 0
            
            for vector_data in vectors:
                # Extract vector
                if "vector" in vector_data:
                    vector = vector_data["vector"]
                elif "embedding" in vector_data:
                    vector = vector_data["embedding"]
                else:
                    continue
                
                # Convert to numpy array
                if isinstance(vector, list):
                    vector = np.array(vector, dtype=np.float32)
                elif isinstance(vector, np.ndarray):
                    vector = vector.astype(np.float32)
                
                vector_id = vector_data.get("id", str(uuid.uuid4()))
                
                # Store vector and metadata
                collection["vectors"][vector_id] = vector
                collection["metadata"][vector_id] = vector_data.get("metadata", {})
                inserted_count += 1
            
            collection["vector_count"] = len(collection["vectors"])
            
            insert_time = (time.time() - insert_start) * 1000
            
            # Update statistics
            self.stats["memory_fallback_usage"] += 1
            self.stats["total_inserts"] += inserted_count
            self.stats["total_vectors"] += inserted_count
            
            # Update collection info
            if collection_name in self.collections:
                self.collections[collection_name].vector_count = collection["vector_count"]
                self.collections[collection_name].last_updated = time.time()
            
            logger.info(f"✅ Inserted {inserted_count} vectors into memory '{collection_name}' in {insert_time:.2f}ms")
            
            return {
                "success": True,
                "vectors_inserted": inserted_count,
                "insert_time_ms": insert_time,
                "collection": collection_name,
                "method": "memory"
            }
            
        except Exception as e:
            logger.error(f"❌ Error inserting vectors into memory {collection_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def search(self,
                    query_vector: np.ndarray,
                    collection_name: str,
                    top_k: int = 5,
                    score_threshold: float = 0.7,
                    filters: Optional[Dict[str, Any]] = None,
                    exact: bool = False) -> Optional[SearchResult]:
        """Search with fallback support."""
        
        if not self.initialized:
            await self.initialize()
        
        search_start = time.time()
        
        # Check if collection exists
        if collection_name not in self.collections:
            logger.debug(f"Collection {collection_name} not found")
            return None
        
        # Use memory search if needed
        if self.using_memory_fallback or collection_name in self.memory_collections:
            return await self._search_memory(query_vector, collection_name, top_k, score_threshold)
        
        # Try Qdrant search
        try:
            if not self.client:
                if not await self._attempt_qdrant_connection():
                    return await self._search_memory(query_vector, collection_name, top_k, score_threshold)
            
            # Prepare query vector
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.astype(np.float32).tolist()
            
            from qdrant_client.http.models import SearchParams
            
            # Build search request
            search_params = SearchParams(
                hnsw_ef=64 if not exact else None,  # Conservative for cloud
                exact=exact
            )
            
            # Build filter if provided
            filter_obj = None
            if filters:
                filter_obj = self._build_filter(filters)
            
            # Execute search
            search_results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_obj,
                limit=top_k,
                score_threshold=score_threshold,
                search_params=search_params
            )
            
            search_time = (time.time() - search_start) * 1000
            
            # Process results
            vectors = []
            scores = []
            
            for result in search_results:
                vectors.append({
                    "id": result.id,
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload
                })
                scores.append(result.score)
            
            # Update statistics
            self._update_search_stats(collection_name, search_time)
            
            logger.debug(f"🎯 Qdrant search completed: {search_time:.2f}ms for {collection_name}")
            
            return SearchResult(
                vectors=vectors,
                scores=scores,
                search_time_ms=search_time,
                collection_used=collection_name,
                metadata={
                    "total_results": len(vectors),
                    "score_threshold": score_threshold,
                    "exact_search": exact,
                    "filtered": filters is not None,
                    "method": "qdrant"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Qdrant search error in {collection_name}: {e}")
            
            # Fallback to memory search
            if self.fallback_to_memory:
                logger.info("🔄 Falling back to memory search")
                return await self._search_memory(query_vector, collection_name, top_k, score_threshold)
            
            return None
    
    async def _search_memory(self, query_vector: np.ndarray, collection_name: str, 
                            top_k: int, score_threshold: float) -> Optional[SearchResult]:
        """Search in-memory collection."""
        search_start = time.time()
        
        try:
            if collection_name not in self.memory_collections:
                return None
            
            collection = self.memory_collections[collection_name]
            vectors_dict = collection["vectors"]
            metadata_dict = collection["metadata"]
            
            if not vectors_dict:
                return SearchResult(
                    vectors=[], scores=[], search_time_ms=0.0, collection_used=collection_name,
                    metadata={"method": "memory", "total_results": 0}
                )
            
            # Ensure query vector is numpy array
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype=np.float32)
            
            # Calculate similarities
            similarities = []
            for vector_id, vector in vectors_dict.items():
                similarity = self._cosine_similarity(query_vector, vector)
                if similarity >= score_threshold:
                    similarities.append((vector_id, similarity))
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            # Format results
            vectors = []
            scores = []
            
            for vector_id, similarity in top_results:
                metadata = metadata_dict.get(vector_id, {})
                vectors.append({
                    "id": vector_id,
                    "text": metadata.get("text", ""),
                    "metadata": metadata
                })
                scores.append(similarity)
            
            search_time = (time.time() - search_start) * 1000
            
            # Update statistics
            self._update_search_stats(collection_name, search_time)
            self.stats["memory_fallback_usage"] += 1
            
            logger.debug(f"🧠 Memory search completed: {search_time:.2f}ms for {collection_name}")
            
            return SearchResult(
                vectors=vectors,
                scores=scores,
                search_time_ms=search_time,
                collection_used=collection_name,
                metadata={
                    "total_results": len(vectors),
                    "score_threshold": score_threshold,
                    "method": "memory"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Memory search error in {collection_name}: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def _build_filter(self, filters: Dict[str, Any]):
        """Build Qdrant filter from dictionary."""
        try:
            from qdrant_client.http.models import Filter, FieldCondition, Range
            from qdrant_client.http import models
            
            conditions = []
            
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Range filters
                    if "$gte" in value or "$lte" in value or "$gt" in value or "$lt" in value:
                        range_filter = {}
                        if "$gte" in value:
                            range_filter["gte"] = value["$gte"]
                        if "$lte" in value:
                            range_filter["lte"] = value["$lte"]
                        if "$gt" in value:
                            range_filter["gt"] = value["$gt"]
                        if "$lt" in value:
                            range_filter["lt"] = value["$lt"]
                        
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(**range_filter)
                            )
                        )
                elif isinstance(value, list):
                    # Match any value in list
                    for val in value:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=models.MatchValue(value=val)
                            )
                        )
                else:
                    # Exact match
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            
            return Filter(must=conditions) if conditions else None
            
        except Exception as e:
            logger.error(f"Error building filter: {e}")
            return None
    
    def _update_search_stats(self, collection_name: str, search_time_ms: float):
        """Update search statistics."""
        self.stats["total_searches"] += 1
        
        # Update global average
        if self.stats["average_search_time_ms"] == 0:
            self.stats["average_search_time_ms"] = search_time_ms
        else:
            self.stats["average_search_time_ms"] = (
                (self.stats["average_search_time_ms"] * (self.stats["total_searches"] - 1) + search_time_ms) /
                self.stats["total_searches"]
            )
        
        # Update collection-specific stats
        if collection_name in self.collections:
            collection_info = self.collections[collection_name]
            collection_info.search_count += 1
            
            if collection_info.average_search_time_ms == 0:
                collection_info.average_search_time_ms = search_time_ms
            else:
                collection_info.average_search_time_ms = (
                    (collection_info.average_search_time_ms * (collection_info.search_count - 1) + search_time_ms) /
                    collection_info.search_count
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "configuration": {
                "host": self.host,
                "port": self.port,
                "vector_dimension": self.vector_dimension,
                "distance_metric": self.distance_metric,
                "optimization_level": self.optimization_level,
                "fallback_enabled": self.fallback_to_memory,
                "using_memory_fallback": self.using_memory_fallback
            },
            "performance": {
                **self.stats,
                "target_latency_ms": 50.0,  # Relaxed for cloud deployment
                "meets_target": self.stats["average_search_time_ms"] <= 50.0
            },
            "collections": {
                name: {
                    "vector_count": info.vector_count,
                    "search_count": info.search_count,
                    "average_search_time_ms": info.average_search_time_ms,
                    "last_updated": info.last_updated,
                    "optimization_level": info.optimization_level
                }
                for name, info in self.collections.items()
            },
            "memory_collections": {
                name: {
                    "vector_count": collection["vector_count"],
                    "created_at": collection["created_at"]
                }
                for name, collection in self.memory_collections.items()
            },
            "health": {
                "status": "healthy" if self.initialized else "down",
                "connected": self.client is not None and not self.using_memory_fallback,
                "memory_fallback_active": self.using_memory_fallback,
                "connection_attempts": self.connection_attempts,
                "performance_acceptable": self.stats["average_search_time_ms"] <= 100.0
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if Qdrant manager is healthy."""
        return (
            self.initialized and
            (self.client is not None or self.using_memory_fallback) and
            self.stats["average_search_time_ms"] <= 100.0  # Relaxed for cloud
        )
    
    async def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            if self.using_memory_fallback:
                return list(self.memory_collections.keys())
            
            if self.client:
                collections_response = await self.client.get_collections()
                return [collection.name for collection in collections_response.collections]
            
            return list(self.collections.keys())
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return list(self.collections.keys())
    
    async def delete_vectors(self,
                           collection_name: str,
                           vector_ids: List[str]) -> bool:
        """Delete vectors from collection."""
        if not self.initialized:
            return False
        
        try:
            if self.using_memory_fallback or collection_name in self.memory_collections:
                # Delete from memory
                if collection_name in self.memory_collections:
                    collection = self.memory_collections[collection_name]
                    deleted_count = 0
                    for vector_id in vector_ids:
                        if vector_id in collection["vectors"]:
                            del collection["vectors"][vector_id]
                            if vector_id in collection["metadata"]:
                                del collection["metadata"][vector_id]
                            deleted_count += 1
                    
                    collection["vector_count"] = len(collection["vectors"])
                    self.stats["total_deletes"] += deleted_count
                    self.stats["total_vectors"] -= deleted_count
                    
                    # Update collection info
                    if collection_name in self.collections:
                        self.collections[collection_name].vector_count = collection["vector_count"]
                        self.collections[collection_name].last_updated = time.time()
                    
                    logger.info(f"Deleted {deleted_count} vectors from memory collection {collection_name}")
                    return deleted_count > 0
                
                return False
            
            # Delete from Qdrant
            if not self.client:
                return False
            
            from qdrant_client.http.models import UpdateStatus
            from qdrant_client.http import models
            
            operation_info = await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=vector_ids
                ),
                wait=True
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                self.stats["total_deletes"] += len(vector_ids)
                self.stats["total_vectors"] -= len(vector_ids)
                
                # Update collection stats
                if collection_name in self.collections:
                    self.collections[collection_name].vector_count -= len(vector_ids)
                    self.collections[collection_name].last_updated = time.time()
                
                logger.info(f"Deleted {len(vector_ids)} vectors from {collection_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting vectors from {collection_name}: {e}")
            return False
    
    async def update_vectors(self,
                           collection_name: str,
                           vectors: List[Dict[str, Any]]) -> bool:
        """Update existing vectors in collection."""
        if not self.initialized:
            return False
        
        try:
            # Use the same logic as add_vectors (upsert behavior)
            result = await self.add_vectors(vectors, collection_name)
            
            if result.get("success"):
                self.stats["total_updates"] += result.get("vectors_inserted", 0)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating vectors in {collection_name}: {e}")
            return False
    
    async def optimize_collection(self, collection_name: str) -> bool:
        """Optimize collection for better performance."""
        if not self.initialized or collection_name not in self.collections:
            return False
        
        try:
            # Skip optimization for memory collections
            if self.using_memory_fallback or collection_name in self.memory_collections:
                logger.info(f"Skipping optimization for memory collection: {collection_name}")
                return True
            
            if not self.client:
                return False
            
            # Trigger optimization
            from qdrant_client.http.models import OptimizersConfig
            
            operation_info = await self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfig(
                    max_optimization_threads=2,  # Conservative for cloud
                    flush_interval_sec=1
                )
            )
            
            self.stats["optimization_runs"] += 1
            logger.info(f"Optimized collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing collection {collection_name}: {e}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a collection."""
        if not self.initialized:
            return None
        
        try:
            if collection_name in self.memory_collections:
                collection = self.memory_collections[collection_name]
                return {
                    "name": collection_name,
                    "vector_count": collection["vector_count"],
                    "vector_size": self.vector_dimension,
                    "distance_metric": self.distance_metric,
                    "status": "memory",
                    "storage_type": "in_memory",
                    "created_at": collection["created_at"]
                }
            
            if not self.client:
                return None
            
            collection_info = await self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "vector_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "status": collection_info.status.value,
                "optimizer_status": collection_info.optimizer_status.dict() if collection_info.optimizer_status else None,
                "indexing_threshold": collection_info.config.optimizer_config.indexing_threshold,
                "segments_count": len(collection_info.segments) if collection_info.segments else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        if not self.initialized:
            return False
        
        try:
            # Delete memory collection
            if collection_name in self.memory_collections:
                collection = self.memory_collections[collection_name]
                self.stats["total_vectors"] -= collection["vector_count"]
                del self.memory_collections[collection_name]
                logger.info(f"Deleted memory collection: {collection_name}")
            
            # Delete Qdrant collection
            if self.client and not self.using_memory_fallback:
                await self.client.delete_collection(collection_name)
                logger.info(f"Deleted Qdrant collection: {collection_name}")
            
            # Update tracking
            if collection_name in self.collections:
                self.stats["total_vectors"] -= self.collections[collection_name].vector_count
                del self.collections[collection_name]
                self.stats["collection_count"] -= 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    async def create_index(self, collection_name: str, field_name: str, field_type: str = "keyword") -> bool:
        """Create an index on a payload field for faster filtering."""
        if not self.initialized:
            return False
        
        try:
            # Skip indexing for memory collections
            if self.using_memory_fallback or collection_name in self.memory_collections:
                logger.info(f"Skipping index creation for memory collection: {collection_name}")
                return True
            
            if not self.client:
                return False
            
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            
            logger.info(f"Created index on {field_name} for collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown Qdrant manager."""
        logger.info("Shutting down Qdrant Manager...")
        
        try:
            if self.client:
                await self.client.close()
            
            if self.sync_client:
                self.sync_client.close()
            
            # Clear memory collections
            self.memory_collections.clear()
            
            self.initialized = False
            logger.info("✅ Qdrant Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Qdrant shutdown: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()