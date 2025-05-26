"""
Qdrant Manager - Tier 3 (Sub-15ms latency)
Optimized Qdrant operations with advanced indexing and filtering.
Primary storage tier for all vectors with high-performance configuration.
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

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, OptimizersConfig, HnswConfig,
    BinaryQuantizationConfig, BinaryQuantization,
    PointStruct, Filter, FieldCondition, Range,
    SearchRequest, SearchParams, UpdateStatus
)

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
    High-performance Qdrant manager for primary vector storage.
    Optimized for <15ms search latency with advanced indexing.
    """
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6333,
                 grpc_port: int = 6334,
                 prefer_grpc: bool = True,
                 timeout: float = 10.0,
                 api_key: Optional[str] = None,
                 vector_dimension: int = 1536,
                 distance_metric: str = "cosine",
                 enable_quantization: bool = True,
                 optimization_level: str = "ultra_high_performance",
                 max_connections: int = 100,
                 enable_replication: bool = False):
        """Initialize Qdrant manager with performance optimizations."""
        
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        self.api_key = api_key
        self.vector_dimension = vector_dimension
        self.distance_metric = distance_metric
        self.enable_quantization = enable_quantization
        self.optimization_level = optimization_level
        self.max_connections = max_connections
        self.enable_replication = enable_replication
        
        # Clients
        self.client: Optional[AsyncQdrantClient] = None
        self.sync_client: Optional[QdrantClient] = None
        
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
            "optimization_runs": 0
        }
        
        # Configuration presets
        self.optimization_configs = {
            "ultra_high_performance": {
                "hnsw_m": 32,
                "hnsw_ef_construct": 256,
                "hnsw_max_indexing_threads": 8,
                "segments": 4,
                "memmap_threshold": 50000,
                "flush_interval": 1
            },
            "high_performance": {
                "hnsw_m": 16,
                "hnsw_ef_construct": 128,
                "hnsw_max_indexing_threads": 4,
                "segments": 2,
                "memmap_threshold": 20000,
                "flush_interval": 2
            },
            "balanced": {
                "hnsw_m": 8,
                "hnsw_ef_construct": 64,
                "hnsw_max_indexing_threads": 2,
                "segments": 1,
                "memmap_threshold": 10000,
                "flush_interval": 5
            }
        }
        
        self.initialized = False
        logger.info(f"Qdrant Manager initialized for {host}:{port} (optimization: {optimization_level})")
    
    async def initialize(self):
        """Initialize Qdrant connections with performance optimizations."""
        logger.info("🚀 Initializing Qdrant Manager...")
        
        try:
            # Initialize async client (preferred for performance)
            if self.prefer_grpc:
                self.client = AsyncQdrantClient(
                    host=self.host,
                    port=self.grpc_port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=True,
                    timeout=self.timeout,
                    api_key=self.api_key
                )
            else:
                self.client = AsyncQdrantClient(
                    host=self.host,
                    port=self.port,
                    timeout=self.timeout,
                    api_key=self.api_key
                )
            
            # Initialize sync client for management operations
            self.sync_client = QdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout,
                api_key=self.api_key
            )
            
            # Test connection
            collections = await self.client.get_collections()
            logger.info(f"✅ Qdrant connection established: {len(collections.collections)} collections found")
            
            # Load existing collections
            await self._load_existing_collections()
            
            self.initialized = True
            logger.info("✅ Qdrant Manager initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            raise
    
    async def _load_existing_collections(self):
        """Load information about existing collections."""
        try:
            collections_response = await self.client.get_collections()
            
            for collection in collections_response.collections:
                collection_info = await self.client.get_collection(collection.name)
                
                self.collections[collection.name] = CollectionInfo(
                    name=collection.name,
                    vector_count=collection_info.points_count or 0,
                    config=collection_info.config.dict() if collection_info.config else {},
                    created_at=time.time(),  # We don't have actual creation time
                    optimization_level=self.optimization_level
                )
                
                self.stats["total_vectors"] += collection_info.points_count or 0
            
            self.stats["collection_count"] = len(self.collections)
            logger.info(f"Loaded {len(self.collections)} existing collections")
            
        except Exception as e:
            logger.error(f"Error loading existing collections: {e}")
    
    async def create_optimized_collection(self,
                                        collection_name: str,
                                        agent_config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a performance-optimized collection for an agent."""
        if not self.initialized:
            raise RuntimeError("Qdrant Manager not initialized")
        
        try:
            # Check if collection already exists
            if collection_name in self.collections:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Get optimization config
            opt_config = self.optimization_configs.get(self.optimization_level, 
                                                      self.optimization_configs["balanced"])
            
            # Create collection with optimized configuration
            create_start = time.time()
            
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE if self.distance_metric == "cosine" else Distance.DOT,
                    on_disk=False  # Keep in memory for speed
                ),
                optimizers_config=OptimizersConfig(
                    default_segment_number=opt_config["segments"],
                    max_segment_size=20000,
                    memmap_threshold=opt_config["memmap_threshold"],
                    indexing_threshold=10000,
                    flush_interval_sec=opt_config["flush_interval"],
                    max_optimization_threads=4
                ),
                hnsw_config=HnswConfig(
                    m=opt_config["hnsw_m"],
                    ef_construct=opt_config["hnsw_ef_construct"],
                    full_scan_threshold=10000,
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
            
            logger.info(f"✅ Created optimized collection '{collection_name}' in {create_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection {collection_name}: {e}")
            return False
    
    async def add_vectors(self,
                         vectors: List[Dict[str, Any]],
                         collection_name: str,
                         batch_size: int = 100) -> Dict[str, Any]:
        """Add vectors to Qdrant with batch optimization."""
        if not self.initialized:
            raise RuntimeError("Qdrant Manager not initialized")
        
        insert_start = time.time()
        
        try:
            # Ensure collection exists
            if collection_name not in self.collections:
                await self.create_optimized_collection(collection_name)
            
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
                
                # Create point
                point = PointStruct(
                    id=vector_data.get("id", str(uuid.uuid4())),
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    payload=vector_data.get("metadata", {})
                )
                points.append(point)
            
            if not points:
                return {"success": False, "error": "No valid vectors found"}
            
            # Insert in batches for optimal performance
            inserted_count = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                operation_info = await self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True  # Wait for operation to complete
                )
                
                if operation_info.status == UpdateStatus.COMPLETED:
                    inserted_count += len(batch)
                else:
                    logger.warning(f"Batch insertion not completed: {operation_info.status}")
            
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
            
            logger.info(f"✅ Inserted {inserted_count} vectors into '{collection_name}' in {insert_time:.2f}ms")
            
            return {
                "success": True,
                "vectors_inserted": inserted_count,
                "insert_time_ms": insert_time,
                "collection": collection_name
            }
            
        except Exception as e:
            logger.error(f"❌ Error inserting vectors into {collection_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def search(self,
                    query_vector: np.ndarray,
                    collection_name: str,
                    top_k: int = 5,
                    score_threshold: float = 0.7,
                    filters: Optional[Dict[str, Any]] = None,
                    exact: bool = False) -> Optional[SearchResult]:
        """
        High-performance vector search with advanced filtering.
        Target: <15ms search latency.
        """
        if not self.initialized:
            raise RuntimeError("Qdrant Manager not initialized")
        
        search_start = time.time()
        
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                logger.debug(f"Collection {collection_name} not found")
                return None
            
            # Prepare query vector
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.astype(np.float32).tolist()
            
            # Build search request
            search_params = SearchParams(
                hnsw_ef=128 if not exact else None,
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
                    "filtered": filters is not None
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Qdrant search error in {collection_name}: {e}")
            return None
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary."""
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
    
    async def delete_vectors(self,
                           collection_name: str,
                           vector_ids: List[str]) -> bool:
        """Delete vectors from collection."""
        if not self.initialized:
            return False
        
        try:
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
            # Prepare points for update
            points = []
            for vector_data in vectors:
                if "vector" in vector_data:
                    vector = vector_data["vector"]
                elif "embedding" in vector_data:
                    vector = vector_data["embedding"]
                else:
                    continue
                
                # Ensure vector is the right type
                if isinstance(vector, np.ndarray):
                    vector = vector.astype(np.float32).tolist()
                
                point = PointStruct(
                    id=vector_data.get("id", str(uuid.uuid4())),
                    vector=vector,
                    payload=vector_data.get("metadata", {})
                )
                points.append(point)
            
            if not points:
                return False
            
            operation_info = await self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                self.stats["total_updates"] += len(points)
                
                # Update collection stats
                if collection_name in self.collections:
                    self.collections[collection_name].last_updated = time.time()
                
                logger.info(f"Updated {len(points)} vectors in {collection_name}")
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
            # Trigger optimization
            operation_info = await self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfig(
                    max_optimization_threads=4,
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
    
    async def list_collections(self) -> List[str]:
        """List all collections."""
        if not self.initialized:
            return []
        
        try:
            collections_response = await self.client.get_collections()
            return [collection.name for collection in collections_response.collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        if not self.initialized:
            return False
        
        try:
            await self.client.delete_collection(collection_name)
            
            # Update tracking
            if collection_name in self.collections:
                self.stats["total_vectors"] -= self.collections[collection_name].vector_count
                del self.collections[collection_name]
                self.stats["collection_count"] -= 1
            
            logger.info(f"Deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    async def create_index(self, collection_name: str, field_name: str, field_type: str = "keyword") -> bool:
        """Create an index on a payload field for faster filtering."""
        if not self.initialized:
            return False
        
        try:
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Qdrant manager statistics."""
        return {
            "configuration": {
                "host": self.host,
                "port": self.port,
                "grpc_port": self.grpc_port,
                "prefer_grpc": self.prefer_grpc,
                "vector_dimension": self.vector_dimension,
                "distance_metric": self.distance_metric,
                "optimization_level": self.optimization_level,
                "quantization_enabled": self.enable_quantization
            },
            "performance": {
                **self.stats,
                "target_latency_ms": 15.0,
                "meets_target": self.stats["average_search_time_ms"] <= 15.0
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
            "health": {
                "status": "healthy" if self.initialized else "down",
                "connected": self.client is not None,
                "performance_target_met": self.stats["average_search_time_ms"] <= 20.0  # Allow 20ms for "healthy"
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if Qdrant manager is healthy."""
        return (
            self.initialized and
            self.client is not None and
            self.stats["average_search_time_ms"] <= 20.0  # Allow up to 20ms for "healthy"
        )
    
    async def shutdown(self):
        """Shutdown Qdrant manager."""
        logger.info("Shutting down Qdrant Manager...")
        
        try:
            if self.client:
                await self.client.close()
            
            if self.sync_client:
                self.sync_client.close()
            
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