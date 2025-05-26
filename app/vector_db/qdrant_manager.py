"""
Optimized Qdrant Manager - High-performance vector database for cold storage
Implements production-ready Qdrant with <50ms query latency
"""
import asyncio
import time
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, Range, MatchValue, ScoredPoint,
    OptimizersConfig, HnswConfig, PayloadIndexParams
)

from app.vector_db.hybrid_vector_store import VectorSearchResult

logger = logging.getLogger(__name__)

@dataclass
class QdrantCollectionConfig:
    """Configuration for Qdrant collection optimization"""
    collection_name: str
    vector_size: int = 1536
    distance: Distance = Distance.COSINE
    hnsw_ef_construct: int = 200
    hnsw_m: int = 16
    hnsw_ef: int = 128
    segments_count: int = 2
    optimizer_deleted_threshold: float = 0.2
    optimizer_vacuum_min_vector_number: int = 1000
    optimizer_default_segment_number: int = 2
    on_disk_payload: bool = False
    replication_factor: int = 1

class QdrantManager:
    """
    Optimized Qdrant Manager for cold storage with <50ms query latency
    
    Features:
    - Async operations for better performance
    - Optimized HNSW parameters for speed
    - Efficient batch operations
    - Smart indexing strategies
    - Memory-optimized configurations
    """
    
    # Performance targets
    TARGET_QUERY_LATENCY = 50.0  # ms
    BATCH_SIZE = 100             # documents per batch
    MAX_RETRIES = 3              # retry attempts
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        timeout: float = 10.0
    ):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        
        # Client connections
        self.client: Optional[AsyncQdrantClient] = None
        self.sync_client: Optional[QdrantClient] = None
        
        # Collection configurations
        self.collections: Dict[str, QdrantCollectionConfig] = {}
        
        # Performance tracking
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "error_count": 0,
            "last_query_time": 0.0
        }
        
        # Agent-specific stats
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        
        self.initialized = False
        
        logger.info(f"🟢 Qdrant Manager initialized - {host}:{port}")
    
    async def init(self):
        """Initialize Qdrant connections with optimized settings"""
        logger.info("🔄 Initializing Qdrant Manager...")
        
        try:
            # Initialize async client for main operations
            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.grpc_port if self.prefer_grpc else self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout
            )
            
            # Initialize sync client for administrative operations
            self.sync_client = QdrantClient(
                host=self.host,
                port=self.grpc_port if self.prefer_grpc else self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout
            )
            
            # Test connection
            await self._test_connection()
            
            # Get existing collections
            await self._discover_existing_collections()
            
            self.initialized = True
            logger.info("✅ Qdrant Manager initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Qdrant Manager initialization failed: {e}")
            raise
    
    async def _test_connection(self):
        """Test Qdrant connection"""
        try:
            collections = await self.client.get_collections()
            logger.info(f"🔗 Connected to Qdrant - {len(collections.collections)} collections found")
        except Exception as e:
            logger.error(f"❌ Qdrant connection test failed: {e}")
            raise
    
    async def _discover_existing_collections(self):
        """Discover and configure existing collections"""
        try:
            collections_response = await self.client.get_collections()
            
            for collection in collections_response.collections:
                collection_name = collection.name
                
                # Get collection info
                info = await self.client.get_collection(collection_name)
                
                # Create configuration
                config = QdrantCollectionConfig(
                    collection_name=collection_name,
                    vector_size=info.config.params.vectors.size,
                    distance=info.config.params.vectors.distance
                )
                
                self.collections[collection_name] = config
                
                # Initialize agent stats if it's an agent collection
                if collection_name.startswith("agent-"):
                    agent_id = collection_name.replace("agent-", "")
                    self.agent_stats[agent_id] = {
                        "collection_name": collection_name,
                        "document_count": 0,
                        "last_updated": time.time(),
                        "query_count": 0,
                        "average_latency": 0.0
                    }
            
            logger.info(f"📚 Discovered {len(self.collections)} existing collections")
            
        except Exception as e:
            logger.error(f"❌ Collection discovery failed: {e}")
    
    async def create_agent_collection(
        self, 
        agent_id: str, 
        knowledge_sources: List[Dict[str, Any]]
    ):
        """Create optimized collection for an agent"""
        collection_name = f"agent-{agent_id}"
        
        logger.info(f"📚 Creating Qdrant collection: {collection_name}")
        
        try:
            # Check if collection already exists
            try:
                existing = await self.client.get_collection(collection_name)
                logger.info(f"✅ Collection {collection_name} already exists")
                return
            except:
                pass  # Collection doesn't exist, create it
            
            # Create optimized collection configuration
            config = QdrantCollectionConfig(collection_name=collection_name)
            
            # Create collection with optimized parameters
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=config.vector_size,
                    distance=config.distance,
                    hnsw_config=HnswConfig(
                        ef_construct=config.hnsw_ef_construct,
                        m=config.hnsw_m,
                        full_scan_threshold=20000,  # Use full scan for small collections
                        max_indexing_threads=4,
                        on_disk=False  # Keep in memory for speed
                    ),
                    on_disk=config.on_disk_payload
                ),
                optimizers_config=OptimizersConfig(
                    deleted_threshold=config.optimizer_deleted_threshold,
                    vacuum_min_vector_number=config.optimizer_vacuum_min_vector_number,
                    default_segment_number=config.optimizer_default_segment_number,
                    max_segment_size=20000,  # Smaller segments for faster queries
                    memmap_threshold=20000,
                    indexing_threshold=20000,
                    flush_interval_sec=5,
                    max_optimization_threads=2
                ),
                replication_factor=config.replication_factor,
                write_consistency_factor=1,
                timeout=60
            )
            
            # Store configuration
            self.collections[collection_name] = config
            
            # Initialize agent stats
            self.agent_stats[agent_id] = {
                "collection_name": collection_name,
                "document_count": 0,
                "last_updated": time.time(),
                "query_count": 0,
                "average_latency": 0.0,
                "knowledge_sources": knowledge_sources
            }
            
            # Create payload indexes for better filtering performance
            await self._create_payload_indexes(collection_name)
            
            logger.info(f"✅ Collection {collection_name} created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating collection {collection_name}: {e}")
            raise
    
    async def _create_payload_indexes(self, collection_name: str):
        """Create payload indexes for efficient filtering"""
        try:
            # Index agent_id for fast agent-specific queries
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name="agent_id",
                field_schema=PayloadIndexParams(type="keyword")
            )
            
            # Index document_type for filtering by document type
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name="document_type",
                field_schema=PayloadIndexParams(type="keyword")
            )
            
            # Index timestamp for time-based filtering
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name="timestamp",
                field_schema=PayloadIndexParams(type="integer")
            )
            
            logger.debug(f"🔍 Payload indexes created for {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Error creating payload indexes: {e}")
    
    async def add_documents_to_agent(
        self, 
        agent_id: str, 
        documents: List[Dict[str, Any]]
    ):
        """Add documents to agent collection with batch optimization"""
        collection_name = f"agent-{agent_id}"
        
        if collection_name not in self.collections:
            logger.error(f"❌ Collection {collection_name} not found")
            return
        
        logger.info(f"📄 Adding {len(documents)} documents to {collection_name}")
        
        try:
            # Process documents in batches for optimal performance
            total_added = 0
            
            for i in range(0, len(documents), self.BATCH_SIZE):
                batch = documents[i:i + self.BATCH_SIZE]
                points = []
                
                for doc in batch:
                    # Generate embeddings if not present
                    if "embedding" not in doc:
                        # In production, this would use the embedding service
                        # For now, we'll skip documents without embeddings
                        continue
                    
                    point_id = doc.get("id", str(uuid.uuid4()))
                    
                    # Create point with optimized payload
                    point = PointStruct(
                        id=point_id,
                        vector=doc["embedding"],
                        payload={
                            "agent_id": agent_id,
                            "text": doc.get("text", ""),
                            "source": doc.get("source", "unknown"),
                            "document_type": doc.get("document_type", "text"),
                            "timestamp": int(time.time()),
                            "metadata": doc.get("metadata", {})
                        }
                    )
                    points.append(point)
                
                if points:
                    # Upsert batch with retry logic
                    await self._upsert_points_with_retry(collection_name, points)
                    total_added += len(points)
                    
                    logger.debug(f"📄 Batch {i//self.BATCH_SIZE + 1}: Added {len(points)} points")
            
            # Update agent stats
            if agent_id in self.agent_stats:
                self.agent_stats[agent_id]["document_count"] += total_added
                self.agent_stats[agent_id]["last_updated"] = time.time()
            
            logger.info(f"✅ Added {total_added} documents to {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to {collection_name}: {e}")
            raise
    
    async def _upsert_points_with_retry(
        self, 
        collection_name: str, 
        points: List[PointStruct]
    ):
        """Upsert points with retry logic"""
        for attempt in range(self.MAX_RETRIES):
            try:
                await self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True  # Wait for operation to complete
                )
                return
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise e
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"⚠️ Upsert attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def query_agent(
        self,
        agent_id: str,
        query: Union[str, np.ndarray],
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """Query agent collection with optimized performance"""
        collection_name = f"agent-{agent_id}"
        
        if collection_name not in self.collections:
            logger.error(f"❌ Collection {collection_name} not found")
            return []
        
        query_start = time.time()
        
        try:
            # Convert query to vector if needed
            if isinstance(query, str):
                # In production, this would use the embedding service
                logger.warning("String query provided but no embedding service configured")
                return []
            
            query_vector = query.tolist() if isinstance(query, np.ndarray) else query
            
            # Prepare search request with optimized parameters
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": top_k,
                "score_threshold": similarity_threshold,
                "with_payload": True,
                "with_vectors": False,  # Don't return vectors to save bandwidth
                "query_filter": Filter(
                    must=[
                        FieldCondition(
                            key="agent_id",
                            match=MatchValue(value=agent_id)
                        )
                    ]
                )
            }
            
            # Execute search with performance tracking
            search_result = await self.client.search(**search_params)
            
            # Convert to VectorSearchResult objects
            results = []
            for point in search_result:
                result = VectorSearchResult(
                    id=str(point.id),
                    text=point.payload.get("text", ""),
                    score=point.score,
                    metadata=point.payload.get("metadata", {}),
                    source_tier="qdrant",
                    retrieval_time_ms=(time.time() - query_start) * 1000,
                    agent_id=agent_id
                )
                results.append(result)
            
            # Update performance stats
            query_latency = (time.time() - query_start) * 1000
            await self._update_query_stats(agent_id, query_latency, len(results))
            
            # Log performance warnings
            if query_latency > self.TARGET_QUERY_LATENCY:
                logger.warning(f"⚠️ Query latency exceeded target: {query_latency:.1f}ms > {self.TARGET_QUERY_LATENCY:.1f}ms")
            
            logger.debug(f"🔍 Query returned {len(results)} results in {query_latency:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Query error for agent {agent_id}: {e}")
            self.query_stats["error_count"] += 1
            return []
    
    async def _update_query_stats(
        self, 
        agent_id: str, 
        latency_ms: float, 
        result_count: int
    ):
        """Update query performance statistics"""
        # Update global stats
        self.query_stats["total_queries"] += 1
        self.query_stats["last_query_time"] = time.time()
        
        if result_count > 0:
            self.query_stats["successful_queries"] += 1
        
        # Update average latency (exponential moving average)
        if self.query_stats["average_latency_ms"] == 0:
            self.query_stats["average_latency_ms"] = latency_ms
        else:
            alpha = 0.1
            self.query_stats["average_latency_ms"] = (
                alpha * latency_ms + 
                (1 - alpha) * self.query_stats["average_latency_ms"]
            )
        
        # Update P95 latency (simplified)
        self.query_stats["p95_latency_ms"] = max(
            self.query_stats["p95_latency_ms"],
            latency_ms
        )
        
        # Update agent-specific stats
        if agent_id in self.agent_stats:
            agent_stats = self.agent_stats[agent_id]
            agent_stats["query_count"] += 1
            
            if agent_stats["average_latency"] == 0:
                agent_stats["average_latency"] = latency_ms
            else:
                alpha = 0.1
                agent_stats["average_latency"] = (
                    alpha * latency_ms + 
                    (1 - alpha) * agent_stats["average_latency"]
                )
    
    async def delete_agent_collection(self, agent_id: str) -> bool:
        """Delete agent collection"""
        collection_name = f"agent-{agent_id}"
        
        try:
            await self.client.delete_collection(collection_name)
            
            # Remove from tracking
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            if agent_id in self.agent_stats:
                del self.agent_stats[agent_id]
            
            logger.info(f"🗑️ Deleted collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting collection {collection_name}: {e}")
            return False
    
    async def get_collection_info(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed collection information"""
        collection_name = f"agent-{agent_id}"
        
        try:
            # Get collection info
            info = await self.client.get_collection(collection_name)
            
            # Get collection stats
            stats = {
                "collection_name": collection_name,
                "agent_id": agent_id,
                "vector_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.value,
                "segments_count": info.segments_count,
                "status": info.status.value,
                "optimizer_status": info.optimizer_status.value if info.optimizer_status else "unknown",
                "indexed_vectors_count": info.indexed_vectors_count or 0
            }
            
            # Add performance stats if available
            if agent_id in self.agent_stats:
                agent_stats = self.agent_stats[agent_id]
                stats.update({
                    "query_count": agent_stats["query_count"],
                    "average_latency_ms": round(agent_stats["average_latency"], 2),
                    "last_updated": agent_stats["last_updated"]
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting collection info for {agent_id}: {e}")
            return {"error": str(e)}
    
    async def optimize_collection(self, agent_id: str) -> bool:
        """Optimize collection for better performance"""
        collection_name = f"agent-{agent_id}"
        
        try:
            # Update collection configuration for optimal performance
            config_update = {
                "optimizers_config": OptimizersConfig(
                    deleted_threshold=0.1,  # More aggressive cleanup
                    vacuum_min_vector_number=500,
                    default_segment_number=1,  # Merge segments
                    max_segment_size=50000,
                    indexing_threshold=10000,
                    flush_interval_sec=3
                )
            }
            
            await self.client.update_collection(
                collection_name=collection_name,
                **config_update
            )
            
            logger.info(f"⚡ Optimized collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error optimizing collection {collection_name}: {e}")
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Get cluster info
            cluster_info = await self.client.cluster_info()
            
            stats = {
                "timestamp": time.time(),
                "cluster_info": {
                    "peer_id": cluster_info.peer_id,
                    "peers_count": len(cluster_info.peers),
                    "raft_info": {
                        "term": cluster_info.raft_info.term,
                        "commit": cluster_info.raft_info.commit,
                        "pending_operations": cluster_info.raft_info.pending_operations,
                        "leader": cluster_info.raft_info.leader,
                        "role": cluster_info.raft_info.role.value
                    }
                },
                "performance": self.query_stats.copy(),
                "collections": {
                    "total_collections": len(self.collections),
                    "agent_collections": len(self.agent_stats),
                    "collection_details": {}
                },
                "agents": {}
            }
            
            # Add collection details
            for collection_name, config in self.collections.items():
                try:
                    info = await self.client.get_collection(collection_name)
                    stats["collections"]["collection_details"][collection_name] = {
                        "points_count": info.points_count,
                        "segments_count": info.segments_count,
                        "status": info.status.value,
                        "vector_size": config.vector_size
                    }
                except:
                    pass
            
            # Add agent-specific stats
            for agent_id, agent_stats in self.agent_stats.items():
                stats["agents"][agent_id] = {
                    "collection_name": agent_stats["collection_name"],
                    "document_count": agent_stats["document_count"],
                    "query_count": agent_stats["query_count"],
                    "average_latency_ms": round(agent_stats["average_latency"], 2),
                    "last_updated": agent_stats["last_updated"],
                    "performance_grade": self._calculate_agent_performance_grade(agent_stats)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {"error": str(e)}
    
    def _calculate_agent_performance_grade(self, agent_stats: Dict[str, Any]) -> str:
        """Calculate performance grade for an agent"""
        avg_latency = agent_stats["average_latency"]
        
        if avg_latency <= self.TARGET_QUERY_LATENCY * 0.5:
            return "A"
        elif avg_latency <= self.TARGET_QUERY_LATENCY:
            return "B"
        elif avg_latency <= self.TARGET_QUERY_LATENCY * 1.5:
            return "C"
        else:
            return "D"
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            "timestamp": time.time(),
            "status": "healthy",
            "issues": [],
            "performance": {},
            "recommendations": []
        }
        
        try:
            # Test basic connectivity
            collections = await self.client.get_collections()
            
            # Check query performance
            if self.query_stats["average_latency_ms"] > self.TARGET_QUERY_LATENCY:
                health["issues"].append({
                    "type": "performance",
                    "severity": "medium",
                    "message": f"Average query latency ({self.query_stats['average_latency_ms']:.1f}ms) exceeds target ({self.TARGET_QUERY_LATENCY:.1f}ms)"
                })
                health["recommendations"].append("Consider optimizing collection configurations or upgrading hardware")
            
            # Check error rate
            if self.query_stats["total_queries"] > 0:
                error_rate = (self.query_stats["error_count"] / self.query_stats["total_queries"]) * 100
                if error_rate > 5:
                    health["issues"].append({
                        "type": "reliability",
                        "severity": "high",
                        "message": f"High error rate: {error_rate:.1f}%"
                    })
                    health["recommendations"].append("Investigate query errors and check system logs")
            
            # Performance summary
            health["performance"] = {
                "query_latency_ms": round(self.query_stats["average_latency_ms"], 2),
                "target_latency_ms": self.TARGET_QUERY_LATENCY,
                "success_rate": ((self.query_stats["successful_queries"] / max(self.query_stats["total_queries"], 1)) * 100),
                "collections_count": len(collections.collections)
            }
            
            # Determine overall status
            if any(issue["severity"] == "high" for issue in health["issues"]):
                health["status"] = "unhealthy"
            elif health["issues"]:
                health["status"] = "degraded"
            
            return health
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "timestamp": time.time(),
                "status": "unhealthy",
                "error": str(e),
                "issues": [{
                    "type": "connectivity",
                    "severity": "critical",
                    "message": f"Cannot connect to Qdrant: {str(e)}"
                }]
            }
    
    async def cleanup(self):
        """Cleanup Qdrant manager resources"""
        logger.info("🧹 Cleaning up Qdrant Manager...")
        
        try:
            if self.client:
                await self.client.close()
            
            if self.sync_client:
                self.sync_client.close()
            
            logger.info("✅ Qdrant Manager cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    # Utility methods for migration and maintenance
    
    async def export_collection_data(self, agent_id: str) -> List[Dict[str, Any]]:
        """Export all data from a collection for migration"""
        collection_name = f"agent-{agent_id}"
        
        try:
            # Scroll through all points in the collection
            points = []
            next_page_offset = None
            
            while True:
                result = await self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                for point in result[0]:
                    points.append({
                        "id": str(point.id),
                        "vector": point.vector,
                        "payload": point.payload
                    })
                
                next_page_offset = result[1]
                if next_page_offset is None:
                    break
            
            logger.info(f"📤 Exported {len(points)} points from {collection_name}")
            return points
            
        except Exception as e:
            logger.error(f"❌ Export error for {collection_name}: {e}")
            return []
    
    async def import_collection_data(
        self, 
        agent_id: str, 
        data: List[Dict[str, Any]]
    ) -> bool:
        """Import data into a collection"""
        collection_name = f"agent-{agent_id}"
        
        try:
            points = []
            for item in data:
                point = PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload=item["payload"]
                )
                points.append(point)
            
            # Batch insert
            for i in range(0, len(points), self.BATCH_SIZE):
                batch = points[i:i + self.BATCH_SIZE]
                await self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
            
            logger.info(f"📥 Imported {len(points)} points to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Import error for {collection_name}: {e}")
            return False