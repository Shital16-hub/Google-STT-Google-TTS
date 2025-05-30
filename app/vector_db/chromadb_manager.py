"""
ChromaDB Manager - RunPod Optimized Replacement for Qdrant
===========================================================

Complete replacement for Qdrant with ChromaDB for better RunPod compatibility.
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
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result from ChromaDB."""
    vectors: List[Dict[str, Any]]
    scores: List[float]
    search_time_ms: float
    collection_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollectionInfo:
    """Information about a ChromaDB collection."""
    name: str
    vector_count: int
    config: Dict[str, Any]
    created_at: float
    last_updated: float = field(default_factory=time.time)
    search_count: int = 0
    average_search_time_ms: float = 0.0

class ChromaDBManager:
    """
    ChromaDB manager optimized for RunPod deployment.
    Provides persistent vector storage with better cloud compatibility.
    """
    
    def __init__(self,
                 persist_directory: str = "./chromadb_storage",
                 vector_dimension: int = 1536,
                 distance_metric: str = "cosine",
                 max_collections: int = 100):
        """Initialize ChromaDB manager."""
        
        self.persist_directory = Path(persist_directory)
        self.vector_dimension = vector_dimension
        self.distance_metric = distance_metric
        self.max_collections = max_collections
        
        # ChromaDB client
        self.client = None
        
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
            "collection_count": 0,
            "total_vectors": 0,
            "connection_failures": 0
        }
        
        self.initialized = False
        logger.info(f"ChromaDB Manager initialized for RunPod: {persist_directory}")
    
    async def initialize(self):
        """Initialize ChromaDB with persistent storage."""
        logger.info("🚀 Initializing ChromaDB Manager for RunPod...")
        
        try:
            # Import ChromaDB
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as e:
                logger.error(f"❌ ChromaDB not available: {e}")
                logger.info("Install with: pip install chromadb")
                raise
            
            # Create persist directory
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Load existing collections
            await self._load_existing_collections()
            
            self.initialized = True
            logger.info("✅ ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {e}")
            raise
    
    async def _load_existing_collections(self):
        """Load information about existing collections."""
        try:
            collections = self.client.list_collections()
            
            for collection in collections:
                collection_name = collection.name
                count = collection.count()
                
                self.collections[collection_name] = CollectionInfo(
                    name=collection_name,
                    vector_count=count,
                    config={"type": "chromadb"},
                    created_at=time.time()
                )
                
                self.stats["total_vectors"] += count
            
            self.stats["collection_count"] = len(self.collections)
            logger.info(f"Loaded {len(self.collections)} existing ChromaDB collections")
            
        except Exception as e:
            logger.error(f"Error loading existing collections: {e}")
    
    async def create_collection(self, collection_name: str, metadata: Dict[str, Any] = None) -> bool:
        """Create a ChromaDB collection."""
        
        try:
            # Check if collection already exists
            if collection_name in self.collections:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Import required functions
            import chromadb.utils.embedding_functions as embedding_functions
            
            # Create embedding function (using OpenAI compatible)
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            # Create collection
            create_start = time.time()
            
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata=metadata or {}
            )
            
            create_time = (time.time() - create_start) * 1000
            
            # Track collection
            self.collections[collection_name] = CollectionInfo(
                name=collection_name,
                vector_count=0,
                config={"type": "chromadb", "metadata": metadata},
                created_at=time.time()
            )
            
            self.stats["collection_count"] += 1
            
            logger.info(f"✅ Created ChromaDB collection '{collection_name}' in {create_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create ChromaDB collection {collection_name}: {e}")
            return False
    
    async def add_vectors(self,
                         vectors: List[Dict[str, Any]],
                         collection_name: str) -> Dict[str, Any]:
        """Add vectors to ChromaDB collection."""
        
        if not self.initialized:
            await self.initialize()
        
        insert_start = time.time()
        
        try:
            # Ensure collection exists
            if collection_name not in self.collections:
                await self.create_collection(collection_name)
            
            # Get collection
            collection = self.client.get_collection(collection_name)
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for vector_data in vectors:
                # Extract ID
                vector_id = vector_data.get("id", str(uuid.uuid4()))
                ids.append(vector_id)
                
                # Extract vector/embedding
                if "vector" in vector_data:
                    embedding = vector_data["vector"]
                elif "embedding" in vector_data:
                    embedding = vector_data["embedding"]
                else:
                    logger.warning(f"No vector found in data: {vector_data}")
                    continue
                
                # Ensure vector is the right type
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                embeddings.append(embedding)
                
                # Extract metadata and document
                metadata = vector_data.get("metadata", {})
                document = metadata.get("text", f"Document {vector_id}")
                
                metadatas.append(metadata)
                documents.append(document)
            
            if not ids:
                return {"success": False, "error": "No valid vectors found"}
            
            # Insert vectors
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            insert_time = (time.time() - insert_start) * 1000
            inserted_count = len(ids)
            
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
            
            logger.info(f"✅ Inserted {inserted_count} vectors into ChromaDB '{collection_name}' in {insert_time:.2f}ms")
            
            return {
                "success": True,
                "vectors_inserted": inserted_count,
                "insert_time_ms": insert_time,
                "collection": collection_name,
                "method": "chromadb"
            }
            
        except Exception as e:
            logger.error(f"❌ Error inserting vectors into ChromaDB {collection_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def search(self,
                    query_vector: np.ndarray,
                    collection_name: str,
                    top_k: int = 5,
                    score_threshold: float = 0.7,
                    filters: Optional[Dict[str, Any]] = None) -> Optional[SearchResult]:
        """Search ChromaDB collection."""
        
        if not self.initialized:
            await self.initialize()
        
        search_start = time.time()
        
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                logger.debug(f"Collection {collection_name} not found")
                return None
            
            # Get collection
            collection = self.client.get_collection(collection_name)
            
            # Prepare query vector
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            # Build where clause for filtering
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Execute search
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause
            )
            
            search_time = (time.time() - search_start) * 1000
            
            # Process results
            vectors = []
            scores = []
            
            if results and results["ids"] and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    # Get metadata and document
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    document = results["documents"][0][i] if results["documents"] and results["documents"][0] else ""
                    distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 0.0
                    
                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity = 1.0 - distance if distance < 1.0 else 0.0
                    
                    # Apply score threshold
                    if similarity >= score_threshold:
                        vectors.append({
                            "id": doc_id,
                            "text": document,
                            "metadata": metadata
                        })
                        scores.append(similarity)
            
            # Update statistics
            self._update_search_stats(collection_name, search_time)
            
            logger.debug(f"🎯 ChromaDB search completed: {search_time:.2f}ms for {collection_name}")
            
            return SearchResult(
                vectors=vectors,
                scores=scores,
                search_time_ms=search_time,
                collection_used=collection_name,
                metadata={
                    "total_results": len(vectors),
                    "score_threshold": score_threshold,
                    "filtered": filters is not None,
                    "method": "chromadb"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ ChromaDB search error in {collection_name}: {e}")
            return None
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        try:
            where_clause = {}
            
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Handle range queries
                    if "$gte" in value:
                        where_clause[key] = {"$gte": value["$gte"]}
                    elif "$lte" in value:
                        where_clause[key] = {"$lte": value["$lte"]}
                    elif "$gt" in value:
                        where_clause[key] = {"$gt": value["$gt"]}
                    elif "$lt" in value:
                        where_clause[key] = {"$lt": value["$lt"]}
                elif isinstance(value, list):
                    # Match any value in list
                    where_clause[key] = {"$in": value}
                else:
                    # Exact match
                    where_clause[key] = {"$eq": value}
            
            return where_clause
            
        except Exception as e:
            logger.error(f"Error building where clause: {e}")
            return {}
    
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
    
    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from collection."""
        if not self.initialized:
            return False
        
        try:
            collection = self.client.get_collection(collection_name)
            
            # Delete vectors
            collection.delete(ids=vector_ids)
            
            deleted_count = len(vector_ids)
            self.stats["total_deletes"] += deleted_count
            self.stats["total_vectors"] -= deleted_count
            
            # Update collection info
            if collection_name in self.collections:
                self.collections[collection_name].vector_count -= deleted_count
                self.collections[collection_name].last_updated = time.time()
            
            logger.info(f"Deleted {deleted_count} vectors from ChromaDB collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors from {collection_name}: {e}")
            return False
    
    async def update_vectors(self, collection_name: str, vectors: List[Dict[str, Any]]) -> bool:
        """Update existing vectors in collection."""
        if not self.initialized:
            return False
        
        try:
            # ChromaDB doesn't have explicit update - use upsert behavior
            result = await self.add_vectors(vectors, collection_name)
            
            if result.get("success"):
                self.stats["total_updates"] += result.get("vectors_inserted", 0)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating vectors in {collection_name}: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            if not self.client:
                return list(self.collections.keys())
            
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return list(self.collections.keys())
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        if not self.initialized:
            return False
        
        try:
            # Delete ChromaDB collection
            self.client.delete_collection(collection_name)
            
            # Update tracking
            if collection_name in self.collections:
                self.stats["total_vectors"] -= self.collections[collection_name].vector_count
                del self.collections[collection_name]
                self.stats["collection_count"] -= 1
            
            logger.info(f"Deleted ChromaDB collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a collection."""
        if not self.initialized:
            return None
        
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            return {
                "name": collection_name,
                "vector_count": count,
                "vector_size": self.vector_dimension,
                "distance_metric": self.distance_metric,
                "status": "active",
                "storage_type": "persistent",
                "created_at": self.collections.get(collection_name, {}).get("created_at", time.time())
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "configuration": {
                "persist_directory": str(self.persist_directory),
                "vector_dimension": self.vector_dimension,
                "distance_metric": self.distance_metric,
                "max_collections": self.max_collections
            },
            "performance": {
                **self.stats,
                "target_latency_ms": 50.0,
                "meets_target": self.stats["average_search_time_ms"] <= 50.0
            },
            "collections": {
                name: {
                    "vector_count": info.vector_count,
                    "search_count": info.search_count,
                    "average_search_time_ms": info.average_search_time_ms,
                    "last_updated": info.last_updated
                }
                for name, info in self.collections.items()
            },
            "health": {
                "status": "healthy" if self.initialized else "down",
                "connected": self.client is not None,
                "performance_acceptable": self.stats["average_search_time_ms"] <= 100.0
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if ChromaDB manager is healthy."""
        return (
            self.initialized and
            self.client is not None and
            self.stats["average_search_time_ms"] <= 100.0
        )
    
    async def shutdown(self):
        """Shutdown ChromaDB manager."""
        logger.info("Shutting down ChromaDB Manager...")
        
        try:
            # ChromaDB client doesn't need explicit closing
            self.initialized = False
            logger.info("✅ ChromaDB Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during ChromaDB shutdown: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()