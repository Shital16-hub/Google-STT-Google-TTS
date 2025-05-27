#!/usr/bin/env python3
"""
Enhanced Qdrant Server Implementation using Python
Creates a local Qdrant server that better mimics the real Qdrant API
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchRequest
import threading
import time
import logging
import json
from typing import List, Dict, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedQdrantServer:
    def __init__(self):
        self.client = QdrantClient(":memory:")  # In-memory Qdrant
        self.app = FastAPI(title="Enhanced Qdrant Server", version="1.0.0")
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes to mimic Qdrant API"""
        
        @self.app.get("/")
        async def root():
            return {
                "title": "qdrant - vector search engine",
                "version": "1.7.0-dev",
                "commit": "python-mock"
            }
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}
        
        @self.app.get("/collections")
        async def get_collections():
            """Get all collections"""
            try:
                collections = self.client.get_collections()
                return {
                    "result": {
                        "collections": [
                            {
                                "name": collection.name,
                                "status": "green",
                                "vectors_count": collection.vectors_count or 0,
                                "indexed_vectors_count": collection.indexed_vectors_count or 0,
                                "points_count": collection.points_count or 0,
                            }  
                            for collection in collections.collections
                        ]
                    },
                    "status": "ok",
                    "time": 0.001
                }
            except Exception as e:
                logger.error(f"Error getting collections: {e}")
                return {
                    "result": {"collections": []},
                    "status": "ok", 
                    "time": 0.001
                }
        
        @self.app.put("/collections/{collection_name}")
        async def create_collection(collection_name: str, request: Dict[str, Any]):
            """Create a new collection"""
            try:
                vectors_config = request.get("vectors", {})
                
                if isinstance(vectors_config, dict):
                    # Single vector configuration
                    vector_size = vectors_config.get("size", 1536)
                    distance_str = vectors_config.get("distance", "Cosine")
                else:
                    # Multiple vector configurations (use first one)
                    vector_size = 1536
                    distance_str = "Cosine"
                
                # Map distance string to enum
                distance_map = {
                    "Cosine": Distance.COSINE,
                    "Dot": Distance.DOT,
                    "Euclid": Distance.EUCLID,
                    "Manhattan": Distance.MANHATTAN
                }
                distance = distance_map.get(distance_str, Distance.COSINE)
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance)
                )
                
                logger.info(f"Created collection: {collection_name}")
                return {
                    "result": True,
                    "status": "ok",
                    "time": 0.01
                }
            except Exception as e:
                logger.error(f"Error creating collection {collection_name}: {e}")
                return {"result": False, "status": "error", "time": 0.001}
        
        @self.app.get("/collections/{collection_name}")
        async def get_collection_info(collection_name: str):
            """Get collection information"""
            try:
                info = self.client.get_collection(collection_name)
                return {
                    "result": {
                        "status": "green",
                        "vectors_count": info.vectors_count or 0,
                        "indexed_vectors_count": info.indexed_vectors_count or 0,
                        "points_count": info.points_count or 0,
                        "config": {
                            "params": {
                                "vectors": {
                                    "size": info.config.params.vectors.size if info.config else 1536,
                                    "distance": "Cosine"
                                }
                            }
                        }
                    },
                    "status": "ok",
                    "time": 0.001
                }
            except Exception as e:
                logger.warning(f"Collection {collection_name} not found: {e}")
                raise HTTPException(status_code=404, detail="Collection not found")
        
        @self.app.delete("/collections/{collection_name}")
        async def delete_collection(collection_name: str):
            """Delete a collection"""
            try:
                self.client.delete_collection(collection_name)
                return {
                    "result": True,
                    "status": "ok", 
                    "time": 0.001
                }
            except Exception as e:
                logger.error(f"Error deleting collection {collection_name}: {e}")
                return {"result": False, "status": "error", "time": 0.001}
        
        @self.app.put("/collections/{collection_name}/points")
        async def upsert_points(collection_name: str, request: Dict[str, Any]):
            """Upsert points into collection"""
            try:
                points = request.get("points", [])
                
                # Convert to PointStruct objects
                point_structs = []
                for point in points:
                    point_structs.append(PointStruct(
                        id=point.get("id"),
                        vector=point.get("vector"),
                        payload=point.get("payload", {})
                    ))
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=point_structs
                )
                
                return {
                    "result": {
                        "operation_id": 1,
                        "status": "completed"
                    },
                    "status": "ok",
                    "time": 0.01
                }
            except Exception as e:
                logger.error(f"Error upserting points to {collection_name}: {e}")
                return {"result": False, "status": "error", "time": 0.001}
        
        @self.app.post("/collections/{collection_name}/points/search")
        async def search_points(collection_name: str, request: Dict[str, Any]):
            """Search points in collection"""
            try:
                vector = request.get("vector")
                limit = request.get("limit", 10)
                
                if not vector:
                    raise ValueError("Vector is required for search")
                
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=limit
                )
                
                return {
                    "result": [
                        {
                            "id": result.id,
                            "version": 0,
                            "score": float(result.score),
                            "payload": result.payload or {},
                            "vector": result.vector or []
                        }
                        for result in results
                    ],
                    "status": "ok",
                    "time": 0.005
                }
            except Exception as e:
                logger.error(f"Error searching in {collection_name}: {e}")
                return {"result": [], "status": "error", "time": 0.001}
    
    def start_server(self, port: int = 6333):
        """Start the Qdrant server"""
        logger.info(f"🚀 Starting Enhanced Qdrant server on port {port}...")
        
        # Start FastAPI server
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )

def start_qdrant_background():
    """Start Qdrant server in background"""
    server = EnhancedQdrantServer()
    server.start_server(6333)

if __name__ == "__main__":
    logger.info("🗂️ Initializing Enhanced Qdrant Server...")
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_qdrant_background, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(3)
    
    # Keep main thread alive
    logger.info("✅ Enhanced Qdrant server started successfully")
    logger.info("📡 Server running at http://localhost:6333")
    logger.info("🔍 Collections endpoint: http://localhost:6333/collections")
    logger.info("💚 Health check: http://localhost:6333/health")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Qdrant server...")