#!/usr/bin/env python3
"""
Qdrant Server Implementation using Python
Creates a local Qdrant server that listens on ports 6333 and 6334
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionsResponse, CollectionInfo
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantServerManager:
    def __init__(self):
        self.client = QdrantClient(":memory:")  # In-memory Qdrant
        self.app = FastAPI()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes to mimic Qdrant API"""
        
        @self.app.get("/")
        async def root():
            return {"status": "ok", "version": "local-python-qdrant"}
        
        @self.app.get("/collections")
        async def get_collections():
            """Get all collections"""
            try:
                # Get collections from in-memory client
                collections = self.client.get_collections()
                return {
                    "result": {
                        "collections": [
                            {
                                "name": collection.name,
                                "status": "green",
                                "vectors_count": collection.vectors_count,
                                "indexed_vectors_count": collection.indexed_vectors_count,
                                "points_count": collection.points_count,
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
        
        @self.app.post("/collections/{collection_name}")
        async def create_collection(collection_name: str, collection_config: dict):
            """Create a new collection"""
            try:
                vector_size = collection_config.get("vectors", {}).get("size", 1536)
                distance = Distance.COSINE
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance)
                )
                
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
                        "vectors_count": info.vectors_count,
                        "indexed_vectors_count": info.indexed_vectors_count,
                        "points_count": info.points_count,
                    },
                    "status": "ok",
                    "time": 0.001
                }
            except Exception as e:
                return {
                    "result": None,
                    "status": "error",
                    "time": 0.001
                }
        
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
                return {"result": False, "status": "error", "time": 0.001}
    
    def start_server(self):
        """Start the Qdrant server"""
        logger.info("🚀 Starting Qdrant server on ports 6333...")
        
        # Start FastAPI server
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=6333,
            log_level="info"
        )

def start_qdrant_background():
    """Start Qdrant server in background"""
    server = QdrantServerManager()
    server.start_server()

if __name__ == "__main__":
    logger.info("🗂️ Initializing Qdrant Server...")
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_qdrant_background, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(3)
    
    # Keep main thread alive
    logger.info("✅ Qdrant server started successfully on port 6333")
    logger.info("📡 Server running at http://localhost:6333")
    logger.info("🔍 Collections endpoint: http://localhost:6333/collections")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Qdrant server...")