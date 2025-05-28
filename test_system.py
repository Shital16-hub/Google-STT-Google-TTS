#!/usr/bin/env python3
"""Quick system test"""
import asyncio
import logging

async def test_imports():
    print("🧪 Testing critical imports...")
    
    # Test typing
    try:
        from typing import Tuple
        print("✅ Tuple import: OK")
    except Exception as e:
        print(f"❌ Tuple import: {e}")
    
    # Test Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(":memory:")  # Test memory mode
        print("✅ Qdrant client: OK")
    except Exception as e:
        print(f"❌ Qdrant client: {e}")
    
    # Test Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_timeout=2)
        r.ping()
        print("✅ Redis connection: OK")
    except Exception as e:
        print(f"❌ Redis connection: {e}")
    
    # Test Google Cloud
    try:
        from google.cloud import speech
        print("✅ Google Cloud Speech: OK")
    except Exception as e:
        print(f"⚠️ Google Cloud Speech: {e} (needs credentials)")
    
    print("\n🎯 System ready for testing!")

if __name__ == "__main__":
    asyncio.run(test_imports())
