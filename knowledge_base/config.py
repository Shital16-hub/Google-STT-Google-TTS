# knowledge_base/config.py
"""
Configuration settings for OpenAI LLM + Pinecone vector store.
Optimized for minimal latency.
"""
import os
from typing import Dict, Any, Optional

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Fastest model
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "256"))  # Shorter for speed
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "5.0"))  # 5 second timeout

# Pinecone Configuration  
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # Fastest OpenAI embedding
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# Retrieval Configuration - Optimized for speed
DEFAULT_RETRIEVE_COUNT = int(os.getenv("DEFAULT_RETRIEVE_COUNT", "2"))  # Reduced for speed
MINIMUM_RELEVANCE_SCORE = float(os.getenv("MINIMUM_RELEVANCE_SCORE", "0.5"))

# Document Processing - Smaller chunks for faster processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))  # Smaller chunks
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "25"))

# Performance Settings
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "True").lower() == "true"
PARALLEL_PROCESSING = os.getenv("PARALLEL_PROCESSING", "True").lower() == "true"

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration optimized for telephony."""
    return {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "timeout": OPENAI_TIMEOUT,
        "stream": True  # Always stream for better perceived latency
    }

def get_pinecone_config() -> Dict[str, Any]:
    """Get Pinecone configuration."""
    return {
        "api_key": PINECONE_API_KEY,
        "environment": PINECONE_ENVIRONMENT,
        "index_name": PINECONE_INDEX_NAME,
        "namespace": PINECONE_NAMESPACE,
        "dimension": EMBEDDING_DIMENSION
    }

def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration."""
    return {
        "model": EMBEDDING_MODEL,
        "api_key": OPENAI_API_KEY,
        "dimension": EMBEDDING_DIMENSION
    }

def get_retrieval_config() -> Dict[str, Any]:
    """Get retrieval configuration optimized for speed."""
    return {
        "top_k": DEFAULT_RETRIEVE_COUNT,
        "min_score": MINIMUM_RELEVANCE_SCORE,
        "enable_caching": ENABLE_CACHING
    }