"""
Configuration settings for RAG using LlamaIndex with OpenAI and Pinecone.
"""
import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGConfig(BaseSettings):
    """Configuration for RAG using LlamaIndex with OpenAI and Pinecone."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        default=os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        description="OpenAI model name"
    )
    openai_embedding_model: str = Field(
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        description="OpenAI embedding model"
    )
    llm_temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        description="Temperature for OpenAI LLM"
    )
    streaming_enabled: bool = Field(
        default=os.getenv("STREAMING_ENABLED", "True").lower() == "true",
        description="Enable streaming responses"
    )
    
    # Pinecone Configuration - Updated for Free Tier
    pinecone_api_key: str = Field(
        default=os.getenv("PINECONE_API_KEY", ""),
        description="Pinecone API key"
    )
    pinecone_cloud: str = Field(
        default=os.getenv("PINECONE_CLOUD", "aws"),
        description="Pinecone cloud provider (aws)"
    )
    pinecone_region: str = Field(
        default=os.getenv("PINECONE_REGION", "us-east-1"),
        description="Pinecone region (us-east-1 for AWS free tier)"
    )
    pinecone_index_name: str = Field(
        default=os.getenv("PINECONE_INDEX_NAME", "voice-ai-agent"),
        description="Pinecone index name"
    )
    pinecone_namespace: str = Field(
        default=os.getenv("PINECONE_NAMESPACE", "voice-assistant"),
        description="Pinecone namespace"
    )
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(
        default=int(os.getenv("RETRIEVAL_TOP_K", "3")),
        description="Number of top documents to retrieve"
    )
    default_retrieve_count: int = Field(
        default=int(os.getenv("DEFAULT_RETRIEVE_COUNT", "3")),
        description="Default number of documents to retrieve"
    )
    similarity_threshold: float = Field(
        default=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
        description="Minimum similarity threshold for retrieval"
    )
    
    # Conversation Configuration
    max_conversation_history: int = Field(
        default=int(os.getenv("MAX_CONVERSATION_HISTORY", "5")),
        description="Maximum conversation turns to keep"
    )
    system_prompt: str = Field(
        default=os.getenv("SYSTEM_PROMPT", 
                          "You are a helpful voice assistant. Provide clear, concise answers optimized for voice conversations. "
                          "Speak naturally and keep your responses short and to the point."),
        description="System prompt for the LLM"
    )
    
    # Performance Configuration
    cache_enabled: bool = Field(
        default=os.getenv("CACHE_ENABLED", "True").lower() == "true",
        description="Enable caching for improved latency"
    )
    storage_dir: str = Field(
        default=os.getenv("STORAGE_DIR", "./storage"),
        description="Storage directory for local data"
    )
    
    # Latency Optimization
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "512")),
        description="Text chunk size for indexing"
    )
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "50")),
        description="Text chunk overlap for indexing"
    )
    
    # Response Generation
    max_tokens: int = Field(
        default=int(os.getenv("MAX_TOKENS", "256")),
        description="Maximum tokens for response generation - smaller for telephony"
    )
    
    class Config:
        env_prefix = "RAG_"
        case_sensitive = False

# Create a global config instance
rag_config = RAGConfig()

# System prompt template for conversational RAG
CONVERSATION_SYSTEM_PROMPT = """You are a helpful voice assistant that provides clear, concise information.
Respond in a natural, conversational way that is easy to follow when spoken aloud.
You should:
- Keep answers brief and to the point (1-3 sentences when possible)
- Use simple, direct language without technical jargon
- Avoid lists or formatting that wouldn't work well in speech
- Maintain a conversational tone with natural transitions
- Only include information that's directly relevant to the question

If you don't know something, say so clearly rather than speculating.
"""

# Retrieve system prompt template for knowledge retrieval
RETRIEVE_SYSTEM_PROMPT = """You are a helpful voice assistant answering questions based on the provided context.
Focus on the following context to answer the user's question:

{context}

Guidelines:
- Answer based ONLY on the context provided
- Keep answers concise and to the point (1-3 sentences)
- Use simple, direct language optimized for speech
- If the context doesn't contain the answer, acknowledge this clearly
- Maintain a conversational, natural tone
"""