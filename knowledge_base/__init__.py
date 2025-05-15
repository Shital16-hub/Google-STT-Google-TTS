# knowledge_base/__init__.py
"""
Knowledge base component for the Voice AI Agent.
Updated with OpenAI LLM + Pinecone vector store for optimal latency.
"""
from knowledge_base.openai_llm import OpenAILLM
from knowledge_base.pinecone_store import PineconeVectorStore
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.document_store import DocumentStore
from knowledge_base.schema import Document, DocumentMetadata

__version__ = "3.0.0"

__all__ = [
    "OpenAILLM",
    "PineconeVectorStore", 
    "ConversationManager",
    "QueryEngine",
    "DocumentStore",
    "Document",
    "DocumentMetadata",
]