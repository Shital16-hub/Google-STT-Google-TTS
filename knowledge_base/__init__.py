"""
Knowledge base package for Voice AI Agent.

This package provides RAG (Retrieval-Augmented Generation) functionality
using the latest LlamaIndex with OpenAI and Pinecone.
"""

from knowledge_base.rag_config import rag_config
from knowledge_base.conversation_manager import ConversationManager, ConversationState
from knowledge_base.query_engine import QueryEngine
from knowledge_base.index_manager import IndexManager
from knowledge_base.document_processor import DocumentProcessor

__version__ = "0.3.0"

__all__ = [
    "rag_config",
    "ConversationManager",
    "ConversationState",
    "QueryEngine",
    "IndexManager",
    "DocumentProcessor"
]