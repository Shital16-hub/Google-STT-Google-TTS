"""
Knowledge base component for the Voice AI Agent.
"""
# Reorganized to prevent circular imports

# Define exports but don't import them here
__version__ = "0.2.0"

__all__ = [
    "Document",
    "DocumentMetadata",
    "DocumentStore",
    "get_embedding_model",
    "IndexManager",
    "QueryEngine",
    "ConversationManager",
    "ConversationState",
    "ConversationTurn",
]

# Import conversation components directly since they don't cause circular imports
from knowledge_base.conversation_manager import ConversationManager, ConversationState, ConversationTurn