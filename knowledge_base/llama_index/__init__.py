"""
LlamaIndex integration components for knowledge base.
"""
# Reorganized to prevent circular imports

# Define exports but don't import them here
__all__ = [
    "DocumentStore",
    "get_embedding_model",
    "IndexManager",
    "QueryEngine",
    "Document",
    "DocumentMetadata"
]

# Don't import directly in __init__ to avoid circular imports