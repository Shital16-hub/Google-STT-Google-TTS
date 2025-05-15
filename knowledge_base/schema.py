# knowledge_base/schema.py
"""
Data schemas for OpenAI + Pinecone knowledge base.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import hashlib

class DocumentMetadata(BaseModel):
    """Metadata for documents stored in Pinecone."""
    source: str = Field(description="Source identifier")
    source_type: str = Field(description="Type of source (file, text, etc.)")
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    created_at: Optional[float] = None
    modified_at: Optional[float] = None
    chunk_index: Optional[int] = None
    chunk_count: Optional[int] = None
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in super().dict(**kwargs).items() if v is not None}

class Document:
    """Document class for knowledge base operations."""
    
    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """Initialize document."""
        self.text = text
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate document ID from content hash."""
        content = self.text + str(self.metadata.get("source", ""))
        return hashlib.md5(content.encode()).hexdigest()
    
    def __str__(self) -> str:
        return f"Document(id={self.doc_id}, text={self.text[:50]}...)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata
        }