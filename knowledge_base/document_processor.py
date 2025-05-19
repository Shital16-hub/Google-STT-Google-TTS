"""
Document processing for RAG with latest LlamaIndex.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader, DocxReader

from knowledge_base.rag_config import rag_config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents for indexing with the latest LlamaIndex."""
    
    def __init__(self, config=None):
        """
        Initialize the document processor.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or rag_config
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        
        # Initialize node parser for text chunking
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Initialize file readers
        self.readers = {
            ".pdf": PyMuPDFReader(),
            ".docx": DocxReader(),
        }
        
        logger.info(f"Document processor initialized with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a single file into documents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of LlamaIndex Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get file extension
        ext = os.path.splitext(file_path.lower())[1]
        
        try:
            # Use appropriate reader based on file type
            if ext in self.readers:
                reader = self.readers[ext]
                docs = reader.load_data(file_path)
            elif ext == ".txt" or ext == ".md":
                # Handle text files directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                docs = [Document(text=text, metadata={"source": file_path})]
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Split documents into nodes/chunks
            nodes = self.node_parser.get_nodes_from_documents(docs)
            
            # Convert nodes back to Documents
            chunked_docs = []
            for i, node in enumerate(nodes):
                metadata = node.metadata.copy() if hasattr(node, 'metadata') else {}
                metadata.update({
                    "chunk_id": i,
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": ext,
                })
                
                doc = Document(
                    text=node.text,
                    metadata=metadata,
                    id_=f"{os.path.basename(file_path)}_{i}"
                )
                chunked_docs.append(doc)
            
            logger.info(f"Processed {file_path} into {len(chunked_docs)} chunks")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of LlamaIndex Document objects
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        all_docs = []
        
        # Process all files in the directory
        for file_path in Path(directory_path).glob("**/*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                # Check if file type is supported
                if ext in self.readers or ext in [".txt", ".md"]:
                    try:
                        docs = self.process_file(str(file_path))
                        all_docs.extend(docs)
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Processed {len(all_docs)} chunks from directory: {directory_path}")
        return all_docs
    
    def process_text(self, text: str, source_name: str = "direct_input") -> List[Document]:
        """
        Process raw text into documents.
        
        Args:
            text: Text content to process
            source_name: Source identifier for the text
            
        Returns:
            List of LlamaIndex Document objects
        """
        # Create a document from the text
        doc = Document(
            text=text, 
            metadata={
                "source": source_name,
                "source_type": "text",
            }
        )
        
        # Split into chunks
        nodes = self.node_parser.get_nodes_from_documents([doc])
        
        # Convert nodes back to Documents
        chunked_docs = []
        for i, node in enumerate(nodes):
            metadata = node.metadata.copy() if hasattr(node, 'metadata') else {}
            metadata.update({
                "chunk_id": i,
                "source": source_name,
                "source_type": "text",
            })
            
            chunked_doc = Document(
                text=node.text,
                metadata=metadata,
                id_=f"{source_name}_{i}"
            )
            chunked_docs.append(chunked_doc)
        
        logger.info(f"Processed text input '{source_name}' into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return list(self.readers.keys()) + [".txt", ".md"]
    
    def estimate_chunk_count(self, text: str) -> int:
        """
        Estimate the number of chunks for a text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of chunks
        """
        # Simple estimation based on text length and chunk size
        return max(1, len(text) // (self.chunk_size - self.chunk_overlap))
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is supported
        """
        if not os.path.exists(file_path):
            return False
            
        ext = os.path.splitext(file_path.lower())[1]
        return ext in self.get_supported_extensions()