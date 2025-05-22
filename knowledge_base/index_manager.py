"""
Index manager for RAG with latest LlamaIndex and Pinecone v3.
"""
import os
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional

from llama_index.core import Document, Settings
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from pinecone import Pinecone, ServerlessSpec  # Use ServerlessSpec for AWS free tier

from knowledge_base.rag_config import rag_config
from knowledge_base.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class IndexManager:
    """Manage vector indices with Pinecone v3 and latest LlamaIndex."""
    
    def __init__(self, config=None):
        """
        Initialize the index manager.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or rag_config
        self.storage_dir = self.config.storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Component placeholders
        self.pinecone_client = None
        self.pinecone_index = None
        self.vector_store = None
        self.embed_model = None
        self.llm = None
        self.storage_context = None
        self.index = None
        self.document_processor = DocumentProcessor(self.config)
        
        self.initialized = False
        
        # Log the index and namespace that will be used
        logger.info(f"IndexManager initialized with:")
        logger.info(f"- storage_dir: {self.storage_dir}")
        logger.info(f"- pinecone_index_name: {self.config.pinecone_index_name}")
        logger.info(f"- pinecone_namespace: {self.config.pinecone_namespace}")
    
    async def init(self):
        """Initialize the index manager asynchronously."""
        if self.initialized:
            return
            
        start_time = time.time()
        
        try:
            # Initialize embedding model
            self.embed_model = OpenAIEmbedding(
                model=self.config.openai_embedding_model,
                api_key=self.config.openai_api_key,
                dimensions=1536  # Default for text-embedding-ada-002
            )
            
            # Initialize OpenAI LLM for completions
            self.llm = OpenAI(
                model=self.config.openai_model,
                temperature=self.config.llm_temperature,
                api_key=self.config.openai_api_key
            )
            
            # Set up global Settings instead of ServiceContext (which is deprecated)
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            # Initialize Pinecone with v3 API
            self.pinecone_client = Pinecone(
                api_key=self.config.pinecone_api_key
            )
            
            # Create Pinecone index if it doesn't exist
            index_name = self.config.pinecone_index_name
            
            # List indexes using the new API
            index_names = [index.name for index in self.pinecone_client.list_indexes()]
            
            logger.info(f"Found existing Pinecone indexes: {', '.join(index_names) if index_names else 'None'}")
            
            if index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {index_name}")
                
                # Create index with ServerlessSpec for AWS free tier
                self.pinecone_client.create_index(
                    name=index_name,
                    dimension=1536,  # Default for text-embedding-ada-002
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # AWS free tier region
                    )
                )
                
                # Wait for index to be ready
                while True:
                    try:
                        index_info = self.pinecone_client.describe_index(index_name)
                        if index_info.status.ready:
                            break
                    except:
                        pass
                    logger.info("Waiting for index to be ready...")
                    await asyncio.sleep(10)
            else:
                logger.info(f"Using existing Pinecone index: {index_name}")
            
            # Connect to the Pinecone index with the new API
            self.pinecone_index = self.pinecone_client.Index(index_name)
            
            # Create vector store with explicit namespace
            namespace = self.config.pinecone_namespace
            logger.info(f"Creating vector store with namespace: {namespace}")
            
            self.vector_store = PineconeVectorStore(
                pinecone_index=self.pinecone_index,
                namespace=namespace
            )
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create vector store index - no need to pass ServiceContext anymore
            # as we've set the global Settings
            self.index = VectorStoreIndex.from_documents(
                documents=[],  # Empty initially
                storage_context=self.storage_context
            )
            
            # Get index stats
            try:
                stats = self.pinecone_index.describe_index_stats()
                namespaces = stats.get("namespaces", {})
                
                # Log all namespaces in the index
                logger.info(f"Available namespaces in Pinecone index '{index_name}':")
                if namespaces:
                    for ns_name, ns_stats in namespaces.items():
                        vector_count = ns_stats.get("vector_count", 0)
                        logger.info(f"  - '{ns_name}': {vector_count} vectors")
                else:
                    logger.info("  No namespaces found (empty index)")
                
                # Log stats for our specific namespace
                namespace_stats = namespaces.get(namespace, {})
                doc_count = namespace_stats.get("vector_count", 0)
                logger.info(f"Connected to Pinecone index '{index_name}' with {doc_count} vectors in namespace '{namespace}'")
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")
            
            self.initialized = True
            logger.info(f"Index manager initialized in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing index manager: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the index.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        if not self.initialized:
            await self.init()
            
        if not documents:
            logger.warning("No documents provided")
            return []
            
        try:
            # Add documents in batches
            doc_ids = []
            batch_size = 50  # Process 50 documents at a time
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Add batch to index
                batch_start = time.time()
                for doc in batch:
                    self.index.insert(doc)
                    doc_ids.append(doc.doc_id if hasattr(doc, 'doc_id') else doc.id_)
                    
                batch_time = time.time() - batch_start
                logger.info(f"Indexed batch of {len(batch)} documents in {batch_time:.2f}s")
            
            logger.info(f"Successfully indexed {len(doc_ids)} documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            raise
    
    async def add_files(self, file_paths: List[str]) -> List[str]:
        """
        Add files to the index.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of document IDs
        """
        if not self.initialized:
            await self.init()
            
        try:
            documents = []
            
            # Process each file
            for file_path in file_paths:
                try:
                    docs = self.document_processor.process_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            # Add documents to index
            return await self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error adding files to index: {e}")
            raise
    
    async def add_directory(self, directory_path: str) -> List[str]:
        """
        Add all files in a directory to the index.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of document IDs
        """
        if not self.initialized:
            await self.init()
            
        try:
            # Process documents from directory
            documents = self.document_processor.process_directory(directory_path)
            
            # Add documents to index
            return await self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error adding directory to index: {e}")
            raise
    
    async def count_documents(self) -> int:
        """
        Count the number of documents in the index.
        
        Returns:
            Number of documents
        """
        if not self.initialized:
            await self.init()
            
        try:
            # Get index stats from Pinecone
            stats = self.pinecone_index.describe_index_stats()
            
            # Get document count for the namespace
            namespace_stats = stats.get("namespaces", {}).get(self.config.pinecone_namespace, {})
            doc_count = namespace_stats.get("vector_count", 0)
            
            return doc_count
            
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    async def clear_index(self) -> bool:
        """
        Clear the index.
        
        Returns:
            True if successful
        """
        if not self.initialized:
            await self.init()
            
        try:
            # Delete all vectors in the namespace using the v3 API
            self.pinecone_index.delete(
                delete_all=True,
                namespace=self.config.pinecone_namespace
            )
            
            logger.info(f"Cleared index namespace: {self.config.pinecone_namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False
            
    async def get_index_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Pinecone index and namespace.
        
        Returns:
            Dictionary with index information
        """
        if not self.initialized:
            await self.init()
            
        index_name = self.config.pinecone_index_name
        namespace = self.config.pinecone_namespace
        
        info = {
            "index_name": index_name,
            "namespace": namespace,
            "vector_count": 0,
            "namespaces": []
        }
        
        try:
            stats = self.pinecone_index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            
            # Get all namespaces
            for ns_name, ns_stats in namespaces.items():
                info["namespaces"].append({
                    "name": ns_name,
                    "vector_count": ns_stats.get("vector_count", 0)
                })
            
            # Get vector count for our namespace
            namespace_stats = namespaces.get(namespace, {})
            info["vector_count"] = namespace_stats.get("vector_count", 0)
            
            # Get index metadata
            index_info = self.pinecone_client.describe_index(index_name)
            if hasattr(index_info, "dimension"):
                info["dimension"] = index_info.dimension
            if hasattr(index_info, "metric"):
                info["metric"] = index_info.metric
            if hasattr(index_info, "status"):
                info["status"] = index_info.status.ready
                
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            info["error"] = str(e)
            
        return info