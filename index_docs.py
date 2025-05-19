#!/usr/bin/env python3
"""
Document indexing script using the new Pinecone API.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def index_documents_new_api(
    directory: str, 
    storage_dir: str = './storage',
    reset_index: bool = False
):
    """
    Index documents with the new Pinecone API.
    """
    logger.info(f"Indexing documents from {directory} using new Pinecone API")
    
    # Import needed components
    from pinecone import Pinecone, ServerlessSpec
    from knowledge_base.llama_index.document_store import DocumentStore
    from knowledge_base.openai_pinecone_config import get_openai_config, get_pinecone_config
    from knowledge_base.llama_index.embedding_setup import get_embedding_model
    from llama_index.core import Settings
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.storage.storage_context import StorageContext
    
    # Get configs
    openai_config = get_openai_config()
    pinecone_config = get_pinecone_config()
    
    # Validate API keys
    if not openai_config["api_key"]:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    if not pinecone_config["api_key"]:
        raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable.")
    
    api_key = pinecone_config["api_key"]
    index_name = pinecone_config["index_name"]
    namespace = pinecone_config["namespace"]
    dimension = pinecone_config.get("dimension", 1536)
    
    # Initialize document store and embedding model
    doc_store = DocumentStore()
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model
    Settings.llm = None  # Explicitly disable LLM usage
    
    # Initialize Pinecone with new API
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    index_list = pc.list_indexes()
    index_exists = index_name in index_list.names() if hasattr(index_list, 'names') else index_name in [idx.name for idx in index_list]
    
    if not index_exists:
        logger.info(f"Creating new Pinecone index: {index_name}")
        try:
            # Try to create with ServerlessSpec if compatible
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        except Exception as e:
            logger.warning(f"ServerlessSpec creation failed, trying standard creation: {e}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
    
    # Reset index if requested
    if reset_index and index_exists:
        logger.info(f"Resetting Pinecone index: {index_name}")
        
        pinecone_index = pc.Index(index_name)
        try:
            # New API uses delete instead of delete_all
            if hasattr(pinecone_index, 'delete'):
                pinecone_index.delete(delete_all=True, namespace=namespace)
            else:
                pinecone_index.delete(filter={}, namespace=namespace)
            logger.info(f"Reset Pinecone index namespace: {namespace}")
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return 0
    
    # Load documents
    try:
        documents = doc_store.load_documents_from_directory(directory)
        
        if not documents:
            logger.warning(f"No documents found in {directory}")
            return 0
            
        logger.info(f"Loaded {len(documents)} document chunks")
        
        # Connect to Pinecone index
        pinecone_index = pc.Index(index_name)
        
        # Create vector store and index
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace=namespace
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Convert to llama_index documents
        llama_docs = [doc.to_llama_index_document() for doc in documents]
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents=llama_docs,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Get count
        stats = pinecone_index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(namespace, {})
        doc_count = namespace_stats.get("vector_count", 0)
        
        logger.info(f"Indexed {len(documents)} documents, total count: {doc_count}")
        return len(documents)
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        import traceback
        traceback.print_exc()
        return 0

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents for Voice AI Agent")
    parser.add_argument('--directory', '-d', type=str, 
                        default='./knowledge_base/knowledge_docs',
                        help='Directory containing documents to index')
    parser.add_argument('--storage', '-s', type=str, 
                        default='./storage',
                        help='Storage directory')
    parser.add_argument('--reset', '-r', action='store_true',
                        help='Reset index before indexing')
    
    args = parser.parse_args()
    
    try:
        # Check if directory exists
        if not os.path.exists(args.directory):
            logger.error(f"Directory not found: {args.directory}")
            return 1
        
        # Index documents
        indexed_count = await index_documents_new_api(args.directory, args.storage, args.reset)
        
        if indexed_count > 0:
            logger.info(f"Successfully indexed {indexed_count} documents")
            return 0
        else:
            logger.warning("No documents were indexed")
            return 1
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))