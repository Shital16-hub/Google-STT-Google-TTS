#!/usr/bin/env python3
"""
Updated simplified document indexing script for Voice AI Agent.
This version uses the latest Pinecone client API and avoids circular imports.
"""
import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Load environment variables
load_dotenv()

async def index_documents(directory: str, storage_dir: str = './storage', reset_index: bool = False) -> int:
    """
    Index documents from a directory using the latest Pinecone API.
    
    Args:
        directory: Directory containing documents
        storage_dir: Directory for storage
        reset_index: Whether to reset the index before indexing
        
    Returns:
        Number of documents indexed
    """
    # Import directly to avoid circular imports
    from knowledge_base.openai_pinecone_config import get_openai_config, get_pinecone_config
    from pinecone import Pinecone, ServerlessSpec
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.core import StorageContext, Settings
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.readers.file import PDFReader, DocxReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    
    # Get configurations
    pinecone_config = get_pinecone_config()
    openai_config = get_openai_config()
    
    # Get API keys
    pinecone_api_key = pinecone_config.get("api_key")
    pinecone_env = pinecone_config.get("environment")
    pinecone_index_name = pinecone_config.get("index_name")
    pinecone_namespace = pinecone_config.get("namespace")
    openai_api_key = openai_config.get("api_key")
    
    if not pinecone_api_key:
        raise ValueError("Pinecone API key is required")
    
    if not openai_api_key:
        raise ValueError("OpenAI API key is required")
    
    try:
        logger.info(f"Indexing documents from {directory}")
        
        # Initialize embedding model
        embed_model = OpenAIEmbedding(
            model=openai_config.get("embedding_model", "text-embedding-ada-002"),
            api_key=openai_api_key
        )
        Settings.embed_model = embed_model
        
        # Initialize Pinecone with new API
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists, create if not
        existing_indexes = pc.list_indexes().names()
        
        if pinecone_index_name not in existing_indexes:
            logger.info(f"Creating index: {pinecone_index_name}")
            
            # Check if environment is a cloud region
            if pinecone_env and pinecone_env in ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]:
                # Create serverless index
                pc.create_index(
                    name=pinecone_index_name,
                    dimension=1536,  # For OpenAI embeddings
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=pinecone_env
                    )
                )
            else:
                # Create standard index
                pc.create_index(
                    name=pinecone_index_name,
                    dimension=1536,  # For OpenAI embeddings
                    metric="cosine"
                )
        
        # Connect to the index
        pinecone_index = pc.Index(pinecone_index_name)
        
        # Reset index if requested
        if reset_index:
            logger.info(f"Resetting index: {pinecone_index_name}")
            pinecone_index.delete(delete_all=True, namespace=pinecone_namespace)
        
        # Create vector store
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace=pinecone_namespace
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        index = VectorStoreIndex.from_documents(
            [],  # Empty to start
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Process documents
        document_count = 0
        
        # Get list of files
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                _, ext = os.path.splitext(file_path)
                if ext.lower() in ['.pdf', '.txt', '.md', '.docx', '.doc']:
                    files.append(file_path)
        
        if not files:
            logger.warning(f"No supported document files found in {directory}")
            return 0
            
        logger.info(f"Found {len(files)} documents to process")
        
        # Process each file
        for file_path in files:
            try:
                logger.info(f"Processing file: {file_path}")
                
                # Load document based on type
                _, ext = os.path.splitext(file_path)
                
                if ext.lower() == '.pdf':
                    reader = PDFReader()
                    docs = reader.load_data(file_path)
                elif ext.lower() in ['.docx', '.doc']:
                    reader = DocxReader()
                    docs = reader.load_data(file_path)
                else:
                    # Text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        docs = [LlamaDocument(text=text)]
                
                # Split into chunks
                splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
                nodes = splitter.get_nodes_from_documents(docs)
                
                # Add to index
                for node in nodes:
                    index.insert(node)
                    document_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Get final stats
        stats = pinecone_index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(pinecone_namespace, {})
        final_count = namespace_stats.get("vector_count", 0) if namespace_stats else 0
        
        logger.info(f"Indexed {document_count} chunks. Total in index: {final_count}")
        return document_count
    
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

async def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Index documents for Voice AI Agent")
    parser.add_argument('--directory', '-d', type=str, default='./knowledge_base/knowledge_docs',
                        help='Directory containing documents to index')
    parser.add_argument('--storage', '-s', type=str, default='./storage',
                        help='Storage directory')
    parser.add_argument('--reset', '-r', action='store_true',
                        help='Reset index before indexing')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        logger.error(f"Directory not found: {args.directory}")
        return 1
    
    # Create storage directory if needed
    os.makedirs(args.storage, exist_ok=True)
    
    # Index documents
    count = await index_documents(args.directory, args.storage, args.reset)
    
    if count > 0:
        logger.info(f"Successfully indexed {count} document chunks")
        return 0
    else:
        logger.warning("No documents were indexed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))