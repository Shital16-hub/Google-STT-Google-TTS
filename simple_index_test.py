# simple_index_test.py
import os
import logging
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_index():
    # Import after configuring logging
    from llama_index.core import Document, ServiceContext, StorageContext, QueryBundle
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    
    from pinecone import Pinecone, ServerlessSpec  # Use ServerlessSpec for free tier in AWS
    
    # Make sure API keys are set
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Initialize Pinecone with v3 API
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Create an index if it doesn't exist
    index_name = "test-index"
    
    # List indexes using the new API
    index_names = [index.name for index in pc.list_indexes()]
    
    if index_name not in index_names:
        logger.info(f"Creating Pinecone index: {index_name}")
        
        # Create index with ServerlessSpec for AWS free tier
        pc.create_index(
            name=index_name,
            dimension=1536,  # Default for text-embedding-ada-002
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # AWS free tier region
            )
        )
    
    # Connect to the Pinecone index with the new API
    pinecone_index = pc.Index(index_name)
    
    # Create vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="test-namespace"
    )
    
    # Create embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key=openai_api_key
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # Create a simple document
    document = Document(
        text="This is a test document for LlamaIndex.",
        metadata={"source": "test"}
    )
    
    # Create vector store index
    index = VectorStoreIndex.from_documents(
        documents=[document],
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    logger.info("Successfully created index!")
    
    # Test simple query
    retriever = index.as_retriever(similarity_top_k=1)
    
    # Create a QueryBundle directly (instead of calling retriever.query_bundle)
    query_text = "What is this document about?"
    query_bundle = QueryBundle(query_str=query_text)
    
    # Use the retriever with the query bundle
    nodes = retriever.retrieve(query_bundle)
    
    logger.info(f"Query: {query_text}")
    logger.info(f"Retrieved {len(nodes)} nodes")
    if nodes:
        logger.info(f"Text: {nodes[0].text}")
        logger.info(f"Score: {nodes[0].score}")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_index())