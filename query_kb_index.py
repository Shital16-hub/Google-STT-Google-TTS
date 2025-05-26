# query_kb_index.py
import os
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def query_knowledge_base(query: str, top_k: int = 3):
    """
    Query the Pinecone knowledge base with the given query.
    
    Args:
        query: The query to search for
        top_k: Number of top results to return
    
    Returns:
        List of results
    """
    # Import required components
    from llama_index.core import QueryBundle, Settings
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.core.retrievers import VectorIndexRetriever
    
    from pinecone import Pinecone
    
    # Check API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Connect to Pinecone
    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Index name and namespace
    index_name = "voice-ai-agent"
    namespace = "voice-assistant"
    
    # Ensure index exists
    index_names = [index.name for index in pc.list_indexes()]
    if index_name not in index_names:
        logger.error(f"Index '{index_name}' does not exist in Pinecone")
        return []
    
    # Connect to the index
    logger.info(f"Connecting to index '{index_name}'...")
    pinecone_index = pc.Index(index_name)
    
    # Get index stats
    try:
        stats = pinecone_index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(namespace, {})
        doc_count = namespace_stats.get("vector_count", 0)
        logger.info(f"Index contains {doc_count} vectors in namespace '{namespace}'")
        
        if doc_count == 0:
            logger.warning(f"No documents found in index namespace '{namespace}'")
            return []
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
    
    # Create vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace
    )
    
    # Create embedding model
    logger.info("Creating embedding model...")
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key=openai_api_key
    )
    
    # Set the embedding model
    Settings.embed_model = embed_model
    
    # Create vector store index
    logger.info("Creating vector index from existing Pinecone store...")
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Create retriever with top_k
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )
    
    # Create query bundle
    logger.info(f"Querying with: '{query}'")
    query_bundle = QueryBundle(query_str=query)
    
    # Retrieve nodes
    nodes = retriever.retrieve(query_bundle)
    
    # Process results
    results = []
    for i, node in enumerate(nodes):
        source = "Unknown"
        if hasattr(node, 'metadata') and node.metadata:
            source = node.metadata.get('source', 'Unknown')
        
        result = {
            "position": i + 1,
            "text": node.text,
            "source": source,
            "score": node.score if hasattr(node, 'score') else 0.0,
            "id": node.node_id
        }
        results.append(result)
    
    return results

async def interactive_query():
    """Run an interactive query session."""
    print("\n===== Knowledge Base Query Tool =====")
    print("Enter your query to search the knowledge base, or type 'exit' to quit.")
    print("========================================\n")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
        
        try:
            results = await query_knowledge_base(query)
            
            if not results:
                print("\nNo relevant documents found for your query.")
                continue
            
            print(f"\nFound {len(results)} relevant documents:")
            print("----------------------------------------")
            
            for i, result in enumerate(results):
                print(f"\n[Result {i+1}] - Relevance: {result['score']:.4f}")
                print(f"Source: {result['source']}")
                print(f"Text: {result['text'][:500]}...")
                if len(result['text']) > 500:
                    print("(text truncated for display)")
                print("----------------------------------------")
            
        except Exception as e:
            print(f"Error querying knowledge base: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_query())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")