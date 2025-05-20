# index_kb_data.py
import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def index_knowledge_base_data():
    """
    Index documents from the /workspace/Google-STT-Google-TTS/knowledge_base/data folder to Pinecone.
    """
    # Import required components
    from llama_index.core import Document, ServiceContext, StorageContext, Settings
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.readers.file import PyMuPDFReader, DocxReader
    from llama_index.core.node_parser import SentenceSplitter
    
    from pinecone import Pinecone, ServerlessSpec  # Use ServerlessSpec for AWS free tier
    
    # Ensure API keys are set
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Set the specific directory to index
    knowledge_dir = "/workspace/Google-STT-Google-TTS/knowledge_base/data"
    
    # Validate directory exists
    if not os.path.exists(knowledge_dir):
        logger.error(f"Directory not found: {knowledge_dir}")
        raise FileNotFoundError(f"Directory not found: {knowledge_dir}")
    
    # Initialize Pinecone with v3 API
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Create an index if it doesn't exist
    index_name = "voice-ai-agent"
    namespace = "voice-assistant"
    
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
        
        # Wait for index to initialize
        logger.info("Waiting for index to be ready...")
        while True:
            try:
                index_info = pc.describe_index(index_name)
                if hasattr(index_info, 'status') and index_info.status.ready:
                    break
            except Exception as e:
                logger.warning(f"Error checking index status: {e}")
            
            await asyncio.sleep(5)
    
    # Connect to the Pinecone index with the new API
    pinecone_index = pc.Index(index_name)
    
    # Create vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace
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
    
    # Create node parser for document chunking
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Initialize document readers
    pdf_reader = PyMuPDFReader()
    docx_reader = DocxReader()
    
    # Process documents
    all_docs = []
    
    # Get list of files in the directory
    files = list(Path(knowledge_dir).glob("**/*"))
    logger.info(f"Found {len(files)} files/directories in {knowledge_dir}")
    
    # Process files in directory
    for file_path in files:
        if file_path.is_file():
            ext = file_path.suffix.lower()
            try:
                logger.info(f"Processing file: {file_path}")
                
                if ext == ".pdf":
                    docs = pdf_reader.load_data(str(file_path))
                elif ext == ".docx":
                    docs = docx_reader.load_data(str(file_path))
                elif ext in [".txt", ".md"]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        docs = [Document(text=text, metadata={"source": file_path.name})]
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    continue
                
                # Add to all docs
                all_docs.extend(docs)
                logger.info(f"Processed {file_path.name}: Added {len(docs)} document(s)")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    logger.info(f"Total documents collected: {len(all_docs)}")
    
    if not all_docs:
        logger.warning("No documents found to index!")
        return 0
    
    # Split documents into nodes/chunks
    nodes = node_parser.get_nodes_from_documents(all_docs)
    logger.info(f"Created {len(nodes)} chunks from {len(all_docs)} documents")
    
    # Create vector store index
    logger.info("Creating vector index in Pinecone...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # Get index stats
    try:
        stats = pinecone_index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(namespace, {})
        doc_count = namespace_stats.get("vector_count", 0)
        logger.info(f"Successfully indexed {len(nodes)} chunks, total vectors in index: {doc_count}")
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
        logger.info(f"Successfully indexed {len(nodes)} chunks")
    
    return len(nodes)

if __name__ == "__main__":
    print(f"Indexing documents from /workspace/Google-STT-Google-TTS/knowledge_base/data to Pinecone...")
    
    try:
        chunks_indexed = asyncio.run(index_knowledge_base_data())
        if chunks_indexed > 0:
            print(f"\n✅ Successfully indexed {chunks_indexed} chunks to Pinecone!")
        else:
            print("\n⚠️ No documents were indexed. Check the logs for details.")
    except Exception as e:
        print(f"\n❌ Error indexing documents: {e}")