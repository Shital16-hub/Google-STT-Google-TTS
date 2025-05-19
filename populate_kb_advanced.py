import os
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def populate_knowledge_base():
    """Populate the knowledge base with documents."""
    # Import required components
    from llama_index.core import Document, ServiceContext, StorageContext, Settings
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.readers.file import PyMuPDFReader, DocxReader
    from llama_index.core.node_parser import SentenceSplitter
    
    from pinecone import Pinecone, ServerlessSpec  # Use ServerlessSpec for AWS free tier
    
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
    index_name = "voice-ai-agent"
    
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
        while True:
            try:
                index_info = pc.describe_index(index_name)
                if hasattr(index_info, 'status') and index_info.status.ready:
                    break
            except:
                pass
            logger.info("Waiting for index to be ready...")
            await asyncio.sleep(10)
    
    # Connect to the Pinecone index with the new API
    pinecone_index = pc.Index(index_name)
    
    # Create vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="voice-assistant"
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
    
    # Directory with documents to process
    knowledge_dir = "./knowledge_base/data"
    
    # Create directory if it doesn't exist
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # Make sure we have at least one document
    sample_doc_path = f"{knowledge_dir}/sample.txt"
    if not os.path.exists(sample_doc_path) and len(os.listdir(knowledge_dir)) == 0:
        logger.info("Creating a sample document")
        with open(sample_doc_path, "w") as f:
            f.write("This is a sample document for testing the knowledge base. " +
                   "VoiceAssist is a product that offers multiple pricing plans. " +
                   "The Basic Plan costs $499/month and includes up to 1,000 conversations per day. " +
                   "The Professional Plan costs $999/month and includes up to 5,000 conversations per day.")
    
    # Process documents
    all_docs = []
    
    # Process files in directory
    for file_path in Path(knowledge_dir).glob("**/*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            try:
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
                logger.info(f"Processed {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    # Split documents into nodes/chunks
    nodes = node_parser.get_nodes_from_documents(all_docs)
    logger.info(f"Created {len(nodes)} chunks from {len(all_docs)} documents")
    
    # Create vector store index
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # Get index stats
    try:
        stats = pinecone_index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get("voice-assistant", {})
        doc_count = namespace_stats.get("vector_count", 0)
        logger.info(f"Successfully indexed {len(nodes)} chunks, total in index: {doc_count}")
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
        logger.info(f"Successfully indexed {len(nodes)} chunks")
    
    return len(nodes)

if __name__ == "__main__":
    asyncio.run(populate_knowledge_base())