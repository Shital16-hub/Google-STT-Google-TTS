# verify_pinecone.py
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

async def verify_pinecone_setup():
    """
    Verify Pinecone setup is working correctly with the free tier.
    """
    from pinecone import Pinecone, ServerlessSpec

    # Check API key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        return False

    try:
        # Initialize Pinecone with v3 API
        pc = Pinecone(api_key=api_key)
        logger.info("Successfully connected to Pinecone")

        # List existing indexes
        indexes = pc.list_indexes()
        index_names = [index.name for index in indexes]
        logger.info(f"Found {len(index_names)} existing indexes: {', '.join(index_names) if index_names else 'none'}")

        # Create a test index using serverless spec (free tier)
        test_index_name = "test-free-tier-index"

        # Delete index if it already exists
        if test_index_name in index_names:
            logger.info(f"Deleting existing index '{test_index_name}'")
            pc.delete_index(test_index_name)
            # Wait a moment for deletion to take effect
            await asyncio.sleep(5)

        # Create the index with serverless spec
        logger.info(f"Creating test index '{test_index_name}' on free tier (AWS serverless)")
        try:
            pc.create_index(
                name=test_index_name,
                dimension=1536,  # For text-embedding-ada-002
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to initialize
            logger.info("Waiting for index to be ready...")
            while True:
                try:
                    index_info = pc.describe_index(test_index_name)
                    if hasattr(index_info, 'status') and index_info.status.ready:
                        logger.info(f"Index {test_index_name} is ready!")
                        break
                except Exception as e:
                    logger.warning(f"Error checking index status: {e}")
                
                logger.info("Still waiting for index to be ready...")
                await asyncio.sleep(10)
        
            # Connect to the index
            index = pc.Index(test_index_name)
            
            # Get stats
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            logger.info("Pinecone free tier setup verified successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(verify_pinecone_setup())
    if result:
        print("\n✅ Pinecone free tier setup is working correctly!")
    else:
        print("\n❌ There was an issue with Pinecone setup. Check the logs above.")