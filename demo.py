#!/usr/bin/env python3
"""
Demo script to test the Voice AI Agent with OpenAI and Pinecone.
This script demonstrates the full pipeline with a simple text sample.
"""
import os
import sys
import asyncio
import logging
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
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Load environment variables
load_dotenv()

async def test_openai_embeddings():
    """Test OpenAI embeddings."""
    print("\n== Testing OpenAI Embeddings ==")
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        
        embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=api_key
        )
        
        # Generate embeddings for a test string
        test_text = "This is a test of the OpenAI embedding model."
        embeddings = embed_model.get_text_embedding(test_text)
        
        print(f"✅ Successfully generated embeddings with dimensions: {len(embeddings)}")
        print(f"   First 5 values: {embeddings[:5]}")
        return True
    except Exception as e:
        print(f"❌ Error testing OpenAI embeddings: {e}")
        return False

async def test_pinecone_storage():
    """Test Pinecone storage."""
    print("\n== Testing Pinecone Storage ==")
    try:
        from pinecone import Pinecone, ServerlessSpec
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        from llama_index.core import StorageContext, Settings
        from llama_index.core.indices.vector_store import VectorStoreIndex
        from llama_index.core.schema import Document
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        # Get config values
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        index_name = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
        namespace = os.getenv("PINECONE_NAMESPACE", "default")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Set up embedding model
        embed_model = OpenAIEmbedding(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            api_key=openai_api_key
        )
        Settings.embed_model = embed_model
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists, create if not
        indexes = pc.list_indexes().names()
        
        if index_name not in indexes:
            print(f"Creating Pinecone index '{index_name}'...")
            
            # Check if environment is a cloud region
            if environment in ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]:
                # Create serverless index
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # For OpenAI embeddings
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=environment
                    )
                )
            else:
                # Create standard index
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # For OpenAI embeddings
                    metric="cosine"
                )
                
            print(f"Created Pinecone index '{index_name}'")
        else:
            print(f"Using existing Pinecone index '{index_name}'")
        
        # Connect to the index
        pinecone_index = pc.Index(index_name)
        
        # Get stats from the Pinecone index directly (not from VectorStoreIndex)
        stats = pinecone_index.describe_index_stats()
        initial_count = stats.get("total_vector_count", 0)
        print(f"Initial document count in Pinecone: {initial_count}")
        
        # Create vector store
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace=namespace
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create a test document
        test_doc = Document(
            text="VoiceAssist is a telephony AI solution that uses Google Cloud STT and TTS with OpenAI and Pinecone for knowledge retrieval. It offers features like real-time speech recognition, natural language understanding, and multilingual support."
        )
        
        # Create index with the test document
        vector_index = VectorStoreIndex.from_documents(
            [test_doc],
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Get updated stats from Pinecone directly
        stats = pinecone_index.describe_index_stats()
        final_count = stats.get("total_vector_count", 0)
        
        print(f"✅ Successfully stored test document in Pinecone index. Vector count: {final_count}")
        
        # Try a simple query
        query = "What features does VoiceAssist offer?"
        query_engine = vector_index.as_query_engine()
        response = query_engine.query(query)
        
        print(f"✅ Successfully queried index. Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Pinecone storage: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def test_openai_llm():
    """Test OpenAI LLM."""
    print("\n== Testing OpenAI LLM ==")
    try:
        from llama_index.llms.openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Initialize OpenAI LLM
        llm = OpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.7
        )
        
        # Test with a simple query
        test_query = "What is the weather like today?"
        response = llm.complete(test_query)
        
        print(f"✅ Successfully generated response from OpenAI: {response}")
        return True
    except Exception as e:
        print(f"❌ Error testing OpenAI LLM: {e}")
        return False

async def test_google_tts():
    """Test Google Cloud TTS."""
    print("\n== Testing Google Cloud TTS ==")
    try:
        from google.cloud import texttospeech
        
        # Use credentials from environment
        client = texttospeech.TextToSpeechClient()
        
        # Set up voice parameters
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-C"
        )
        
        # Set up audio config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        
        # Create synthesis input
        synthesis_input = texttospeech.SynthesisInput(
            text="This is a test of the Google Cloud Text-to-Speech API."
        )
        
        # Generate speech
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        print(f"✅ Successfully generated {len(response.audio_content)} bytes of audio from Google Cloud TTS")
        return True
    except Exception as e:
        print(f"❌ Error testing Google Cloud TTS: {e}")
        return False

async def main():
    """Main function."""
    print("\n" + "="*60)
    print("Voice AI Agent Demo - Testing Core Components")
    print("="*60)
    
    # Test OpenAI embeddings
    embeddings_ok = await test_openai_embeddings()
    
    # Test Pinecone storage
    pinecone_ok = await test_pinecone_storage()
    
    # Test OpenAI LLM
    openai_ok = await test_openai_llm()
    
    # Test Google Cloud TTS
    tts_ok = await test_google_tts()
    
    # Summary
    print("\n" + "="*60)
    print("Demo Summary")
    print("="*60)
    
    print(f"OpenAI Embeddings: {'✅ OK' if embeddings_ok else '❌ Failed'}")
    print(f"Pinecone Storage: {'✅ OK' if pinecone_ok else '❌ Failed'}")
    print(f"OpenAI LLM: {'✅ OK' if openai_ok else '❌ Failed'}")
    print(f"Google Cloud TTS: {'✅ OK' if tts_ok else '❌ Failed'}")
    
    # Overall status
    all_ok = embeddings_ok and pinecone_ok and openai_ok and tts_ok
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ All components are working correctly!")
        print("You can now use the Voice AI Agent.")
    else:
        print("❌ Some components failed. Please fix the issues above and try again.")
    print("="*60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))