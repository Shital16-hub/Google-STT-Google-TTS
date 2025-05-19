#!/usr/bin/env python3
"""
Updated troubleshooting script to verify OpenAI and Pinecone connections.
Uses the latest Pinecone client API.
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

# Load environment variables
load_dotenv()

def check_environment_variables():
    """Check required environment variables."""
    print("\n== Checking Environment Variables ==")
    
    # Required variables
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'PINECONE_API_KEY': 'Pinecone API Key',
        'GOOGLE_APPLICATION_CREDENTIALS': 'Google Cloud Credentials',
        'GOOGLE_CLOUD_PROJECT': 'Google Cloud Project ID'
    }
    
    all_good = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"❌ {description} ({var}) is missing")
            all_good = False
        else:
            # Only show masked value for keys
            if 'KEY' in var or 'TOKEN' in var:
                masked = value[:4] + '...' + value[-4:]
                print(f"✅ {description} ({var}) is set: {masked}")
            else:
                print(f"✅ {description} ({var}) is set: {value}")
    
    return all_good

def check_openai_connection():
    """Check OpenAI API connection."""
    print("\n== Checking OpenAI Connection ==")
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OpenAI API key is not set")
            return False
        
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        
        if models and len(models.data) > 0:
            print(f"✅ OpenAI connection successful. Found {len(models.data)} models.")
            print(f"   Available models include: {', '.join([model.id for model in models.data[:3]])}...")
            return True
        else:
            print("❌ OpenAI connection successful but no models found")
            return False
    except ImportError:
        print("❌ OpenAI package not installed. Run: pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def check_pinecone_connection():
    """Check Pinecone connection using the latest API."""
    print("\n== Checking Pinecone Connection ==")
    try:
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not api_key:
            print("❌ Pinecone API key is not set")
            return False
        if not environment:
            print("❌ Pinecone environment is not set")
            return False
        if not index_name:
            print("❌ Pinecone index name is not set")
            return False
        
        # Initialize Pinecone with new API
        pc = Pinecone(api_key=api_key)
        
        # List indexes
        indexes = pc.list_indexes().names()
        print(f"✅ Pinecone connection successful. Available indexes: {indexes}")
        
        # Check for specific index
        if index_name in indexes:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            vector_count = stats.get("total_vector_count", 0)
            print(f"✅ Index '{index_name}' exists with {vector_count} vectors")
            
            # List namespaces
            namespaces = stats.get("namespaces", {})
            if namespaces:
                namespace_list = list(namespaces.keys())
                print(f"   Namespaces: {namespace_list}")
            else:
                print("   No namespaces found in the index")
            
            return True
        else:
            print(f"❌ Index '{index_name}' not found. Available indexes: {indexes}")
            print(f"   You need to create an index named '{index_name}' with dimension 1536")
            return False
    except ImportError:
        print("❌ Pinecone package not installed. Run: pip install pinecone-client")
        return False
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False

def check_google_credentials():
    """Check Google Cloud credentials."""
    print("\n== Checking Google Cloud Credentials ==")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    if not creds_path:
        print("❌ Google Cloud credentials path is not set")
        return False
    
    if not os.path.exists(creds_path):
        print(f"❌ Google Cloud credentials file not found at: {creds_path}")
        return False
    
    print(f"✅ Google Cloud credentials file exists at: {creds_path}")
    
    try:
        from google.oauth2 import service_account
        from google.cloud import texttospeech
        
        # Try to create a simple client
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        
        # Get voices to test connection
        request = texttospeech.ListVoicesRequest()
        response = client.list_voices(request=request)
        
        print(f"✅ Google Cloud authentication successful. Found {len(response.voices)} TTS voices.")
        return True
    except ImportError:
        print("❌ Google Cloud packages not installed. Run: pip install google-cloud-texttospeech")
        return False
    except Exception as e:
        print(f"❌ Google Cloud authentication failed: {e}")
        return False

def check_directories():
    """Check required directories."""
    print("\n== Checking Required Directories ==")
    
    # Directories to check
    directories = [
        './storage',
        './knowledge_base/knowledge_docs'
    ]
    
    all_good = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Directory missing: {directory}")
            all_good = False
            try:
                os.makedirs(directory)
                print(f"  Created directory: {directory}")
            except Exception as e:
                print(f"  Failed to create directory: {e}")
    
    return all_good

def check_llama_index():
    """Check LlamaIndex installation."""
    print("\n== Checking LlamaIndex Installation ==")
    try:
        import llama_index
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        
        # Use a safer version check
        print(f"✅ LlamaIndex is installed")
        print("✅ OpenAI embeddings module is installed")
        print("✅ Pinecone vector store module is installed")
        
        # Try to import core modules
        import llama_index.core
        print("✅ LlamaIndex core module is installed")
        
        return True
    except ImportError as e:
        print(f"❌ LlamaIndex import error: {e}")
        print("   Make sure to install llama-index, llama-index-embeddings-openai, and llama-index-vector-stores-pinecone")
        return False

def main():
    """Main function."""
    print("\n" + "="*60)
    print("Voice AI Agent Troubleshooter (Updated for latest Pinecone API)")
    print("="*60)
    
    # Check environment variables
    env_vars_ok = check_environment_variables()
    
    # Check directories
    dirs_ok = check_directories()
    
    # Check OpenAI
    openai_ok = check_openai_connection()
    
    # Check Pinecone
    pinecone_ok = check_pinecone_connection()
    
    # Check Google credentials
    google_ok = check_google_credentials()
    
    # Check LlamaIndex
    llama_index_ok = check_llama_index()
    
    # Summary
    print("\n" + "="*60)
    print("Troubleshooting Summary")
    print("="*60)
    
    print(f"Environment Variables: {'✅ OK' if env_vars_ok else '❌ Issues Found'}")
    print(f"Required Directories: {'✅ OK' if dirs_ok else '❌ Issues Found'}")
    print(f"OpenAI Connection: {'✅ OK' if openai_ok else '❌ Issues Found'}")
    print(f"Pinecone Connection: {'✅ OK' if pinecone_ok else '❌ Issues Found'}")
    print(f"Google Cloud Authentication: {'✅ OK' if google_ok else '❌ Issues Found'}")
    print(f"LlamaIndex Installation: {'✅ OK' if llama_index_ok else '❌ Issues Found'}")
    
    # Overall status
    all_ok = env_vars_ok and dirs_ok and openai_ok and pinecone_ok and google_ok and llama_index_ok
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ All checks passed! Your setup appears to be working correctly.")
        print("You can now try using the Voice AI Agent.")
    else:
        print("❌ Some checks failed. Please fix the issues above and try again.")
    print("="*60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())