# setup_openai_pinecone.py
"""
Setup script for migrating to OpenAI + Pinecone architecture.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_knowledge_base():
    """Setup Pinecone knowledge base with sample data."""
    try:
        # Import after ensuring path is set
        from knowledge_base.examples.setup_knowledge_base import setup_knowledge_base
        await setup_knowledge_base()
        logger.info("Knowledge base setup complete")
    except Exception as e:
        logger.error(f"Error setting up knowledge base: {e}")
        raise

def check_environment():
    """Check required environment variables."""
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set these variables in your .env file or environment")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies."""
    import subprocess
    
    logger.info("Installing OpenAI + Pinecone dependencies...")
    
    # Core dependencies for OpenAI + Pinecone
    dependencies = [
        "openai>=1.3.0",
        "pinecone-client>=6.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.0",
        "asyncio-extras>=1.3.2"
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"Installed {dep}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing {dep}: {e}")
            return False
    
    return True

async def test_setup():
    """Test the OpenAI + Pinecone setup."""
    logger.info("Testing OpenAI + Pinecone setup...")
    
    try:
        # Test OpenAI connection
        from knowledge_base.openai_llm import OpenAILLM
        llm = OpenAILLM()
        logger.info("✓ OpenAI connection successful")
        
        # Test Pinecone connection
        from knowledge_base.pinecone_store import PineconeVectorStore
        vector_store = PineconeVectorStore()
        logger.info("✓ Pinecone connection successful")
        
        # Test end-to-end
        from knowledge_base.query_engine import QueryEngine
        query_engine = QueryEngine(vector_store=vector_store, llm=llm)
        
        # Simple test query
        result = await query_engine.query("What is VoiceAI Technologies?")
        if result.get("response"):
            logger.info("✓ End-to-end test successful")
            logger.info(f"  Test response: {result['response'][:100]}...")
        else:
            logger.warning("! No response from test query (knowledge base may be empty)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Setup test failed: {e}")
        return False

async def main():
    """Main setup function."""
    print("="*60)
    print("OpenAI + Pinecone Voice AI Agent Setup")
    print("="*60)
    
    # Step 1: Check environment
    print("\n1. Checking environment variables...")
    if not check_environment():
        return 1
    print("✓ Environment variables configured")
    
    # Step 2: Install dependencies (optional)
    print("\n2. Installing dependencies...")
    if "--install-deps" in sys.argv:
        if not install_dependencies():
            return 1
        print("✓ Dependencies installed")
    else:
        print("Skipping dependency installation (use --install-deps to install)")
    
    # Step 3: Setup knowledge base
    print("\n3. Setting up knowledge base...")
    try:
        await setup_knowledge_base()
        print("✓ Knowledge base initialized")
    except Exception as e:
        print(f"✗ Knowledge base setup failed: {e}")
        return 1
    
    # Step 4: Test setup
    print("\n4. Testing setup...")
    if await test_setup():
        print("✓ Setup test successful")
    else:
        print("✗ Setup test failed")
        return 1
    
    # Step 5: Final instructions
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the server: python twilio_app_openai_pinecone.py")
    print("2. Configure your Twilio webhook to point to your server")
    print("3. Test with a phone call")
    print("\nExpected performance:")
    print("- STT: <0.5 seconds")
    print("- Knowledge Base: <1.0 seconds")
    print("- TTS: <0.5 seconds")
    print("- Total: <2.0 seconds")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup OpenAI + Pinecone Voice AI")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install required dependencies")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run tests, skip setup")
    
    args = parser.parse_args()
    
    if args.test_only:
        # Only run tests
        async def test_only():
            if await test_setup():
                print("✓ All tests passed")
                return 0
            else:
                print("✗ Tests failed")
                return 1
        
        exit_code = asyncio.run(test_only())
    else:
        # Full setup
        exit_code = asyncio.run(main())
    
    sys.exit(exit_code)