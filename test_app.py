# test_app.py - Quick test to verify components work

import asyncio
import os
import sys
import logging

# Add current directory to path
sys.path.append('.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_components():
    """Test that all components can be initialized properly."""
    
    print("🧪 Testing Voice AI Agent Components...")
    
    try:
        # Test 1: Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'GOOGLE_CLOUD_PROJECT']
        for var in required_vars:
            if not os.getenv(var):
                print(f"❌ Missing environment variable: {var}")
                return False
            else:
                print(f"✅ Found environment variable: {var}")
        
        # Test 2: Import core modules
        print("\n📦 Testing imports...")
        from core.config import Settings
        from knowledge_base.rag_config import rag_config
        from knowledge_base.index_manager import IndexManager
        from knowledge_base.query_engine import QueryEngine
        from core.conversation_manager import ConversationManager
        print("✅ All imports successful")
        
        # Test 3: Initialize settings
        print("\n⚙️ Testing settings...")
        settings = Settings()
        print(f"✅ Settings loaded - Debug: {settings.debug}")
        
        # Test 4: Initialize knowledge base
        print("\n🧠 Testing knowledge base...")
        try:
            index_manager = IndexManager(config=rag_config)
            await index_manager.init()
            
            doc_count = await index_manager.count_documents()
            print(f"✅ Knowledge base initialized with {doc_count} documents")
            
            # If empty, add sample documents
            if doc_count == 0:
                print("📚 Adding sample documents...")
                if os.path.exists('./knowledge_base/data'):
                    doc_ids = await index_manager.add_directory('./knowledge_base/data')
                    print(f"✅ Added {len(doc_ids)} sample documents")
                else:
                    print("⚠️ No sample data directory found")
            
        except Exception as e:
            print(f"❌ Knowledge base error: {e}")
            return False
        
        # Test 5: Initialize query engine
        print("\n🔍 Testing query engine...")
        try:
            query_engine = QueryEngine(index_manager=index_manager, config=rag_config)
            await query_engine.init()
            
            # Test a simple query
            result = await query_engine.query("What is this system?")
            print(f"✅ Query engine working - Response: {result.get('response', '')[:50]}...")
            
        except Exception as e:
            print(f"❌ Query engine error: {e}")
            return False
        
        # Test 6: Initialize conversation manager
        print("\n💬 Testing conversation manager...")
        try:
            conversation_config = {
                "max_conversation_history": 5,
                "context_window_size": 4096,
                "max_tokens": 256,
                "temperature": 0.7
            }
            
            conversation_manager = ConversationManager(
                query_engine=query_engine,
                config=conversation_config
            )
            await conversation_manager.init()
            print("✅ Conversation manager initialized")
            
        except Exception as e:
            print(f"❌ Conversation manager error: {e}")
            return False
        
        # Test 7: Test Google Cloud TTS
        print("\n🗣️ Testing Google Cloud TTS...")
        try:
            from text_to_speech.google_cloud_tts import GoogleCloudTTS
            
            tts_client = GoogleCloudTTS(
                credentials_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
                voice_name='en-US-Neural2-C',
                language_code='en-US',
                container_format='mulaw',
                sample_rate=8000
            )
            
            # Test synthesis
            audio_data = await tts_client.synthesize("Hello, this is a test.")
            print(f"✅ TTS working - Generated {len(audio_data)} bytes of audio")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False
        
        print("\n🎉 All component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_pipeline():
    """Test the complete voice AI pipeline."""
    
    print("\n🔄 Testing full pipeline...")
    
    try:
        from voice_ai_agent import VoiceAIAgent
        
        # Initialize agent
        agent = VoiceAIAgent(
            storage_dir='./storage',
            credentials_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
        
        await agent.init()
        print("✅ Voice AI Agent initialized")
        
        # Test text processing
        test_text = "Hello, how can I help you today?"
        
        # Create some dummy audio data for testing
        import numpy as np
        sample_audio = np.random.randint(-32768, 32767, 8000, dtype=np.int16)  # 1 second of audio
        audio_bytes = sample_audio.tobytes()
        
        # Test audio processing (this will likely fail without real speech)
        try:
            result = await agent.process_audio(audio_bytes)
            print(f"✅ Audio processing test completed: {result.get('status', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Audio processing test failed (expected): {e}")
        
        print("✅ Full pipeline test completed")
        
        # Cleanup
        await agent.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    print("🚀 Starting Voice AI Agent Tests...")
    
    # Run component tests
    success = asyncio.run(test_components())
    
    if success:
        print("\n" + "="*50)
        print("✅ All tests passed! Your application should work.")
        print("You can now start the server with:")
        print("python main.py")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ Some tests failed. Please fix the issues above.")
        print("="*50)
        sys.exit(1)