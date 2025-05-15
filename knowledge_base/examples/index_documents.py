# knowledge_base/examples/index_documents.py
"""
Script to index documents from knowledge_docs directory into Pinecone.
Updated for optimized OpenAI + Pinecone setup.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_base.pinecone_store import PineconeVectorStore
from knowledge_base.document_store import DocumentStore
from knowledge_base.query_engine import QueryEngine
from knowledge_base.openai_llm import OpenAILLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def index_knowledge_docs():
    """Index all documents from knowledge_docs directory into Pinecone."""
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable not found")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        return False
    
    if not pinecone_key:
        print("❌ PINECONE_API_KEY environment variable not found")
        print("Please set it with: export PINECONE_API_KEY=your_api_key")
        return False
    
    print(f"🔧 Using Pinecone index: {pinecone_index}")
    
    # Get the knowledge docs directory
    knowledge_docs_dir = Path(__file__).parent.parent / "knowledge_docs"
    
    if not knowledge_docs_dir.exists():
        print(f"❌ Knowledge docs directory not found: {knowledge_docs_dir}")
        print("Creating directory for you...")
        knowledge_docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {knowledge_docs_dir}")
        print("Please add your documents to this directory and run the script again.")
        return False
    
    print(f"📁 Looking for documents in: {knowledge_docs_dir}")
    
    # Find all supported files
    supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.html'}
    files_found = []
    
    for file_path in knowledge_docs_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files_found.append(file_path)
            print(f"  📄 Found: {file_path.name} ({file_path.suffix})")
    
    if not files_found:
        print(f"❌ No supported documents found in {knowledge_docs_dir}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        print("\nExample files you can add:")
        print("  - company_info.txt")
        print("  - pricing.md")
        print("  - features.txt")
        return False
    
    print(f"\n📄 Found {len(files_found)} documents to index")
    
    # Initialize components
    print("\n🔧 Initializing Pinecone and OpenAI...")
    
    try:
        # Initialize Pinecone vector store with config
        pinecone_config = {
            "api_key": pinecone_key,
            "index_name": pinecone_index,
            "dimension": 1536,  # OpenAI text-embedding-3-small
            "namespace": "default"
        }
        vector_store = PineconeVectorStore(config=pinecone_config)
        
        # Initialize document store with optimized settings
        doc_store = DocumentStore(
            chunk_size=256,  # Smaller chunks for better retrieval
            chunk_overlap=25  # Minimal overlap for speed
        )
        
        print("✅ Components initialized successfully")
        
        # Verify Pinecone connection
        stats = await vector_store.get_stats()
        print(f"📊 Connected to Pinecone index: {stats.get('index_name')}")
        print(f"   Current vectors: {stats.get('total_vectors', 0)}")
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        print(f"   Please check your API keys and Pinecone index name")
        return False
    
    # Process each file
    print(f"\n📚 Indexing documents into Pinecone...")
    
    total_chunks = 0
    successful_files = 0
    failed_files = []
    processing_times = []
    
    for file_path in files_found:
        try:
            import time
            file_start = time.time()
            
            print(f"\n📄 Processing: {file_path.name}")
            
            # Load and chunk the document
            documents = doc_store.load_file(str(file_path))
            chunk_time = time.time() - file_start
            print(f"  📝 Created {len(documents)} chunks ({chunk_time:.2f}s)")
            
            # Add to Pinecone with timing
            embed_start = time.time()
            doc_ids = await vector_store.add_documents(documents)
            embed_time = time.time() - embed_start
            
            total_time = time.time() - file_start
            processing_times.append(total_time)
            
            print(f"  ✅ Indexed {len(doc_ids)} chunks ({embed_time:.2f}s)")
            print(f"  ⏱️  Total time: {total_time:.2f}s")
            
            total_chunks += len(doc_ids)
            successful_files += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {e}")
            failed_files.append(file_path.name)
            import traceback
            print(f"     Details: {traceback.format_exc()}")
    
    # Print summary
    print(f"\n📊 Indexing Summary:")
    print(f"  ✅ Successfully processed: {successful_files}/{len(files_found)} files")
    print(f"  📦 Total chunks indexed: {total_chunks}")
    print(f"  ⚡ Average processing time: {sum(processing_times)/len(processing_times):.2f}s per file" if processing_times else "N/A")
    
    if failed_files:
        print(f"  ❌ Failed files: {', '.join(failed_files)}")
    
    # Get final Pinecone stats
    try:
        final_stats = await vector_store.get_stats()
        print(f"\n📈 Final Pinecone Index Stats:")
        print(f"  Index: {final_stats.get('index_name', 'N/A')}")
        print(f"  Total vectors: {final_stats.get('total_vectors', 0)}")
        print(f"  Dimension: {final_stats.get('dimension', 'N/A')}")
        print(f"  Namespace: {final_stats.get('namespace', 'default')}")
    except Exception as e:
        print(f"⚠️  Could not get final Pinecone stats: {e}")
    
    return successful_files > 0

async def test_indexed_knowledge():
    """Test the indexed knowledge with sample queries."""
    print("\n🧪 Testing indexed knowledge with sample queries...")
    
    try:
        # Initialize components
        pinecone_key = os.getenv("PINECONE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
        
        pinecone_config = {
            "api_key": pinecone_key,
            "index_name": pinecone_index,
            "dimension": 1536,
            "namespace": "default"
        }
        
        vector_store = PineconeVectorStore(config=pinecone_config)
        llm = OpenAILLM(
            api_key=openai_key,
            model="gpt-4o-mini",
            max_tokens=256,
            timeout=2.0
        )
        query_engine = QueryEngine(vector_store=vector_store, llm=llm)
        
        # Smart test queries based on common knowledge docs
        test_queries = [
            "What is VoiceAI Technologies?",
            "How much does the Basic plan cost?",
            "What features does VoiceAssist include?",
            "What languages are supported?",
            "Tell me about pricing plans",
            "What is the difference between Basic and Pro plans?",
            "How do I contact support?",
            "What are the main products offered?"
        ]
        
        print("🔍 Running performance tests...")
        print(f"🎯 Target: <2 seconds total response time")
        
        response_times = []
        successful_queries = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            try:
                # Time the query
                import time
                start_time = time.time()
                
                result = await query_engine.query(query)
                
                query_time = time.time() - start_time
                response_times.append(query_time)
                
                response = result.get('response', 'No response')
                sources = result.get('sources', [])
                retrieval_time = result.get('retrieval_time', 0)
                llm_time = result.get('llm_time', 0)
                
                print(f"   ⏱️  Total time: {query_time:.2f}s")
                print(f"   🔍 Retrieval: {retrieval_time:.2f}s | 🤖 LLM: {llm_time:.2f}s")
                print(f"   💬 Answer: {response[:100]}...")
                print(f"   📚 Sources: {len(sources)} documents")
                
                # Performance feedback
                if query_time < 1.5:
                    print(f"   🚀 Excellent performance!")
                elif query_time < 2.0:
                    print(f"   ✅ Good performance")
                else:
                    print(f"   ⚠️  Warning: Exceeded 2s target ({query_time:.2f}s)")
                
                successful_queries += 1
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                print(f"      Details: {traceback.format_exc()}")
        
        # Performance summary
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            under_target = len([t for t in response_times if t < 2.0])
            
            print(f"\n📊 Performance Summary:")
            print(f"  ✅ Successful queries: {successful_queries}/{len(test_queries)}")
            print(f"  ⚡ Average response time: {avg_time:.2f}s")
            print(f"  🏃 Fastest: {min_time:.2f}s | 🐌 Slowest: {max_time:.2f}s")
            print(f"  🎯 Under 2s target: {under_target}/{len(response_times)} ({under_target/len(response_times)*100:.1f}%)")
            
            if avg_time < 2.0:
                print(f"  🎉 System meets latency target!")
            else:
                print(f"  ⚠️  System needs optimization for target latency")
        
        print("\n✅ Testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        print(f"Details: {traceback.format_exc()}")

async def interactive_query():
    """Interactive query mode for testing specific questions."""
    print("\n💬 Interactive Query Mode")
    print("Type your questions to test the knowledge base.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    try:
        # Initialize components
        pinecone_key = os.getenv("PINECONE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
        
        pinecone_config = {
            "api_key": pinecone_key,
            "index_name": pinecone_index,
            "dimension": 1536,
            "namespace": "default"
        }
        
        vector_store = PineconeVectorStore(config=pinecone_config)
        llm = OpenAILLM(api_key=openai_key, model="gpt-4o-mini")
        query_engine = QueryEngine(vector_store=vector_store, llm=llm)
        
        # Get current status
        stats = await vector_store.get_stats()
        print(f"📊 Connected to: {stats.get('index_name')}")
        print(f"📦 Available vectors: {stats.get('total_vectors', 0)}\n")
        
        while True:
            query = input("🤔 Your question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q', '']:
                break
            
            try:
                import time
                start_time = time.time()
                
                result = await query_engine.query(query)
                query_time = time.time() - start_time
                
                print(f"\n⏱️  Response time: {query_time:.2f}s")
                print(f"💬 Answer: {result.get('response', 'No response')}")
                
                if result.get('sources'):
                    print(f"📚 Sources used: {len(result['sources'])}")
                    for src in result['sources'][:3]:  # Show top 3 sources
                        print(f"   - {src.get('name', 'Unknown')}")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print("-" * 50)
        
        print("👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")

async def main():
    """Main function with enhanced options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index knowledge documents into Pinecone")
    parser.add_argument('--test', action='store_true', help='Test indexed knowledge after indexing')
    parser.add_argument('--reset', action='store_true', help='Reset Pinecone index before indexing')
    parser.add_argument('--test-only', action='store_true', help='Only run tests, don\'t index')
    parser.add_argument('--interactive', action='store_true', help='Start interactive query mode')
    parser.add_argument('--stats', action='store_true', help='Show current index statistics')
    
    args = parser.parse_args()
    
    print("🚀 Pinecone Knowledge Indexer for OpenAI + Pinecone")
    print("=" * 55)
    
    # Show stats if requested
    if args.stats:
        print("\n📊 Current Index Statistics:")
        try:
            pinecone_config = {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "index_name": os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge"),
                "dimension": 1536,
                "namespace": "default"
            }
            vector_store = PineconeVectorStore(config=pinecone_config)
            stats = await vector_store.get_stats()
            
            print(f"  Index: {stats.get('index_name')}")
            print(f"  Total vectors: {stats.get('total_vectors', 0)}")
            print(f"  Dimension: {stats.get('dimension')}")
            print(f"  Namespace: {stats.get('namespace')}")
        except Exception as e:
            print(f"  ❌ Error getting stats: {e}")
        return
    
    # Reset index if requested
    if args.reset:
        print("\n🗑️  Resetting Pinecone index...")
        try:
            pinecone_config = {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "index_name": os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge"),
                "dimension": 1536,
                "namespace": "default"
            }
            vector_store = PineconeVectorStore(config=pinecone_config)
            await vector_store.reset_index()
            print("✅ Index reset successfully")
        except Exception as e:
            print(f"❌ Error resetting index: {e}")
            return
    
    # Index documents unless test-only or interactive
    if not args.test_only and not args.interactive:
        success = await index_knowledge_docs()
        if not success:
            print("\n❌ Indexing failed. Please check the errors above.")
            return
    
    # Run tests if requested
    if args.test or args.test_only:
        await test_indexed_knowledge()
    
    # Start interactive mode if requested
    if args.interactive:
        await interactive_query()
    
    if not any([args.test, args.test_only, args.interactive, args.stats]):
        print("\n🎉 Indexing completed successfully!")
        print("💡 Try running with --test to verify the indexed knowledge")
        print("💡 Or use --interactive for manual testing")

if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set them before running this script.")
        print("Example:")
        print("export OPENAI_API_KEY=your_openai_key")
        print("export PINECONE_API_KEY=your_pinecone_key")
        print("export PINECONE_INDEX_NAME=voice-ai-knowledge  # Optional")
        sys.exit(1)
    
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()