# voice_ai_agent.py
"""
Voice AI Agent optimized for sub-2-second latency using OpenAI + Pinecone.
Simplified architecture for telephony applications.
"""
import os
import logging
import asyncio
import json
from typing import Optional, Dict, Any, Union, Callable, Awaitable

# Import optimized components
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.query_engine import QueryEngine
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.pinecone_store import PineconeVectorStore
from knowledge_base.openai_llm import OpenAILLM
from knowledge_base.document_store import DocumentStore
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """
    Voice AI Agent optimized for minimal latency using OpenAI LLM + Pinecone.
    Designed for sub-2-second response times in telephony applications.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: str = "voice-ai-knowledge",
        openai_model: str = "gpt-4o-mini",  # Fastest GPT-4 model
        credentials_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize Voice AI Agent with optimized components."""
        # API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key is required")
        
        # Component configuration
        self.pinecone_index_name = pinecone_index_name
        self.openai_model = openai_model
        self.credentials_file = credentials_file
        
        # Get project ID for Google Cloud services
        self.project_id = self._get_project_id()
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.vector_store = None
        self.llm = None
        self.query_engine = None
        self.conversation_manager = None
        self.tts_client = None
        
        self._initialized = False
        
        logger.info(f"VoiceAIAgent initialized for OpenAI ({openai_model}) + Pinecone")
    
    def _get_project_id(self) -> str:
        """Get Google Cloud project ID."""
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        if not project_id and self.credentials_file and os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                        logger.info(f"Extracted project ID: {project_id}")
            except Exception as e:
                logger.error(f"Error reading credentials: {e}")
        
        if not project_id:
            raise ValueError("Google Cloud project ID is required")
        
        return project_id
    
    async def init(self):
        """Initialize all components with parallel initialization where possible."""
        logger.info("Initializing Voice AI Agent with optimized components...")
        
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        
        if self.credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
        
        # Initialize components in parallel where possible
        await asyncio.gather(
            self._init_speech_components(),
            self._init_knowledge_components(),
            self._init_tts_component()
        )
        
        self._initialized = True
        logger.info("Voice AI Agent initialization complete")
    
    async def _init_speech_components(self):
        """Initialize speech recognition components."""
        # Google Cloud STT v2 for speech recognition
        self.speech_recognizer = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,
            project_id=self.project_id,
            location="global",
            credentials_file=self.credentials_file
        )
        
        # STT integration with zero preprocessing
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language="en-US"
        )
        await self.stt_integration.init(project_id=self.project_id)
        
        logger.info("Speech components initialized")
    
    async def _init_knowledge_components(self):
        """Initialize knowledge base components."""
        # Initialize Pinecone vector store
        pinecone_config = {
            "api_key": self.pinecone_api_key,
            "index_name": self.pinecone_index_name,
            "dimension": 1536,  # OpenAI text-embedding-3-small dimension
            "namespace": "default"
        }
        
        self.vector_store = PineconeVectorStore(config=pinecone_config)
        
        # Initialize OpenAI LLM
        self.llm = OpenAILLM(
            api_key=self.openai_api_key,
            model=self.openai_model,
            temperature=0.7,
            max_tokens=256,  # Shorter responses for telephony
            timeout=1.5  # Aggressive timeout for speed
        )
        
        # Initialize query engine
        self.query_engine = QueryEngine(
            vector_store=self.vector_store,
            llm=self.llm
        )
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            max_history=4  # Keep minimal history for speed
        )
        
        logger.info("Knowledge base components initialized")
    
    async def _init_tts_component(self):
        """Initialize TTS component."""
        # Google Cloud TTS optimized for telephony
        self.tts_client = GoogleCloudTTS(
            credentials_file=self.credentials_file,
            voice_name="en-US-Neural2-C",
            voice_gender=None,  # Don't set for Neural2
            language_code="en-US",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True,
            voice_type="NEURAL2"
        )
        
        logger.info("TTS component initialized")
    
    async def process_audio(
        self,
        audio_data: Union[bytes],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio with optimized pipeline for minimal latency.
        Target: <2 seconds total processing time.
        """
        if not self._initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: Speech to Text (target: <0.5s)
            stt_start = time.time()
            result = await self.stt_integration.transcribe_audio_data(
                audio_data, 
                callback=callback
            )
            stt_time = time.time() - stt_start
            
            if not result.get("is_valid", False) or not result.get("transcription"):
                return {
                    "status": "invalid_transcription",
                    "transcription": result.get("transcription", ""),
                    "processing_time": time.time() - start_time,
                    "error": "No valid speech detected"
                }
            
            transcription = result["transcription"]
            logger.info(f"Transcription ({stt_time:.2f}s): {transcription}")
            
            # Step 2: Knowledge Base Query with Streaming (target: <1s)
            kb_start = time.time()
            
            # Use streaming for immediate response
            response_text = ""
            async for chunk in self.conversation_manager.generate_streaming_response(transcription):
                if chunk.get("chunk"):
                    response_text += chunk["chunk"]
                
                # Break on completion or timeout
                if chunk.get("done", False):
                    break
                
                # Additional safety timeout
                if time.time() - kb_start > 1.0:
                    logger.warning("KB query taking too long, using partial response")
                    break
            
            kb_time = time.time() - kb_start
            
            if not response_text:
                response_text = "I couldn't find an answer to that question."
            
            logger.info(f"Knowledge base response ({kb_time:.2f}s): {response_text[:50]}...")
            
            # Step 3: Text to Speech (target: <0.5s)
            tts_start = time.time()
            try:
                speech_audio = await self.tts_client.synthesize(response_text)
                tts_time = time.time() - tts_start
                logger.info(f"TTS generation ({tts_time:.2f}s): {len(speech_audio)} bytes")
            except Exception as e:
                logger.error(f"TTS error: {e}")
                speech_audio = b""
                tts_time = time.time() - tts_start
            
            total_time = time.time() - start_time
            
            return {
                "transcription": transcription,
                "response": response_text,
                "speech_audio": speech_audio,
                "status": "success",
                "timing": {
                    "stt_time": stt_time,
                    "kb_time": kb_time,
                    "tts_time": tts_time,
                    "total_time": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_audio: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def add_knowledge(self, text: str, source: str = "manual_input") -> bool:
        """Add knowledge to the vector store."""
        if not self._initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        try:
            # Create document
            from knowledge_base.document_store import DocumentStore
            doc_store = DocumentStore()
            documents = doc_store.load_text(text, source)
            
            # Add to vector store
            doc_ids = await self.vector_store.add_documents(documents)
            
            logger.info(f"Added {len(doc_ids)} document chunks to knowledge base")
            return len(doc_ids) > 0
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
    
    async def add_documents_from_file(self, file_path: str) -> bool:
        """Add documents from file to knowledge base."""
        if not self._initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        try:
            from knowledge_base.document_store import DocumentStore
            doc_store = DocumentStore()
            documents = doc_store.load_file(file_path)
            
            doc_ids = await self.vector_store.add_documents(documents)
            
            logger.info(f"Added {len(doc_ids)} chunks from {file_path}")
            return len(doc_ids) > 0
            
        except Exception as e:
            logger.error(f"Error adding documents from file: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        stats = {
            "initialized": self._initialized,
            "openai_model": self.openai_model,
            "pinecone_index": self.pinecone_index_name
        }
        
        # Add component stats
        if self.query_engine:
            stats["query_engine"] = await self.query_engine.get_stats()
        
        if self.conversation_manager:
            stats["conversation"] = self.conversation_manager.get_stats()
        
        if self.vector_store:
            stats["vector_store"] = await self.vector_store.get_stats()
        
        if self.tts_client and hasattr(self.tts_client, 'get_stats'):
            stats["tts"] = self.tts_client.get_stats()
        
        return stats
    
    @property
    def initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._initialized
    
    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Stop STT streaming if active
        if (self.speech_recognizer and 
            hasattr(self.speech_recognizer, 'is_streaming') and 
            self.speech_recognizer.is_streaming):
            await self.speech_recognizer.stop_streaming()
        
        # Reset conversation
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        self._initialized = False
        logger.info("Voice AI Agent shutdown complete")