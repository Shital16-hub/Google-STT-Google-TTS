# voice_ai_agent.py

"""
Voice AI Agent main class updated with low-latency optimizations and better error handling.
"""
import os
import logging
import asyncio
import time
import json
from typing import Optional, Dict, Any, Union, Callable, Awaitable, List

# Import our optimized STT implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration

# Knowledge base imports
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.index_manager import IndexManager
from knowledge_base.rag_config import rag_config

# Google Cloud TTS import
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Voice AI Agent optimized for low-latency voice interactions."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: Optional[str] = None,
        llm_temperature: float = 0.7,
        credentials_file: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with low-latency optimizations.
        
        Args:
            storage_dir: Directory for local storage
            model_name: OpenAI model name (defaults to config)
            llm_temperature: Temperature for sampling
            credentials_file: Path to Google Cloud credentials file
            openai_api_key: OpenAI API key (defaults to environment)
            pinecone_api_key: Pinecone API key (defaults to environment)
            **kwargs: Additional parameters
        """
        # Update config with provided parameters
        if storage_dir:
            rag_config.storage_dir = storage_dir
        if model_name:
            rag_config.openai_model = model_name
        if llm_temperature:
            rag_config.llm_temperature = llm_temperature
        
        # Set API keys if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            rag_config.openai_api_key = openai_api_key
        if pinecone_api_key:
            os.environ["PINECONE_API_KEY"] = pinecone_api_key
            rag_config.pinecone_api_key = pinecone_api_key
        
        # Validate API keys
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        if not os.environ.get("PINECONE_API_KEY"):
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable.")
        
        # Google Cloud credentials handling
        self.credentials_file = credentials_file
        if not self.credentials_file:
            # Try common locations
            possible_paths = [
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
                '/workspace/credentials/my-tts-project-458404-8ab56bac7265.json',
                './credentials/my-tts-project-458404-8ab56bac7265.json',
            ]
            for path in possible_paths:
                if path and os.path.exists(path):
                    self.credentials_file = path
                    logger.info(f"Found credentials file: {path}")
                    break
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        
        # TTS Parameters for Google Cloud TTS
        self.tts_voice_name = kwargs.get('tts_voice_name', os.getenv('TTS_VOICE_NAME', 'en-US-Neural2-C'))
        self.tts_voice_gender = kwargs.get('tts_voice_gender', os.getenv('TTS_VOICE_GENDER', 'NEUTRAL'))
        self.tts_language_code = kwargs.get('tts_language_code', os.getenv('TTS_LANGUAGE_CODE', 'en-US'))
        
        # Get project ID from environment or credentials file for Google Cloud services
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # If not in environment, try to extract from credentials file
        if not self.project_id and self.credentials_file and os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    self.project_id = creds_data.get('project_id')
                    # Set environment variable for consistency
                    if self.project_id:
                        os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
                        logger.info(f"Extracted and set project ID: {self.project_id}")
            except Exception as e:
                logger.error(f"Error reading credentials file: {e}")
        
        # Validate Google Cloud project ID
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID is required for STT/TTS. Set GOOGLE_CLOUD_PROJECT environment variable."
            )
        
        # Set the environment variable for Google Cloud clients
        if self.credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
            logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {self.credentials_file}")
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.index_manager = None
        self.query_engine = None
        self.conversation_manager = None
        self.tts_client = None
        
        logger.info("VoiceAIAgent initialized with low-latency optimizations")
        
    async def init(self):
        """Initialize all components with better error handling."""
        logger.info("Initializing Voice AI Agent components...")
        
        try:
            # Initialize speech recognizer with optimized settings
            self.speech_recognizer = GoogleCloudStreamingSTT(
                language=self.stt_language,
                sample_rate=8000,
                encoding="MULAW",
                channels=1,
                interim_results=True,  # Enable interim results for early processing
                project_id=self.project_id,
                location="global",
                credentials_file=self.credentials_file
            )
            
            # Initialize STT integration with optimized settings
            self.stt_integration = STTIntegration(
                speech_recognizer=self.speech_recognizer,
                language=self.stt_language
            )
            await self.stt_integration.init(project_id=self.project_id)
            
            # Initialize index manager with Pinecone
            self.index_manager = IndexManager(config=rag_config)
            await self.index_manager.init()
            
            # Initialize query engine with streaming support
            self.query_engine = QueryEngine(
                index_manager=self.index_manager,
                config=rag_config
            )
            await self.query_engine.init()
            
            # Initialize conversation manager with memory
            self.conversation_manager = ConversationManager(
                query_engine=self.query_engine,
                config=rag_config,
                skip_greeting=True  # Better for telephony
            )
            await self.conversation_manager.init()
            
            # Initialize Google Cloud TTS with telephony optimization
            try:
                self.tts_client = GoogleCloudTTS(
                    credentials_file=self.credentials_file,
                    voice_name=self.tts_voice_name,
                    voice_gender=self.tts_voice_gender if not "Neural2" in self.tts_voice_name else None,
                    language_code=self.tts_language_code,
                    container_format="mulaw",
                    sample_rate=8000,
                    enable_caching=True,
                    voice_type="NEURAL2"
                )
                
                logger.info(f"Initialized Google Cloud TTS with voice: {self.tts_voice_name}")
            except Exception as e:
                logger.error(f"Error initializing Google Cloud TTS: {e}")
                raise
            
            # Mark as initialized
            self._initialized = True
            logger.info("Voice AI Agent initialization complete with low-latency optimizations")
            
        except Exception as e:
            logger.error(f"Error initializing Voice AI Agent: {e}")
            # Attempt cleanup of partial initialization
            await self.shutdown(force=True)
            raise
        
    async def process_audio(
        self,
        audio_data: Union[bytes],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with low-latency optimizations.
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Reset speaking state for optimal processing
        if hasattr(self.speech_recognizer, 'set_speaking_state'):
            self.speech_recognizer.set_speaking_state(False)
        
        # Pass audio directly to STT with minimal overhead
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Set speaking state during response generation
            if hasattr(self.speech_recognizer, 'set_speaking_state'):
                self.speech_recognizer.set_speaking_state(True)
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(transcription)
            
            # Generate speech
            if response and response.get("response"):
                try:
                    speech_audio = await self.tts_client.synthesize(response["response"])
                    
                    # Reset speaking state after response generation
                    if hasattr(self.speech_recognizer, 'set_speaking_state'):
                        self.speech_recognizer.set_speaking_state(False)
                    
                    return {
                        "transcription": transcription,
                        "response": response.get("response", ""),
                        "speech_audio": speech_audio,
                        "status": "success",
                        "context": response.get("context"),
                        "sources": response.get("sources", [])
                    }
                except Exception as e:
                    logger.error(f"Error synthesizing speech: {e}")
                    
                    # Reset speaking state in case of error
                    if hasattr(self.speech_recognizer, 'set_speaking_state'):
                        self.speech_recognizer.set_speaking_state(False)
                        
                    return {
                        "transcription": transcription,
                        "response": response.get("response", ""),
                        "error": f"Speech synthesis error: {str(e)}",
                        "status": "tts_error"
                    }
            else:
                # Reset speaking state if no response
                if hasattr(self.speech_recognizer, 'set_speaking_state'):
                    self.speech_recognizer.set_speaking_state(False)
                    
                return {
                    "transcription": transcription,
                    "response": response.get("response", ""),
                    "status": "success"
                }
        else:
            logger.info("Invalid or empty transcription")
            return {
                "status": "invalid_transcription",
                "transcription": result.get("transcription", ""),
                "error": "No valid speech detected"
            }
    
    async def process_audio_streaming(
        self,
        audio_data: Union[bytes],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process audio with optimized streaming response.
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Create optimized pipeline on demand if not already available
        from integration.pipeline import VoiceAIAgentPipeline
        
        pipeline = VoiceAIAgentPipeline(
            speech_recognizer=self.speech_recognizer,
            conversation_manager=self.conversation_manager,
            query_engine=self.query_engine,
            tts_integration=self.tts_client
        )
        
        return await pipeline.process_audio_streaming(audio_data, audio_callback)
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return getattr(self, '_initialized', False)
                
    async def shutdown(self, force=False):
        """Shut down all components properly with force option."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Close Google Cloud streaming session if active
        if self.speech_recognizer:
            try:
                if hasattr(self.speech_recognizer, 'cleanup'):
                    await self.speech_recognizer.cleanup()
                elif hasattr(self.speech_recognizer, 'stop_streaming'):
                    await self.speech_recognizer.stop_streaming()
            except Exception as e:
                if not force:
                    raise
                logger.error(f"Error shutting down speech recognizer: {e}")
        
        # Reset conversation if active
        if self.conversation_manager:
            try:
                self.conversation_manager.reset()
            except Exception as e:
                if not force:
                    raise
                logger.error(f"Error resetting conversation manager: {e}")
        
        # Mark as not initialized
        self._initialized = False
    
    async def add_documents_to_index(self, directory_path: str) -> List[str]:
        """
        Add documents from a directory to the index.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of document IDs
        """
        if not self.initialized:
            await self.init()
            
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        # Add directory to index
        return await self.index_manager.add_directory(directory_path)