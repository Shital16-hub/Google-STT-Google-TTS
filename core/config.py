# core/fixed_config.py

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAISettings(BaseModel):
    """OpenAI configuration."""
    api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = Field(default=float(os.getenv("OPENAI_TEMPERATURE", "0.7")))
    max_tokens: int = Field(default=int(os.getenv("OPENAI_MAX_TOKENS", "256")))
    timeout: float = Field(default=float(os.getenv("OPENAI_TIMEOUT", "1.5")))
    embedding_model: str = Field(default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    streaming: bool = Field(default=True)

class PineconeSettings(BaseModel):
    """Pinecone configuration."""
    api_key: str = Field(default=os.getenv("PINECONE_API_KEY", ""))
    index_name: str = Field(default=os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge"))
    environment: str = Field(default=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"))
    namespace: str = Field(default=os.getenv("PINECONE_NAMESPACE", "default"))
    
class GoogleCloudSettings(BaseModel):
    """Google Cloud configuration."""
    credentials_file: str = Field(default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
    project_id: str = Field(default=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    
class STTSettings(BaseModel):
    """STT configuration."""
    language: str = Field(default=os.getenv("STT_LANGUAGE", "en-US"))
    model: str = Field(default=os.getenv("STT_MODEL", "telephony"))
    sample_rate: int = Field(default=int(os.getenv("STT_SAMPLE_RATE", "8000")))
    encoding: str = Field(default=os.getenv("STT_ENCODING", "MULAW"))
    channels: int = Field(default=int(os.getenv("STT_CHANNELS", "1")))
    location: str = Field(default=os.getenv("STT_LOCATION", "global"))
    recognizer_id: str = Field(default=os.getenv("STT_RECOGNIZER_ID", "_"))
    interim_results: bool = Field(default=os.getenv("STT_INTERIM_RESULTS", "false").lower() == "true")
    enable_automatic_punctuation: bool = Field(default=os.getenv("STT_ENABLE_AUTOMATIC_PUNCTUATION", "true").lower() == "true")
    enable_word_time_offsets: bool = Field(default=os.getenv("STT_ENABLE_WORD_TIME_OFFSETS", "true").lower() == "true")
    enable_word_confidence: bool = Field(default=os.getenv("STT_ENABLE_WORD_CONFIDENCE", "true").lower() == "true")
    enable_voice_activity_events: bool = Field(default=os.getenv("STT_ENABLE_VOICE_ACTIVITY_EVENTS", "true").lower() == "true")
    speech_start_timeout: int = Field(default=int(os.getenv("STT_SPEECH_START_TIMEOUT", "5")))
    speech_end_timeout: int = Field(default=int(os.getenv("STT_SPEECH_END_TIMEOUT", "1")))
    max_alternatives: int = Field(default=int(os.getenv("STT_MAX_ALTERNATIVES", "1")))
    profanity_filter: bool = Field(default=os.getenv("STT_PROFANITY_FILTER", "false").lower() == "true")
    use_enhanced_model: bool = Field(default=os.getenv("STT_USE_ENHANCED_MODEL", "true").lower() == "true")

class TTSSettings(BaseModel):
    """TTS configuration."""
    voice_type: str = Field(default=os.getenv("TTS_VOICE_TYPE", "NEURAL2"))
    voice_name: str = Field(default=os.getenv("TTS_VOICE_NAME", "en-US-Neural2-C"))
    voice_gender: str = Field(default=os.getenv("TTS_VOICE_GENDER", ""))
    language_code: str = Field(default=os.getenv("TTS_LANGUAGE_CODE", "en-US"))
    container_format: str = Field(default=os.getenv("TTS_CONTAINER_FORMAT", "mulaw"))
    sample_rate: int = Field(default=int(os.getenv("TTS_SAMPLE_RATE", "8000")))
    enable_caching: bool = Field(default=os.getenv("TTS_ENABLE_CACHING", "true").lower() == "true")

class KnowledgeBaseSettings(BaseModel):
    """Knowledge base configuration."""
    storage_dir: str = Field(default=os.getenv("STORAGE_DIR", "./storage"))
    max_document_size_mb: int = Field(default=int(os.getenv("MAX_DOCUMENT_SIZE_MB", "10")))
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "50")))
    default_retrieve_count: int = Field(default=int(os.getenv("DEFAULT_RETRIEVE_COUNT", "3")))
    minimum_relevance_score: float = Field(default=float(os.getenv("MINIMUM_RELEVANCE_SCORE", "0.6")))
    embedding_batch_size: int = Field(default=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
    enable_caching: bool = Field(default=os.getenv("ENABLE_CACHING", "true").lower() == "true")
    parallel_processing: bool = Field(default=os.getenv("PARALLEL_PROCESSING", "true").lower() == "true")
    use_gpu: bool = Field(default=os.getenv("USE_GPU", "false").lower() == "true")
    persist_dir: str = Field(default=os.getenv("PERSIST_DIR", "./storage"))
    reranking_enabled: bool = Field(default=os.getenv("RERANKING_ENABLED", "false").lower() == "true")
    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dimension: str = Field(default=os.getenv("EMBEDDING_DIMENSION", "1536"))
    vector_dimension: str = Field(default=os.getenv("VECTOR_DIMENSION", "1536"))

class ConversationSettings(BaseModel):
    """Conversation configuration."""
    max_conversation_history: int = Field(default=int(os.getenv("MAX_CONVERSATION_HISTORY", "5")))
    context_window_size: int = Field(default=int(os.getenv("CONTEXT_WINDOW_SIZE", "4096")))
    max_tokens: int = Field(default=int(os.getenv("MAX_TOKENS", "150")))
    temperature: float = Field(default=float(os.getenv("TEMPERATURE", "0.7")))

class PerformanceSettings(BaseModel):
    """Performance configuration."""
    target_stt_latency: float = Field(default=float(os.getenv("TARGET_STT_LATENCY", "0.5")))
    target_kb_latency: float = Field(default=float(os.getenv("TARGET_KB_LATENCY", "1.0")))
    target_tts_latency: float = Field(default=float(os.getenv("TARGET_TTS_LATENCY", "0.5")))
    target_total_latency: float = Field(default=float(os.getenv("TARGET_TOTAL_LATENCY", "2.0")))
    enable_performance_logging: bool = Field(default=os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true")
    enable_prometheus: bool = Field(default=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true")
    metrics_port: int = Field(default=int(os.getenv("METRICS_PORT", "9090")))

class TwilioSettings(BaseModel):
    """Twilio integration configuration."""
    account_sid: str = Field(default=os.getenv("TWILIO_ACCOUNT_SID", ""))
    auth_token: str = Field(default=os.getenv("TWILIO_AUTH_TOKEN", ""))
    phone_number: str = Field(default=os.getenv("TWILIO_PHONE_NUMBER", ""))
    twiml_app_sid: Optional[str] = Field(default=os.getenv("TWILIO_TWIML_APP_SID"))
    status_callback_url: Optional[str] = Field(default=None)

class RedisSettings(BaseModel):
    """Redis configuration."""
    url: str = Field(default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))

class FeatureFlags(BaseModel):
    """Feature flags."""
    enable_conversation_context: bool = Field(default=os.getenv("ENABLE_CONVERSATION_CONTEXT", "true").lower() == "true")
    enable_source_citations: bool = Field(default=os.getenv("ENABLE_SOURCE_CITATIONS", "true").lower() == "true")

class Settings(BaseSettings):
    """Main application settings."""
    # Basic settings
    debug: bool = Field(default=os.getenv("DEBUG", "false").lower() == "true")
    environment: str = Field(default="development")
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    
    # Application paths
    base_dir: str = Field(default=".")
    prompts_dir: str = Field(default="./prompts")
    
    # Component configurations
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    google_cloud: GoogleCloudSettings = Field(default_factory=GoogleCloudSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    knowledge_base: KnowledgeBaseSettings = Field(default_factory=KnowledgeBaseSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    twilio: TwilioSettings = Field(default_factory=TwilioSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    
    # API configuration
    host: str = Field(default=os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default=int(os.getenv("PORT", "5000")))
    base_url: str = Field(default=os.getenv("BASE_URL", "http://localhost:5000"))
    
    # Security settings
    api_key: Optional[str] = Field(default=os.getenv("API_KEY"))
    allowed_origins: list = Field(default=["*"])
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"