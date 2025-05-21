# core/config.py

"""
Core configuration settings for the application.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class KnowledgeBaseSettings(BaseModel):
    """Knowledge base configuration."""
    storage_dir: str = Field(default="./storage")
    pinecone_api_key: str = Field(default=os.getenv("PINECONE_API_KEY", ""))
    pinecone_environment: str = Field(default=os.getenv("PINECONE_ENV", "gcp-starter"))
    pinecone_index: str = Field(default=os.getenv("PINECONE_INDEX", "roadside-assistance"))
    index_batch_size: int = Field(default=100)
    cache_enabled: bool = Field(default=True)

class ConversationSettings(BaseModel):
    """Conversation management configuration."""
    max_turns: int = Field(default=20)
    timeout_seconds: int = Field(default=300)
    max_silence_seconds: float = Field(default=10.0)
    early_response_threshold: int = Field(default=3)

class AgentSettings(BaseModel):
    """Agent system configuration."""
    default_agent: str = Field(default="dispatcher")
    handoff_timeout: int = Field(default=60)
    max_retry_attempts: int = Field(default=3)

class TwilioSettings(BaseModel):
    """Twilio integration configuration."""
    account_sid: str = Field(default=os.getenv("TWILIO_ACCOUNT_SID", ""))
    auth_token: str = Field(default=os.getenv("TWILIO_AUTH_TOKEN", ""))
    phone_number: str = Field(default=os.getenv("TWILIO_PHONE_NUMBER", ""))
    twiml_app_sid: Optional[str] = Field(default=os.getenv("TWILIO_TWIML_APP_SID"))
    status_callback_url: Optional[str] = Field(default=None)

class OpenAISettings(BaseModel):
    """OpenAI configuration."""
    api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=150)
    streaming: bool = Field(default=True)

class Settings(BaseSettings):
    """Main application settings."""
    # Basic settings
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    
    # Application paths
    base_dir: str = Field(default=".")
    prompts_dir: str = Field(default="./prompts")
    
    # Component configurations
    knowledge_base: KnowledgeBaseSettings = Field(default_factory=KnowledgeBaseSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    twilio: TwilioSettings = Field(default_factory=TwilioSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    
    # API configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    base_url: str = Field(default=os.getenv("BASE_URL", "http://localhost:8000"))
    
    # Security settings
    api_key: Optional[str] = Field(default=os.getenv("API_KEY"))
    allowed_origins: list = Field(default=["*"])
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_component_settings(self, component: str) -> Dict[str, Any]:
        """Get settings for a specific component."""
        components = {
            "knowledge_base": self.knowledge_base,
            "conversation": self.conversation,
            "agent": self.agent,
            "twilio": self.twilio,
            "openai": self.openai
        }
        return components.get(component, {})
    
    def get_knowledge_base_config(self) -> Dict[str, Any]:
        """Get configuration for knowledge base initialization."""
        return {
            "storage_dir": self.knowledge_base.storage_dir,
            "pinecone_api_key": self.knowledge_base.pinecone_api_key,
            "pinecone_environment": self.knowledge_base.pinecone_environment,
            "pinecone_index": self.knowledge_base.pinecone_index,
            "cache_enabled": self.knowledge_base.cache_enabled
        }
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return {
            "api_key": self.openai.api_key,
            "model": self.openai.model,
            "temperature": self.openai.temperature,
            "max_tokens": self.openai.max_tokens,
            "streaming": self.openai.streaming
        }
    
    def get_twilio_config(self) -> Dict[str, Any]:
        """Get Twilio configuration."""
        config = {
            "account_sid": self.twilio.account_sid,
            "auth_token": self.twilio.auth_token,
            "phone_number": self.twilio.phone_number
        }
        
        if self.twilio.twiml_app_sid:
            config["application_sid"] = self.twilio.twiml_app_sid
            
        if self.twilio.status_callback_url:
            config["status_callback"] = self.twilio.status_callback_url
        elif self.base_url:
            config["status_callback"] = f"{self.base_url}/voice/status"
            
        return config