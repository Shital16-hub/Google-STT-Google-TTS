# knowledge_base/openai_llm.py
"""
OpenAI LLM client with streaming support optimized for telephony latency.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, AsyncIterator
import openai
from openai import AsyncOpenAI

from knowledge_base.config import get_openai_config

logger = logging.getLogger(__name__)

class OpenAILLM:
    """
    OpenAI LLM client optimized for minimal latency with async streaming.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """Initialize OpenAI LLM client."""
        self.config = get_openai_config()
        
        # Override config with provided values
        self.api_key = api_key or self.config["api_key"]
        self.model = model or self.config["model"]
        self.temperature = temperature or self.config["temperature"]
        self.max_tokens = max_tokens or self.config["max_tokens"]
        self.timeout = timeout or self.config["timeout"]
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize async client with timeout
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        # System prompt optimized for telephony
        self.system_prompt = """You are a helpful AI assistant for telephone conversations. 
        Keep responses concise and conversational. Use simple language suitable for speech.
        Don't use bullet points or lists. Keep answers under 100 words when possible."""
        
        logger.info(f"Initialized OpenAI LLM with model: {self.model}")
    
    async def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a single response (non-streaming) for simple use cases.
        """
        messages = self._prepare_messages(query, context, chat_history)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process that request right now."
    
    async def generate_streaming_response(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate streaming response for real-time applications.
        Optimized for minimal latency.
        """
        messages = self._prepare_messages(query, context, chat_history)
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True}  # Get token usage
            )
            
            full_response = ""
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    
                    # Yield chunk immediately for TTS processing
                    yield {
                        "chunk": content,
                        "done": False,
                        "full_response": full_response
                    }
                
                # Check for completion
                if chunk.choices and chunk.choices[0].finish_reason == "stop":
                    # Get usage stats if available
                    usage = chunk.usage if hasattr(chunk, 'usage') else None
                    
                    yield {
                        "chunk": "",
                        "done": True,
                        "full_response": full_response,
                        "usage": usage.dict() if usage else None
                    }
                    break
                    
        except asyncio.TimeoutError:
            logger.error(f"OpenAI API timeout after {self.timeout} seconds")
            yield {
                "chunk": "",
                "done": True,
                "full_response": "I'm sorry, the response is taking too long. Please try again.",
                "error": "timeout"
            }
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield {
                "chunk": "",
                "done": True,
                "full_response": "I'm sorry, I encountered an error processing your request.",
                "error": str(e)
            }
    
    def _prepare_messages(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenAI API call.
        Optimized for minimal token usage while maintaining context.
        """
        messages = []
        
        # System message with context if available
        system_content = self.system_prompt
        if context:
            system_content += f"\n\nRelevant information:\n{context}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add chat history (keep last 3 exchanges for context)
        if chat_history:
            recent_history = chat_history[-6:]  # Last 3 exchanges (6 messages)
            messages.extend(recent_history)
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM statistics."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "api_key_configured": bool(self.api_key)
        }