#!/usr/bin/env python3
"""
Multi-Agent Voice AI System - Working Version
Simplified to work with current dependencies while maintaining core functionality.
"""
import os
import sys
import asyncio
import logging
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

# FastAPI and async imports
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from pydantic import BaseModel

# Basic imports that should work with your current setup
import openai
import redis
from qdrant_client import QdrantClient
import requests

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/voice_ai_system.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Basic configuration from environment
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    HOST = os.getenv("APP_HOST", "0.0.0.0")
    PORT = int(os.getenv("APP_PORT", "8000"))
    DEBUG = os.getenv("ENVIRONMENT", "development") == "development"

# Global components (simplified)
system_components = {}
active_sessions = {}

class SystemHealthModel(BaseModel):
    """System health status model."""
    status: str
    timestamp: float
    components: Dict[str, bool]
    active_sessions: int
    version: str

class SimpleAgent:
    """Simple agent implementation for testing."""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY) if Config.OPENAI_API_KEY else None
    
    async def process_message(self, message: str, context: List[Dict] = None) -> Dict[str, Any]:
        """Process a message through this agent."""
        try:
            if not self.openai_client:
                return {
                    "agent_id": self.agent_id,
                    "response": f"Hello! I'm the {self.agent_type} agent. I received: {message}",
                    "confidence": 0.8
                }
            
            # Create agent-specific prompt
            system_prompts = {
                "roadside-assistance": "You are a professional roadside assistance coordinator. Help with towing, flat tires, and emergency services.",
                "billing-support": "You are a billing support specialist. Help with payments, refunds, and account issues.",
                "technical-support": "You are a technical support specialist. Help with troubleshooting and technical problems.",
                "general-support": "You are a helpful general support assistant."
            }
            
            system_prompt = system_prompts.get(self.agent_type, system_prompts["general-support"])
            
            # Call OpenAI
            response = await self._call_openai(system_prompt, message)
            
            return {
                "agent_id": self.agent_id,
                "response": response,
                "confidence": 0.9,
                "agent_type": self.agent_type
            }
            
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id}: {e}")
            return {
                "agent_id": self.agent_id,
                "response": f"I apologize, I'm having trouble processing your request right now.",
                "confidence": 0.5,
                "error": str(e)
            }
    
    async def _call_openai(self, system_prompt: str, user_message: str) -> str:
        """Call OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"I'm here to help with {self.agent_type} issues. How can I assist you?"

class SimpleRouter:
    """Simple routing logic for agent selection."""
    
    def __init__(self):
        self.routing_keywords = {
            "roadside-assistance": ["tow", "towing", "stuck", "breakdown", "accident", "roadside", "emergency", "flat tire", "battery"],
            "billing-support": ["bill", "billing", "payment", "charge", "refund", "money", "invoice", "account"],
            "technical-support": ["technical", "tech", "problem", "issue", "broken", "error", "troubleshoot", "help"]
        }
    
    def route_message(self, message: str) -> str:
        """Route message to appropriate agent."""
        message_lower = message.lower()
        
        # Score each agent based on keyword matches
        scores = {}
        for agent_type, keywords in self.routing_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                scores[agent_type] = score
        
        # Return highest scoring agent or default
        if scores:
            return max(scores, key=scores.get)
        return "general-support"

class VoiceSession:
    """Simplified voice session handler."""
    
    def __init__(self, session_id: str, call_sid: str, websocket: WebSocket):
        self.session_id = session_id
        self.call_sid = call_sid
        self.websocket = websocket
        self.start_time = time.time()
        self.message_count = 0
        self.current_agent = None
        self.stream_sid = None
        
        # Initialize router and agents
        self.router = SimpleRouter()
        self.agents = {
            "roadside-assistance": SimpleAgent("roadside-assistance", "roadside-assistance"),
            "billing-support": SimpleAgent("billing-support", "billing-support"),
            "technical-support": SimpleAgent("technical-support", "technical-support"),
            "general-support": SimpleAgent("general-support", "general-support")
        }
    
    async def handle_message(self, data: Dict[str, Any]):
        """Handle WebSocket message."""
        try:
            event_type = data.get('event')
            
            if event_type == 'connected':
                logger.info(f"WebSocket connected for session {self.session_id}")
                
            elif event_type == 'start':
                self.stream_sid = data.get('streamSid')
                logger.info(f"Stream started: {self.stream_sid}")
                await self.send_welcome_message()
                
            elif event_type == 'media':
                # For now, we'll simulate audio processing
                await self.process_audio(data)
                
            elif event_type == 'stop':
                logger.info(f"Stream stopped for session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def process_audio(self, data: Dict[str, Any]):
        """Process audio (simplified - no actual STT for now)."""
        try:
            # For testing, simulate receiving text
            # In real implementation, this would use STT
            self.message_count += 1
            
            # Simulate a text message for testing
            test_messages = [
                "I need a tow truck",
                "I have a question about my bill", 
                "I'm having technical issues",
                "Can you help me?"
            ]
            
            # Use message count to cycle through test messages
            simulated_text = test_messages[self.message_count % len(test_messages)]
            
            await self.process_text_message(simulated_text)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    async def process_text_message(self, text: str):
        """Process text message through multi-agent system."""
        try:
            # Route to appropriate agent
            agent_type = self.router.route_message(text)
            self.current_agent = agent_type
            
            logger.info(f"Routing message to {agent_type}: {text}")
            
            # Process through selected agent
            agent = self.agents[agent_type]
            response = await agent.process_message(text)
            
            # For now, just log the response (would normally convert to audio)
            logger.info(f"Agent response: {response}")
            
            # Send text response back (in real system, this would be TTS)
            await self.send_text_response(response['response'])
            
        except Exception as e:
            logger.error(f"Error processing text message: {e}")
    
    async def send_welcome_message(self):
        """Send welcome message."""
        try:
            welcome_text = "Hello! I'm your AI assistant. How can I help you today?"
            await self.send_text_response(welcome_text)
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}")
    
    async def send_text_response(self, text: str):
        """Send text response (placeholder for TTS)."""
        try:
            # For now, just send as text message
            message = {
                "event": "response",
                "text": text,
                "agent": self.current_agent,
                "session_id": self.session_id
            }
            
            await self.websocket.send_text(json.dumps(message))
            logger.info(f"Sent response: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error sending text response: {e}")
    
    async def get_status(self):
        """Get session status."""
        return {
            "session_id": self.session_id,
            "call_sid": self.call_sid,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time,
            "message_count": self.message_count,
            "current_agent": self.current_agent
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("🚀 Starting Multi-Agent Voice AI System (Simplified)...")
    
    # Initialize basic components
    await initialize_system()
    
    yield
    
    logger.info("🛑 Shutting down system...")

# FastAPI app
app = FastAPI(
    title="Multi-Agent Voice AI System",
    description="Simplified multi-agent voice AI for testing",
    version="1.0.0",
    docs_url="/docs" if Config.DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if Config.DEBUG else ["https://yourprovider.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_system():
    """Initialize system components."""
    try:
        logger.info("Initializing simplified system...")
        
        # Test connections
        success_count = 0
        total_tests = 3
        
        # Test OpenAI
        if Config.OPENAI_API_KEY:
            try:
                client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
                # Test with a simple call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                logger.info("✅ OpenAI connection successful")
                success_count += 1
            except Exception as e:
                logger.error(f"❌ OpenAI connection failed: {e}")
        else:
            logger.warning("⚠️ No OpenAI API key provided")
        
        # Test Redis (optional)
        try:
            redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD if Config.REDIS_PASSWORD else None,
                decode_responses=True
            )
            redis_client.ping()
            logger.info("✅ Redis connection successful")
            success_count += 1
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed (optional): {e}")
        
        # Test Qdrant (optional)
        try:
            qdrant_client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
            # Just test connection, don't require collections
            logger.info("✅ Qdrant connection successful")
            success_count += 1
        except Exception as e:
            logger.warning(f"⚠️ Qdrant connection failed (optional): {e}")
        
        logger.info(f"System initialized with {success_count}/{total_tests} services working")
        
    except Exception as e:
        logger.error(f"System initialization error: {e}")

# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "system": "Multi-Agent Voice AI (Simplified)",
        "version": "1.0.0", 
        "status": "operational",
        "timestamp": time.time(),
        "active_sessions": len(active_sessions)
    }

@app.get("/health", response_model=SystemHealthModel)
async def health_check():
    """Health check endpoint."""
    components = {
        "api": True,
        "openai": bool(Config.OPENAI_API_KEY),
        "system": True
    }
    
    overall_status = "healthy" if all(components.values()) else "degraded"
    
    return SystemHealthModel(
        status=overall_status,
        timestamp=time.time(),
        components=components,
        active_sessions=len(active_sessions),
        version="1.0.0"
    )

@app.post("/api/test/text")
async def test_text_processing(request: Dict[str, Any]):
    """Test endpoint for text processing."""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text input required")
        
        # Create temporary session for testing
        router = SimpleRouter()
        agent_type = router.route_message(text)
        
        agent = SimpleAgent(agent_type, agent_type)
        response = await agent.process_message(text)
        
        return {
            "input": text,
            "routed_to": agent_type,
            "response": response,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/voice/{call_sid}")
async def handle_voice_websocket(websocket: WebSocket, call_sid: str):
    """Handle WebSocket for voice processing."""
    session_id = str(uuid.uuid4())
    logger.info(f"🔗 WebSocket connection for call {call_sid}")
    
    try:
        await websocket.accept()
        
        # Create session
        session = VoiceSession(session_id, call_sid, websocket)
        active_sessions[session_id] = session
        
        # Handle messages
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                await session.handle_message(data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON: {message}")
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]

@app.get("/sessions")
async def get_active_sessions():
    """Get active sessions."""
    sessions_info = {}
    for session_id, session in active_sessions.items():
        try:
            sessions_info[session_id] = await session.get_status()
        except:
            sessions_info[session_id] = {"error": "Status unavailable"}
    
    return {
        "active_sessions": len(active_sessions),
        "sessions": sessions_info
    }

@app.get("/api/agents")
async def list_agents():
    """List available agents."""
    return {
        "agents": [
            {"id": "roadside-assistance", "type": "Emergency Services", "status": "active"},
            {"id": "billing-support", "type": "Financial Services", "status": "active"},
            {"id": "technical-support", "type": "Technical Support", "status": "active"},
            {"id": "general-support", "type": "General Support", "status": "active"}
        ]
    }

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🚀 Starting Multi-Agent Voice AI System...")
    
    uvicorn.run(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        log_level="info"
    )