#!/usr/bin/env python3
"""
Complete Multi-Agent Voice AI System - Production FastAPI Application
Optimized for <2-second end-to-end latency with hot deployment capabilities.
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
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

# Twilio imports
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client as TwilioClient

# Core system imports
from app.core.orchestrator import MultiAgentOrchestrator
from app.core.latency_optimizer import LatencyOptimizer
from app.core.conversation_manager import EnhancedConversationManager

# Vector database imports
from app.vector_db.hybrid_vector_store import HybridVectorStore
from app.vector_db.qdrant_manager import QdrantManager
from app.vector_db.redis_cache import RedisCache

# Agent imports
from app.agents.agent_registry import AgentRegistry
from app.agents.intelligent_router import IntelligentRouter

# Voice processing imports
from app.voice.optimized_stt import OptimizedSTT
from app.voice.dual_streaming_tts import DualStreamingTTS

# Configuration imports
from app.config.production_settings import ProductionConfig
from app.config.latency_config import LatencyConfig

# Monitoring imports
from app.monitoring.performance_tracker import PerformanceTracker
from app.monitoring.business_analytics import BusinessAnalytics
from app.monitoring.alerting_system import AlertingSystem

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/voice_ai_system.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('redis').setLevel(logging.WARNING)

# Global system components
system_components = {}
performance_tracker = None
business_analytics = None
alerting_system = None

# Session tracking
active_sessions = {}
session_metrics = {}

class SystemHealthModel(BaseModel):
    """System health status model."""
    status: str
    timestamp: float
    components: Dict[str, bool]
    performance_metrics: Dict[str, float]
    active_sessions: int

class AgentDeploymentModel(BaseModel):
    """Agent deployment model."""
    agent_id: str
    version: str
    config: Dict[str, Any]
    hot_deploy: bool = True

class VoiceSessionModel(BaseModel):
    """Voice session model."""
    session_id: str
    call_sid: str
    agent_type: Optional[str] = None
    metadata: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management with graceful startup/shutdown."""
    logger.info("🚀 Starting Multi-Agent Voice AI System...")
    
    # Initialize system components
    await initialize_system()
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down Multi-Agent Voice AI System...")
    await cleanup_system()

# FastAPI app with production optimizations
app = FastAPI(
    title="Multi-Agent Voice AI System",
    description="Production-ready multi-agent voice AI with <2s latency optimization",
    version="3.0.0",
    docs_url="/docs" if ProductionConfig.DEBUG else None,
    redoc_url="/redoc" if ProductionConfig.DEBUG else None,
    lifespan=lifespan
)

# Add middleware stack for production
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ProductionConfig.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware for request tracking
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID and performance tracking to all requests."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    # Add to logging context
    import logging
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    # Track request timing
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add performance headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    # Track in performance system
    if performance_tracker:
        await performance_tracker.track_request(
            endpoint=str(request.url.path),
            method=request.method,
            duration=process_time,
            status_code=response.status_code
        )
    
    # Restore logging factory
    logging.setLogRecordFactory(old_factory)
    
    return response

async def initialize_system():
    """Initialize all system components with optimized startup sequence."""
    global system_components, performance_tracker, business_analytics, alerting_system
    
    try:
        logger.info("Initializing system components...")
        
        # 1. Initialize monitoring systems first
        performance_tracker = PerformanceTracker()
        business_analytics = BusinessAnalytics()
        alerting_system = AlertingSystem()
        
        await performance_tracker.init()
        await business_analytics.init()
        await alerting_system.init()
        
        # 2. Initialize vector database infrastructure
        logger.info("Setting up vector database infrastructure...")
        redis_cache = RedisCache()
        qdrant_manager = QdrantManager()
        
        await redis_cache.init()
        await qdrant_manager.init()
        
        hybrid_vector_store = HybridVectorStore(
            redis_cache=redis_cache,
            qdrant_manager=qdrant_manager
        )
        await hybrid_vector_store.init()
        
        # 3. Initialize voice processing components
        logger.info("Initializing voice processing...")
        optimized_stt = OptimizedSTT()
        dual_streaming_tts = DualStreamingTTS()
        
        await optimized_stt.init()
        await dual_streaming_tts.init()
        
        # 4. Initialize agent system
        logger.info("Setting up multi-agent system...")
        agent_registry = AgentRegistry()
        intelligent_router = IntelligentRouter()
        
        await agent_registry.init()
        await intelligent_router.init(agent_registry)
        
        # 5. Initialize core orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = MultiAgentOrchestrator(
            vector_store=hybrid_vector_store,
            agent_registry=agent_registry,
            router=intelligent_router,
            stt=optimized_stt,
            tts=dual_streaming_tts
        )
        await orchestrator.init()
        
        # 6. Initialize conversation manager
        logger.info("Setting up conversation management...")
        conversation_manager = EnhancedConversationManager(
            orchestrator=orchestrator,
            performance_tracker=performance_tracker
        )
        await conversation_manager.init()
        
        # 7. Initialize latency optimizer
        logger.info("Activating latency optimizer...")
        latency_optimizer = LatencyOptimizer(
            orchestrator=orchestrator,
            performance_tracker=performance_tracker
        )
        await latency_optimizer.init()
        
        # Store all components
        system_components.update({
            'orchestrator': orchestrator,
            'conversation_manager': conversation_manager,
            'latency_optimizer': latency_optimizer,
            'vector_store': hybrid_vector_store,
            'agent_registry': agent_registry,
            'router': intelligent_router,
            'stt': optimized_stt,
            'tts': dual_streaming_tts,
            'redis_cache': redis_cache,
            'qdrant_manager': qdrant_manager
        })
        
        # Load default agents
        await load_default_agents()
        
        logger.info("✅ Multi-Agent Voice AI System initialized successfully!")
        
        # Start background optimization tasks
        asyncio.create_task(run_optimization_loop())
        asyncio.create_task(run_analytics_loop())
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}", exc_info=True)
        raise

async def load_default_agents():
    """Load the three specialized agents with hot deployment."""
    agent_registry = system_components['agent_registry']
    
    # Define default agents
    default_agents = [
        {
            'agent_id': 'roadside-assistance',
            'version': '1.0.0',
            'config_file': 'app/config/agents/roadside_agent.yaml'
        },
        {
            'agent_id': 'billing-support',
            'version': '1.0.0', 
            'config_file': 'app/config/agents/billing_agent.yaml'
        },
        {
            'agent_id': 'technical-support',
            'version': '1.0.0',
            'config_file': 'app/config/agents/technical_agent.yaml'
        }
    ]
    
    for agent_config in default_agents:
        try:
            await agent_registry.deploy_agent(
                agent_id=agent_config['agent_id'],
                version=agent_config['version'],
                config_file=agent_config['config_file'],
                hot_deploy=True
            )
            logger.info(f"✅ Deployed agent: {agent_config['agent_id']}")
        except Exception as e:
            logger.error(f"❌ Failed to deploy agent {agent_config['agent_id']}: {e}")

async def cleanup_system():
    """Clean up all system components gracefully."""
    logger.info("Starting graceful system shutdown...")
    
    try:
        # Close active sessions
        for session_id, session in list(active_sessions.items()):
            try:
                await session.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
        
        # Shutdown system components
        for name, component in system_components.items():
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                elif hasattr(component, 'cleanup'):
                    await component.cleanup()
                logger.info(f"✅ Cleaned up {name}")
            except Exception as e:
                logger.error(f"❌ Error cleaning up {name}: {e}")
        
        # Final monitoring cleanup
        if performance_tracker:
            await performance_tracker.shutdown()
        if business_analytics:
            await business_analytics.shutdown()
        if alerting_system:
            await alerting_system.shutdown()
            
        logger.info("✅ System shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during system shutdown: {e}", exc_info=True)

async def run_optimization_loop():
    """Background loop for continuous system optimization."""
    while True:
        try:
            if 'latency_optimizer' in system_components:
                await system_components['latency_optimizer'].optimize_system()
            await asyncio.sleep(LatencyConfig.OPTIMIZATION_INTERVAL)
        except Exception as e:
            logger.error(f"Error in optimization loop: {e}")
            await asyncio.sleep(30)  # Wait before retrying

async def run_analytics_loop():
    """Background loop for business analytics collection."""
    while True:
        try:
            if business_analytics:
                await business_analytics.collect_metrics()
            await asyncio.sleep(60)  # Run every minute
        except Exception as e:
            logger.error(f"Error in analytics loop: {e}")
            await asyncio.sleep(60)

# Root endpoint
@app.get("/")
async def root():
    """System status endpoint."""
    return {
        "system": "Multi-Agent Voice AI",
        "version": "3.0.0",
        "status": "operational",
        "timestamp": time.time(),
        "active_sessions": len(active_sessions),
        "agents_deployed": len(system_components.get('agent_registry', {}).get_active_agents() if 'agent_registry' in system_components else [])
    }

# Health check endpoint
@app.get("/health", response_model=SystemHealthModel)
async def health_check():
    """Comprehensive system health check."""
    components_status = {}
    performance_metrics = {}
    
    # Check all system components
    for name, component in system_components.items():
        try:
            if hasattr(component, 'health_check'):
                components_status[name] = await component.health_check()
            else:
                components_status[name] = True
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            components_status[name] = False
    
    # Get performance metrics
    if performance_tracker:
        performance_metrics = await performance_tracker.get_current_metrics()
    
    # Determine overall status
    overall_status = "healthy" if all(components_status.values()) else "degraded"
    
    return SystemHealthModel(
        status=overall_status,
        timestamp=time.time(),
        components=components_status,
        performance_metrics=performance_metrics,
        active_sessions=len(active_sessions)
    )

# Voice call endpoints
@app.post("/voice/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming voice calls with multi-agent routing."""
    try:
        # Parse Twilio webhook data
        form_data = await request.form()
        call_sid = form_data.get('CallSid')
        from_number = form_data.get('From')
        to_number = form_data.get('To')
        
        logger.info(f"📞 Incoming call: {call_sid} from {from_number} to {to_number}")
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Create WebSocket URL for media streaming
        ws_url = f"{ProductionConfig.BASE_URL.replace('https://', 'wss://')}/ws/voice/{call_sid}"
        
        # Set up media stream
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"
        )
        connect.append(stream)
        response.append(connect)
        
        # Track call in business analytics
        if business_analytics:
            await business_analytics.track_call_initiated(call_sid, from_number, to_number)
        
        return HTMLResponse(content=str(response), media_type="text/xml")
        
    except Exception as e:
        logger.error(f"❌ Error handling incoming call: {e}", exc_info=True)
        
        # Return error TwiML
        response = VoiceResponse()
        response.say("I'm sorry, there's a technical issue. Please try again later.")
        response.hangup()
        
        return HTMLResponse(content=str(response), media_type="text/xml")

@app.websocket("/ws/voice/{call_sid}")
async def handle_voice_websocket(websocket: WebSocket, call_sid: str):
    """Handle WebSocket connection for voice processing with multi-agent capabilities."""
    session_id = str(uuid.uuid4())
    
    logger.info(f"🔗 WebSocket connection for call {call_sid}, session {session_id}")
    
    try:
        await websocket.accept()
        
        # Create voice session
        session = VoiceSession(
            session_id=session_id,
            call_sid=call_sid,
            websocket=websocket,
            conversation_manager=system_components['conversation_manager'],
            performance_tracker=performance_tracker
        )
        
        active_sessions[session_id] = session
        
        # Initialize session
        await session.initialize()
        
        # Handle WebSocket messages
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                await session.handle_message(data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
    finally:
        # Cleanup session
        if session_id in active_sessions:
            try:
                await active_sessions[session_id].cleanup()
                del active_sessions[session_id]
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

# Agent management endpoints
@app.post("/agents/deploy")
async def deploy_agent(deployment: AgentDeploymentModel):
    """Deploy or update an agent with hot deployment."""
    try:
        agent_registry = system_components['agent_registry']
        
        result = await agent_registry.deploy_agent(
            agent_id=deployment.agent_id,
            version=deployment.version,
            config=deployment.config,
            hot_deploy=deployment.hot_deploy
        )
        
        logger.info(f"✅ Agent deployed: {deployment.agent_id} v{deployment.version}")
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        logger.error(f"❌ Agent deployment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/status")
async def get_agents_status():
    """Get status of all deployed agents."""
    try:
        agent_registry = system_components['agent_registry']
        return await agent_registry.get_agents_status()
    except Exception as e:
        logger.error(f"Error getting agents status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance monitoring endpoints
@app.get("/metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics."""
    try:
        metrics = {}
        
        if performance_tracker:
            metrics['performance'] = await performance_tracker.get_detailed_metrics()
        
        if business_analytics:
            metrics['business'] = await business_analytics.get_analytics()
        
        # Add system metrics
        metrics['system'] = {
            'active_sessions': len(active_sessions),
            'uptime': time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            'components_health': {name: True for name in system_components.keys()}
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latency/report")
async def get_latency_report():
    """Get detailed latency analysis report."""
    try:
        if 'latency_optimizer' in system_components:
            return await system_components['latency_optimizer'].get_latency_report()
        else:
            raise HTTPException(status_code=503, detail="Latency optimizer not available")
    except Exception as e:
        logger.error(f"Error getting latency report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Session management endpoints
@app.get("/sessions")
async def get_active_sessions():
    """Get information about active voice sessions."""
    sessions_info = {}
    
    for session_id, session in active_sessions.items():
        try:
            sessions_info[session_id] = await session.get_status()
        except Exception as e:
            sessions_info[session_id] = {"error": str(e)}
    
    return {
        "active_sessions": len(active_sessions),
        "sessions": sessions_info
    }

@app.delete("/sessions/{session_id}")
async def terminate_session(session_id: str):
    """Terminate a specific voice session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        await active_sessions[session_id].cleanup()
        del active_sessions[session_id]
        return {"status": "terminated"}
    except Exception as e:
        logger.error(f"Error terminating session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class VoiceSession:
    """Enhanced voice session with multi-agent capabilities."""
    
    def __init__(self, session_id: str, call_sid: str, websocket: WebSocket, 
                 conversation_manager, performance_tracker):
        self.session_id = session_id
        self.call_sid = call_sid
        self.websocket = websocket
        self.conversation_manager = conversation_manager
        self.performance_tracker = performance_tracker
        
        self.stream_sid = None
        self.start_time = time.time()
        self.message_count = 0
        self.current_agent = None
        
    async def initialize(self):
        """Initialize the voice session."""
        logger.info(f"Initializing voice session {self.session_id}")
        
        # Create conversation context
        await self.conversation_manager.create_session(
            session_id=self.session_id,
            call_sid=self.call_sid
        )
        
    async def handle_message(self, data: Dict[str, Any]):
        """Handle WebSocket message with performance tracking."""
        start_time = time.time()
        self.message_count += 1
        
        try:
            event_type = data.get('event')
            
            if event_type == 'connected':
                logger.info(f"WebSocket connected for session {self.session_id}")
                
            elif event_type == 'start':
                self.stream_sid = data.get('streamSid')
                logger.info(f"Stream started: {self.stream_sid}")
                
                # Send welcome message
                await self.send_welcome_message()
                
            elif event_type == 'media':
                # Process audio with multi-agent system
                await self.process_audio(data)
                
            elif event_type == 'stop':
                logger.info(f"Stream stopped for session {self.session_id}")
                await self.cleanup()
                
        except Exception as e:
            logger.error(f"Error handling message in session {self.session_id}: {e}")
        finally:
            # Track message processing time
            processing_time = time.time() - start_time
            if self.performance_tracker:
                await self.performance_tracker.track_message_processing(
                    session_id=self.session_id,
                    event_type=event_type,
                    processing_time=processing_time
                )
    
    async def process_audio(self, data: Dict[str, Any]):
        """Process audio through multi-agent system."""
        try:
            # Extract audio payload
            media = data.get('media', {})
            payload = media.get('payload')
            
            if not payload:
                return
            
            # Process through conversation manager
            response = await self.conversation_manager.process_audio(
                session_id=self.session_id,
                audio_payload=payload
            )
            
            if response and response.get('audio_response'):
                await self.send_audio(response['audio_response'])
                
                # Track agent used
                if response.get('agent_used'):
                    self.current_agent = response['agent_used']
                    
        except Exception as e:
            logger.error(f"Error processing audio in session {self.session_id}: {e}")
    
    async def send_welcome_message(self):
        """Send welcome message through TTS."""
        try:
            welcome_text = "Hello! I'm your AI assistant. How can I help you today?"
            
            # Process through TTS
            if 'tts' in system_components:
                audio_data = await system_components['tts'].synthesize(
                    text=welcome_text,
                    optimize_for='telephony'
                )
                
                await self.send_audio(audio_data)
                
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data through WebSocket."""
        try:
            if not self.stream_sid:
                return
            
            # Split into chunks for streaming
            chunk_size = 400  # Optimal for telephony
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                
                import base64
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(chunk).decode('utf-8')}
                }
                
                await self.websocket.send_text(json.dumps(message))
                await asyncio.sleep(0.02)  # Pacing for smooth playback
                
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
    
    async def get_status(self):
        """Get session status information."""
        return {
            "session_id": self.session_id,
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time,
            "message_count": self.message_count,
            "current_agent": self.current_agent
        }
    
    async def cleanup(self):
        """Clean up session resources."""
        logger.info(f"Cleaning up session {self.session_id}")
        
        try:
            # Cleanup conversation manager session
            if self.conversation_manager:
                await self.conversation_manager.cleanup_session(self.session_id)
            
            # Track session completion
            if self.performance_tracker:
                await self.performance_tracker.track_session_completion(
                    session_id=self.session_id,
                    duration=time.time() - self.start_time,
                    message_count=self.message_count
                )
                
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with logging."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "request_id": getattr(request.state, 'request_id', 'unknown')}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with logging."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Unexpected error [req:{request_id}]: {exc}", exc_info=True)
    
    # Alert system administrators
    if alerting_system:
        await alerting_system.send_alert(
            level="critical",
            message=f"Unexpected system error in request {request_id}: {str(exc)}"
        )
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": request_id}
    )

if __name__ == "__main__":
    # Set start time for uptime tracking
    app.state.start_time = time.time()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🚀 Starting Multi-Agent Voice AI System...")
    
    # Production server configuration
    uvicorn.run(
        "app.main:app",
        host=ProductionConfig.HOST,
        port=ProductionConfig.PORT,
        reload=ProductionConfig.DEBUG,
        workers=ProductionConfig.WORKERS,
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False
    )