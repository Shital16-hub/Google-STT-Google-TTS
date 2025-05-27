#!/usr/bin/env python3
"""
Revolutionary Multi-Agent Voice AI System - Main FastAPI Application
Enhanced with LangGraph orchestration, hybrid vector architecture, and hot agent deployment.
Target: <377ms end-to-end latency with 84% improvement over current system.
"""
import os
import sys
import asyncio
import logging
import json
import time
import threading
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import uuid

# FastAPI imports with enhanced performance
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from pydantic import BaseModel

# Twilio integration
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream, Say
from dotenv import load_dotenv

# Core system imports
from app.core.agent_orchestrator import MultiAgentOrchestrator
from app.core.state_manager import ConversationStateManager
from app.core.health_monitor import SystemHealthMonitor

# Agent and vector systems
from app.agents.registry import AgentRegistry
from app.agents.router import IntelligentAgentRouter
from app.vector_db.hybrid_vector_system import HybridVectorSystem

# Voice processing with enhanced pipeline
from app.voice.enhanced_stt import EnhancedSTTSystem
from app.voice.dual_streaming_tts import DualStreamingTTSEngine
from app.telephony.advanced_websocket_handler import AdvancedWebSocketHandler

# Tool orchestration
from app.tools.tool_orchestrator import ComprehensiveToolOrchestrator

# Load environment variables
load_dotenv()

# Ensure logs directory exists
try:
    os.makedirs('./logs', exist_ok=True)
    log_handlers = [
        logging.StreamHandler(),
        logging.FileHandler('./logs/voice_ai_system.log')
    ]
except (OSError, PermissionError) as e:
    print(f"Warning: Could not create log file: {e}")
    # Fall back to console logging only
    log_handlers = [logging.StreamHandler()]

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# Suppress noisy logs for production performance
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# System components - global for performance
orchestrator: Optional[MultiAgentOrchestrator] = None
state_manager: Optional[ConversationStateManager] = None
health_monitor: Optional[SystemHealthMonitor] = None
agent_registry: Optional[AgentRegistry] = None
agent_router: Optional[IntelligentAgentRouter] = None
hybrid_vector_system: Optional[HybridVectorSystem] = None
stt_system: Optional[EnhancedSTTSystem] = None
tts_engine: Optional[DualStreamingTTSEngine] = None
tool_orchestrator: Optional[ComprehensiveToolOrchestrator] = None

# Configuration and state
BASE_URL = None
SYSTEM_INITIALIZED = False
initialization_complete = asyncio.Event()

# Active sessions tracking with enhanced metrics
active_sessions = {}
session_metrics = {
    "total_sessions": 0,
    "active_count": 0,
    "avg_latency_ms": 0.0,
    "success_rate": 0.0,
    "agent_usage": {},
    "tool_usage": {},
    "error_count": 0
}

# Request/Response models
class AgentDeploymentRequest(BaseModel):
    agent_config: Dict[str, Any]
    deployment_strategy: str = "blue_green"
    health_check_enabled: bool = True

class ConversationRequest(BaseModel):
    session_id: str
    input_text: str
    context: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: float
    components: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    active_sessions: int
    system_version: str = "2.0.0"

async def initialize_revolutionary_system():
    """Initialize the complete multi-agent system with all enhancements."""
    global orchestrator, state_manager, health_monitor, agent_registry, agent_router
    global hybrid_vector_system, stt_system, tts_engine, tool_orchestrator
    global BASE_URL, SYSTEM_INITIALIZED
    
    logger.info("🚀 Initializing Revolutionary Multi-Agent Voice AI System...")
    start_time = time.time()
    
    try:
        # Validate environment
        BASE_URL = os.getenv('BASE_URL')
        if not BASE_URL:
            raise ValueError("BASE_URL environment variable must be set")
        
        # 1. Initialize Hybrid Vector System (Tier 1: Redis + FAISS + Qdrant)
        logger.info("📊 Initializing 3-tier hybrid vector architecture...")
        hybrid_vector_system = HybridVectorSystem(
            redis_config={
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "cache_size": 50000,
                "ttl_seconds": 3600
            },
            faiss_config={
                "memory_limit_gb": 4,
                "promotion_threshold": 100,
                "index_type": "HNSW"
            },
            qdrant_config = {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", "6334")),
                "prefer_grpc": False,  # ← CHANGED: Force HTTP connection
                "timeout": 5.0
            }
        )
        await hybrid_vector_system.initialize()
        
        # 2. Initialize Enhanced STT System
        logger.info("🎤 Initializing enhanced STT with voice activity detection...")
        stt_system = EnhancedSTTSystem(
            primary_provider="google_cloud_v2",
            backup_provider="assemblyai",
            enable_vad=True,
            enable_echo_cancellation=True,
            target_latency_ms=80
        )
        await stt_system.initialize()
        
        # 3. Initialize Dual Streaming TTS Engine
        logger.info("🔊 Initializing dual streaming TTS engine...")
        tts_engine = DualStreamingTTSEngine()

        await tts_engine.initialize()
        
        # 4. Initialize Comprehensive Tool Orchestrator
        logger.info("🛠️ Initializing tool orchestration framework...")
        tool_orchestrator = ComprehensiveToolOrchestrator(
            enable_business_workflows=True,
            enable_external_apis=True,
            dummy_mode=True,  # Enable dummy mode for development
            max_concurrent_tools=10
        )
        await tool_orchestrator.initialize()
        
        # 5. Initialize Agent Registry with Hot Deployment
        logger.info("🤖 Initializing agent registry with hot deployment...")
        agent_registry = AgentRegistry(
            hybrid_vector_system=hybrid_vector_system,
            tool_orchestrator=tool_orchestrator,
            deployment_strategy="blue_green",
            enable_health_checks=True
        )
        await agent_registry.initialize()
        
        # 6. Initialize Intelligent Agent Router
        logger.info("🧠 Initializing ML-based intelligent agent router...")
        agent_router = IntelligentAgentRouter(
            agent_registry=agent_registry,
            hybrid_vector_system=hybrid_vector_system,
            confidence_threshold=0.85,
            fallback_threshold=0.6,
            enable_ml_routing=True
        )
        await agent_router.initialize()
        
        # 7. Initialize Conversation State Manager
        logger.info("💾 Initializing stateful conversation management...")
        state_manager = ConversationStateManager(
            redis_client=hybrid_vector_system.redis_cache.client,
            enable_persistence=True,
            max_context_length=2048,
            context_compression="intelligent_summarization"
        )
        await state_manager.initialize()
        
        # 8. Initialize Multi-Agent Orchestrator (LangGraph)
        logger.info("🎭 Initializing LangGraph multi-agent orchestrator...")
        orchestrator = MultiAgentOrchestrator(
            agent_registry=agent_registry,
            agent_router=agent_router,
            state_manager=state_manager,
            hybrid_vector_system=hybrid_vector_system,
            stt_system=stt_system,
            tts_engine=tts_engine,
            tool_orchestrator=tool_orchestrator,
            target_latency_ms=377
        )
        await orchestrator.initialize()
        
        # 9. Initialize System Health Monitor
        logger.info("📈 Initializing comprehensive health monitoring...")
        health_monitor = SystemHealthMonitor(
            orchestrator=orchestrator,
            hybrid_vector_system=hybrid_vector_system,
            target_latency_ms=377,
            enable_predictive_analytics=True,
            alert_thresholds={
                "latency_ms": 500,
                "error_rate": 0.02,
                "memory_usage": 0.85
            }
        )
        await health_monitor.initialize()
        
        # 10. Deploy default specialized agents
        logger.info("🚀 Deploying specialized agents...")
        await deploy_default_agents()
        
        # Mark system as initialized
        SYSTEM_INITIALIZED = True
        initialization_time = time.time() - start_time
        
        logger.info(f"✅ Revolutionary Multi-Agent System initialized in {initialization_time:.2f}s")
        logger.info(f"🎯 Target end-to-end latency: <377ms")
        logger.info(f"🔄 Agents deployed: {len(await agent_registry.list_active_agents())}")
        
        # Set initialization event
        initialization_complete.set()
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}", exc_info=True)
        raise

async def deploy_default_agents():
    """Deploy the three core specialized agents."""
    logger.info("Deploying core specialized agents...")
    
    # Deploy Roadside Assistance Agent
    roadside_config = {
        "agent_id": "roadside-assistance-v2",
        "version": "2.1.0",
        "specialization": {
            "domain_expertise": "emergency_roadside_assistance",
            "personality_profile": "professional_urgent_empathetic",
            "voice_settings": {
                "tts_voice": "en-US-Neural2-C",
                "speaking_rate": 1.1,
                "emotion_detection": True,
                "stress_response_mode": True
            }
        },
        "tools": [
            "dispatch_tow_truck_workflow",
            "emergency_escalation_workflow",
            "send_location_sms",
            "search_service_coverage"
        ],
        "routing": {
            "primary_keywords": ["tow", "stuck", "breakdown", "accident", "emergency", "stranded"],
            "confidence_threshold": 0.85
        }
    }
    
    # Deploy Billing Support Agent
    billing_config = {
        "agent_id": "billing-support-v2",
        "version": "2.1.0",
        "specialization": {
            "domain_expertise": "billing_and_payments",
            "personality_profile": "empathetic_solution_oriented",
            "voice_settings": {
                "tts_voice": "en-US-Neural2-J",
                "speaking_rate": 1.0,
                "tone_adaptation": "financial_empathy"
            }
        },
        "tools": [
            "stripe_payment_api",
            "process_refund_workflow",
            "update_subscription_workflow",
            "billing_inquiry_search"
        ],
        "routing": {
            "primary_keywords": ["payment", "refund", "bill", "charge", "subscription"],
            "confidence_threshold": 0.80
        }
    }
    
    # Deploy Technical Support Agent
    technical_config = {
        "agent_id": "technical-support-v2",
        "version": "2.1.0",
        "specialization": {
            "domain_expertise": "technical_troubleshooting",
            "personality_profile": "patient_instructional_expert",
            "voice_settings": {
                "tts_voice": "en-US-Neural2-D",
                "speaking_rate": 0.9,
                "instructional_mode": True
            }
        },
        "tools": [
            "create_support_ticket",
            "run_diagnostics_workflow",
            "schedule_callback_workflow",
            "technical_knowledge_search"
        ],
        "routing": {
            "primary_keywords": ["not working", "error", "setup", "install", "technical"],
            "confidence_threshold": 0.75
        }
    }
    
    # Deploy agents with validation
    for config in [roadside_config, billing_config, technical_config]:
        try:
            result = await agent_registry.deploy_agent(config)
            if result.success:
                logger.info(f"✅ Deployed agent: {config['agent_id']}")
            else:
                logger.error(f"❌ Failed to deploy agent {config['agent_id']}: {result.error}")
        except Exception as e:
            logger.error(f"❌ Error deploying agent {config['agent_id']}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper initialization and cleanup."""
    # Startup
    logger.info("🚀 Starting Revolutionary Multi-Agent Voice AI System...")
    try:
        # Initialize system in background to avoid blocking startup
        asyncio.create_task(initialize_revolutionary_system())
        yield
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Revolutionary Multi-Agent Voice AI System...")
        await cleanup_system()

async def cleanup_system():
    """Clean up all system resources."""
    global active_sessions
    
    # Clean up active sessions
    logger.info(f"Cleaning up {len(active_sessions)} active sessions...")
    for session_id, handler in list(active_sessions.items()):
        try:
            await handler.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    # Cleanup system components
    if health_monitor:
        await health_monitor.shutdown()
    if orchestrator:
        await orchestrator.shutdown()
    if hybrid_vector_system:
        await hybrid_vector_system.shutdown()
    
    logger.info("✅ System cleanup completed")

# FastAPI app with enhanced configuration
app = FastAPI(
    title="Revolutionary Multi-Agent Voice AI System",
    description="Ultra-low latency multi-agent conversation system with <377ms response time",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("DEBUG", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("DEBUG", "false").lower() == "true" else None
)

# Enhanced middleware for performance
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for system initialization check
async def ensure_system_initialized():
    """Ensure system is initialized before handling requests."""
    if not SYSTEM_INITIALIZED:
        # Wait for initialization with timeout
        try:
            await asyncio.wait_for(initialization_complete.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="System still initializing, please try again"
            )
    
    if not SYSTEM_INITIALIZED:
        raise HTTPException(
            status_code=503,
            detail="System initialization failed"
        )

@app.get("/", response_model=Dict[str, Any])
async def root():
    """System status and welcome endpoint."""
    return {
        "system": "Revolutionary Multi-Agent Voice AI System",
        "version": "2.0.0",
        "status": "operational" if SYSTEM_INITIALIZED else "initializing",
        "features": [
            "Multi-agent specialization with hot deployment",
            "Hybrid 3-tier vector architecture (Redis+FAISS+Qdrant)",
            "Enhanced STT/TTS with dual streaming",
            "LangGraph orchestration with stateful execution",
            "Intelligent agent routing with ML confidence scoring",
            "Advanced tool integration framework",
            "Real-time performance monitoring"
        ],
        "target_latency_ms": 377,
        "active_sessions": len(active_sessions),
        "timestamp": time.time()
    }

@app.get("/health", response_model=SystemHealthResponse)
async def comprehensive_health_check(
    _: None = Depends(ensure_system_initialized)
):
    """Comprehensive system health check with detailed metrics."""
    health_data = await health_monitor.get_comprehensive_health()
    
    return SystemHealthResponse(
        status=health_data["status"],
        timestamp=health_data["timestamp"],
        components=health_data["components"],
        performance_metrics=health_data["performance_metrics"],
        active_sessions=len(active_sessions)
    )

@app.post("/agents/deploy")
async def deploy_agent(
    request: AgentDeploymentRequest,
    _: None = Depends(ensure_system_initialized)
):
    """Deploy a new agent with hot deployment and validation."""
    try:
        result = await agent_registry.deploy_agent_with_validation(
            config=request.agent_config,
            deployment_strategy=request.deployment_strategy,
            health_check_enabled=request.health_check_enabled
        )
        
        if result.success:
            logger.info(f"✅ Successfully deployed agent: {result.agent_id}")
            return {
                "success": True,
                "agent_id": result.agent_id,
                "deployment_id": result.deployment_id,
                "version": result.version,
                "message": "Agent deployed successfully with zero downtime"
            }
        else:
            logger.error(f"❌ Agent deployment failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Error in agent deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/list")
async def list_active_agents(
    _: None = Depends(ensure_system_initialized)
):
    """List all active agents with their status and metrics."""
    agents = await agent_registry.list_active_agents()
    
    agent_list = []
    for agent in agents:
        agent_stats = await agent_registry.get_agent_stats(agent.agent_id)
        agent_list.append({
            "agent_id": agent.agent_id,
            "version": agent.version,
            "status": agent.status,
            "specialization": agent.specialization,
            "deployment_time": agent.deployment_time,
            "stats": agent_stats
        })
    
    return {
        "agents": agent_list,
        "total_count": len(agent_list),
        "timestamp": time.time()
    }

@app.post("/conversation")
async def process_conversation(
    request: ConversationRequest,
    _: None = Depends(ensure_system_initialized)
):
    """Process a conversation request through the multi-agent system."""
    try:
        start_time = time.time()
        
        # Process through orchestrator
        result = await orchestrator.process_conversation(
            session_id=request.session_id,
            input_text=request.input_text,
            context=request.context,
            preferred_agent_id=request.agent_id
        )
        
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Update session metrics
        session_metrics["total_sessions"] += 1
        session_metrics["avg_latency_ms"] = (
            (session_metrics["avg_latency_ms"] * (session_metrics["total_sessions"] - 1) + total_time) /
            session_metrics["total_sessions"]
        )
        
        logger.info(f"Conversation processed in {total_time:.2f}ms for session {request.session_id}")
        
        return {
            "success": True,
            "session_id": request.session_id,
            "response": result.response,
            "agent_used": result.agent_id,
            "confidence": result.confidence,
            "latency_ms": total_time,
            "tools_used": result.tools_used,
            "sources": result.sources,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error processing conversation: {e}", exc_info=True)
        session_metrics["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/incoming")
async def handle_incoming_call(
    request: Request,
    _: None = Depends(ensure_system_initialized)
):
    """Handle incoming voice calls with multi-agent TwiML generation."""
    logger.info("📞 Incoming call received")
    
    # Parse Twilio form data
    form_data = await request.form()
    from_number = form_data.get('From')
    to_number = form_data.get('To')
    call_sid = form_data.get('CallSid')
    
    logger.info(f"Call details - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
    
    try:
        # Generate optimized TwiML for multi-agent system
        response = VoiceResponse()
        
        # Create WebSocket URL for advanced handler
        ws_url = f'{BASE_URL.replace("https://", "wss://")}/ws/voice/{call_sid}'
        
        # Configure stream for optimal performance
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"
        )
        connect.append(stream)
        response.append(connect)
        
        logger.info(f"✅ Generated TwiML for multi-agent call: {call_sid}")
        
        return HTMLResponse(
            content=str(response),
            media_type="text/xml"
        )
        
    except Exception as e:
        logger.error(f"❌ Error handling incoming call: {e}", exc_info=True)
        
        # Fallback TwiML
        response = VoiceResponse()
        response.say("I'm sorry, our system is temporarily unavailable. Please try again later.")
        response.hangup()
        
        return HTMLResponse(
            content=str(response),
            media_type="text/xml"
        )

@app.websocket("/ws/voice/{call_sid}")
async def handle_voice_websocket(
    websocket: WebSocket,
    call_sid: str
):
    """Enhanced WebSocket handler for multi-agent voice conversations."""
    if not SYSTEM_INITIALIZED:
        await websocket.close(code=1013, reason="System not initialized")
        return
    
    logger.info(f"🔗 WebSocket connection for call: {call_sid}")
    
    handler = None
    try:
        # Accept connection
        await websocket.accept()
        
        # Create advanced WebSocket handler
        handler = AdvancedWebSocketHandler(
            call_sid=call_sid,
            orchestrator=orchestrator,
            state_manager=state_manager,
            target_latency_ms=377
        )
        
        # Register active session
        active_sessions[call_sid] = handler
        session_metrics["active_count"] = len(active_sessions)
        
        # Handle WebSocket communication
        await handler.handle_websocket_session(websocket)
        
    except WebSocketDisconnect:
        logger.info(f"📞 Call {call_sid} disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error for call {call_sid}: {e}", exc_info=True)
    finally:
        # Cleanup
        if handler:
            await handler.cleanup()
        
        if call_sid in active_sessions:
            del active_sessions[call_sid]
            session_metrics["active_count"] = len(active_sessions)
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info(f"🧹 Cleaned up call session: {call_sid}")

@app.get("/metrics")
async def get_system_metrics(
    _: None = Depends(ensure_system_initialized)
):
    """Get comprehensive system metrics and performance data."""
    metrics = await health_monitor.get_performance_metrics()
    
    return {
        "system_metrics": metrics,
        "session_metrics": session_metrics,
        "agent_metrics": await agent_registry.get_usage_metrics(),
        "vector_metrics": await hybrid_vector_system.get_performance_stats(),
        "tool_metrics": await tool_orchestrator.get_usage_statistics(),
        "timestamp": time.time()
    }

@app.get("/agents/{agent_id}/status")
async def get_agent_status(
    agent_id: str,
    _: None = Depends(ensure_system_initialized)
):
    """Get detailed status and metrics for a specific agent."""
    try:
        agent = await agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        stats = await agent_registry.get_agent_stats(agent_id)
        health = await agent_registry.get_agent_health(agent_id)
        
        return {
            "agent_id": agent_id,
            "status": agent.status,
            "version": agent.version,
            "specialization": agent.specialization,
            "health": health,
            "statistics": stats,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance monitoring endpoint
@app.get("/performance")
async def get_performance_dashboard(
    _: None = Depends(ensure_system_initialized)
):
    """Get comprehensive performance dashboard data."""
    dashboard = await health_monitor.generate_executive_dashboard()
    
    return {
        "dashboard": dashboard,
        "target_latency_ms": 377,
        "current_performance": {
            "avg_latency_ms": session_metrics["avg_latency_ms"],
            "success_rate": session_metrics.get("success_rate", 0.0),
            "active_sessions": len(active_sessions),
            "error_rate": session_metrics["error_count"] / max(session_metrics["total_sessions"], 1)
        },
        "system_health": await health_monitor.get_system_health_score(),
        "timestamp": time.time()
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    print("🚀 Starting Revolutionary Multi-Agent Voice AI System...")
    print(f"🎯 Target latency: <377ms (84% improvement)")
    print(f"🔧 Base URL: {os.getenv('BASE_URL', 'Not configured')}")
    print(f"📊 Vector DB: Hybrid 3-tier (Redis+FAISS+Qdrant)")
    print(f"🤖 Agents: Hot deployment with specialization")
    print(f"🛠️ Tools: Comprehensive orchestration framework")
    
    # Create logs directory
    os.makedirs('./logs', exist_ok=True)
    
    # Run with optimized settings
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
        workers=1,  # Single worker for stateful system
        loop="asyncio",
        http="httptools",
        lifespan="on"
    )