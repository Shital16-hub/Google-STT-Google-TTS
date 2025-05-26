#!/usr/bin/env python3
"""
Complete Multi-Agent Voice AI System - FastAPI Application
Implements comprehensive system transformation with <650ms end-to-end latency target
"""
import os
import sys
import asyncio
import logging
import json
import time
import uuid
from typing import Dict, Any, Optional, List

# FastAPI imports
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from pydantic import BaseModel

# Twilio imports
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from dotenv import load_dotenv

# Core system imports
from app.core.orchestrator import MultiAgentOrchestrator
from app.core.latency_optimizer import LatencyOptimizer
from app.core.conversation_manager import EnhancedConversationManager

# Load environment variables
load_dotenv()

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/voice_ai_agent.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# FastAPI app setup with optimization
app = FastAPI(
    title="Multi-Agent Voice AI System",
    description="Complete multi-agent transformation with <650ms latency target",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware for performance
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system components
orchestrator: Optional[MultiAgentOrchestrator] = None
latency_optimizer: Optional[LatencyOptimizer] = None
base_url: Optional[str] = None

# Enhanced session management
active_sessions = {}  # call_sid -> session data
session_stats = {}    # session_id -> performance stats
initialization_complete = asyncio.Event()

# Performance tracking
system_metrics = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "average_latency": 0.0,
    "agent_routing_stats": {},
    "start_time": time.time()
}

class CallSession(BaseModel):
    """Enhanced call session model"""
    call_sid: str
    session_id: str
    start_time: float
    agent_assignments: List[str] = []
    conversation_turns: int = 0
    total_latency: float = 0.0
    error_count: int = 0
    status: str = "active"

class SystemHealthResponse(BaseModel):
    """System health response model"""
    status: str
    timestamp: float
    components: Dict[str, Any]
    performance: Dict[str, Any]
    agents: Dict[str, Any]

async def initialize_multi_agent_system():
    """Initialize the complete multi-agent system"""
    global orchestrator, latency_optimizer, base_url
    
    logger.info("🚀 Initializing Multi-Agent Voice AI System...")
    
    # Validate environment
    base_url = os.getenv('BASE_URL')
    if not base_url:
        raise ValueError("BASE_URL environment variable must be set")
    
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not google_creds or not os.path.exists(google_creds):
        raise FileNotFoundError(f"Google credentials file not found: {google_creds}")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    pinecone_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY environment variable must be set")
    
    try:
        # Initialize latency optimizer first
        latency_optimizer = LatencyOptimizer()
        await latency_optimizer.init()
        
        # Initialize multi-agent orchestrator
        orchestrator = MultiAgentOrchestrator(
            latency_optimizer=latency_optimizer,
            credentials_file=google_creds
        )
        await orchestrator.init()
        
        # Load and deploy agents
        await orchestrator.load_agents_from_config("app/config/agents/")
        
        logger.info("✅ Multi-Agent System initialized successfully")
        logger.info(f"📊 Loaded {len(orchestrator.active_agents)} agents")
        logger.info(f"🎯 Target latency: <650ms end-to-end")
        
        initialization_complete.set()
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with system initialization"""
    try:
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("app/config/agents", exist_ok=True)
        os.makedirs("cache", exist_ok=True)
        
        # Start initialization
        asyncio.create_task(initialize_multi_agent_system())
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced shutdown with proper cleanup"""
    logger.info("🔄 Shutting down Multi-Agent Voice AI System...")
    
    # Clean up active sessions
    for session_id, session in list(active_sessions.items()):
        try:
            if 'handler' in session:
                await session['handler'].cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    # Shutdown orchestrator
    if orchestrator:
        await orchestrator.shutdown()
    
    # Shutdown latency optimizer
    if latency_optimizer:
        await latency_optimizer.shutdown()
    
    logger.info("✅ System shutdown complete")

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Enhanced root endpoint with system status"""
    uptime = time.time() - system_metrics["start_time"]
    
    return {
        "service": "Multi-Agent Voice AI System",
        "version": "3.0.0",
        "status": "operational" if initialization_complete.is_set() else "initializing",
        "uptime_seconds": round(uptime, 2),
        "active_sessions": len(active_sessions),
        "total_calls": system_metrics["total_calls"],
        "success_rate": (system_metrics["successful_calls"] / max(system_metrics["total_calls"], 1)) * 100,
        "target_latency": "<650ms",
        "features": {
            "multi_agent_orchestration": True,
            "hot_deployment": True,
            "latency_optimization": True,
            "agent_specialization": True,
            "tool_integration": True
        }
    }

@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """Comprehensive health check with agent status"""
    initialized = initialization_complete.is_set()
    current_time = time.time()
    
    # Component health
    components = {
        "orchestrator": orchestrator is not None and orchestrator.initialized,
        "latency_optimizer": latency_optimizer is not None,
        "active_sessions": len(active_sessions),
        "initialization_complete": initialized
    }
    
    # Performance metrics
    performance = {
        "uptime": current_time - system_metrics["start_time"],
        "total_calls": system_metrics["total_calls"],
        "success_rate": (system_metrics["successful_calls"] / max(system_metrics["total_calls"], 1)) * 100,
        "average_latency": system_metrics["average_latency"],
        "memory_usage_mb": 0  # Would be implemented with psutil
    }
    
    # Agent status
    agents = {}
    if orchestrator and orchestrator.initialized:
        for agent_id, agent in orchestrator.active_agents.items():
            agents[agent_id] = {
                "status": "active",
                "version": agent.get("version", "unknown"),
                "last_used": agent.get("last_used", 0),
                "total_queries": agent.get("total_queries", 0),
                "success_rate": agent.get("success_rate", 0)
            }
    
    # Add latency metrics if available
    if latency_optimizer:
        performance.update(await latency_optimizer.get_current_metrics())
    
    return SystemHealthResponse(
        status="healthy" if initialized and all(components.values()) else "degraded",
        timestamp=current_time,
        components=components,
        performance=performance,
        agents=agents
    )

@app.post("/voice/incoming")
async def handle_incoming_call(request: Request, background_tasks: BackgroundTasks):
    """Enhanced incoming call handler with agent pre-selection"""
    logger.info("📞 Incoming call received")
    
    # Wait for system initialization
    try:
        await asyncio.wait_for(initialization_complete.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("❌ System initialization timeout")
        return HTMLResponse(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">System is initializing. Please try again in a few moments.</Say>
    <Hangup/>
</Response>''',
            media_type="text/xml"
        )
    
    if not orchestrator or not orchestrator.initialized:
        logger.error("❌ System not initialized")
        return HTMLResponse(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Service temporarily unavailable. Please try again later.</Say>
    <Hangup/>
</Response>''',
            media_type="text/xml"
        )
    
    # Parse Twilio form data
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    from_number = form_data.get('From')
    to_number = form_data.get('To')
    
    logger.info(f"📞 Call {call_sid}: {from_number} → {to_number}")
    
    # Create session
    session_id = str(uuid.uuid4())
    session = CallSession(
        call_sid=call_sid,
        session_id=session_id,
        start_time=time.time()
    )
    
    active_sessions[call_sid] = {
        "session": session,
        "metadata": {
            "from": from_number,
            "to": to_number,
            "start_time": time.time()
        }
    }
    
    # Update system metrics
    system_metrics["total_calls"] += 1
    
    try:
        # Create optimized TwiML
        response = VoiceResponse()
        
        # WebSocket URL for streaming
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        
        # Enhanced streaming configuration
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"
        )
        
        # Add stream parameters for optimization
        stream.parameter(name="latencyOptimization", value="true")
        stream.parameter(name="sessionId", value=session_id)
        
        connect.append(stream)
        response.append(connect)
        
        logger.info(f"✅ TwiML generated for call {call_sid}")
        
        return HTMLResponse(
            content=str(response),
            media_type="text/xml"
        )
        
    except Exception as e:
        logger.error(f"❌ Error handling call {call_sid}: {e}")
        system_metrics["failed_calls"] += 1
        
        # Update session status
        if call_sid in active_sessions:
            active_sessions[call_sid]["metadata"]["error"] = str(e)
        
        return HTMLResponse(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">We're experiencing technical difficulties. Please try again later.</Say>
    <Hangup/>
</Response>''',
            media_type="text/xml"
        )

@app.post("/voice/status")
async def handle_status_callback(request: Request, background_tasks: BackgroundTasks):
    """Enhanced status callback with session analytics"""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    call_status = form_data.get('CallStatus')
    call_duration = form_data.get('CallDuration', '0')
    
    logger.info(f"📊 Call {call_sid} status: {call_status}, duration: {call_duration}s")
    
    # Update session
    if call_sid in active_sessions:
        session_data = active_sessions[call_sid]
        session_data["metadata"].update({
            "status": call_status,
            "duration": call_duration,
            "end_time": time.time()
        })
        
        # Calculate session metrics
        if call_status in ['completed', 'failed']:
            session = session_data["session"]
            total_time = time.time() - session.start_time
            
            # Store session stats
            session_stats[session.session_id] = {
                "duration": total_time,
                "turns": session.conversation_turns,
                "latency": session.total_latency,
                "agent_changes": len(session.agent_assignments),
                "success": call_status == 'completed'
            }
            
            # Update system metrics
            if call_status == 'completed':
                system_metrics["successful_calls"] += 1
            else:
                system_metrics["failed_calls"] += 1
            
            # Cleanup session
            background_tasks.add_task(cleanup_session, call_sid)
    
    return Response(status_code=204)

async def cleanup_session(call_sid: str):
    """Clean up session resources"""
    if call_sid in active_sessions:
        session_data = active_sessions[call_sid]
        
        # Cleanup handler if exists
        if 'handler' in session_data:
            try:
                await session_data['handler'].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up handler for {call_sid}: {e}")
        
        # Remove from active sessions
        del active_sessions[call_sid]
        logger.info(f"🧹 Cleaned up session {call_sid}")

@app.websocket("/ws/stream/{call_sid}")
async def handle_media_stream(websocket: WebSocket, call_sid: str):
    """Enhanced WebSocket handler with multi-agent orchestration"""
    logger.info(f"🔗 WebSocket connection request for call {call_sid}")
    
    # Wait for initialization
    try:
        await asyncio.wait_for(initialization_complete.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        await websocket.close(code=1013, reason="Service unavailable")
        return
    
    if not orchestrator or not orchestrator.initialized:
        await websocket.close(code=1013, reason="Service not ready")
        return
    
    # Accept connection
    await websocket.accept()
    logger.info(f"✅ WebSocket connected for call {call_sid}")
    
    # Get session
    session_data = active_sessions.get(call_sid)
    if not session_data:
        logger.error(f"❌ Session not found for call {call_sid}")
        await websocket.close(code=1008, reason="Session not found")
        return
    
    session = session_data["session"]
    handler = None
    
    try:
        # Create enhanced conversation handler
        handler = await orchestrator.create_conversation_handler(
            call_sid=call_sid,
            session_id=session.session_id,
            websocket=websocket
        )
        
        # Store handler reference
        session_data["handler"] = handler
        
        # Start latency tracking
        if latency_optimizer:
            await latency_optimizer.start_session_tracking(session.session_id)
        
        # Main message processing loop
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                if not message:
                    continue
                
                # Parse message
                try:
                    data = json.loads(message)
                    event_type = data.get('event')
                    
                    if event_type == 'connected':
                        logger.info(f"🔗 Stream connected for {call_sid}")
                        
                    elif event_type == 'start':
                        stream_sid = data.get('streamSid')
                        logger.info(f"🎵 Stream started: {stream_sid}")
                        
                        # Initialize conversation
                        await handler.start_conversation(stream_sid)
                        
                        # Update session
                        session.status = "streaming"
                        
                    elif event_type == 'media':
                        # Process audio with latency tracking
                        start_time = time.time()
                        
                        await handler.process_audio_chunk(data)
                        
                        # Track latency
                        if latency_optimizer:
                            processing_time = (time.time() - start_time) * 1000
                            await latency_optimizer.record_processing_time(
                                session.session_id, 
                                "audio_processing", 
                                processing_time
                            )
                        
                    elif event_type == 'stop':
                        logger.info(f"🛑 Stream stopped for {call_sid}")
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received for {call_sid}")
                    continue
                    
            except asyncio.TimeoutError:
                # Check connection health
                continue
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected for {call_sid}")
                break
            except Exception as e:
                logger.error(f"❌ Error processing message for {call_sid}: {e}")
                break
    
    except Exception as e:
        logger.error(f"❌ WebSocket handler error for {call_sid}: {e}")
    
    finally:
        # Cleanup
        if handler:
            try:
                await handler.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up handler: {e}")
        
        # Stop latency tracking
        if latency_optimizer and session:
            await latency_optimizer.stop_session_tracking(session.session_id)
        
        # Update session
        if session:
            session.status = "completed"
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info(f"🧹 WebSocket cleanup complete for {call_sid}")

@app.get("/agents")
async def list_agents():
    """List all available agents"""
    if not orchestrator or not orchestrator.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    agents = {}
    for agent_id, agent in orchestrator.active_agents.items():
        agents[agent_id] = {
            "id": agent_id,
            "name": agent.get("display_name", agent_id),
            "version": agent.get("version", "unknown"),
            "status": agent.get("status", "unknown"),
            "specialization": agent.get("specialization", {}),
            "tools": list(agent.get("tools", {}).keys()),
            "performance": {
                "total_queries": agent.get("total_queries", 0),
                "success_rate": agent.get("success_rate", 0),
                "avg_response_time": agent.get("avg_response_time", 0)
            }
        }
    
    return {
        "total_agents": len(agents),
        "agents": agents
    }

@app.post("/agents/{agent_id}/deploy")
async def deploy_agent(agent_id: str, config: Dict[str, Any]):
    """Hot deploy a new agent"""
    if not orchestrator or not orchestrator.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await orchestrator.deploy_agent(agent_id, config)
        return {
            "status": "success",
            "agent_id": agent_id,
            "deployment_time": result.get("deployment_time", 0),
            "message": f"Agent {agent_id} deployed successfully"
        }
    except Exception as e:
        logger.error(f"Error deploying agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics"""
    metrics = {
        "system": system_metrics.copy(),
        "timestamp": time.time()
    }
    
    # Add latency metrics
    if latency_optimizer:
        metrics["latency"] = await latency_optimizer.get_current_metrics()
    
    # Add agent metrics
    if orchestrator and orchestrator.initialized:
        metrics["agents"] = {}
        for agent_id, agent in orchestrator.active_agents.items():
            metrics["agents"][agent_id] = {
                "queries": agent.get("total_queries", 0),
                "success_rate": agent.get("success_rate", 0),
                "avg_latency": agent.get("avg_latency", 0)
            }
    
    # Add session metrics
    metrics["sessions"] = {
        "active": len(active_sessions),
        "completed": len(session_stats),
        "success_rate": sum(1 for s in session_stats.values() if s["success"]) / max(len(session_stats), 1) * 100
    }
    
    return metrics

@app.get("/performance")
async def get_performance_analysis():
    """Get detailed performance analysis"""
    if not latency_optimizer:
        raise HTTPException(status_code=503, detail="Latency optimizer not available")
    
    return await latency_optimizer.get_performance_analysis()

if __name__ == '__main__':
    print("🚀 Starting Multi-Agent Voice AI System...")
    print(f"📊 Target latency: <650ms end-to-end")
    print(f"🔗 Base URL: {os.getenv('BASE_URL', 'Not set')}")
    
    uvicorn.run(
        "app.main:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000)),
        reload=os.getenv('DEBUG', 'False').lower() == 'true',
        log_level="info",
        workers=1,
        access_log=True
    )