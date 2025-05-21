# main.py

"""
Main application entry point with FastAPI implementation.
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import Settings
from core.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from prompts.prompt_manager import PromptManager
from agents.router import AgentRouter
from services.dispatcher import DispatcherService
from api.twilio_routes import router as twilio_router
from api.health import router as health_router

# Load settings
settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
conversation_manager = None
query_engine = None
prompt_manager = None
agent_router = None
dispatcher_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global conversation_manager, query_engine, prompt_manager, agent_router, dispatcher_service
    
    # Startup
    logger.info("Initializing application components...")
    try:
        # Initialize knowledge base components
        query_engine = QueryEngine(config=settings.knowledge_base)
        await query_engine.init()
        
        conversation_manager = ConversationManager(
            query_engine=query_engine,
            config=settings.conversation
        )
        await conversation_manager.init()
        
        # Initialize prompt system
        prompt_manager = PromptManager(prompt_dir=settings.prompts_dir)
        
        # Initialize agent router
        agent_router = AgentRouter(
            conversation_manager=conversation_manager,
            query_engine=query_engine,
            prompt_manager=prompt_manager
        )
        
        # Initialize dispatcher service
        dispatcher_service = DispatcherService()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application components...")
    try:
        # Clean up conversation manager
        if conversation_manager:
            await conversation_manager.cleanup()
        
        # Clean up query engine
        if query_engine:
            await query_engine.cleanup()
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Roadside Assistance Voice AI",
    description="Voice AI system for roadside assistance services",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(twilio_router, prefix="/voice", tags=["voice"])
app.include_router(health_router, prefix="/health", tags=["health"])

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication."""
    try:
        await websocket.accept()
        active_connections[session_id] = websocket
        
        # Session tracking
        session_start = None
        current_agent = None
        
        try:
            session_start = asyncio.get_event_loop().time()
            logger.info(f"WebSocket connection established for session {session_id}")
            
            while True:
                # Receive message
                message = await websocket.receive_text()
                
                # Route message through agent system
                response = await agent_router.route_message(
                    session_id=session_id,
                    message=message
                )
                
                # Check if we need to handle handoff
                if response.get("requires_handoff"):
                    # Create service request
                    request_id = await dispatcher_service.create_service_request(
                        session_id=session_id,
                        agent_type=current_agent.agent_type if current_agent else None,
                        customer_info=response.get("collected_info", {}),
                        service_requirements=response.get("service_requirements", {}),
                        handoff_reason=response.get("handoff_reason", "unspecified")
                    )
                    
                    # Add request ID to response
                    response["service_request_id"] = request_id
                
                # Send response
                await websocket.send_json(response)
                
                # Update current agent if needed
                if response.get("agent_type"):
                    current_agent = agent_router.get_agent(
                        session_id,
                        response["agent_type"]
                    )
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        finally:
            # Clean up session
            if session_id in active_connections:
                del active_connections[session_id]
            
            # Clean up agent
            agent_router.cleanup_session(session_id)
            
            # Log session duration if available
            if session_start:
                duration = asyncio.get_event_loop().time() - session_start
                logger.info(f"Session {session_id} ended after {duration:.2f} seconds")
                
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "status": "running",
        "version": "2.0.0",
        "active_sessions": len(active_connections),
        "components": {
            "conversation_manager": conversation_manager is not None,
            "query_engine": query_engine is not None,
            "agent_router": agent_router is not None,
            "dispatcher_service": dispatcher_service is not None
        }
    }

@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics."""
    stats = {
        "active_sessions": len(active_connections),
        "components": {}
    }
    
    # Add component stats
    if agent_router:
        stats["components"]["agent_router"] = agent_router.get_stats()
    
    if dispatcher_service:
        stats["components"]["dispatcher"] = dispatcher_service.get_stats()
    
    if conversation_manager:
        stats["components"]["conversation"] = await conversation_manager.get_stats()
    
    if query_engine:
        stats["components"]["knowledge_base"] = await query_engine.get_stats()
    
    return stats

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        workers=1  # Use single worker for shared state
    )