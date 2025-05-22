# main.py

"""
Fixed main application entry point with improved WebSocket handling.
"""
import os
import logging
import asyncio
import json
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

# Import configurations and core modules
from core.config import Settings
from core.state_manager import StateManager, ConversationState
from core.conversation_manager import ConversationManager
from core.session_manager import SessionManager

# Import knowledge base components
from knowledge_base.query_engine import QueryEngine
from knowledge_base.rag_config import rag_config
from prompts.prompt_manager import PromptManager
from agents.router import AgentRouter
from services.dispatcher import DispatcherService

# Import API routes
from api.twilio_routes import router as twilio_router
from api.health import router as health_router

# Load settings
settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
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
        logger.info("Initializing query engine...")
        try:
            query_engine = QueryEngine(config=rag_config)
            await query_engine.init()
            logger.info("Query engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            # Try with minimal configuration
            from knowledge_base.index_manager import IndexManager
            index_manager = IndexManager(config=rag_config)
            await index_manager.init()
            query_engine = QueryEngine(index_manager=index_manager, config=rag_config)
            await query_engine.init()
            logger.info("Query engine initialized with fallback method")
        
        # Initialize prompt system
        logger.info("Initializing prompt manager...")
        try:
            prompt_manager = PromptManager(prompt_dir=settings.prompts_dir)
            logger.info("Prompt manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize prompt manager: {e}")
            # Create a minimal prompt manager
            prompt_manager = PromptManager(prompt_dir="./prompts")
            logger.info("Prompt manager initialized with default directory")
        
        # Initialize conversation manager with proper config conversion
        logger.info("Initializing conversation manager...")
        try:
            # Convert settings object to dictionary for conversation manager
            conversation_config = {
                "max_conversation_history": getattr(settings.conversation, 'max_conversation_history', 5),
                "context_window_size": getattr(settings.conversation, 'context_window_size', 4096),
                "max_tokens": getattr(settings.conversation, 'max_tokens', 256),
                "temperature": getattr(settings.conversation, 'temperature', 0.7)
            }
            
            conversation_manager = ConversationManager(
                query_engine=query_engine,
                prompt_manager=prompt_manager,
                config=conversation_config  # Pass dict instead of settings object
            )
            await conversation_manager.init()
            logger.info("Conversation manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize conversation manager: {e}")
            # Create with minimal config
            conversation_manager = ConversationManager(
                query_engine=query_engine,
                config={}
            )
            await conversation_manager.init()
            logger.info("Conversation manager initialized with minimal config")
        
        # Initialize agent router
        logger.info("Initializing agent router...")
        try:
            agent_router = AgentRouter(
                conversation_manager=conversation_manager,
                query_engine=query_engine,
                prompt_manager=prompt_manager
            )
            logger.info("Agent router initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent router: {e}")
            raise
        
        # Initialize dispatcher service
        logger.info("Initializing dispatcher service...")
        try:
            dispatcher_service = DispatcherService()
            logger.info("Dispatcher service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dispatcher service: {e}")
            raise
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application components...")
    try:
        # Clean up conversation manager
        if conversation_manager:
            try:
                await conversation_manager.cleanup()
                logger.info("Conversation manager cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up conversation manager: {e}")
        
        # Clean up query engine
        if query_engine:
            try:
                await query_engine.cleanup()
                logger.info("Query engine cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up query engine: {e}")
        
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

# Active WebSocket connections with improved tracking
active_connections: Dict[str, Dict[str, Any]] = {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Enhanced WebSocket endpoint for Twilio Media Streams with proper error handling."""
    handler = None
    
    try:
        await websocket.accept()
        
        # Track connection
        connection_info = {
            "websocket": websocket,
            "connected_at": time.time(),
            "session_id": session_id,
            "status": "connected"
        }
        active_connections[session_id] = connection_info
        
        logger.info(f"WebSocket connection established for session {session_id}")
        
        # Import voice components
        from telephony.simple_websocket_handler import SimpleWebSocketHandler
        
        # Create pipeline components if available
        if query_engine and conversation_manager:
            try:
                # Create a simplified pipeline for voice calls
                class SimplePipeline:
                    def __init__(self):
                        self.query_engine = query_engine
                        self.conversation_manager = conversation_manager
                
                pipeline = SimplePipeline()
                
                # Create WebSocket handler with improved error handling
                handler = SimpleWebSocketHandler(session_id, pipeline)
                
                # Start the conversation
                await handler.start_conversation(websocket)
                
                # Update connection status
                active_connections[session_id]["status"] = "active"
                active_connections[session_id]["handler"] = handler
                
                # Handle WebSocket messages with proper error handling
                while True:
                    try:
                        # Receive message from Twilio with timeout
                        message = await asyncio.wait_for(
                            websocket.receive_text(), 
                            timeout=60.0  # 60 second timeout
                        )
                        
                        data = json.loads(message)
                        event_type = data.get('event')
                        
                        if event_type == 'start':
                            # Media stream started
                            stream_sid = data.get('streamSid')
                            handler.stream_sid = stream_sid
                            logger.info(f"Media stream started: {stream_sid}")
                            
                            # Send initial greeting after media stream starts
                            await handler.send_initial_greeting()
                            
                        elif event_type == 'media':
                            # Audio data received - handle with connection checks
                            if not handler.connection_closed:
                                await handler._handle_audio(data, websocket)
                            else:
                                logger.debug("Skipping audio processing - connection marked as closed")
                            
                        elif event_type == 'stop':
                            # Media stream stopped
                            logger.info(f"Media stream stopped for session {session_id}")
                            break
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"WebSocket timeout for session {session_id}")
                        # Send a ping to check if connection is still alive
                        try:
                            await websocket.send_text(json.dumps({"event": "ping"}))
                        except:
                            logger.info(f"WebSocket appears disconnected for session {session_id}")
                            break
                            
                    except WebSocketDisconnect:
                        logger.info(f"WebSocket disconnected for session {session_id}")
                        break
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received: {e}")
                        continue
                        
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        # Continue processing other messages instead of breaking
                        continue
                        
            except Exception as e:
                logger.error(f"Error setting up voice handler: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "error": f"Setup error: {str(e)}"
                    }))
                except:
                    pass  # Connection might already be closed
                
        else:
            logger.error("Voice AI components not available")
            try:
                await websocket.send_text(json.dumps({
                    "error": "Voice AI components not initialized"
                }))
            except:
                pass  # Connection might already be closed
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during setup for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        
    finally:
        # Enhanced cleanup with proper error handling
        try:
            # Update connection status
            if session_id in active_connections:
                active_connections[session_id]["status"] = "disconnected"
                active_connections[session_id]["disconnected_at"] = time.time()
            
            # Clean up handler
            if handler:
                try:
                    await handler._cleanup()
                except Exception as e:
                    logger.error(f"Error during handler cleanup: {e}")
            
            # Clean up agent if available
            if agent_router:
                try:
                    agent_router.cleanup_session(session_id)
                except Exception as e:
                    logger.error(f"Error cleaning up agent session: {e}")
            
            # Remove from active connections after a delay (for debugging)
            async def delayed_cleanup():
                await asyncio.sleep(5)  # Keep connection info for 5 seconds
                if session_id in active_connections:
                    del active_connections[session_id]
            
            asyncio.create_task(delayed_cleanup())
            
            logger.info(f"WebSocket connection closed and cleaned up for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error during WebSocket cleanup: {e}")

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

@app.get("/connections")
async def get_connections():
    """Get information about active WebSocket connections."""
    return {
        "active_connections": len(active_connections),
        "connections": {
            session_id: {
                "session_id": info["session_id"],
                "connected_at": info["connected_at"],
                "status": info["status"],
                "duration": time.time() - info["connected_at"],
                "disconnected_at": info.get("disconnected_at"),
                "has_handler": "handler" in info
            }
            for session_id, info in active_connections.items()
        }
    }

@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics."""
    stats = {
        "active_sessions": len([c for c in active_connections.values() if c["status"] == "active"]),
        "total_connections": len(active_connections),
        "components": {},
        "timestamp": time.time()
    }
    
    # Add component stats
    if agent_router:
        try:
            stats["components"]["agent_router"] = agent_router.get_stats()
        except Exception as e:
            logger.error(f"Error getting agent router stats: {e}")
            stats["components"]["agent_router"] = {"error": str(e)}
    
    if dispatcher_service:
        try:
            stats["components"]["dispatcher"] = dispatcher_service.get_stats()
        except Exception as e:
            logger.error(f"Error getting dispatcher stats: {e}")
            stats["components"]["dispatcher"] = {"error": str(e)}
    
    if conversation_manager:
        try:
            stats["components"]["conversation"] = await conversation_manager.get_stats()
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            stats["components"]["conversation"] = {"error": str(e)}
    
    if query_engine:
        try:
            stats["components"]["knowledge_base"] = await query_engine.get_stats()
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            stats["components"]["knowledge_base"] = {"error": str(e)}
    
    # Add connection statistics
    connection_stats = {
        "total": len(active_connections),
        "active": len([c for c in active_connections.values() if c["status"] == "active"]),
        "disconnected": len([c for c in active_connections.values() if c["status"] == "disconnected"])
    }
    
    if active_connections:
        durations = [
            time.time() - info["connected_at"] 
            for info in active_connections.values()
            if info["status"] == "active"
        ]
        if durations:
            connection_stats["avg_duration"] = sum(durations) / len(durations)
            connection_stats["max_duration"] = max(durations)
    
    stats["connections"] = connection_stats
    
    return stats

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the application is working."""
    try:
        # Test knowledge base
        kb_status = "unknown"
        if query_engine:
            try:
                kb_stats = await query_engine.get_stats()
                kb_status = f"ready with {kb_stats.get('document_count', 0)} documents"
            except Exception as e:
                kb_status = f"error: {str(e)}"
        
        # Test conversation manager
        conv_status = "unknown"
        if conversation_manager:
            try:
                conv_stats = await conversation_manager.get_stats()
                conv_status = f"ready (session: {conv_stats.get('session_id', 'unknown')})"
            except Exception as e:
                conv_status = f"error: {str(e)}"
        
        return {
            "status": "ok",
            "timestamp": time.time(),
            "components": {
                "knowledge_base": kb_status,
                "conversation_manager": conv_status,
                "agent_router": "ready" if agent_router else "not initialized",
                "dispatcher_service": "ready" if dispatcher_service else "not initialized"
            },
            "environment": {
                "google_credentials": os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")),
                "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
                "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY")),
                "twilio_configured": bool(os.getenv("TWILIO_ACCOUNT_SID"))
            },
            "connections": {
                "active": len([c for c in active_connections.values() if c["status"] == "active"]),
                "total": len(active_connections)
            }
        }
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
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
    port = int(os.getenv("PORT", 5000))
    
    logger.info(f"Starting Voice AI Agent on port {port}")
    
    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        workers=1,  # Use single worker for shared state
        log_level=settings.log_level.lower()
    )