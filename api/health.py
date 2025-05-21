# api/health.py

"""
Health check endpoints for monitoring system status.
"""
import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter()

class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    checks: Dict[str, Any]
    timestamp: float

class ComponentStatus(BaseModel):
    """Component status response model."""
    status: str
    details: Dict[str, Any]
    last_check: float

def get_settings():
    """Get application settings."""
    return Settings()

@router.get("/", response_model=HealthStatus)
async def health_check(settings: Settings = Depends(get_settings)):
    """Basic health check endpoint."""
    checks = {
        "api": {
            "status": "healthy",
            "timestamp": time.time()
        }
    }
    
    # Check components
    from main import conversation_manager, query_engine, agent_router, dispatcher_service
    
    if conversation_manager:
        try:
            conv_stats = await conversation_manager.get_stats()
            checks["conversation_manager"] = {
                "status": "healthy",
                "details": conv_stats
            }
        except Exception as e:
            logger.error(f"Error checking conversation manager: {e}")
            checks["conversation_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    if query_engine:
        try:
            kb_stats = await query_engine.get_stats()
            checks["knowledge_base"] = {
                "status": "healthy",
                "details": kb_stats
            }
        except Exception as e:
            logger.error(f"Error checking knowledge base: {e}")
            checks["knowledge_base"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    if agent_router:
        try:
            agent_stats = agent_router.get_stats()
            checks["agent_router"] = {
                "status": "healthy",
                "details": agent_stats
            }
        except Exception as e:
            logger.error(f"Error checking agent router: {e}")
            checks["agent_router"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    if dispatcher_service:
        try:
            dispatch_stats = dispatcher_service.get_stats()
            checks["dispatcher"] = {
                "status": "healthy",
                "details": dispatch_stats
            }
        except Exception as e:
            logger.error(f"Error checking dispatcher service: {e}")
            checks["dispatcher"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Determine overall status
    status = "healthy"
    if any(check.get("status") == "unhealthy" for check in checks.values()):
        status = "degraded"
    if all(check.get("status") == "unhealthy" for check in checks.values()):
        status = "unhealthy"
    
    return HealthStatus(
        status=status,
        checks=checks,
        timestamp=time.time()
    )

@router.get("/readiness")
async def readiness_check():
    """
    Readiness probe for Kubernetes.
    Checks if system is ready to handle requests.
    """
    from main import conversation_manager, query_engine, agent_router, dispatcher_service
    
    # Check all required components
    components_ready = all([
        conversation_manager is not None,
        query_engine is not None,
        agent_router is not None,
        dispatcher_service is not None
    ])
    
    if not components_ready:
        return {"status": "not_ready", "message": "Components still initializing"}
    
    return {"status": "ready"}

@router.get("/liveness")
async def liveness_check():
    """
    Liveness probe for Kubernetes.
    Quick check to ensure service is responsive.
    """
    return {"status": "alive", "timestamp": time.time()}