#!/usr/bin/env python3
"""
Revolutionary Multi-Agent Voice AI System - Main FastAPI Application
Enhanced with YAML configuration loading and proper agent deployment.
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
from pathlib import Path
import yaml

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

# Configuration Management
class ConfigManager:
    """Centralized configuration manager for YAML configs."""
    
    def __init__(self, config_base_path: str = "app/config"):
        self.config_base_path = Path(config_base_path)
        self.agents_config_path = self.config_base_path / "agents"
        self._configs_cache = {}
        logger.info(f"ConfigManager initialized with base path: {self.config_base_path}")
    
    def load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            logger.info(f"✅ Loaded config: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to load config {config_path}: {e}")
            raise
    
    def load_agent_config(self, agent_config_file: str) -> Dict[str, Any]:
        """Load specific agent configuration."""
        config_path = self.agents_config_path / agent_config_file
        return self.load_yaml_config(config_path)
    
    def load_all_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all agent configurations from the agents directory."""
        agent_configs = {}
        
        if not self.agents_config_path.exists():
            logger.error(f"❌ Agents config directory not found: {self.agents_config_path}")
            return agent_configs
        
        # Get all YAML files in agents directory
        yaml_files = list(self.agents_config_path.glob("*.yaml")) + list(self.agents_config_path.glob("*.yml"))
        
        for yaml_file in yaml_files:
            # Skip template files
            if "template" in yaml_file.name.lower():
                continue
                
            try:
                config = self.load_yaml_config(yaml_file)
                agent_id = config.get("agent_id")
                
                if agent_id:
                    agent_configs[agent_id] = config
                    logger.info(f"✅ Loaded agent config: {agent_id} from {yaml_file.name}")
                else:
                    logger.warning(f"⚠️ No agent_id found in {yaml_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load agent config from {yaml_file}: {e}")
        
        return agent_configs
    
    def load_system_config(self, config_name: str) -> Dict[str, Any]:
        """Load system configuration (monitoring, qdrant, etc.)."""
        config_path = self.config_base_path / f"{config_name}.yaml"
        return self.load_yaml_config(config_path)
    
    def validate_agent_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance agent configuration."""
        required_fields = [
            "agent_id", "version", "specialization", 
            "voice_settings", "tools", "routing"
        ]
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Validate voice_settings structure
        if "voice_settings" in config:
            voice_settings = config["voice_settings"]
            if not isinstance(voice_settings, dict):
                validation_result["valid"] = False
                validation_result["errors"].append("voice_settings must be a dictionary")
            else:
                # Ensure required voice settings
                if "tts_voice" not in voice_settings:
                    validation_result["warnings"].append("No tts_voice specified in voice_settings")
        
        # Validate tools
        if "tools" in config:
            if not isinstance(config["tools"], list):
                validation_result["valid"] = False
                validation_result["errors"].append("tools must be a list")
        
        # Validate routing
        if "routing" in config:
            routing = config["routing"]
            if not isinstance(routing, dict):
                validation_result["valid"] = False
                validation_result["errors"].append("routing must be a dictionary")
            else:
                if "primary_keywords" not in routing:
                    validation_result["warnings"].append("No primary_keywords specified in routing")
        
        return validation_result

# Global configuration manager
config_manager = ConfigManager()

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
        
        # Load system configurations
        logger.info("📋 Loading system configurations...")
        try:
            # Load system configs if they exist
            monitoring_config = {}
            qdrant_config = {}
            
            try:
                monitoring_config = config_manager.load_system_config("monitoring")
                logger.info("✅ Loaded monitoring configuration")
            except FileNotFoundError:
                logger.warning("⚠️ monitoring.yaml not found, using defaults")
            
            try:
                qdrant_config = config_manager.load_system_config("qdrant")
                logger.info("✅ Loaded Qdrant configuration")
            except FileNotFoundError:
                logger.warning("⚠️ qdrant.yaml not found, using defaults")
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading system configs: {e}, using defaults")
            monitoring_config = {}
            qdrant_config = {}
        
        # 1. Initialize Hybrid Vector System (Tier 1: Redis + FAISS + Qdrant)
        logger.info("📊 Initializing 3-tier hybrid vector architecture...")
        
        # Use qdrant config if available, otherwise use defaults
        qdrant_connection_config = qdrant_config.get('service', {})
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
            qdrant_config={
                "host": os.getenv("QDRANT_HOST", qdrant_connection_config.get('host', 'localhost')),
                "port": int(os.getenv("QDRANT_PORT", str(qdrant_connection_config.get('http_port', 6333)))),
                "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", str(qdrant_connection_config.get('grpc_port', 6334)))),
                "prefer_grpc": False,  # Force HTTP connection for RunPod
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
        
        # 10. Deploy specialized agents from YAML configs
        logger.info("🚀 Deploying specialized agents from YAML configurations...")
        await deploy_agents_from_yaml()
        
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

async def deploy_agents_from_yaml():
    """Deploy agents from YAML configuration files."""
    logger.info("🤖 Loading agent configurations from YAML files...")
    
    try:
        # Load all agent configurations
        agent_configs = config_manager.load_all_agent_configs()
        
        if not agent_configs:
            logger.error("❌ No agent configurations found in config/agents/")
            return
        
        deployed_count = 0
        failed_count = 0
        
        for agent_id, config in agent_configs.items():
            try:
                logger.info(f"🚀 Deploying agent: {agent_id}")
                
                # Validate configuration
                validation_result = config_manager.validate_agent_config(config)
                
                if not validation_result["valid"]:
                    logger.error(f"❌ Invalid configuration for {agent_id}: {validation_result['errors']}")
                    failed_count += 1
                    continue
                
                if validation_result["warnings"]:
                    logger.warning(f"⚠️ Configuration warnings for {agent_id}: {validation_result['warnings']}")
                
                # Deploy the agent
                result = await agent_registry.deploy_agent(config)
                
                if result.success:
                    logger.info(f"✅ Successfully deployed agent: {agent_id}")
                    deployed_count += 1
                else:
                    logger.error(f"❌ Failed to deploy agent {agent_id}: {result.error}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Error deploying agent {agent_id}: {e}")
                failed_count += 1
        
        logger.info(f"🎯 Agent deployment summary: {deployed_count} successful, {failed_count} failed")
        
        if deployed_count == 0:
            logger.warning("⚠️ No agents were successfully deployed!")
        
    except Exception as e:
        logger.error(f"❌ Failed to deploy agents from YAML: {e}", exc_info=True)

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
            "Real-time performance monitoring",
            "YAML-based configuration management"
        ],
        "target_latency_ms": 377,
        "active_sessions": len(active_sessions),
        "config_loaded": len(config_manager._configs_cache) > 0,
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

@app.get("/config/agents")
async def list_agent_configs():
    """List all available agent configurations."""
    agent_configs = config_manager.load_all_agent_configs()
    
    config_summary = {}
    for agent_id, config in agent_configs.items():
        config_summary[agent_id] = {
            "version": config.get("version", "unknown"),
            "domain_expertise": config.get("specialization", {}).get("domain_expertise", "unknown"),
            "status": config.get("status", "unknown"),
            "tools_count": len(config.get("tools", [])),
            "routing_keywords": len(config.get("routing", {}).get("primary_keywords", []))
        }
    
    return {
        "available_configs": config_summary,
        "total_count": len(config_summary),
        "config_directory": str(config_manager.agents_config_path)
    }

@app.post("/agents/deploy")
async def deploy_agent(
    request: AgentDeploymentRequest,
    _: None = Depends(ensure_system_initialized)
):
    """Deploy a new agent with hot deployment and validation."""
    try:
        # Validate configuration
        validation_result = config_manager.validate_agent_config(request.agent_config)
        
        if not validation_result["valid"]:
            logger.error(f"❌ Invalid agent configuration: {validation_result['errors']}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid configuration: {validation_result['errors']}"
            )
        
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
                "message": "Agent deployed successfully with zero downtime",
                "validation_warnings": validation_result.get("warnings", [])
            }
        else:
            logger.error(f"❌ Agent deployment failed: {result.error}")
            raise HTTPException(status_code=400, detail=result.error)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agent deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/deploy-from-file/{config_filename}")
async def deploy_agent_from_file(
    config_filename: str,
    _: None = Depends(ensure_system_initialized)
):
    """Deploy agent from YAML configuration file."""
    try:
        # Load configuration from file
        config = config_manager.load_agent_config(config_filename)
        
        # Validate configuration
        validation_result = config_manager.validate_agent_config(config)
        
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid configuration in {config_filename}: {validation_result['errors']}"
            )
        
        # Deploy agent
        result = await agent_registry.deploy_agent(config)
        
        if result.success:
            return {
                "success": True,
                "agent_id": result.agent_id,
                "config_file": config_filename,
                "deployment_id": result.deployment_id,
                "version": result.version,
                "validation_warnings": validation_result.get("warnings", [])
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Configuration file not found: {config_filename}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying from file {config_filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# [Rest of the endpoints remain the same as in the original main.py]
# ... (continuing with the remaining endpoints like /agents/list, /conversation, etc.)

if __name__ == "__main__":
    print("🚀 Starting Revolutionary Multi-Agent Voice AI System...")
    print(f"🎯 Target latency: <377ms (84% improvement)")
    print(f"🔧 Base URL: {os.getenv('BASE_URL', 'Not configured')}")
    print(f"📊 Vector DB: Hybrid 3-tier (Redis+FAISS+Qdrant)")
    print(f"🤖 Agents: Hot deployment with YAML configuration")
    print(f"🛠️ Tools: Comprehensive orchestration framework")
    print(f"📋 Config Directory: app/config/")
    
    # Verify config directory exists
    config_path = Path("app/config/agents")
    if config_path.exists():
        yaml_files = list(config_path.glob("*.yaml"))
        print(f"📄 Found {len(yaml_files)} agent config files")
    else:
        print("⚠️ Config directory not found - please check app/config/agents/")
    
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
        lifespan="on"
    )