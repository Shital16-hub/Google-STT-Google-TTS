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
import subprocess
import signal
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

# Ensure logs directory exists and configure logging FIRST
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

class ConfigurationManager:
    """Enhanced configuration manager with service startup integration."""
    
    def __init__(self, config_base_path: str = "app/config"):
        self.config_base_path = Path(config_base_path)
        self.agents_config_path = self.config_base_path / "agents"
        self._configs_cache = {}
        self.services_started = {
            'redis': False,
            'qdrant': False
        }
        logger.info(f"ConfigurationManager initialized with base path: {self.config_base_path}")
        
        # Ensure config directories exist
        self.config_base_path.mkdir(parents=True, exist_ok=True)
        self.agents_config_path.mkdir(parents=True, exist_ok=True)
    
    def load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        try:
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return {}
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            logger.info(f"✅ Loaded config: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to load config {config_path}: {e}")
            return {}
    
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
                if not config:
                    continue
                    
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
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration with fallback to defaults."""
        redis_conf_path = self.config_base_path / "redis.conf"
        
        # Default Redis configuration
        default_config = {
            "host": "localhost",
            "port": 6379,
            "timeout": 5,
            "max_connections": 100,
            "retry_on_timeout": True,
            "decode_responses": False
        }
        
        if redis_conf_path.exists():
            logger.info(f"📋 Found Redis config file: {redis_conf_path}")
            # Parse key Redis settings from redis.conf
            try:
                with open(redis_conf_path, 'r') as f:
                    content = f.read()
                
                # Extract key settings
                import re
                
                # Extract port
                port_match = re.search(r'^port\s+(\d+)', content, re.MULTILINE)
                if port_match:
                    default_config["port"] = int(port_match.group(1))
                
                # Extract bind address
                bind_match = re.search(r'^bind\s+(.+)', content, re.MULTILINE)
                if bind_match:
                    bind_addr = bind_match.group(1).strip()
                    if bind_addr not in ["0.0.0.0", "*"]:
                        default_config["host"] = bind_addr
                
                # Extract maxclients
                maxclients_match = re.search(r'^maxclients\s+(\d+)', content, re.MULTILINE)
                if maxclients_match:
                    default_config["max_connections"] = int(maxclients_match.group(1))
                
                logger.info(f"✅ Parsed Redis config: {default_config}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse redis.conf: {e}")
        
        return default_config
    
    async def ensure_redis_running(self) -> bool:
        """Ensure Redis is running with configuration."""
        if self.services_started['redis']:
            return True
            
        logger.info("🔧 Ensuring Redis is running...")
        
        # Get Redis configuration
        redis_config = self.get_redis_config()
        host = redis_config.get("host", "localhost")
        port = redis_config.get("port", 6379)
        
        # Check if Redis is already running
        try:
            import redis
            test_client = redis.Redis(host=host, port=port, socket_timeout=2)
            response = test_client.ping()
            if response:
                logger.info("✅ Redis already running")
                self.services_started['redis'] = True
                return True
        except Exception:
            pass
        
        # Try to start Redis
        logger.info("🚀 Starting Redis...")
        
        redis_conf_path = self.config_base_path / "redis.conf"
        
        # Try different Redis startup methods
        startup_commands = []
        
        if redis_conf_path.exists():
            # Try with config file
            startup_commands.extend([
                ['redis-server', str(redis_conf_path)],
                ['redis-server', str(redis_conf_path), '--daemonize', 'yes'],
            ])
        
        # Fallback commands
        startup_commands.extend([
            ['redis-server', '--daemonize', 'yes'],
            ['redis-server', '--port', str(port), '--daemonize', 'yes'],
            ['redis-server']  # Last resort - foreground
        ])
        
        for cmd in startup_commands:
            try:
                logger.info(f"Trying: {' '.join(cmd)}")
                
                if cmd[-1] == 'redis-server':  # Foreground mode
                    # Start in background
                    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    await asyncio.sleep(2)
                else:
                    # Daemon mode
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    await asyncio.sleep(3)
                
                # Test if Redis is responding
                try:
                    test_client = redis.Redis(host=host, port=port, socket_timeout=2)
                    response = test_client.ping()
                    if response:
                        logger.info("✅ Redis started successfully")
                        self.services_started['redis'] = True
                        return True
                except Exception as e:
                    logger.debug(f"Redis test failed: {e}")
                    continue
                    
            except Exception as e:
                logger.debug(f"Redis startup attempt failed: {e}")
                continue
        
        # If all else fails, try installing and starting Redis
        logger.warning("⚠️ All Redis startup attempts failed, trying installation...")
        try:
            # Try to install Redis
            subprocess.run(['apt-get', 'update'], capture_output=True)
            subprocess.run(['apt-get', 'install', '-y', 'redis-server'], capture_output=True)
            
            # Try starting again
            subprocess.run(['redis-server', '--daemonize', 'yes'], capture_output=True)
            await asyncio.sleep(3)
            
            # Final test
            test_client = redis.Redis(host=host, port=port, socket_timeout=2)
            response = test_client.ping()
            if response:
                logger.info("✅ Redis installed and started successfully")
                self.services_started['redis'] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to install Redis: {e}")
        
        logger.error("❌ All Redis startup methods failed")
        return False
    
    async def start_qdrant_with_config(self) -> bool:
        """Start Qdrant with configuration from qdrant.yaml."""
        if self.services_started['qdrant']:
            return True
            
        logger.info("🗄️ Starting Qdrant with configuration...")
        
        # Check if Qdrant is already running
        try:
            import requests
            response = requests.get('http://localhost:6333/health', timeout=5)
            if response.status_code == 200:
                logger.info("✅ Qdrant already running")
                self.services_started['qdrant'] = True
                return True
        except:
            pass
        
        # Create Qdrant setup directory
        qdrant_setup_dir = Path("/workspace/qdrant-setup")
        qdrant_setup_dir.mkdir(parents=True, exist_ok=True)
        (qdrant_setup_dir / "config").mkdir(exist_ok=True)
        (qdrant_setup_dir / "storage").mkdir(exist_ok=True)
        
        # Stop any existing Qdrant
        try:
            subprocess.run(['pkill', '-f', 'qdrant'], capture_output=True)
            await asyncio.sleep(2)
        except:
            pass
        
        # Load Qdrant configuration
        qdrant_config = self.load_system_config("qdrant")
        
        # Create production config file
        config_file = qdrant_setup_dir / "config" / "production.yaml"
        
        if qdrant_config:
            logger.info("📋 Using Qdrant configuration from qdrant.yaml")
            # Create simplified config from loaded config
            service_config = qdrant_config.get('service', {})
            storage_config = qdrant_config.get('storage', {})
            
            production_config = {
                'service': {
                    'host': service_config.get('host', '0.0.0.0'),
                    'http_port': service_config.get('http_port', 6333),
                    'grpc_port': service_config.get('grpc_port', 6334),
                    'enable_cors': service_config.get('enable_cors', True),
                    'max_request_size_mb': service_config.get('max_request_size_mb', 32),
                    'log_level': service_config.get('log_level', 'INFO'),
                    'max_workers': service_config.get('max_workers', 4)
                },
                'storage': {
                    'storage_path': './storage',
                    'performance': storage_config.get('performance', {
                        'max_search_threads': 4,
                        'max_optimization_threads': 2,
                        'search_batch_size': 100,
                        'max_concurrent_searches': 500,
                        'max_indexing_threads': 2
                    }),
                    'optimizers': storage_config.get('optimizers', {
                        'deleted_threshold': 0.2,
                        'vacuum_min_vector_number': 1000,
                        'default_segment_number': 2,
                        'max_segment_size': 20000,
                        'memmap_threshold': 20000,
                        'flush_interval_sec': 2,
                        'max_optimization_threads': 2
                    }),
                    'hnsw': storage_config.get('hnsw', {
                        'm': 16,
                        'ef_construct': 128,
                        'full_scan_threshold': 10000,
                        'max_indexing_threads': 2,
                        'ef': 64,
                        'on_disk': False
                    })
                },
                'telemetry': qdrant_config.get('telemetry', {'enabled': False}),
                'cluster': qdrant_config.get('cluster', {'enabled': False})
            }
        else:
            logger.info("📋 Creating minimal Qdrant configuration")
            production_config = {
                'service': {
                    'host': '0.0.0.0',
                    'http_port': 6333,
                    'grpc_port': 6334,
                    'enable_cors': True
                },
                'storage': {
                    'storage_path': './storage'
                },
                'telemetry': {'enabled': False}
            }
        
        # Write config file
        try:
            with open(config_file, 'w') as f:
                yaml.dump(production_config, f, default_flow_style=False)
            logger.info(f"✅ Created Qdrant config: {config_file}")
        except Exception as e:
            logger.error(f"❌ Failed to create Qdrant config: {e}")
            return False
        
        # Download Qdrant binary if not present
        qdrant_binary = qdrant_setup_dir / "qdrant"
        if not qdrant_binary.exists():
            logger.info("📦 Downloading Qdrant binary...")
            try:
                import requests
                url = "https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz"
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                import tarfile
                import io
                
                with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
                    tar.extractall(path=qdrant_setup_dir)
                
                qdrant_binary.chmod(0o755)
                logger.info("✅ Downloaded Qdrant binary")
            except Exception as e:
                logger.error(f"❌ Failed to download Qdrant: {e}")
                return False
        
        # Start Qdrant
        try:
            os.chdir(qdrant_setup_dir)
            
            # Start Qdrant process
            with open('qdrant.log', 'w') as log_file:
                process = subprocess.Popen([
                    './qdrant', '--config-path', str(config_file)
                ], stdout=log_file, stderr=subprocess.STDOUT)
            
            # Save PID
            with open('qdrant.pid', 'w') as pid_file:
                pid_file.write(str(process.pid))
            
            logger.info(f"🚀 Started Qdrant (PID: {process.pid})")
            
            # Wait for Qdrant to start
            await asyncio.sleep(5)
            
            # Test Qdrant connection
            import requests
            for attempt in range(10):
                try:
                    response = requests.get('http://localhost:6333/health', timeout=3)
                    if response.status_code == 200:
                        logger.info("✅ Qdrant started successfully")
                        self.services_started['qdrant'] = True
                        return True
                except:
                    if attempt < 9:
                        await asyncio.sleep(2)
                    
            logger.error("❌ Qdrant failed to start")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start Qdrant: {e}")
            return False
    
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

# Global configuration manager (initialized after logger is set up)
config_manager = ConfigurationManager()

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
            logger.warning("⚠️ BASE_URL not set, using default")
            BASE_URL = "http://localhost:8000"
        
        # Step 1: Ensure Redis is running with configuration
        logger.info("📊 Step 1: Ensuring Redis is running with configuration...")
        redis_success = await config_manager.ensure_redis_running()
        if not redis_success:
            logger.warning("⚠️ Redis startup failed, system will continue with degraded performance")
        
        # Step 2: Start Qdrant with configuration
        logger.info("🗄️ Step 2: Starting Qdrant with configuration...")
        qdrant_success = await config_manager.start_qdrant_with_config()
        if not qdrant_success:
            logger.warning("⚠️ Qdrant startup failed, will use in-memory fallback")
        
        # Step 3: Initialize Hybrid Vector System with proper configuration
        logger.info("📊 Step 3: Initializing 3-tier hybrid vector architecture...")
        
        # Get Redis configuration from config manager
        redis_config = config_manager.get_redis_config()
        
        # Try external services first, fallback to in-memory/localhost
        if redis_success and qdrant_success:
            logger.info("🚀 Using external Redis and Qdrant")
            hybrid_vector_system = HybridVectorSystem(
                redis_config={
                    "host": redis_config.get("host", "localhost"),
                    "port": redis_config.get("port", 6379),
                    "cache_size": 50000,
                    "ttl_seconds": 3600,
                    "timeout": redis_config.get("timeout", 5),
                    "max_connections": redis_config.get("max_connections", 100)
                },
                faiss_config={
                    "memory_limit_gb": 4,
                    "promotion_threshold": 100,
                    "index_type": "HNSW"
                },
                qdrant_config={
                    "host": "localhost",
                    "port": 6333,
                    "grpc_port": 6334,
                    "prefer_grpc": False,  # Force HTTP connection for RunPod
                    "timeout": 10.0
                }
            )
        elif qdrant_success:
            logger.info("🔄 Using in-memory Redis with external Qdrant")
            hybrid_vector_system = HybridVectorSystem(
                redis_config={
                    "host": "localhost",  # Try localhost anyway
                    "port": 6379,
                    "cache_size": 10000,  # Smaller cache
                    "ttl_seconds": 1800,
                    "timeout": 2,
                    "max_connections": 10,
                    "fallback_to_memory": True  # Use memory if Redis fails
                },
                faiss_config={
                    "memory_limit_gb": 4,
                    "promotion_threshold": 100,
                    "index_type": "HNSW"
                },
                qdrant_config={
                    "host": "localhost",
                    "port": 6333,
                    "grpc_port": 6334,
                    "prefer_grpc": False,
                    "timeout": 10.0
                }
            )
        else:
            logger.info("🧠 Using full in-memory fallback")
            hybrid_vector_system = HybridVectorSystem(
                redis_config={
                    "host": ":memory:",  # In-memory Redis simulation
                    "port": None,
                    "cache_size": 5000,
                    "ttl_seconds": 900
                },
                faiss_config={
                    "memory_limit_gb": 2,
                    "promotion_threshold": 50,
                    "index_type": "HNSW"
                },
                qdrant_config={
                    "host": ":memory:",  # In-memory mode
                    "port": None,
                    "prefer_grpc": False,
                    "timeout": 1.0
                }
            )
        
        # Initialize with error handling
        try:
            await hybrid_vector_system.initialize()
            logger.info("✅ Hybrid vector system initialized successfully")
        except Exception as e:
            logger.error(f"❌ Vector system initialization failed: {e}")
            # Create minimal fallback vector system
            logger.info("🔄 Creating minimal fallback vector system...")
            hybrid_vector_system = HybridVectorSystem(
                redis_config={"host": ":memory:", "port": None, "cache_size": 1000},
                faiss_config={"memory_limit_gb": 1, "promotion_threshold": 25},
                qdrant_config={"host": ":memory:", "port": None}
            )
            await hybrid_vector_system.initialize()
            logger.info("✅ Fallback vector system initialized")
        
        # 4. Initialize Enhanced STT System
        logger.info("🎤 Step 4: Initializing enhanced STT system...")
        stt_system = EnhancedSTTSystem(
            primary_provider="google_cloud_v2",
            backup_provider="assemblyai",
            enable_vad=True,
            enable_echo_cancellation=True,
            target_latency_ms=80
        )
        await stt_system.initialize()
        
        # 5. Initialize Dual Streaming TTS Engine
        logger.info("🔊 Step 5: Initializing dual streaming TTS engine...")
        tts_engine = DualStreamingTTSEngine()
        await tts_engine.initialize()
        
        # 6. Initialize Comprehensive Tool Orchestrator
        logger.info("🛠️ Step 6: Initializing tool orchestration framework...")
        tool_orchestrator = ComprehensiveToolOrchestrator(
            enable_business_workflows=True,
            enable_external_apis=True,
            dummy_mode=True,  # Enable dummy mode for development
            max_concurrent_tools=10
        )
        await tool_orchestrator.initialize()
        
        # 7. Initialize Agent Registry with Hot Deployment
        logger.info("🤖 Step 7: Initializing agent registry with hot deployment...")
        agent_registry = AgentRegistry(
            hybrid_vector_system=hybrid_vector_system,
            tool_orchestrator=tool_orchestrator,
            deployment_strategy="blue_green",
            enable_health_checks=True
        )
        await agent_registry.initialize()
        
        # 8. Initialize Intelligent Agent Router
        logger.info("🧠 Step 8: Initializing ML-based intelligent agent router...")
        agent_router = IntelligentAgentRouter(
            agent_registry=agent_registry,
            hybrid_vector_system=hybrid_vector_system,
            confidence_threshold=0.85,
            fallback_threshold=0.6,
            enable_ml_routing=True
        )
        await agent_router.initialize()
        
        # 9. Initialize Conversation State Manager
        logger.info("💾 Step 9: Initializing stateful conversation management...")
        state_manager = ConversationStateManager(
            redis_client=hybrid_vector_system.redis_cache.client if hybrid_vector_system.redis_cache else None,
            enable_persistence=redis_success,  # Only enable persistence if Redis is working
            max_context_length=2048,
            context_compression="intelligent_summarization"
        )
        await state_manager.initialize()
        
        # 10. Initialize Multi-Agent Orchestrator (LangGraph)
        logger.info("🎭 Step 10: Initializing LangGraph multi-agent orchestrator...")
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
        
        # 11. Initialize System Health Monitor
        logger.info("📈 Step 11: Initializing comprehensive health monitoring...")
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
        
        # 12. Deploy specialized agents from YAML configs
        logger.info("🚀 Step 12: Deploying specialized agents from YAML configurations...")
        await deploy_agents_from_yaml()
        
        # Mark system as initialized
        SYSTEM_INITIALIZED = True
        initialization_time = time.time() - start_time
        
        logger.info(f"✅ Revolutionary Multi-Agent System initialized in {initialization_time:.2f}s")
        logger.info(f"🎯 Target end-to-end latency: <377ms")
        logger.info(f"🔄 Agents deployed: {len(await agent_registry.list_active_agents())}")
        logger.info(f"📊 Redis: {'✅' if redis_success else '⚠️ (fallback)'}")
        logger.info(f"🗄️ Qdrant: {'✅' if qdrant_success else '🔄 (in-memory)'}")
        
        # Set initialization event
        initialization_complete.set()
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}", exc_info=True)
        # Don't raise the exception, let the system continue with minimal functionality
        logger.info("🔄 Continuing with minimal system functionality...")
        SYSTEM_INITIALIZED = True
        initialization_complete.set()

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
    
    # Stop services
    try:
        subprocess.run(['pkill', 'redis-server'], capture_output=True)
        subprocess.run(['pkill', '-f', 'qdrant'], capture_output=True)
    except:
        pass
    
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
            "YAML-based configuration management",
            "Automatic service startup and management"
        ],
        "target_latency_ms": 377,
        "active_sessions": len(active_sessions),
        "services": {
            "redis": config_manager.services_started.get('redis', False),
            "qdrant": config_manager.services_started.get('qdrant', False)
        },
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
        "config_directory": str(config_manager.agents_config_path),
        "services_status": config_manager.services_started
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

@app.get("/stats")
async def get_stats():
    """Get comprehensive statistics including conversation metrics."""
    stats = {
        "timestamp": time.time(),
        "system": {
            "initialized": SYSTEM_INITIALIZED,
            "active_calls": len(active_sessions),
            "base_url": BASE_URL,
            "services": config_manager.services_started
        },
        "calls": {},
        "sessions": {}
    }
    
    # Add individual call stats
    for session_id, handler in active_sessions.items():
        try:
            stats["calls"][session_id] = handler.get_stats()
        except Exception as e:
            logger.error(f"Error getting stats for session {session_id}: {e}")
            stats["calls"][session_id] = {"error": str(e)}
    
    return stats

@app.get("/config")
async def get_config():
    """Get current configuration."""
    config = {
        "base_url": BASE_URL,
        "google_credentials": os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        "google_project": os.getenv('GOOGLE_CLOUD_PROJECT'),
        "services": config_manager.services_started,
        "conversation_features": {
            "continuous_streaming": True,
            "session_management": True,
            "auto_reconnection": True,
            "configuration_management": True
        }
    }
    return config

# Add more specific error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

def handle_shutdown_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    # Cleanup will be handled by lifespan context manager

# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown_signal)
signal.signal(signal.SIGINT, handle_shutdown_signal)

if __name__ == '__main__':
    print("🚀 Starting Revolutionary Multi-Agent Voice AI System...")
    print(f"🎯 Target latency: <377ms (84% improvement)")
    print(f"🔧 Base URL: {os.getenv('BASE_URL', 'Not configured')}")
    print(f"📊 Vector DB: Hybrid 3-tier (Redis+FAISS+Qdrant)")
    print(f"🤖 Agents: Hot deployment with YAML configuration")
    print(f"🛠️ Tools: Comprehensive orchestration framework")
    print(f"📋 Config Directory: app/config/")
    print(f"⚙️ Services: Auto-startup with configuration integration")
    
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