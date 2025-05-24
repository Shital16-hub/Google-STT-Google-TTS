"""
Agent Registry with Hot Deployment Support.
Manages agent lifecycle, configuration, and zero-downtime deployment.
"""
import asyncio
import logging
import time
import os
import yaml
import json
import hashlib
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from app.agents.base_agent import BaseAgent, AgentStatus
from app.config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Agent deployment status."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    UPDATING = "updating"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    DEPRECATED = "deprecated"

@dataclass
class AgentDeployment:
    """Agent deployment information."""
    agent_id: str
    version: str
    config_hash: str
    status: DeploymentStatus
    deployed_at: float
    config_path: str
    instance: Optional[BaseAgent] = None
    rollback_version: Optional[str] = None
    health_check_failures: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "config_hash": self.config_hash,
            "status": self.status.value,
            "deployed_at": self.deployed_at,
            "config_path": self.config_path,
            "rollback_version": self.rollback_version,
            "health_check_failures": self.health_check_failures,
            "instance_available": self.instance is not None,
            "uptime": time.time() - self.deployed_at if self.status == DeploymentStatus.ACTIVE else 0
        }

class AgentRegistry:
    """
    Agent Registry with Hot Deployment Capabilities.
    
    Features:
    - Zero-downtime agent deployment and updates
    - Configuration validation and rollback
    - Health monitoring and automatic recovery
    - Version management and A/B testing
    - Performance tracking per agent version
    """
    
    def __init__(self, config_dir: str = "app/config/agents", vector_store=None):
        """Initialize agent registry."""
        self.config_dir = Path(config_dir)
        self.vector_store = vector_store
        
        # Agent deployments tracking
        self.deployments: Dict[str, AgentDeployment] = {}
        self.agent_classes: Dict[str, Type[BaseAgent]] = {}
        
        # Configuration management
        self.config_cache = {}
        self.config_watchers = {}
        
        # Health monitoring
        self.health_check_interval = 60  # seconds
        self.max_health_failures = 3
        
        # Performance tracking
        self.deployment_metrics = {
            'total_deployments': 0,
            'successful_deployments': 0,
            'failed_deployments': 0,
            'rollbacks': 0,
            'avg_deployment_time': 0.0
        }
        
        # Background tasks
        self.health_monitor_task = None
        self.config_watcher_task = None
        
        logger.info(f"AgentRegistry initialized with config dir: {self.config_dir}")
    
    async def init(self):
        """Initialize the agent registry."""
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Register built-in agent classes
        await self._register_builtin_agents()
        
        # Load existing agent configurations
        await self._load_existing_configurations()
        
        # Start background monitoring
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.config_watcher_task = asyncio.create_task(self._config_watcher_loop())
        
        logger.info("✅ Agent registry initialized")
    
    async def _register_builtin_agents(self):
        """Register built-in agent classes."""
        # Import agent classes
        try:
            from app.agents.roadside_agent import RoadsideAssistanceAgent
            from app.agents.billing_agent import BillingSupportAgent
            from app.agents.technical_agent import TechnicalSupportAgent
            
            self.agent_classes.update({
                "roadside-assistance": RoadsideAssistanceAgent,
                "billing-support": BillingSupportAgent,
                "technical-support": TechnicalSupportAgent
            })
            
            logger.info(f"Registered {len(self.agent_classes)} built-in agent classes")
            
        except ImportError as e:
            logger.warning(f"Could not import some agent classes: {e}")
    
    async def _load_existing_configurations(self):
        """Load existing agent configurations from disk."""
        if not self.config_dir.exists():
            return
        
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        
        for config_file in config_files:
            try:
                config = await self._load_config_file(config_file)
                agent_id = config.get("agent_id")
                
                if agent_id and agent_id in self.agent_classes:
                    await self._deploy_from_config(config, config_file)
                    
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
    
    async def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load and validate agent configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ["agent_id", "version", "status"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Cache configuration
            config_hash = self._calculate_config_hash(config)
            self.config_cache[str(config_path)] = {
                "config": config,
                "hash": config_hash,
                "loaded_at": time.time()
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            raise
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for change detection."""
        # Remove timestamp fields that shouldn't affect hash
        config_copy = config.copy()
        config_copy.pop("deployed_at", None)
        config_copy.pop("last_updated", None)
        
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    async def deploy_agent(
        self,
        agent_id: str,
        version: str,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        hot_deploy: bool = True
    ) -> Dict[str, Any]:
        """
        Deploy or update an agent with hot deployment support.
        
        Args:
            agent_id: Unique agent identifier
            version: Agent version
            config: Agent configuration dictionary
            config_file: Path to configuration file
            hot_deploy: Enable zero-downtime deployment
            
        Returns:
            Deployment result with status and metrics
        """
        deployment_start = time.time()
        
        try:
            logger.info(f"🚀 Starting deployment: {agent_id} v{version}")
            
            # Load configuration
            if config_file:
                config_path = Path(config_file)
                config = await self._load_config_file(config_path)
            elif config:
                # Save config to file for persistence
                config_path = self.config_dir / f"{agent_id}.yaml"
                await self._save_config_file(config, config_path)
            else:
                raise ValueError("Either config or config_file must be provided")
            
            # Validate configuration
            await self._validate_agent_config(config)
            
            # Calculate config hash
            config_hash = self._calculate_config_hash(config)
            
            # Check if agent class is registered
            if agent_id not in self.agent_classes:
                raise ValueError(f"Agent class not registered: {agent_id}")
            
            # Check if this is an update
            existing_deployment = self.deployments.get(agent_id)
            is_update = existing_deployment is not None
            
            if is_update and hot_deploy:
                # Hot deployment - prepare new instance while keeping old one running
                result = await self._hot_deploy_agent(
                    agent_id, version, config, config_path, config_hash, existing_deployment
                )
            else:
                # Fresh deployment or cold update
                result = await self._deploy_fresh_agent(
                    agent_id, version, config, config_path, config_hash
                )
            
            # Update deployment metrics
            deployment_time = time.time() - deployment_start
            self._update_deployment_metrics(True, deployment_time)
            
            result["deployment_time"] = deployment_time
            
            logger.info(f"✅ Successfully deployed {agent_id} v{version} in {deployment_time:.2f}s")
            
            return result
            
        except Exception as e:
            deployment_time = time.time() - deployment_start
            self._update_deployment_metrics(False, deployment_time)
            
            logger.error(f"❌ Failed to deploy {agent_id} v{version}: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "deployment_time": deployment_time,
                "agent_id": agent_id,
                "version": version
            }
    
    async def _save_config_file(self, config: Dict[str, Any], config_path: Path):
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    async def _validate_agent_config(self, config: Dict[str, Any]):
        """Validate agent configuration."""
        required_sections = ["agent_id", "version", "specialization"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specialization section
        specialization = config["specialization"]
        if "system_prompt" not in specialization:
            raise ValueError("Missing system_prompt in specialization section")
        
        # Validate tools if present
        if "tools" in config:
            for tool in config["tools"]:
                if "name" not in tool or "type" not in tool:
                    raise ValueError("Tools must have 'name' and 'type' fields")
        
        # Validate routing if present
        if "routing" in config:
            routing = config["routing"]
            if "primary_keywords" not in routing:
                raise ValueError("Routing configuration must include primary_keywords")
    
    async def _hot_deploy_agent(
        self,
        agent_id: str,
        version: str,
        config: Dict[str, Any],
        config_path: Path,
        config_hash: str,
        existing_deployment: AgentDeployment
    ) -> Dict[str, Any]:
        """Perform hot deployment with zero downtime."""
        
        # Mark as updating
        existing_deployment.status = DeploymentStatus.UPDATING
        
        try:
            # Create new agent instance
            logger.info(f"🔄 Creating new instance for {agent_id} v{version}")
            new_instance = await self._create_agent_instance(agent_id, config)
            
            # Perform health check on new instance
            if not await new_instance.health_check():
                raise Exception("New agent instance failed health check")
            
            # Prepare new deployment record
            new_deployment = AgentDeployment(
                agent_id=agent_id,
                version=version,
                config_hash=config_hash,
                status=DeploymentStatus.ACTIVE,
                deployed_at=time.time(),
                config_path=str(config_path),
                instance=new_instance,
                rollback_version=existing_deployment.version
            )
            
            # Atomic swap - replace old with new
            old_instance = existing_deployment.instance
            self.deployments[agent_id] = new_deployment
            
            # Gracefully shutdown old instance
            if old_instance:
                try:
                    await old_instance.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down old instance: {e}")
            
            logger.info(f"🔄 Hot deployment completed for {agent_id}")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "version": version,
                "deployment_type": "hot_deploy",
                "previous_version": existing_deployment.version
            }
            
        except Exception as e:
            # Rollback on failure
            existing_deployment.status = DeploymentStatus.ACTIVE
            logger.error(f"Hot deployment failed, maintaining existing version: {e}")
            raise
    
    async def _deploy_fresh_agent(
        self,
        agent_id: str,
        version: str,
        config: Dict[str, Any],
        config_path: Path,
        config_hash: str
    ) -> Dict[str, Any]:
        """Deploy fresh agent instance."""
        
        # Create deployment record
        deployment = AgentDeployment(
            agent_id=agent_id,
            version=version,
            config_hash=config_hash,
            status=DeploymentStatus.DEPLOYING,
            deployed_at=time.time(),
            config_path=str(config_path)
        )
        
        try:
            # Create agent instance
            logger.info(f"🆕 Creating fresh instance for {agent_id} v{version}")
            instance = await self._create_agent_instance(agent_id, config)
            
            # Perform health check
            if not await instance.health_check():
                raise Exception("Agent instance failed health check")
            
            # Update deployment
            deployment.instance = instance
            deployment.status = DeploymentStatus.ACTIVE
            
            # Store deployment
            self.deployments[agent_id] = deployment
            
            logger.info(f"🆕 Fresh deployment completed for {agent_id}")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "version": version,
                "deployment_type": "fresh_deploy"
            }
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            self.deployments[agent_id] = deployment
            raise
    
    async def _create_agent_instance(self, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create and initialize agent instance."""
        if agent_id not in self.agent_classes:
            raise ValueError(f"Agent class not registered: {agent_id}")
        
        agent_class = self.agent_classes[agent_id]
        
        # Create instance
        instance = agent_class(
            agent_id=agent_id,
            agent_config=config,
            vector_store=self.vector_store
        )
        
        # Initialize
        await instance.init()
        
        return instance
    
    async def _deploy_from_config(self, config: Dict[str, Any], config_path: Path):
        """Deploy agent from existing configuration file."""
        agent_id = config["agent_id"]
        version = config["version"]
        
        if config.get("status") == "active":
            try:
                await self.deploy_agent(
                    agent_id=agent_id,
                    version=version,
                    config=config,
                    hot_deploy=False  # Cold start for initial load
                )
            except Exception as e:
                logger.error(f"Failed to deploy {agent_id} from config: {e}")
    
    def _update_deployment_metrics(self, success: bool, deployment_time: float):
        """Update deployment metrics."""
        self.deployment_metrics['total_deployments'] += 1
        
        if success:
            self.deployment_metrics['successful_deployments'] += 1
        else:
            self.deployment_metrics['failed_deployments'] += 1
        
        # Update average deployment time
        total = self.deployment_metrics['total_deployments']
        current_avg = self.deployment_metrics['avg_deployment_time']
        self.deployment_metrics['avg_deployment_time'] = (
            (current_avg * (total - 1) + deployment_time) / total
        )
    
    async def rollback_agent(self, agent_id: str) -> Dict[str, Any]:
        """Rollback agent to previous version."""
        if agent_id not in self.deployments:
            raise ValueError(f"Agent not found: {agent_id}")
        
        deployment = self.deployments[agent_id]
        
        if not deployment.rollback_version:
            raise ValueError(f"No rollback version available for {agent_id}")
        
        logger.info(f"🔄 Rolling back {agent_id} to version {deployment.rollback_version}")
        
        deployment.status = DeploymentStatus.ROLLING_BACK
        
        try:
            # Find rollback configuration
            rollback_config_path = self.config_dir / f"{agent_id}_v{deployment.rollback_version}.yaml"
            
            if rollback_config_path.exists():
                rollback_config = await self._load_config_file(rollback_config_path)
            else:
                # Try to reconstruct config for rollback (simplified approach)
                rollback_config = deployment.instance.config.copy()
                rollback_config["version"] = deployment.rollback_version
            
            # Deploy rollback version
            result = await self.deploy_agent(
                agent_id=agent_id,
                version=deployment.rollback_version,
                config=rollback_config,
                hot_deploy=True
            )
            
            if result["success"]:
                self.deployment_metrics['rollbacks'] += 1
                logger.info(f"✅ Successfully rolled back {agent_id}")
            
            return result
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            logger.error(f"❌ Rollback failed for {agent_id}: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    async def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs."""
        return [
            agent_id for agent_id, deployment in self.deployments.items()
            if deployment.status == DeploymentStatus.ACTIVE and deployment.instance
        ]
    
    async def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration."""
        if agent_id not in self.deployments:
            return None
        
        deployment = self.deployments[agent_id]
        
        if deployment.instance:
            return deployment.instance.config
        
        return None
    
    async def create_agent_instance(self, agent_id: str, config: Dict[str, Any]) -> Optional[BaseAgent]:
        """Create agent instance (for orchestrator use)."""
        try:
            return await self._create_agent_instance(agent_id, config)
        except Exception as e:
            logger.error(f"Failed to create agent instance {agent_id}: {e}")
            return None
    
    async def get_agent_instance(self, agent_id: str) -> Optional[BaseAgent]:
        """Get active agent instance."""
        if agent_id not in self.deployments:
            return None
        
        deployment = self.deployments[agent_id]
        
        if deployment.status == DeploymentStatus.ACTIVE:
            return deployment.instance
        
        return None
    
    async def get_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {
            "total_agents": len(self.deployments),
            "active_agents": len([d for d in self.deployments.values() if d.status == DeploymentStatus.ACTIVE]),
            "agents": {}
        }
        
        for agent_id, deployment in self.deployments.items():
            agent_status = deployment.to_dict()
            
            # Add instance metrics if available
            if deployment.instance:
                try:
                    agent_metrics = await deployment.instance.get_metrics()
                    agent_status["metrics"] = agent_metrics
                except Exception as e:
                    agent_status["metrics_error"] = str(e)
            
            status["agents"][agent_id] = agent_status
        
        # Add deployment metrics
        status["deployment_metrics"] = self.deployment_metrics.copy()
        
        return status
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _perform_health_checks(self):
        """Perform health checks on all active agents."""
        for agent_id, deployment in self.deployments.items():
            if deployment.status != DeploymentStatus.ACTIVE or not deployment.instance:
                continue
            
            try:
                is_healthy = await deployment.instance.health_check()
                
                if is_healthy:
                    deployment.health_check_failures = 0
                else:
                    deployment.health_check_failures += 1
                    logger.warning(f"Health check failed for {agent_id} "
                                 f"({deployment.health_check_failures}/{self.max_health_failures})")
                    
                    # Auto-restart or rollback if too many failures
                    if deployment.health_check_failures >= self.max_health_failures:
                        await self._handle_unhealthy_agent(agent_id, deployment)
                
            except Exception as e:
                deployment.health_check_failures += 1
                logger.error(f"Health check error for {agent_id}: {e}")
    
    async def _handle_unhealthy_agent(self, agent_id: str, deployment: AgentDeployment):
        """Handle unhealthy agent - attempt restart or rollback."""
        logger.warning(f"🚨 Agent {agent_id} is unhealthy, attempting recovery")
        
        try:
            # First try to restart the agent
            if deployment.instance:
                await deployment.instance.shutdown()
            
            # Recreate instance
            new_instance = await self._create_agent_instance(agent_id, deployment.instance.config)
            
            if await new_instance.health_check():
                deployment.instance = new_instance
                deployment.health_check_failures = 0
                logger.info(f"✅ Successfully restarted {agent_id}")
                return
            
        except Exception as e:
            logger.error(f"Failed to restart {agent_id}: {e}")
        
        # If restart failed and rollback is available, try rollback
        if deployment.rollback_version:
            try:
                await self.rollback_agent(agent_id)
                logger.info(f"🔄 Rolled back unhealthy agent {agent_id}")
            except Exception as e:
                logger.error(f"Failed to rollback {agent_id}: {e}")
                deployment.status = DeploymentStatus.FAILED
        else:
            deployment.status = DeploymentStatus.FAILED
            logger.error(f"❌ Agent {agent_id} marked as failed - no recovery options")
    
    async def _config_watcher_loop(self):
        """Background configuration file watcher."""
        last_check = time.time()
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check for configuration file changes
                for config_path in self.config_dir.glob("*.yaml"):
                    if config_path.stat().st_mtime > last_check:
                        await self._handle_config_change(config_path)
                
                last_check = time.time()
                
            except Exception as e:
                logger.error(f"Error in config watcher loop: {e}")
                await asyncio.sleep(60)
    
    async def _handle_config_change(self, config_path: Path):
        """Handle configuration file change."""
        try:
            config = await self._load_config_file(config_path)
            agent_id = config.get("agent_id")
            
            if not agent_id or agent_id not in self.agent_classes:
                return
            
            # Check if this is a new configuration hash
            config_hash = self._calculate_config_hash(config)
            
            if agent_id in self.deployments:
                current_deployment = self.deployments[agent_id]
                if current_deployment.config_hash != config_hash:
                    logger.info(f"🔄 Configuration changed for {agent_id}, triggering hot deployment")
                    
                    await self.deploy_agent(
                        agent_id=agent_id,
                        version=config["version"],
                        config=config,
                        hot_deploy=True
                    )
            else:
                # New agent configuration
                if config.get("status") == "active":
                    logger.info(f"🆕 New agent configuration detected: {agent_id}")
                    await self._deploy_from_config(config, config_path)
                    
        except Exception as e:
            logger.error(f"Error handling config change for {config_path}: {e}")
    
    async def health_check(self) -> bool:
        """Perform health check on the registry itself."""
        try:
            # Check if we have active agents
            active_agents = await self.get_active_agents()
            if not active_agents:
                logger.warning("No active agents in registry")
                return False
            
            # Check if background tasks are running
            if self.health_monitor_task and self.health_monitor_task.done():
                logger.error("Health monitor task has stopped")
                return False
            
            if self.config_watcher_task and self.config_watcher_task.done():
                logger.error("Config watcher task has stopped")
                return False
            
            # Check agent health
            unhealthy_agents = 0
            for agent_id in active_agents:
                deployment = self.deployments[agent_id]
                if deployment.health_check_failures > 0:
                    unhealthy_agents += 1
            
            # If more than 50% of agents are unhealthy, consider registry unhealthy
            if unhealthy_agents > len(active_agents) * 0.5:
                logger.warning(f"Too many unhealthy agents: {unhealthy_agents}/{len(active_agents)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Registry health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown agent registry gracefully."""
        logger.info("🛑 Shutting down agent registry...")
        
        # Cancel background tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        if self.config_watcher_task:
            self.config_watcher_task.cancel()
        
        # Shutdown all agents
        for agent_id, deployment in self.deployments.items():
            if deployment.instance:
                try:
                    await deployment.instance.shutdown()
                    logger.info(f"✅ Shutdown agent: {agent_id}")
                except Exception as e:
                    logger.error(f"Error shutting down {agent_id}: {e}")
        
        # Clear deployments
        self.deployments.clear()
        
        logger.info("✅ Agent registry shutdown complete")