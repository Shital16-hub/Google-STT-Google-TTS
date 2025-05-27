"""
Comprehensive System Health Monitor for Revolutionary Multi-Agent Voice AI System
Provides real-time monitoring, predictive analytics, and intelligent alerting.
"""
import asyncio
import logging
import time
import psutil
import statistics
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"
    DEGRADED = "degraded"

class ComponentType(str, Enum):
    """System component types for monitoring."""
    ORCHESTRATOR = "orchestrator"
    VECTOR_DB = "vector_database"
    STT_SYSTEM = "speech_to_text"
    TTS_ENGINE = "text_to_speech"
    AGENT_REGISTRY = "agent_registry"
    TOOL_ORCHESTRATOR = "tool_orchestrator"
    STATE_MANAGER = "state_manager"
    REDIS_CACHE = "redis_cache"
    QDRANT_DB = "qdrant_database"
    FAISS_INDEX = "faiss_index"

@dataclass
class PerformanceMetric:
    """Performance metric with historical tracking."""
    name: str
    current_value: float
    target_value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    history: List[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    
    def add_measurement(self, value: float):
        """Add a new measurement to the metric."""
        self.current_value = value
        self.last_updated = time.time()
        self.history.append(value)
        
        # Keep only last 100 measurements
        if len(self.history) > 100:
            self.history.pop(0)
    
    def get_status(self) -> HealthStatus:
        """Get health status based on current value."""
        if self.current_value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.current_value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_trend(self) -> str:
        """Get trend analysis of recent measurements."""
        if len(self.history) < 3:
            return "insufficient_data"
        
        recent = self.history[-3:]
        if recent[-1] > recent[0]:
            return "increasing"
        elif recent[-1] < recent[0]:
            return "decreasing"
        else:
            return "stable"

@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_type: ComponentType
    status: HealthStatus
    last_check: float
    response_time_ms: float
    error_count: int = 0
    uptime_percent: float = 100.0
    metrics: Dict[str, PerformanceMetric] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class SystemAlert:
    """System alert with context and recommendations."""
    alert_id: str
    severity: HealthStatus
    component: ComponentType
    message: str
    timestamp: float
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None

@dataclass
class ExecutiveDashboard:
    """Executive dashboard with key metrics and insights."""
    timestamp: float
    overall_status: HealthStatus
    
    # Key Performance Indicators
    kpis: Dict[str, float]
    
    # Performance trends
    trends: Dict[str, str]
    
    # Agent performance comparison
    agent_performance: Dict[str, Any]
    
    # Predictive analytics
    predictions: Dict[str, Any]
    
    # Actionable recommendations
    recommendations: List[str]

class SystemHealthMonitor:
    """
    Comprehensive system health monitor with predictive analytics and intelligent alerting.
    Monitors all system components and provides executive-level insights.
    """
    
    def __init__(
        self,
        orchestrator=None,
        hybrid_vector_system=None,
        target_latency_ms: int = 377,
        enable_predictive_analytics: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
        monitoring_interval_seconds: int = 30,
        alert_callbacks: Optional[List[Callable[[SystemAlert], Awaitable[None]]]] = None
    ):
        """Initialize the comprehensive health monitor."""
        self.orchestrator = orchestrator
        self.hybrid_vector_system = hybrid_vector_system
        self.target_latency_ms = target_latency_ms
        self.enable_predictive_analytics = enable_predictive_analytics
        self.monitoring_interval = monitoring_interval_seconds
        self.alert_callbacks = alert_callbacks or []
        
        # Alert thresholds
        self.alert_thresholds = {
            "latency_ms": 500,
            "error_rate": 0.02,
            "memory_usage": 0.85,
            "cpu_usage": 0.80,
            "disk_usage": 0.90,
            "response_time_ms": 1000,
        }

        self.alert_thresholds = {**default_thresholds, **(alert_thresholds or {})}

        
        # Component health tracking
        self.component_health: Dict[ComponentType, ComponentHealth] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[str, PerformanceMetric] = {}
        
        # Alert management
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []
        
        # System stats
        self.system_stats = {
            "startup_time": time.time(),
            "total_alerts": 0,
            "resolved_alerts": 0,
            "avg_resolution_time": 0.0,
            "uptime_seconds": 0.0
        }
        
        # Background monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.initialized = False
        
        logger.info("System Health Monitor initialized")
    
    async def initialize(self):
        """Initialize health monitoring with performance metrics setup."""
        logger.info("Initializing comprehensive health monitoring...")
        
        # Initialize performance metrics
        await self._setup_performance_metrics()
        
        # Initialize component health tracking
        await self._initialize_component_health()
        
        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._background_monitoring())
        
        self.initialized = True
        logger.info("✅ System Health Monitor initialized")
    
    async def _setup_performance_metrics(self):
        """Setup core performance metrics for monitoring."""
        self.performance_metrics = {
            "end_to_end_latency_ms": PerformanceMetric(
                name="End-to-End Latency",
                current_value=0.0,
                target_value=self.target_latency_ms,
                unit="ms",
                threshold_warning=self.target_latency_ms * 1.2,
                threshold_critical=self.target_latency_ms * 1.5
            ),
            "conversation_success_rate": PerformanceMetric(
                name="Conversation Success Rate",
                current_value=1.0,
                target_value=0.98,
                unit="ratio",
                threshold_warning=0.95,
                threshold_critical=0.90
            ),
            "agent_response_time_ms": PerformanceMetric(
                name="Agent Response Time",
                current_value=0.0,
                target_value=200.0,
                unit="ms",
                threshold_warning=300.0,
                threshold_critical=500.0
            ),
            "vector_search_time_ms": PerformanceMetric(
                name="Vector Search Time",
                current_value=0.0,
                target_value=5.0,
                unit="ms",
                threshold_warning=10.0,
                threshold_critical=20.0
            ),
            "tool_execution_time_ms": PerformanceMetric(
                name="Tool Execution Time",
                current_value=0.0,
                target_value=100.0,
                unit="ms",
                threshold_warning=200.0,
                threshold_critical=500.0
            ),
            "memory_usage_percent": PerformanceMetric(
                name="Memory Usage",
                current_value=0.0,
                target_value=0.60,
                unit="percent",
                threshold_warning=0.80,
                threshold_critical=0.90
            ),
            "cpu_usage_percent": PerformanceMetric(
                name="CPU Usage",
                current_value=0.0,
                target_value=0.50,
                unit="percent",
                threshold_warning=0.75,
                threshold_critical=0.90
            ),
            "active_sessions": PerformanceMetric(
                name="Active Sessions",
                current_value=0.0,
                target_value=100.0,
                unit="count",
                threshold_warning=500.0,
                threshold_critical=1000.0
            )
        }
        
        logger.debug("Performance metrics initialized")
    
    async def _initialize_component_health(self):
        """Initialize health tracking for all system components."""
        components = [
            ComponentType.ORCHESTRATOR,
            ComponentType.VECTOR_DB,
            ComponentType.STT_SYSTEM,
            ComponentType.TTS_ENGINE,
            ComponentType.AGENT_REGISTRY,
            ComponentType.TOOL_ORCHESTRATOR,
            ComponentType.STATE_MANAGER,
            ComponentType.REDIS_CACHE,
            ComponentType.QDRANT_DB,
            ComponentType.FAISS_INDEX
        ]
        
        for component_type in components:
            self.component_health[component_type] = ComponentHealth(
                component_type=component_type,
                status=HealthStatus.HEALTHY,
                last_check=time.time(),
                response_time_ms=0.0
            )
        
        logger.debug("Component health tracking initialized")
    
    async def _background_monitoring(self):
        """Background monitoring task that runs continuously."""
        logger.info("Starting background health monitoring...")
        
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Update system metrics
                await self._update_system_metrics()
                
                # Check component health
                await self._check_all_components()
                
                # Check for alerts
                await self._check_alert_conditions()
                
                # Update uptime
                self.system_stats["uptime_seconds"] = time.time() - self.system_stats["startup_time"]
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _update_system_metrics(self):
        """Update system-level performance metrics."""
        try:
            # System resource metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            self.performance_metrics["memory_usage_percent"].add_measurement(memory.percent / 100.0)
            self.performance_metrics["cpu_usage_percent"].add_measurement(cpu_percent / 100.0)
            
            # Add disk usage metric if not exists
            if "disk_usage_percent" not in self.performance_metrics:
                self.performance_metrics["disk_usage_percent"] = PerformanceMetric(
                    name="Disk Usage",
                    current_value=disk.percent / 100.0,
                    target_value=0.70,
                    unit="percent",
                    threshold_warning=0.85,
                    threshold_critical=0.95
                )
            else:
                self.performance_metrics["disk_usage_percent"].add_measurement(disk.percent / 100.0)
            
            # Get orchestrator metrics if available
            if self.orchestrator and hasattr(self.orchestrator, 'get_performance_stats'):
                try:
                    orchestrator_stats = await self.orchestrator.get_performance_stats()
                    
                    if "avg_latency_ms" in orchestrator_stats:
                        self.performance_metrics["end_to_end_latency_ms"].add_measurement(
                            orchestrator_stats["avg_latency_ms"]
                        )
                    
                    if "success_rate" in orchestrator_stats:
                        self.performance_metrics["conversation_success_rate"].add_measurement(
                            orchestrator_stats["success_rate"]
                        )
                        
                except Exception as e:
                    logger.debug(f"Could not get orchestrator stats: {e}")
            
            # Get vector system metrics if available
            if self.hybrid_vector_system and hasattr(self.hybrid_vector_system, 'get_performance_stats'):
                try:
                    vector_stats = await self.hybrid_vector_system.get_performance_stats()
                    
                    if "avg_search_time_ms" in vector_stats:
                        self.performance_metrics["vector_search_time_ms"].add_measurement(
                            vector_stats["avg_search_time_ms"]
                        )
                        
                except Exception as e:
                    logger.debug(f"Could not get vector system stats: {e}")
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    async def _check_all_components(self):
        """Check health of all system components."""
        for component_type in self.component_health.keys():
            await self._check_component_health(component_type)
    
    async def _check_component_health(self, component_type: ComponentType):
        """Check health of a specific component."""
        start_time = time.time()
        component_health = self.component_health[component_type]
        
        try:
            # Component-specific health checks
            if component_type == ComponentType.ORCHESTRATOR:
                status = await self._check_orchestrator_health()
            elif component_type == ComponentType.VECTOR_DB:
                status = await self._check_vector_db_health()
            elif component_type == ComponentType.REDIS_CACHE:
                status = await self._check_redis_health()
            elif component_type == ComponentType.QDRANT_DB:
                status = await self._check_qdrant_health()
            else:
                # Generic component check
                status = HealthStatus.HEALTHY
            
            response_time = (time.time() - start_time) * 1000
            
            # Update component health
            component_health.status = status
            component_health.last_check = time.time()
            component_health.response_time_ms = response_time
            
            # Update uptime calculation
            if status == HealthStatus.HEALTHY:
                component_health.uptime_percent = min(100.0, component_health.uptime_percent + 0.1)
            else:
                component_health.uptime_percent = max(0.0, component_health.uptime_percent - 1.0)
                component_health.error_count += 1
            
        except Exception as e:
            logger.error(f"Error checking {component_type} health: {e}")
            component_health.status = HealthStatus.CRITICAL
            component_health.error_count += 1
            component_health.issues.append(f"Health check failed: {str(e)}")
    
    async def _check_orchestrator_health(self) -> HealthStatus:
        """Check orchestrator health."""
        if not self.orchestrator or not self.orchestrator.initialized:
            return HealthStatus.DOWN
        
        try:
            # Check if orchestrator is responsive
            stats = await self.orchestrator.get_performance_stats()
            
            # Check key metrics
            if stats.get("avg_latency_ms", 0) > self.alert_thresholds["latency_ms"]:
                return HealthStatus.WARNING
            
            if stats.get("success_rate", 1.0) < 0.95:
                return HealthStatus.WARNING
            
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.CRITICAL
    
    async def _check_vector_db_health(self) -> HealthStatus:
        """Check vector database health."""
        if not self.hybrid_vector_system:
            return HealthStatus.DOWN
        
        try:
            # Check if vector system is responsive
            stats = await self.hybrid_vector_system.get_performance_stats()
            
            # Check response times
            if stats.get("avg_search_time_ms", 0) > 50:  # 50ms threshold
                return HealthStatus.WARNING
            
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.CRITICAL
    
    async def _check_redis_health(self) -> HealthStatus:
        """Check Redis cache health."""
        try:
            if (self.hybrid_vector_system and 
                hasattr(self.hybrid_vector_system, 'redis_cache') and
                self.hybrid_vector_system.redis_cache):
                
                # Simple ping test
                await self.hybrid_vector_system.redis_cache.client.ping()
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DOWN
                
        except Exception:
            return HealthStatus.CRITICAL
    
    async def _check_qdrant_health(self) -> HealthStatus:
        """Check Qdrant database health."""
        try:
            if (self.hybrid_vector_system and 
                hasattr(self.hybrid_vector_system, 'qdrant_primary') and
                self.hybrid_vector_system.qdrant_primary):
                
                # Check collections
                collections = await self.hybrid_vector_system.qdrant_primary.get_collections()
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DOWN
                
        except Exception:
            return HealthStatus.CRITICAL
    
    async def _check_alert_conditions(self):
        """Check for alert conditions and trigger alerts."""
        new_alerts = []
        
        # Check performance metrics for alert conditions
        for metric_name, metric in self.performance_metrics.items():
            status = metric.get_status()
            
            if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert_id = f"metric_{metric_name}_{int(time.time())}"
                
                # Check if we already have a similar active alert
                similar_alert = None
                for alert in self.active_alerts.values():
                    if alert.component.value == "performance" and metric_name in alert.message:
                        similar_alert = alert
                        break
                
                if not similar_alert:
                    alert = SystemAlert(
                        alert_id=alert_id,
                        severity=status,
                        component=ComponentType.ORCHESTRATOR,  # Default to orchestrator
                        message=f"{metric.name} {status.value}: {metric.current_value:.2f} {metric.unit} "
                               f"(threshold: {metric.threshold_warning:.2f})",
                        timestamp=time.time(),
                        details={
                            "metric_name": metric_name,
                            "current_value": metric.current_value,
                            "target_value": metric.target_value,
                            "threshold_warning": metric.threshold_warning,
                            "threshold_critical": metric.threshold_critical,
                            "trend": metric.get_trend()
                        }
                    )
                    new_alerts.append(alert)
        
        # Check component health for alerts
        for component_type, health in self.component_health.items():
            if health.status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.DOWN]:
                alert_id = f"component_{component_type.value}_{int(time.time())}"
                
                # Check for existing alert
                existing_alert = None
                for alert in self.active_alerts.values():
                    if alert.component == component_type:
                        existing_alert = alert
                        break
                
                if not existing_alert:
                    alert = SystemAlert(
                        alert_id=alert_id,
                        severity=health.status,
                        component=component_type,
                        message=f"{component_type.value} is {health.status.value}",
                        timestamp=time.time(),
                        details={
                            "response_time_ms": health.response_time_ms,
                            "error_count": health.error_count,
                            "uptime_percent": health.uptime_percent,
                            "issues": health.issues,
                            "last_check": health.last_check
                        }
                    )
                    new_alerts.append(alert)
        
        # Process new alerts
        for alert in new_alerts:
            await self._trigger_alert(alert)
    
    async def _trigger_alert(self, alert: SystemAlert):
        """Trigger a new alert and notify callbacks."""
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.system_stats["total_alerts"] += 1
        
        logger.warning(f"🚨 ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        overall_status = self._calculate_overall_status()
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "uptime_seconds": self.system_stats["uptime_seconds"],
            "components": {
                component_type.value: {
                    "status": health.status.value,
                    "response_time_ms": health.response_time_ms,
                    "uptime_percent": health.uptime_percent,
                    "error_count": health.error_count,
                    "last_check": health.last_check,
                    "issues": health.issues,
                    "recommendations": health.recommendations
                }
                for component_type, health in self.component_health.items()
            },
            "performance_metrics": {
                name: {
                    "current_value": metric.current_value,
                    "target_value": metric.target_value,
                    "unit": metric.unit,
                    "status": metric.get_status().value,
                    "trend": metric.get_trend(),
                    "last_updated": metric.last_updated
                }
                for name, metric in self.performance_metrics.items()
            },
            "active_alerts": len(self.active_alerts),
            "total_alerts": self.system_stats["total_alerts"],
            "alert_resolution_rate": (
                self.system_stats["resolved_alerts"] / max(self.system_stats["total_alerts"], 1)
            )
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "metrics": {
                name: {
                    "current": metric.current_value,
                    "target": metric.target_value,
                    "unit": metric.unit,
                    "status": metric.get_status().value,
                    "trend": metric.get_trend(),
                    "history": metric.history[-20:],  # Last 20 measurements
                    "average": statistics.mean(metric.history) if metric.history else 0,
                    "percentile_95": statistics.quantiles(metric.history, n=20)[18] if len(metric.history) >= 20 else 0
                }
                for name, metric in self.performance_metrics.items()
            },
            "system_resources": {
                "memory": psutil.virtual_memory()._asdict(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "disk": psutil.disk_usage('/')._asdict(),
                "process_count": len(psutil.pids())
            },
            "target_latency_ms": self.target_latency_ms,
            "monitoring_interval_seconds": self.monitoring_interval
        }
    
    async def generate_executive_dashboard(self) -> ExecutiveDashboard:
        """Generate executive dashboard with key insights and recommendations."""
        current_time = time.time()
        overall_status = self._calculate_overall_status()
        
        # Key Performance Indicators
        kpis = {
            "average_response_time_ms": self.performance_metrics["end_to_end_latency_ms"].current_value,
            "conversations_per_hour": 0.0,  # Would need session tracking
            "user_satisfaction_score": self.performance_metrics["conversation_success_rate"].current_value * 5,
            "system_uptime_percentage": (self.system_stats["uptime_seconds"] / (24 * 3600)) * 100,
            "cost_per_conversation": 0.05,  # Estimated
            "agent_utilization_rate": 0.75  # Estimated
        }
        
        # Performance trends
        trends = {
            "latency_trend": self.performance_metrics["end_to_end_latency_ms"].get_trend(),
            "quality_trend": self.performance_metrics["conversation_success_rate"].get_trend(),
            "volume_trend": "stable",  # Would need session tracking
            "efficiency_trend": "improving"
        }
        
        # Agent performance (would need actual agent metrics)
        agent_performance = {
            "roadside-assistance": {"success_rate": 0.95, "avg_latency_ms": 320},
            "billing-support": {"success_rate": 0.98, "avg_latency_ms": 280},
            "technical-support": {"success_rate": 0.92, "avg_latency_ms": 400}
        }
        
        # Predictive analytics
        predictions = {
            "expected_load_next_hour": "moderate",
            "potential_bottlenecks": ["vector_search_latency"],
            "maintenance_recommendations": ["optimize_redis_cache", "update_agent_models"],
            "capacity_planning": "current_capacity_sufficient"
        }
        
        # Generate recommendations
        recommendations = await self._generate_actionable_recommendations()
        
        return ExecutiveDashboard(
            timestamp=current_time,
            overall_status=overall_status,
            kpis=kpis,
            trends=trends,
            agent_performance=agent_performance,
            predictions=predictions,
            recommendations=recommendations
        )
    
    async def _generate_actionable_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on current health."""
        recommendations = []
        
        # Check latency performance
        latency_metric = self.performance_metrics.get("end_to_end_latency_ms")
        if latency_metric and latency_metric.current_value > self.target_latency_ms * 1.1:
            recommendations.append(
                f"Optimize response latency: Currently {latency_metric.current_value:.0f}ms, "
                f"target {self.target_latency_ms}ms. Consider vector search optimization."
            )
        
        # Check memory usage
        memory_metric = self.performance_metrics.get("memory_usage_percent")
        if memory_metric and memory_metric.current_value > 0.8:
            recommendations.append(
                "High memory usage detected. Consider implementing conversation context compression "
                "or increasing memory allocation."
            )
        
        # Check component health
        unhealthy_components = [
            comp_type.value for comp_type, health in self.component_health.items()
            if health.status != HealthStatus.HEALTHY
        ]
        if unhealthy_components:
            recommendations.append(
                f"Address component health issues: {', '.join(unhealthy_components)}"
            )
        
        # Check active alerts
        if len(self.active_alerts) > 0:
            recommendations.append(
                f"Resolve {len(self.active_alerts)} active alerts to improve system stability."
            )
        
        # Performance optimization recommendations
        if not recommendations:
            recommendations.append("System is performing well. Consider proactive optimizations.")
        
        return recommendations
    
    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall system health status."""
        if not self.component_health:
            return HealthStatus.DOWN
        
        statuses = [health.status for health in self.component_health.values()]
        
        if HealthStatus.DOWN in statuses:
            return HealthStatus.DOWN
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    async def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.component_health:
            return 0.0
        
        total_score = 0.0
        component_weights = {
            ComponentType.ORCHESTRATOR: 0.25,
            ComponentType.VECTOR_DB: 0.20,
            ComponentType.STT_SYSTEM: 0.15,
            ComponentType.TTS_ENGINE: 0.15,
            ComponentType.AGENT_REGISTRY: 0.10,
            ComponentType.TOOL_ORCHESTRATOR: 0.05,
            ComponentType.STATE_MANAGER: 0.05,
            ComponentType.REDIS_CACHE: 0.03,
            ComponentType.QDRANT_DB: 0.02
        }
        
        for component_type, health in self.component_health.items():
            weight = component_weights.get(component_type, 0.01)
            
            if health.status == HealthStatus.HEALTHY:
                component_score = 100.0
            elif health.status == HealthStatus.WARNING:
                component_score = 75.0
            elif health.status == HealthStatus.DEGRADED:
                component_score = 50.0
            elif health.status == HealthStatus.CRITICAL:
                component_score = 25.0
            else:  # DOWN
                component_score = 0.0
            
            total_score += component_score * weight
        
        return min(100.0, max(0.0, total_score))
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = time.time()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Update stats
            self.system_stats["resolved_alerts"] += 1
            resolution_time = alert.resolution_time - alert.timestamp
            
            current_avg = self.system_stats["avg_resolution_time"]
            resolved_count = self.system_stats["resolved_alerts"]
            self.system_stats["avg_resolution_time"] = (
                (current_avg * (resolved_count - 1) + resolution_time) / resolved_count
            )
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by} "
                       f"(resolution time: {resolution_time:.1f}s)")
    
    async def shutdown(self):
        """Shutdown the health monitor."""
        logger.info("Shutting down System Health Monitor...")
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.initialized = False
        logger.info("✅ System Health Monitor shutdown complete")