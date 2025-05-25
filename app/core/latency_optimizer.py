"""
Advanced Latency Optimizer for Multi-Agent Voice AI System.
Monitors, analyzes, and optimizes system performance to maintain <2-second response times.
"""
import asyncio
import logging
import time
import statistics
import functools
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

from config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

# Global reference to the latency optimizer instance
_global_optimizer: Optional['LatencyOptimizer'] = None

def set_global_optimizer(optimizer: 'LatencyOptimizer'):
    """Set the global optimizer instance for the decorator"""
    global _global_optimizer
    _global_optimizer = optimizer

def latency_monitor(component_name: str):
    """
    Decorator to monitor function execution latency
    Integrates with the existing LatencyOptimizer system
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            session_id = kwargs.get('session_id', 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # milliseconds
                
                # Track with the global optimizer if available
                if _global_optimizer:
                    await _global_optimizer.track_component_latency(
                        component=component_name,
                        duration=execution_time,
                        session_id=session_id
                    )
                else:
                    logger.debug(
                        f"Component {component_name} completed",
                        execution_time_ms=execution_time,
                        session_id=session_id
                    )
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    f"Component {component_name} failed",
                    execution_time_ms=execution_time,
                    session_id=session_id,
                    error=str(e)
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            session_id = kwargs.get('session_id', 'unknown')
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # For sync functions, we can't directly await, so just log
                logger.debug(
                    f"Component {component_name} completed",
                    execution_time_ms=execution_time,
                    session_id=session_id
                )
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    f"Component {component_name} failed",
                    execution_time_ms=execution_time,
                    session_id=session_id,
                    error=str(e)
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class PerformanceAlert(Enum):
    """Performance alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class LatencyMeasurement:
    """Individual latency measurement."""
    timestamp: float
    session_id: str
    component: str
    duration: float
    target: float
    
    @property
    def is_over_target(self) -> bool:
        return self.duration > self.target
    
    @property
    def target_ratio(self) -> float:
        return self.duration / self.target if self.target > 0 else 0.0

@dataclass
class ComponentPerformance:
    """Performance statistics for a system component."""
    component_name: str
    measurements: deque = field(default_factory=lambda: deque(maxlen=1000))
    target_latency: float = 0.0
    
    def add_measurement(self, duration: float, session_id: str):
        """Add a new measurement."""
        measurement = LatencyMeasurement(
            timestamp=time.time(),
            session_id=session_id,
            component=self.component_name,
            duration=duration,
            target=self.target_latency
        )
        self.measurements.append(measurement)
    
    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if not self.measurements:
            return 0.0
        return statistics.mean(m.duration for m in self.measurements)
    
    @property
    def p95_latency(self) -> float:
        """Calculate 95th percentile latency."""
        if not self.measurements:
            return 0.0
        durations = [m.duration for m in self.measurements]
        return statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max(durations)
    
    @property
    def target_met_percentage(self) -> float:
        """Percentage of measurements meeting target."""
        if not self.measurements:
            return 100.0
        met_target = sum(1 for m in self.measurements if not m.is_over_target)
        return (met_target / len(self.measurements)) * 100.0
    
    @property
    def recent_trend(self) -> str:
        """Analyze recent performance trend."""
        if len(self.measurements) < 20:
            return "insufficient_data"
        
        recent = list(self.measurements)[-10:]
        older = list(self.measurements)[-20:-10]
        
        recent_avg = statistics.mean(m.duration for m in recent)
        older_avg = statistics.mean(m.duration for m in older)
        
        if recent_avg > older_avg * 1.1:
            return "degrading"
        elif recent_avg < older_avg * 0.9:
            return "improving"
        else:
            return "stable"

@dataclass
class OptimizationAction:
    """Represents an optimization action to take."""
    component: str
    action_type: str
    description: str
    priority: int
    estimated_improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "action_type": self.action_type,
            "description": self.description,
            "priority": self.priority,
            "estimated_improvement": self.estimated_improvement
        }

class LatencyOptimizer:
    """
    Comprehensive latency optimization system for multi-agent voice AI.
    
    Features:
    - Real-time performance monitoring
    - Automated bottleneck detection
    - Dynamic optimization recommendations
    - System health alerting
    - Performance trend analysis
    """
    
    def __init__(self, orchestrator=None, performance_tracker=None):
        """Initialize the latency optimizer."""
        self.orchestrator = orchestrator
        self.performance_tracker = performance_tracker
        
        # Component performance tracking
        self.component_stats = {
            "stt": ComponentPerformance("stt", target_latency=120),  # ms
            "routing": ComponentPerformance("routing", target_latency=15),
            "retrieval": ComponentPerformance("retrieval", target_latency=10),
            "agent": ComponentPerformance("agent", target_latency=280),
            "tools": ComponentPerformance("tools", target_latency=30),
            "tts": ComponentPerformance("tts", target_latency=150),
            "total": ComponentPerformance("total", target_latency=650)
        }
        
        # System state tracking
        self.system_health_score = 100.0
        self.last_optimization_time = time.time()
        self.optimization_actions_taken = []
        
        # Alert thresholds
        self.alert_thresholds = {
            PerformanceAlert.WARNING: 0.8,  # 80% of target met
            PerformanceAlert.CRITICAL: 0.6   # 60% of target met
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            "stt": self._optimize_stt,
            "routing": self._optimize_routing,
            "retrieval": self._optimize_retrieval,
            "agent": self._optimize_agent,
            "tools": self._optimize_tools,
            "tts": self._optimize_tts
        }
        
        # Performance baselines
        self.performance_baselines = {}
        
        logger.info("LatencyOptimizer initialized")
    
    async def init(self):
        """Initialize the latency optimizer."""
        logger.info("Initializing latency optimizer...")
        
        # Set this instance as the global optimizer for the decorator
        set_global_optimizer(self)
        
        # Establish performance baselines
        await self._establish_baselines()
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        logger.info("✅ Latency optimizer ready")
    
    async def track_component_latency(
        self,
        component: str,
        duration: float,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track latency for a specific component."""
        if component in self.component_stats:
            self.component_stats[component].add_measurement(duration, session_id)
            
            # Check for immediate alerts
            if duration > self.component_stats[component].target_latency * 2:
                await self._send_alert(
                    level=PerformanceAlert.CRITICAL,
                    message=f"Extreme latency in {component}: {duration:.3f}ms (target: {self.component_stats[component].target_latency:.3f}ms)",
                    component=component,
                    session_id=session_id
                )
    
    async def analyze_session_performance(self, session_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance for a completed session."""
        session_id = session_metrics.get("session_id", "unknown")
        
        # Track individual component latencies
        for component, duration in session_metrics.items():
            if component.endswith("_time") and isinstance(duration, (int, float)):
                component_name = component.replace("_time", "")
                await self.track_component_latency(component_name, duration, session_id)
        
        # Track total latency
        total_time = session_metrics.get("total_time", 0.0)
        if total_time > 0:
            await self.track_component_latency("total", total_time, session_id)
        
        # Generate performance analysis
        analysis = {
            "session_id": session_id,
            "total_latency": total_time,
            "target_met": total_time <= 650,  # ms
            "bottlenecks": await self._identify_session_bottlenecks(session_metrics),
            "optimization_suggestions": await self._generate_session_optimizations(session_metrics)
        }
        
        return analysis
    
    async def _identify_session_bottlenecks(self, session_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify bottlenecks in a specific session."""
        bottlenecks = []
        
        # Check each component against its target
        component_mappings = {
            "routing_time": ("routing", 15),
            "retrieval_time": ("retrieval", 10),
            "agent_time": ("agent", 280),
            "tool_time": ("tools", 30),
            "tts_time": ("tts", 150)
        }
        
        for metric_key, (component, target) in component_mappings.items():
            duration = session_metrics.get(metric_key, 0.0)
            if duration > target:
                bottlenecks.append({
                    "component": component,
                    "duration": duration,
                    "target": target,
                    "excess": duration - target,
                    "severity": "critical" if duration > target * 2 else "warning"
                })
        
        # Sort by excess time (biggest bottlenecks first)
        bottlenecks.sort(key=lambda x: x["excess"], reverse=True)
        
        return bottlenecks
    
    async def _generate_session_optimizations(self, session_metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """Generate optimization recommendations for a session."""
        optimizations = []
        
        # Analyze each component
        for component, stats in self.component_stats.items():
            if component == "total":
                continue
                
            component_time_key = f"{component}_time"
            if component_time_key in session_metrics:
                duration = session_metrics[component_time_key]
                
                if duration > stats.target_latency * 1.5:
                    # Generate component-specific optimization
                    if component in self.optimization_strategies:
                        action = await self.optimization_strategies[component](duration, stats.target_latency)
                        if action:
                            optimizations.append(action)
        
        return optimizations
    
    async def optimize_system(self):
        """Run system-wide optimization analysis and actions."""
        logger.info("🔧 Running system optimization cycle...")
        
        try:
            # Calculate current system health
            self.system_health_score = await self._calculate_health_score()
            
            # Identify system-wide bottlenecks
            bottlenecks = await self._identify_system_bottlenecks()
            
            # Generate optimization actions
            actions = await self._generate_optimization_actions(bottlenecks)
            
            # Execute high-priority actions
            await self._execute_optimization_actions(actions)
            
            # Update optimization time
            self.last_optimization_time = time.time()
            
            logger.info(f"✅ Optimization cycle complete. Health score: {self.system_health_score:.1f}/100")
            
        except Exception as e:
            logger.error(f"❌ Error in system optimization: {e}", exc_info=True)
    
    async def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        scores = []
        
        for component, stats in self.component_stats.items():
            if len(stats.measurements) == 0:
                continue
                
            # Target met percentage contributes to health
            target_score = stats.target_met_percentage
            
            # Recent trend affects score
            trend = stats.recent_trend
            trend_modifier = {
                "improving": 1.1,
                "stable": 1.0,
                "degrading": 0.9,
                "insufficient_data": 1.0
            }
            
            component_score = target_score * trend_modifier.get(trend, 1.0)
            scores.append(min(100.0, component_score))
        
        return statistics.mean(scores) if scores else 100.0
    
    async def _identify_system_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify system-wide performance bottlenecks."""
        bottlenecks = []
        
        for component, stats in self.component_stats.items():
            if len(stats.measurements) < 10:  # Need sufficient data
                continue
            
            # Check if component is consistently over target
            target_met_pct = stats.target_met_percentage
            
            if target_met_pct < 80:  # Less than 80% meeting target
                bottlenecks.append({
                    "component": component,
                    "target_met_percentage": target_met_pct,
                    "avg_latency": stats.avg_latency,
                    "p95_latency": stats.p95_latency,
                    "target": stats.target_latency,
                    "trend": stats.recent_trend,
                    "severity": "critical" if target_met_pct < 60 else "warning"
                })
        
        # Sort by severity and impact
        bottlenecks.sort(key=lambda x: (x["severity"] == "critical", 100 - x["target_met_percentage"]), reverse=True)
        
        return bottlenecks
    
    async def _generate_optimization_actions(self, bottlenecks: List[Dict[str, Any]]) -> List[OptimizationAction]:
        """Generate optimization actions based on identified bottlenecks."""
        actions = []
        
        for bottleneck in bottlenecks:
            component = bottleneck["component"]
            
            if component in self.optimization_strategies:
                avg_latency = bottleneck["avg_latency"]
                target = bottleneck["target"]
                
                action = await self.optimization_strategies[component](avg_latency, target)
                if action:
                    actions.append(action)
        
        return actions
    
    async def _execute_optimization_actions(self, actions: List[OptimizationAction]):
        """Execute optimization actions based on priority."""
        # Sort by priority (higher numbers = higher priority)
        actions.sort(key=lambda x: x.priority, reverse=True)
        
        executed_actions = []
        
        for action in actions[:5]:  # Limit to top 5 actions per cycle
            try:
                success = await self._execute_action(action)
                if success:
                    executed_actions.append(action)
                    logger.info(f"✅ Executed optimization: {action.description}")
                else:
                    logger.warning(f"⚠️ Failed to execute optimization: {action.description}")
                    
            except Exception as e:
                logger.error(f"❌ Error executing optimization action: {e}")
        
        # Track executed actions
        self.optimization_actions_taken.extend(executed_actions)
        
        # Keep only recent actions (last 100)
        if len(self.optimization_actions_taken) > 100:
            self.optimization_actions_taken = self.optimization_actions_taken[-100:]
    
    async def _execute_action(self, action: OptimizationAction) -> bool:
        """Execute a specific optimization action."""
        try:
            if action.action_type == "cache_warming":
                await self._warm_caches(action.component)
            elif action.action_type == "index_optimization":
                await self._optimize_vector_index(action.component)
            elif action.action_type == "connection_pooling":
                await self._optimize_connections(action.component)
            elif action.action_type == "model_optimization":
                await self._optimize_model_settings(action.component)
            elif action.action_type == "resource_scaling":
                await self._scale_resources(action.component)
            else:
                logger.warning(f"Unknown optimization action type: {action.action_type}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {e}")
            return False
    
    # Component-specific optimization strategies
    
    async def _optimize_stt(self, current_latency: float, target: float) -> Optional[OptimizationAction]:
        """Generate STT optimization recommendations."""
        if current_latency > target * 1.5:
            return OptimizationAction(
                component="stt",
                action_type="model_optimization",
                description="Optimize STT model settings for lower latency",
                priority=8,
                estimated_improvement=0.2
            )
        return None
    
    async def _optimize_routing(self, current_latency: float, target: float) -> Optional[OptimizationAction]:
        """Generate routing optimization recommendations."""
        if current_latency > target * 1.2:
            return OptimizationAction(
                component="routing",
                action_type="cache_warming",
                description="Pre-warm routing decision cache",
                priority=7,
                estimated_improvement=0.3
            )
        return None
    
    async def _optimize_retrieval(self, current_latency: float, target: float) -> Optional[OptimizationAction]:
        """Generate retrieval optimization recommendations."""
        if current_latency > target * 1.3:
            return OptimizationAction(
                component="retrieval",
                action_type="index_optimization",
                description="Optimize vector index performance",
                priority=9,
                estimated_improvement=0.4
            )
        return None
    
    async def _optimize_agent(self, current_latency: float, target: float) -> Optional[OptimizationAction]:
        """Generate agent optimization recommendations."""
        if current_latency > target * 1.4:
            return OptimizationAction(
                component="agent",
                action_type="model_optimization",
                description="Optimize LLM inference parameters",
                priority=8,
                estimated_improvement=0.25
            )
        return None
    
    async def _optimize_tools(self, current_latency: float, target: float) -> Optional[OptimizationAction]:
        """Generate tool optimization recommendations."""
        if current_latency > target * 1.5:
            return OptimizationAction(
                component="tools",
                action_type="connection_pooling",
                description="Optimize external API connection pooling",
                priority=6,
                estimated_improvement=0.3
            )
        return None
    
    async def _optimize_tts(self, current_latency: float, target: float) -> Optional[OptimizationAction]:
        """Generate TTS optimization recommendations."""
        if current_latency > target * 1.3:
            return OptimizationAction(
                component="tts",
                action_type="cache_warming",
                description="Pre-generate common TTS responses",
                priority=7,
                estimated_improvement=0.35
            )
        return None
    
    # Optimization action implementations
    
    async def _warm_caches(self, component: str):
        """Warm up caches for better performance."""
        logger.info(f"🔥 Warming caches for {component}")
        
        if component == "routing" and self.orchestrator:
            # Pre-warm routing decisions for common queries
            common_queries = [
                "I need roadside assistance",
                "Help with my bill",
                "Technical support needed",
                "Check my account balance"
            ]
            
            for query in common_queries:
                try:
                    if hasattr(self.orchestrator, 'router'):
                        await self.orchestrator.router.route_request(
                            user_input=query,
                            conversation_history=[],
                            user_context={},
                            session_metadata={}
                        )
                except Exception as e:
                    logger.error(f"Error warming routing cache: {e}")
        
        elif component == "tts":
            # Pre-generate common responses
            common_responses = [
                "How can I help you today?",
                "I understand. Let me help you with that.",
                "Is there anything else I can assist you with?",
                "Thank you for contacting us."
            ]
            
            if self.orchestrator and hasattr(self.orchestrator, 'tts'):
                for response in common_responses:
                    try:
                        await self.orchestrator.tts.synthesize(response)
                    except Exception as e:
                        logger.error(f"Error warming TTS cache: {e}")
    
    async def _optimize_vector_index(self, component: str):
        """Optimize vector database index performance."""
        logger.info(f"🔍 Optimizing vector index for {component}")
        
        if self.orchestrator and hasattr(self.orchestrator, 'vector_store'):
            try:
                # Trigger index optimization
                if hasattr(self.orchestrator.vector_store, 'optimize_performance'):
                    await self.orchestrator.vector_store.optimize_performance()
            except Exception as e:
                logger.error(f"Error optimizing vector index: {e}")
    
    async def _optimize_connections(self, component: str):
        """Optimize connection pooling and network settings."""
        logger.info(f"🔗 Optimizing connections for {component}")
        
        # This would optimize connection pools, timeouts, etc.
        # Implementation depends on specific components
        pass
    
    async def _optimize_model_settings(self, component: str):
        """Optimize model inference settings."""
        logger.info(f"🤖 Optimizing model settings for {component}")
        
        # This would adjust model parameters for better performance
        # Implementation depends on specific models used
        pass
    
    async def _scale_resources(self, component: str):
        """Scale resources for better performance."""
        logger.info(f"📈 Scaling resources for {component}")
        
        # This would trigger resource scaling (if in cloud environment)
        # Implementation depends on deployment environment
        pass
    
    async def _establish_baselines(self):
        """Establish performance baselines for comparison."""
        logger.info("📊 Establishing performance baselines...")
        
        # Set initial baselines from configuration
        self.performance_baselines = {
            "stt": 120,
            "routing": 15,
            "retrieval": 10,
            "agent": 280,
            "tools": 30,
            "tts": 150,
            "total": 650
        }
        
        logger.info("✅ Performance baselines established")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while True:
            try:
                # Check system health every minute
                await asyncio.sleep(60)
                
                # Run optimization if needed
                if time.time() - self.last_optimization_time > 300:  # 5 minutes
                    await self.optimize_system()
                
                # Send health alerts if needed
                await self._check_health_alerts()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _check_health_alerts(self):
        """Check if health alerts need to be sent."""
        current_health = await self._calculate_health_score()
        
        if current_health < 60:  # Critical threshold
            await self._send_alert(
                level=PerformanceAlert.CRITICAL,
                message=f"System health critically low: {current_health:.1f}/100",
                component="system"
            )
        elif current_health < 80:  # Warning threshold
            await self._send_alert(
                level=PerformanceAlert.WARNING,
                message=f"System health degraded: {current_health:.1f}/100",
                component="system"
            )
    
    async def _send_alert(
        self,
        level: PerformanceAlert,
        message: str,
        component: str,
        session_id: Optional[str] = None
    ):
        """Send performance alert."""
        alert = {
            "timestamp": time.time(),
            "level": level.value,
            "message": message,
            "component": component,
            "session_id": session_id,
            "system_health": self.system_health_score
        }
        
        logger.warning(f"🚨 Performance Alert [{level.value.upper()}]: {message}")
        
        # Here you would integrate with your alerting system
        # (PagerDuty, Slack, email, etc.)
    
    async def get_latency_report(self) -> Dict[str, Any]:
        """Generate comprehensive latency report."""
        report = {
            "timestamp": time.time(),
            "system_health_score": self.system_health_score,
            "components": {},
            "recent_optimizations": [action.to_dict() for action in self.optimization_actions_taken[-10:]],
            "bottlenecks": await self._identify_system_bottlenecks(),
            "recommendations": []
        }
        
        # Add component statistics
        for component, stats in self.component_stats.items():
            if len(stats.measurements) > 0:
                report["components"][component] = {
                    "avg_latency": stats.avg_latency,
                    "p95_latency": stats.p95_latency,
                    "target_latency": stats.target_latency,
                    "target_met_percentage": stats.target_met_percentage,
                    "trend": stats.recent_trend,
                    "measurement_count": len(stats.measurements)
                }
        
        # Generate recommendations
        bottlenecks = await self._identify_system_bottlenecks()
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            component = bottleneck["component"]
            if component in self.optimization_strategies:
                action = await self.optimization_strategies[component](
                    bottleneck["avg_latency"],
                    bottleneck["target"]
                )
                if action:
                    report["recommendations"].append(action.to_dict())
        
        return report
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {
            "system_health_score": self.system_health_score,
            "last_optimization": self.last_optimization_time,
            "components": {}
        }
        
        for component, stats in self.component_stats.items():
            if len(stats.measurements) > 0:
                recent_measurements = list(stats.measurements)[-10:]  # Last 10 measurements
                
                metrics["components"][component] = {
                    "current_avg": statistics.mean(m.duration for m in recent_measurements),
                    "target": stats.target_latency,
                    "recent_trend": stats.recent_trend,
                    "target_met_recent": sum(1 for m in recent_measurements if not m.is_over_target) / len(recent_measurements) * 100
                }
        
        return metrics
    
    async def health_check(self) -> bool:
        """Health check for the latency optimizer."""
        try:
            # Check if we have recent measurements
            recent_activity = any(
                len(stats.measurements) > 0 and 
                (time.time() - stats.measurements[-1].timestamp) < 300  # Last 5 minutes
                for stats in self.component_stats.values()
            )
            
            # Check system health score
            health_ok = self.system_health_score > 50
            
            return recent_activity and health_ok
            
        except Exception as e:
            logger.error(f"Latency optimizer health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the latency optimizer."""
        logger.info("🛑 Shutting down latency optimizer...")
        
        # Final optimization report
        if self.optimization_actions_taken:
            logger.info(f"📊 Total optimizations performed: {len(self.optimization_actions_taken)}")
        
        logger.info("✅ Latency optimizer shutdown complete")