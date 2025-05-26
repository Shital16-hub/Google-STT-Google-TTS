"""
Latency Optimizer - Performance monitoring and optimization for <650ms target
Implements comprehensive latency tracking, bottleneck detection, and real-time optimization
"""
import time
import asyncio
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    """Comprehensive latency metrics tracking"""
    component: str
    session_id: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceBudget:
    """Performance budget for different components"""
    stt_target: float = 120.0      # <120ms STT processing
    routing_target: float = 15.0    # <15ms agent routing
    retrieval_target: float = 5.0   # <5ms vector retrieval (hybrid)
    llm_target: float = 280.0      # <280ms LLM generation
    tool_target: float = 30.0      # <30ms tool execution
    tts_target: float = 150.0      # <150ms TTS synthesis
    network_target: float = 50.0   # <50ms network overhead
    total_target: float = 650.0    # <650ms total (73% improvement target)

class LatencyOptimizer:
    """
    Advanced latency optimizer for achieving <650ms end-to-end response times
    """
    
    def __init__(self):
        # Performance targets from transformation plan
        self.budget = PerformanceBudget()
        
        # Real-time tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_metrics: Dict[str, List[LatencyMetrics]] = defaultdict(list)
        
        # Performance analytics
        self.component_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "measurements": deque(maxlen=1000),  # Last 1000 measurements
            "average": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "success_rate": 100.0,
            "target_compliance": 100.0,
            "trend": "stable"
        })
        
        # Bottleneck detection
        self.bottleneck_alerts: List[Dict[str, Any]] = []
        self.optimization_recommendations: List[Dict[str, Any]] = []
        
        # System health
        self.system_health = {
            "overall_health": "excellent",
            "performance_grade": "A",
            "target_compliance": 100.0,
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Configuration
        self.alert_thresholds = {
            "target_miss_rate": 10.0,  # Alert if >10% of requests miss target
            "performance_degradation": 20.0,  # Alert if >20% degradation
            "consecutive_failures": 5  # Alert after 5 consecutive failures
        }
        
        logger.info("🎯 Latency Optimizer initialized with <650ms target")
    
    async def init(self):
        """Initialize latency optimizer"""
        logger.info("🔄 Initializing Latency Optimizer...")
        
        # Start background monitoring tasks
        asyncio.create_task(self._performance_analyzer())
        asyncio.create_task(self._bottleneck_detector())
        asyncio.create_task(self._system_health_monitor())
        
        logger.info("✅ Latency Optimizer initialized")
    
    async def start_session_tracking(self, session_id: str) -> Dict[str, Any]:
        """Start tracking latency for a session"""
        start_time = time.time()
        
        self.active_sessions[session_id] = {
            "start_time": start_time,
            "components": {},
            "total_latency": 0.0,
            "target_met": True,
            "bottlenecks": [],
            "optimization_applied": []
        }
        
        logger.debug(f"📊 Started latency tracking for session {session_id}")
        
        return {
            "session_id": session_id,
            "start_time": start_time,
            "budget": self.budget.__dict__
        }
    
    async def record_processing_time(
        self, 
        session_id: str, 
        component: str, 
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record processing time for a component"""
        current_time = time.time()
        
        # Create latency metric
        metric = LatencyMetrics(
            component=component,
            session_id=session_id,
            start_time=current_time - (duration_ms / 1000),
            end_time=current_time,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata or {}
        )
        
        # Store metric
        self.session_metrics[session_id].append(metric)
        
        # Update component statistics
        self._update_component_stats(component, duration_ms, success)
        
        # Update session tracking
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session["components"][component] = duration_ms
            session["total_latency"] += duration_ms
            
            # Check against targets
            target = self._get_component_target(component)
            if duration_ms > target:
                session["target_met"] = False
                session["bottlenecks"].append({
                    "component": component,
                    "actual": duration_ms,
                    "target": target,
                    "overage": duration_ms - target
                })
                
                logger.warning(f"⚠️ {component} exceeded target: {duration_ms:.1f}ms > {target:.1f}ms")
            
            # Real-time optimization
            await self._apply_real_time_optimization(session_id, component, duration_ms)
        
        # Log performance
        logger.debug(f"📊 {component}: {duration_ms:.1f}ms (target: {self._get_component_target(component):.1f}ms)")
    
    def _update_component_stats(self, component: str, duration_ms: float, success: bool):
        """Update statistical data for a component"""
        stats = self.component_stats[component]
        
        # Add measurement
        stats["measurements"].append({
            "duration": duration_ms,
            "success": success,
            "timestamp": time.time()
        })
        
        # Calculate statistics
        durations = [m["duration"] for m in stats["measurements"]]
        if durations:
            stats["average"] = statistics.mean(durations)
            stats["p50"] = statistics.median(durations)
            stats["p95"] = statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations)
            stats["p99"] = statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations)
        
        # Calculate success rate
        successes = sum(1 for m in stats["measurements"] if m["success"])
        stats["success_rate"] = (successes / len(stats["measurements"])) * 100
        
        # Calculate target compliance
        target = self._get_component_target(component)
        compliant = sum(1 for d in durations if d <= target)
        stats["target_compliance"] = (compliant / len(durations)) * 100 if durations else 100.0
        
        # Detect trend
        if len(durations) >= 10:
            recent = durations[-5:]
            older = durations[-10:-5]
            
            recent_avg = statistics.mean(recent)
            older_avg = statistics.mean(older)
            
            if recent_avg > older_avg * 1.1:
                stats["trend"] = "degrading"
            elif recent_avg < older_avg * 0.9:
                stats["trend"] = "improving"
            else:
                stats["trend"] = "stable"
    
    def _get_component_target(self, component: str) -> float:
        """Get performance target for a component"""
        component_map = {
            "stt": self.budget.stt_target,
            "agent_routing": self.budget.routing_target,
            "vector_retrieval": self.budget.retrieval_target,
            "agent_execution": self.budget.llm_target,
            "tool_orchestration": self.budget.tool_target,
            "tts": self.budget.tts_target,
            "network": self.budget.network_target,
            "audio_processing": self.budget.stt_target,  # Alias for STT
            "response_synthesis": self.budget.llm_target  # Alias for LLM
        }
        
        return component_map.get(component, 100.0)  # Default 100ms
    
    async def _apply_real_time_optimization(self, session_id: str, component: str, duration_ms: float):
        """Apply real-time optimizations based on performance"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        optimizations = []