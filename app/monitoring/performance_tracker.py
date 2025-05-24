"""
Performance Tracker for Multi-Agent Voice AI System
Real-time monitoring of latency, throughput, and system performance metrics
"""

import asyncio
import time
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
from enum import Enum
import json

import redis
import psutil
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import structlog

# Configure structured logging
logger = structlog.get_logger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    AGENT_SPECIFIC = "agent_specific"

class ComponentType(Enum):
    """System components being monitored"""
    STT = "stt"
    AGENT_ROUTING = "agent_routing"
    VECTOR_SEARCH = "vector_search"
    LLM_GENERATION = "llm_generation"
    TOOL_EXECUTION = "tool_execution"
    TTS = "tts"
    NETWORK = "network"
    TOTAL = "total"

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: datetime
    component: ComponentType
    agent_id: Optional[str]
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component.value,
            "agent_id": self.agent_id,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "metadata": self.metadata
        }

@dataclass
class LatencyBreakdown:
    """Detailed latency breakdown for end-to-end requests"""
    request_id: str
    total_latency: float
    stt_latency: float
    routing_latency: float
    vector_latency: float
    llm_latency: float
    tool_latency: float
    tts_latency: float
    network_latency: float
    agent_id: Optional[str] = None
    success: bool = True
    error_type: Optional[str] = None
    
    def get_component_latencies(self) -> Dict[ComponentType, float]:
        """Get latencies by component"""
        return {
            ComponentType.STT: self.stt_latency,
            ComponentType.AGENT_ROUTING: self.routing_latency,
            ComponentType.VECTOR_SEARCH: self.vector_latency,
            ComponentType.LLM_GENERATION: self.llm_latency,
            ComponentType.TOOL_EXECUTION: self.tool_latency,
            ComponentType.TTS: self.tts_latency,
            ComponentType.NETWORK: self.network_latency,
            ComponentType.TOTAL: self.total_latency
        }

class PrometheusMetrics:
    """Prometheus metrics for performance tracking"""
    
    def __init__(self):
        # Latency metrics
        self.request_duration = Histogram(
            'voice_ai_request_duration_seconds',
            'Request duration in seconds',
            ['component', 'agent_id', 'status']
        )
        
        self.component_duration = Histogram(
            'voice_ai_component_duration_seconds',
            'Component duration in seconds',
            ['component', 'agent_id'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Throughput metrics
        self.requests_total = Counter(
            'voice_ai_requests_total',
            'Total number of requests',
            ['agent_id', 'status']
        )
        
        self.concurrent_requests = Gauge(
            'voice_ai_concurrent_requests',
            'Current number of concurrent requests',
            ['agent_id']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'voice_ai_errors_total',
            'Total number of errors',
            ['component', 'agent_id', 'error_type']
        )
        
        # Resource usage metrics
        self.cpu_usage = Gauge(
            'voice_ai_cpu_usage_percent',
            'CPU usage percentage',
            ['instance']
        )
        
        self.memory_usage = Gauge(
            'voice_ai_memory_usage_bytes',
            'Memory usage in bytes',
            ['instance', 'type']
        )
        
        self.vector_db_latency = Histogram(
            'voice_ai_vector_db_latency_seconds',
            'Vector database query latency',
            ['db_type', 'agent_id', 'cache_hit']
        )
        
        # Agent-specific metrics
        self.agent_routing_accuracy = Gauge(
            'voice_ai_agent_routing_accuracy',
            'Agent routing accuracy percentage',
            ['agent_id']
        )
        
        self.tool_execution_success = Counter(
            'voice_ai_tool_execution_total',
            'Tool execution attempts',
            ['tool_name', 'agent_id', 'status']
        )

class PerformanceWindow:
    """Sliding window for performance metrics"""
    
    def __init__(self, window_size: int = 1000, time_window_minutes: int = 60):
        self.window_size = window_size
        self.time_window = timedelta(minutes=time_window_minutes)
        self.metrics: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a metric to the window"""
        with self._lock:
            self.metrics.append(metric)
            self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than time window"""
        now = datetime.utcnow()
        cutoff = now - self.time_window
        
        while self.metrics and self.metrics[0].timestamp < cutoff:
            self.metrics.popleft()
    
    def get_metrics(self, 
                   component: Optional[ComponentType] = None,
                   agent_id: Optional[str] = None,
                   metric_type: Optional[MetricType] = None) -> List[PerformanceMetric]:
        """Get filtered metrics from window"""
        with self._lock:
            metrics = list(self.metrics)
        
        filtered = metrics
        if component:
            filtered = [m for m in filtered if m.component == component]
        if agent_id:
            filtered = [m for m in filtered if m.agent_id == agent_id]
        if metric_type:
            filtered = [m for m in filtered if m.metric_type == metric_type]
        
        return filtered
    
    def get_percentiles(self, 
                       component: Optional[ComponentType] = None,
                       agent_id: Optional[str] = None) -> Dict[str, float]:
        """Calculate latency percentiles"""
        latency_metrics = self.get_metrics(component, agent_id, MetricType.LATENCY)
        
        if not latency_metrics:
            return {}
        
        values = [m.value for m in latency_metrics]
        
        return {
            'p50': np.percentile(values, 50),
            'p90': np.percentile(values, 90),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }

class PerformanceTracker:
    """Main performance tracking system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.prometheus_metrics = PrometheusMetrics()
        self.performance_window = PerformanceWindow()
        self.active_requests: Dict[str, datetime] = {}
        self.active_components: Dict[str, Dict[ComponentType, datetime]] = {}
        self._lock = threading.Lock()
        
        # Performance targets from config
        self.latency_targets = {
            ComponentType.STT: 120,          # ms
            ComponentType.AGENT_ROUTING: 15, # ms
            ComponentType.VECTOR_SEARCH: 10, # ms
            ComponentType.LLM_GENERATION: 280, # ms
            ComponentType.TOOL_EXECUTION: 30,  # ms
            ComponentType.TTS: 150,          # ms
            ComponentType.NETWORK: 50,       # ms
            ComponentType.TOTAL: 650         # ms
        }
        
        # Start background monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self._monitor_thread.start()
    
    @asynccontextmanager
    async def track_request(self, request_id: str, agent_id: Optional[str] = None):
        """Context manager for tracking full request lifecycle"""
        start_time = datetime.utcnow()
        
        try:
            with self._lock:
                self.active_requests[request_id] = start_time
                self.active_components[request_id] = {}
            
            # Update concurrent requests metric
            if agent_id:
                self.prometheus_metrics.concurrent_requests.labels(agent_id=agent_id).inc()
            
            yield RequestTracker(self, request_id, agent_id)
            
        except Exception as e:
            # Track error
            await self.track_error(ComponentType.TOTAL, str(type(e).__name__), agent_id)
            raise
        finally:
            # Calculate total duration
            end_time = datetime.utcnow()
            total_duration = (end_time - start_time).total_seconds() * 1000  # ms
            
            # Record metrics
            await self.record_latency(ComponentType.TOTAL, total_duration, agent_id, request_id)
            
            # Update Prometheus
            status = "success"  # Would be "error" if exception occurred
            if agent_id:
                self.prometheus_metrics.requests_total.labels(
                    agent_id=agent_id, status=status
                ).inc()
                self.prometheus_metrics.concurrent_requests.labels(agent_id=agent_id).dec()
            
            # Cleanup
            with self._lock:
                self.active_requests.pop(request_id, None)
                self.active_components.pop(request_id, None)
    
    @asynccontextmanager
    async def track_component(self, request_id: str, component: ComponentType, agent_id: Optional[str] = None):
        """Context manager for tracking individual component performance"""
        start_time = datetime.utcnow()
        
        try:
            with self._lock:
                if request_id in self.active_components:
                    self.active_components[request_id][component] = start_time
            
            yield
            
        finally:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000  # ms
            
            # Record component latency
            await self.record_latency(component, duration, agent_id, request_id)
            
            # Update Prometheus component metric
            self.prometheus_metrics.component_duration.labels(
                component=component.value, agent_id=agent_id or "unknown"
            ).observe(duration / 1000)  # Convert to seconds for Prometheus
    
    async def record_latency(self, 
                           component: ComponentType, 
                           duration_ms: float, 
                           agent_id: Optional[str] = None,
                           request_id: Optional[str] = None):
        """Record latency metric"""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            component=component,
            agent_id=agent_id,
            metric_type=MetricType.LATENCY,
            value=duration_ms,
            metadata={"request_id": request_id} if request_id else {}
        )
        
        self.performance_window.add_metric(metric)
        
        # Store in Redis for real-time dashboards
        if self.redis_client:
            await self._store_metric_redis(metric)
        
        # Check against targets and log warnings
        target = self.latency_targets.get(component)
        if target and duration_ms > target:
            logger.warning(
                "Latency target exceeded",
                component=component.value,
                agent_id=agent_id,
                duration_ms=duration_ms,
                target_ms=target,
                request_id=request_id
            )
    
    async def record_throughput(self, 
                              component: ComponentType, 
                              requests_per_second: float, 
                              agent_id: Optional[str] = None):
        """Record throughput metric"""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            component=component,
            agent_id=agent_id,
            metric_type=MetricType.THROUGHPUT,
            value=requests_per_second
        )
        
        self.performance_window.add_metric(metric)
        
        if self.redis_client:
            await self._store_metric_redis(metric)
    
    async def track_error(self, 
                         component: ComponentType, 
                         error_type: str, 
                         agent_id: Optional[str] = None):
        """Track error occurrence"""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            component=component,
            agent_id=agent_id,
            metric_type=MetricType.ERROR_RATE,
            value=1.0,
            metadata={"error_type": error_type}
        )
        
        self.performance_window.add_metric(metric)
        
        # Update Prometheus error counter
        self.prometheus_metrics.errors_total.labels(
            component=component.value,
            agent_id=agent_id or "unknown",
            error_type=error_type
        ).inc()
        
        logger.error(
            "Error tracked",
            component=component.value,
            error_type=error_type,
            agent_id=agent_id
        )
    
    async def track_vector_db_query(self, 
                                  db_type: str, 
                                  duration_ms: float, 
                                  agent_id: str, 
                                  cache_hit: bool = False):
        """Track vector database query performance"""
        self.prometheus_metrics.vector_db_latency.labels(
            db_type=db_type,
            agent_id=agent_id,
            cache_hit="true" if cache_hit else "false"
        ).observe(duration_ms / 1000)
        
        # Record as component latency too
        await self.record_latency(ComponentType.VECTOR_SEARCH, duration_ms, agent_id)
    
    async def track_tool_execution(self, 
                                 tool_name: str, 
                                 agent_id: str, 
                                 success: bool, 
                                 duration_ms: float):
        """Track tool execution performance"""
        status = "success" if success else "error"
        
        self.prometheus_metrics.tool_execution_success.labels(
            tool_name=tool_name,
            agent_id=agent_id,
            status=status
        ).inc()
        
        # Record as component latency
        await self.record_latency(ComponentType.TOOL_EXECUTION, duration_ms, agent_id)
    
    def get_current_performance(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current performance summary"""
        summary = {}
        
        for component in ComponentType:
            percentiles = self.performance_window.get_percentiles(component, agent_id)
            if percentiles:
                summary[component.value] = percentiles
        
        # Add system resource usage
        summary["system_resources"] = self._get_system_resources()
        
        # Add SLA compliance
        summary["sla_compliance"] = self._calculate_sla_compliance(agent_id)
        
        return summary
    
    def get_latency_breakdown(self, request_id: str) -> Optional[LatencyBreakdown]:
        """Get detailed latency breakdown for a specific request"""
        # This would typically be populated during request processing
        # For now, return None - would be implemented with request correlation
        return None
    
    def _get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Update Prometheus metrics
        self.prometheus_metrics.cpu_usage.labels(instance="main").set(cpu_percent)
        self.prometheus_metrics.memory_usage.labels(
            instance="main", type="used"
        ).set(memory.used)
        self.prometheus_metrics.memory_usage.labels(
            instance="main", type="available"
        ).set(memory.available)
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "disk_free_gb": disk.free / (1024**3)
        }
    
    def _calculate_sla_compliance(self, agent_id: Optional[str] = None) -> Dict[str, float]:
        """Calculate SLA compliance percentages"""
        compliance = {}
        
        for component, target_ms in self.latency_targets.items():
            metrics = self.performance_window.get_metrics(component, agent_id, MetricType.LATENCY)
            
            if metrics:
                compliant_count = sum(1 for m in metrics if m.value <= target_ms)
                compliance_rate = (compliant_count / len(metrics)) * 100
                compliance[component.value] = compliance_rate
        
        return compliance
    
    async def _store_metric_redis(self, metric: PerformanceMetric):
        """Store metric in Redis for real-time access"""
        if not self.redis_client:
            return
        
        try:
            key = f"metrics:{metric.component.value}:{metric.agent_id or 'global'}"
            value = json.dumps(metric.to_dict())
            
            # Store with TTL of 1 hour
            await self.redis_client.setex(key, 3600, value)
            
            # Also add to time series for trends
            ts_key = f"metrics:ts:{metric.component.value}:{metric.metric_type.value}"
            timestamp = int(metric.timestamp.timestamp() * 1000)
            await self.redis_client.zadd(ts_key, {value: timestamp})
            
            # Keep only last 24 hours of time series data
            cutoff = timestamp - (24 * 60 * 60 * 1000)
            await self.redis_client.zremrangebyscore(ts_key, 0, cutoff)
            
        except Exception as e:
            logger.error("Failed to store metric in Redis", error=str(e))
    
    def _background_monitoring(self):
        """Background thread for system monitoring"""
        while self._monitoring_active:
            try:
                # Update system resource metrics
                self._get_system_resources()
                
                # Calculate and update derived metrics
                self._update_derived_metrics()
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error("Background monitoring error", error=str(e))
                time.sleep(30)  # Wait longer on error
    
    def _update_derived_metrics(self):
        """Update derived metrics like error rates, trends"""
        try:
            # Calculate error rates for last 5 minutes
            now = datetime.utcnow()
            five_min_ago = now - timedelta(minutes=5)
            
            recent_metrics = [
                m for m in self.performance_window.metrics 
                if m.timestamp >= five_min_ago
            ]
            
            # Group by component and agent
            error_counts = defaultdict(int)
            total_counts = defaultdict(int)
            
            for metric in recent_metrics:
                key = (metric.component, metric.agent_id)
                total_counts[key] += 1
                if metric.metric_type == MetricType.ERROR_RATE:
                    error_counts[key] += 1
            
            # Update error rate metrics (would be sent to monitoring system)
            for key, total in total_counts.items():
                if total > 0:
                    error_rate = error_counts[key] / total
                    component, agent_id = key
                    
                    # Log high error rates
                    if error_rate > 0.05:  # 5% error rate threshold
                        logger.warning(
                            "High error rate detected",
                            component=component.value,
                            agent_id=agent_id,
                            error_rate=error_rate,
                            total_requests=total,
                            errors=error_counts[key]
                        )
            
        except Exception as e:
            logger.error("Error updating derived metrics", error=str(e))
    
    def shutdown(self):
        """Shutdown the performance tracker"""
        self._monitoring_active = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=5.0)

class RequestTracker:
    """Helper class for tracking individual request performance"""
    
    def __init__(self, performance_tracker: PerformanceTracker, request_id: str, agent_id: Optional[str]):
        self.performance_tracker = performance_tracker
        self.request_id = request_id
        self.agent_id = agent_id
        self.component_times: Dict[ComponentType, float] = {}
    
    async def track_component(self, component: ComponentType):
        """Track a specific component"""
        return self.performance_tracker.track_component(self.request_id, component, self.agent_id)
    
    async def record_component_time(self, component: ComponentType, duration_ms: float):
        """Record component time manually"""
        self.component_times[component] = duration_ms
        await self.performance_tracker.record_latency(component, duration_ms, self.agent_id, self.request_id)
    
    def get_latency_breakdown(self) -> LatencyBreakdown:
        """Get complete latency breakdown for this request"""
        return LatencyBreakdown(
            request_id=self.request_id,
            total_latency=sum(self.component_times.values()),
            stt_latency=self.component_times.get(ComponentType.STT, 0),
            routing_latency=self.component_times.get(ComponentType.AGENT_ROUTING, 0),
            vector_latency=self.component_times.get(ComponentType.VECTOR_SEARCH, 0),
            llm_latency=self.component_times.get(ComponentType.LLM_GENERATION, 0),
            tool_latency=self.component_times.get(ComponentType.TOOL_EXECUTION, 0),
            tts_latency=self.component_times.get(ComponentType.TTS, 0),
            network_latency=self.component_times.get(ComponentType.NETWORK, 0),
            agent_id=self.agent_id
        )

# Global performance tracker instance
performance_tracker: Optional[PerformanceTracker] = None

def initialize_performance_tracker(redis_client: Optional[redis.Redis] = None, 
                                 prometheus_port: int = 8000):
    """Initialize the global performance tracker"""
    global performance_tracker
    
    if performance_tracker is None:
        performance_tracker = PerformanceTracker(redis_client)
        
        # Start Prometheus metrics server
        try:
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    return performance_tracker

def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    if performance_tracker is None:
        raise RuntimeError("Performance tracker not initialized. Call initialize_performance_tracker() first.")
    return performance_tracker

# Decorator for automatic performance tracking
def track_performance(component: ComponentType, agent_id: Optional[str] = None):
    """Decorator for automatic performance tracking"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            request_id = kwargs.get('request_id', f"req_{int(time.time()*1000)}")
            
            async with tracker.track_component(request_id, component, agent_id):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async context
            async def run_with_tracking():
                tracker = get_performance_tracker()
                request_id = kwargs.get('request_id', f"req_{int(time.time()*1000)}")
                
                async with tracker.track_component(request_id, component, agent_id):
                    return func(*args, **kwargs)
            
            return asyncio.create_task(run_with_tracking())
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator