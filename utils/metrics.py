# utils/metrics.py

"""
Metrics collection and monitoring for the voice AI system.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    response_times: List[float]
    error_counts: Dict[str, int]
    success_rate: float
    throughput: float
    latency_p50: float
    latency_p90: float
    latency_p99: float

@dataclass
class ServiceMetrics:
    """Service-specific metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_duration: float
    customer_satisfaction: float

class MetricsCollector:
    """
    Collect and monitor system metrics.
    
    Features:
    1. Performance monitoring
    2. Error tracking
    3. Service quality metrics
    4. Resource utilization
    5. Customer satisfaction
    """
    
    def __init__(self, storage_dir: str = "./metrics"):
        """
        Initialize metrics collector.
        
        Args:
            storage_dir: Directory for metrics storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Prometheus metrics
        self.request_counter = Counter(
            'voice_ai_requests_total',
            'Total number of requests processed',
            ['service_type']
        )
        
        self.response_time = Histogram(
            'voice_ai_response_time_seconds',
            'Response time in seconds',
            ['service_type']
        )
        
        self.active_sessions = Gauge(
            'voice_ai_active_sessions',
            'Number of active sessions'
        )
        
        self.error_counter = Counter(
            'voice_ai_errors_total',
            'Total number of errors',
            ['error_type']
        )
        
        self.customer_satisfaction = Gauge(
            'voice_ai_customer_satisfaction',
            'Customer satisfaction score'
        )
        
        self.system_info = Info(
            'voice_ai_system_info',
            'System information'
        )
        
        # Performance tracking
        self.response_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        
        # Service tracking
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        
        # Resource tracking
        self.resource_usage: Dict[str, List[float]] = {
            "cpu": [],
            "memory": [],
            "network": []
        }
        
        logger.info("Initialized metrics collector")
    
    async def track_request(
        self,
        service_type: str,
        duration: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """
        Track service request metrics.
        
        Args:
            service_type: Type of service
            duration: Request duration
            success: Whether request was successful
            error_type: Optional error type if failed
        """
        # Update Prometheus metrics
        self.request_counter.labels(service_type=service_type).inc()
        self.response_time.labels(service_type=service_type).observe(duration)
        
        # Track success/failure
        if success:
            if service_type not in self.success_counts:
                self.success_counts[service_type] = 0
            self.success_counts[service_type] += 1
        elif error_type:
            self.error_counter.labels(error_type=error_type).inc()
            if error_type not in self.error_counts:
                self.error_counts[error_type] = 0
            self.error_counts[error_type] += 1
        
        # Track response time
        self.response_times.append(duration)
        
        # Update service metrics
        if service_type not in self.service_metrics:
            self.service_metrics[service_type] = ServiceMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_duration=0.0,
                customer_satisfaction=0.0
            )
        
        metrics = self.service_metrics[service_type]
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Update average duration
        metrics.average_duration = (
            (metrics.average_duration * (metrics.total_requests - 1) + duration) /
            metrics.total_requests
        )
        
        # Schedule periodic save
        asyncio.create_task(self._save_metrics())
    
    async def track_session(
        self,
        session_id: str,
        active: bool
    ):
        """
        Track session metrics.
        
        Args:
            session_id: Session identifier
            active: Whether session is active
        """
        if active:
            self.active_sessions.inc()
        else:
            self.active_sessions.dec()
    
    async def track_customer_satisfaction(
        self,
        service_type: str,
        rating: float
    ):
        """
        Track customer satisfaction metrics.
        
        Args:
            service_type: Type of service
            rating: Customer rating (1-5)
        """
        self.customer_satisfaction.set(rating)
        
        if service_type in self.service_metrics:
            metrics = self.service_metrics[service_type]
            metrics.customer_satisfaction = (
                (metrics.customer_satisfaction * metrics.total_requests + rating) /
                (metrics.total_requests + 1)
            )
    
    async def track_resource_usage(
        self,
        cpu_percent: float,
        memory_percent: float,
        network_usage: float
    ):
        """
        Track system resource usage.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            network_usage: Network usage (bytes/s)
        """
        self.resource_usage["cpu"].append(cpu_percent)
        self.resource_usage["memory"].append(memory_percent)
        self.resource_usage["network"].append(network_usage)
        
        # Keep only recent history
        max_history = 1000
        for key in self.resource_usage:
            if len(self.resource_usage[key]) > max_history:
                self.resource_usage[key] = self.resource_usage[key][-max_history:]
    
    async def get_performance_metrics(
        self,
        window: Optional[int] = None
    ) -> PerformanceMetrics:
        """
        Get performance metrics.
        
        Args:
            window: Optional time window in seconds
            
        Returns:
            Performance metrics
        """
        # Filter metrics by time window if specified
        response_times = self.response_times
        if window:
            cutoff_time = time.time() - window
            response_times = [
                t for t in response_times
                if t >= cutoff_time
            ]
        
        if not response_times:
            return PerformanceMetrics(
                response_times=[],
                error_counts=self.error_counts.copy(),
                success_rate=0.0,
                throughput=0.0,
                latency_p50=0.0,
                latency_p90=0.0,
                latency_p99=0.0
            )
        
        # Calculate metrics
        total_requests = sum(self.success_counts.values())
        total_errors = sum(self.error_counts.values())
        
        if total_requests + total_errors > 0:
            success_rate = total_requests / (total_requests + total_errors)
        else:
            success_rate = 0.0
        
        # Calculate throughput (requests per second)
        if window:
            throughput = len(response_times) / window
        else:
            throughput = len(response_times) / (time.time() - self.start_time)
        
        # Calculate latency percentiles
        sorted_times = sorted(response_times)
        return PerformanceMetrics(
            response_times=response_times,
            error_counts=self.error_counts.copy(),
            success_rate=success_rate,
            throughput=throughput,
            latency_p50=sorted_times[len(sorted_times) // 2],
            latency_p90=sorted_times[int(len(sorted_times) * 0.9)],
            latency_p99=sorted_times[int(len(sorted_times) * 0.99)]
        )
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service-specific statistics."""
        stats = {
            "services": {},
            "total_requests": 0,
            "total_errors": sum(self.error_counts.values()),
            "average_satisfaction": 0.0
        }
        
        total_satisfaction = 0.0
        rated_services = 0
        
        for service_type, metrics in self.service_metrics.items():
            stats["services"][service_type] = {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": (
                    metrics.successful_requests / metrics.total_requests
                    if metrics.total_requests > 0 else 0.0
                ),
                "average_duration": metrics.average_duration,
                "customer_satisfaction": metrics.customer_satisfaction
            }
            
            stats["total_requests"] += metrics.total_requests
            if metrics.customer_satisfaction > 0:
                total_satisfaction += metrics.customer_satisfaction
                rated_services += 1
        
        if rated_services > 0:
            stats["average_satisfaction"] = total_satisfaction / rated_services
        
        return stats
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        stats = {}
        
        for resource, values in self.resource_usage.items():
            if not values:
                continue
                
            stats[resource] = {
                "current": values[-1],
                "average": sum(values) / len(values),
                "max": max(values),
                "min": min(values)
            }
        
        return stats
    
    async def _save_metrics(self):
        """Save metrics to storage."""
        try:
            # Prepare data for storage
            data = {
                "timestamp": time.time(),
                "response_times": self.response_times[-1000:],  # Keep last 1000
                "error_counts": self.error_counts,
                "success_counts": self.success_counts,
                "service_metrics": {
                    service_type: metrics.__dict__
                    for service_type, metrics in self.service_metrics.items()
                },
                "resource_usage": self.resource_usage
            }
            
            # Save to file
            metrics_file = self.storage_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    async def load_metrics(self):
        """Load metrics from storage."""
        try:
            metrics_file = self.storage_dir / "metrics.json"
            if not metrics_file.exists():
                return
                
            with open(metrics_file) as f:
                data = json.load(f)
                
            # Load metrics
            self.response_times = data.get("response_times", [])
            self.error_counts = data.get("error_counts", {})
            self.success_counts = data.get("success_counts", {})
            
            # Load service metrics
            for service_type, metrics in data.get("service_metrics", {}).items():
                self.service_metrics[service_type] = ServiceMetrics(**metrics)
            
            # Load resource usage
            self.resource_usage = data.get("resource_usage", {
                "cpu": [],
                "memory": [],
                "network": []
            })
            
            logger.info("Loaded metrics from storage")
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        from prometheus_client import generate_latest
        return generate_latest().decode('utf-8')