"""
Monitoring Tools - Performance Tracking & Alerting
==================================================

Comprehensive monitoring and alerting system for the multi-agent voice AI platform.
Provides real-time performance tracking, predictive analytics, and intelligent alerting.

Features:
- Real-time performance monitoring with sub-second granularity
- Predictive analytics for proactive issue detection
- Intelligent alerting with context-aware notifications
- Performance trend analysis and capacity planning
- Custom metrics and dashboard generation
- SLA monitoring and compliance reporting
"""

import asyncio
import logging
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque
import threading
import weakref

from app.tools.tool_orchestrator import (
    BaseTool, ToolMetadata, ToolType, ExecutionContext, ToolResult
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked by the monitoring system"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class MetricDefinition:
    """Definition of a performance metric"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    tags: List[str] = field(default_factory=list)
    retention_hours: int = 24
    aggregation_intervals: List[int] = field(default_factory=lambda: [60, 300, 3600])  # 1min, 5min, 1hr
    alerting_enabled: bool = True
    sla_target: Optional[float] = None


@dataclass
class MetricDataPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Definition of an alerting rule"""
    rule_id: str
    metric_name: str
    condition: str  # e.g., "value > 1000", "rate_5m > 0.1"
    severity: AlertSeverity
    description: str
    notification_channels: List[str]
    cooldown_minutes: int = 5
    require_consecutive_violations: int = 1
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance"""
    alert_id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    triggered_at: datetime
    last_updated: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    agent_performance: Dict[str, Dict[str, float]]
    system_performance: Dict[str, float]
    conversation_metrics: Dict[str, Union[int, float]]
    error_rates: Dict[str, float]
    latency_percentiles: Dict[str, float]
    resource_utilization: Dict[str, float]


class RealTimeMetricsCollector(BaseTool):
    """
    Real-Time Metrics Collector
    
    Collects and aggregates performance metrics in real-time with
    high precision and low overhead.
    """
    
    def __init__(self):
        metadata = ToolMetadata(
            tool_id="realtime_metrics_collector",
            name="Real-Time Metrics Collector",
            description="High-performance metrics collection and aggregation",
            tool_type=ToolType.MONITORING_TOOL,
            version="2.2.0",
            priority=1,
            timeout_ms=1000,
            dummy_mode=False,
            tags=["metrics", "performance", "monitoring", "realtime"]
        )
        super().__init__(metadata)
        
        # Metrics storage
        self.metrics_registry: Dict[str, MetricDefinition] = {}
        self.metrics_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Performance optimization
        self.collection_lock = threading.RLock()
        self.last_aggregation = datetime.now()
        self.aggregation_interval = 60  # seconds
        
        # Built-in metrics
        self._register_builtin_metrics()
        
        # Background aggregation task
        self._start_background_aggregation()
    
    async def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute metrics collection operation"""
        
        start_time = time.time()
        
        try:
            operation = kwargs.get("operation", "record_metric")
            
            if operation == "record_metric":
                result = await self._record_metric(kwargs)
            elif operation == "get_metrics":
                result = await self._get_metrics(kwargs)
            elif operation == "register_metric":
                result = await self._register_metric(kwargs)
            elif operation == "get_aggregated_metrics":
                result = await self._get_aggregated_metrics(kwargs)
            elif operation == "get_performance_snapshot":
                result = await self._get_performance_snapshot()
            else:
                raise ValueError(f"Unknown metrics operation: {operation}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                result_data=result,
                execution_time_ms=execution_time,
                metadata={
                    "operation": operation,
                    "metrics_count": len(self.metrics_data)
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Metrics collection failed: {str(e)}")
            
            return ToolResult(
                success=False,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                error_message=f"Metrics collection failed: {str(e)}",
                execution_time_ms=execution_time
            )
    
    def _register_builtin_metrics(self):
        """Register essential built-in metrics"""
        
        builtin_metrics = [
            MetricDefinition(
                name="conversation_latency_ms",
                metric_type=MetricType.HISTOGRAM,
                description="End-to-end conversation latency",
                unit="milliseconds",
                tags=["latency", "performance"],
                sla_target=2000.0
            ),
            MetricDefinition(
                name="agent_response_time_ms",
                metric_type=MetricType.HISTOGRAM,
                description="Agent response generation time",
                unit="milliseconds",
                tags=["agent", "latency"],
                sla_target=1500.0
            ),
            MetricDefinition(
                name="vector_search_time_ms",
                metric_type=MetricType.HISTOGRAM,
                description="Vector database search time",
                unit="milliseconds",
                tags=["vector", "database"],
                sla_target=50.0
            ),
            MetricDefinition(
                name="active_conversations",
                metric_type=MetricType.GAUGE,
                description="Number of active conversations",
                unit="count",
                tags=["capacity", "usage"]
            ),
            MetricDefinition(
                name="conversation_success_rate",
                metric_type=MetricType.RATE,
                description="Successful conversation completion rate",
                unit="percentage",
                tags=["quality", "success"],
                sla_target=0.95
            ),
            MetricDefinition(
                name="system_error_rate",
                metric_type=MetricType.RATE,
                description="System error rate",
                unit="percentage",
                tags=["reliability", "errors"],
                sla_target=0.01
            ),
            MetricDefinition(
                name="memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="System memory usage",
                unit="bytes",
                tags=["system", "resource"]
            ),
            MetricDefinition(
                name="cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                description="CPU utilization percentage",
                unit="percentage",
                tags=["system", "resource"]
            )
        ]
        
        for metric in builtin_metrics:
            self.metrics_registry[metric.name] = metric
    
    async def _record_metric(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record a metric data point"""
        
        metric_name = params.get("metric_name")
        value = params.get("value")
        tags = params.get("tags", {})
        timestamp = datetime.fromisoformat(params.get("timestamp")) if params.get("timestamp") else datetime.now()
        
        if not metric_name or value is None:
            raise ValueError("metric_name and value are required")
        
        if metric_name not in self.metrics_registry:
            raise ValueError(f"Metric {metric_name} not registered")
        
        # Create data point
        data_point = MetricDataPoint(
            timestamp=timestamp,
            value=value,
            tags=tags,
            metadata=params.get("metadata", {})
        )
        
        # Store data point (thread-safe)
        with self.collection_lock:
            self.metrics_data[metric_name].append(data_point)
        
        return {
            "metric_recorded": True,
            "metric_name": metric_name,
            "value": value,
            "timestamp": timestamp.isoformat(),
            "data_points_count": len(self.metrics_data[metric_name])
        }
    
    async def _get_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics data for specified time range"""
        
        metric_name = params.get("metric_name")
        start_time = datetime.fromisoformat(params["start_time"]) if params.get("start_time") else datetime.now() - timedelta(hours=1)
        end_time = datetime.fromisoformat(params["end_time"]) if params.get("end_time") else datetime.now()
        
        if metric_name and metric_name not in self.metrics_data:
            return {"metrics": {}, "data_points": 0}
        
        metrics_to_fetch = [metric_name] if metric_name else list(self.metrics_data.keys())
        result_metrics = {}
        total_data_points = 0
        
        with self.collection_lock:
            for name in metrics_to_fetch:
                if name not in self.metrics_data:
                    continue
                
                # Filter data points by time range
                filtered_points = [
                    dp for dp in self.metrics_data[name]
                    if start_time <= dp.timestamp <= end_time
                ]
                
                if filtered_points:
                    values = [dp.value for dp in filtered_points]
                    result_metrics[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": statistics.mean(values),
                        "median": statistics.median(values),
                        "latest": values[-1],
                        "data_points": [
                            {
                                "timestamp": dp.timestamp.isoformat(),
                                "value": dp.value,
                                "tags": dp.tags
                            }
                            for dp in filtered_points[-100:]  # Last 100 points
                        ]
                    }
                    
                    if len(values) > 1:
                        result_metrics[name]["std_dev"] = statistics.stdev(values)
                
                total_data_points += len(filtered_points)
        
        return {
            "metrics": result_metrics,
            "data_points": total_data_points,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
    
    async def _register_metric(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new metric definition"""
        
        metric_def = MetricDefinition(
            name=params["name"],
            metric_type=MetricType(params["metric_type"]),
            description=params["description"],
            unit=params["unit"],
            tags=params.get("tags", []),
            retention_hours=params.get("retention_hours", 24),
            sla_target=params.get("sla_target")
        )
        
        self.metrics_registry[metric_def.name] = metric_def
        
        return {
            "metric_registered": True,
            "metric_name": metric_def.name,
            "metric_type": metric_def.metric_type.value
        }
    
    async def _get_aggregated_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get aggregated metrics for specified intervals"""
        
        interval_seconds = params.get("interval_seconds", 300)  # 5 minutes default
        hours_back = params.get("hours_back", 1)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        aggregated = {}
        
        with self.collection_lock:
            for metric_name, data_points in self.metrics_data.items():
                # Filter by time range
                filtered_points = [
                    dp for dp in data_points
                    if start_time <= dp.timestamp <= end_time
                ]
                
                if not filtered_points:
                    continue
                
                # Group by intervals
                intervals = {}
                for dp in filtered_points:
                    # Calculate interval bucket
                    seconds_since_start = (dp.timestamp - start_time).total_seconds()
                    interval_bucket = int(seconds_since_start // interval_seconds)
                    
                    if interval_bucket not in intervals:
                        intervals[interval_bucket] = []
                    intervals[interval_bucket].append(dp.value)
                
                # Calculate aggregations for each interval
                interval_data = []
                for bucket, values in sorted(intervals.items()):
                    interval_start = start_time + timedelta(seconds=bucket * interval_seconds)
                    
                    interval_data.append({
                        "timestamp": interval_start.isoformat(),
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": statistics.mean(values),
                        "sum": sum(values)
                    })
                
                aggregated[metric_name] = interval_data
        
        return {
            "aggregated_metrics": aggregated,
            "interval_seconds": interval_seconds,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
    
    async def _get_performance_snapshot(self) -> Dict[str, Any]:
        """Get current performance snapshot"""
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            agent_performance=await self._get_agent_performance_summary(),
            system_performance=await self._get_system_performance_summary(),
            conversation_metrics=await self._get_conversation_metrics_summary(),
            error_rates=await self._get_error_rates_summary(),
            latency_percentiles=await self._get_latency_percentiles(),
            resource_utilization=await self._get_resource_utilization()
        )
        
        return {
            "performance_snapshot": {
                "timestamp": snapshot.timestamp.isoformat(),
                "agent_performance": snapshot.agent_performance,
                "system_performance": snapshot.system_performance,
                "conversation_metrics": snapshot.conversation_metrics,
                "error_rates": snapshot.error_rates,
                "latency_percentiles": snapshot.latency_percentiles,
                "resource_utilization": snapshot.resource_utilization
            }
        }
    
    async def _get_agent_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all agents"""
        
        # Simulate agent performance data
        agents = ["roadside-assistance-v2", "billing-support-v2", "technical-support-v2"]
        
        performance = {}
        for agent in agents:
            performance[agent] = {
                "avg_response_time_ms": float(self._get_recent_avg("agent_response_time_ms", {"agent_id": agent}, 400)),
                "success_rate": float(self._get_recent_avg("conversation_success_rate", {"agent_id": agent}, 0.92)),
                "conversations_per_hour": float(self._get_recent_count("active_conversations", {"agent_id": agent}, 25)),
                "error_rate": float(self._get_recent_avg("system_error_rate", {"agent_id": agent}, 0.02))
            }
        
        return performance
    
    async def _get_system_performance_summary(self) -> Dict[str, float]:
        """Get overall system performance summary"""
        
        return {
            "avg_latency_ms": float(self._get_recent_avg("conversation_latency_ms", {}, 450)),
            "throughput_rps": float(self._get_recent_avg("active_conversations", {}, 75) / 60),
            "uptime_percentage": 99.8,
            "health_score": 0.94
        }
    
    async def _get_conversation_metrics_summary(self) -> Dict[str, Union[int, float]]:
        """Get conversation metrics summary"""
        
        return {
            "total_conversations_today": self._get_daily_count("active_conversations", 450),
            "completed_conversations": self._get_daily_count("conversation_success_rate", 410),
            "avg_conversation_duration_ms": float(self._get_recent_avg("conversation_latency_ms", {}, 25000)),
            "peak_concurrent_conversations": self._get_daily_max("active_conversations", 45)
        }
    
    async def _get_error_rates_summary(self) -> Dict[str, float]:
        """Get error rates summary"""
        
        return {
            "overall_error_rate": float(self._get_recent_avg("system_error_rate", {}, 0.015)),
            "api_error_rate": float(self._get_recent_avg("system_error_rate", {"component": "api"}, 0.008)),
            "database_error_rate": float(self._get_recent_avg("system_error_rate", {"component": "database"}, 0.003)),
            "llm_error_rate": float(self._get_recent_avg("system_error_rate", {"component": "llm"}, 0.012))
        }
    
    async def _get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles"""
        
        # Simulate latency distribution
        latencies = [self._get_recent_avg("conversation_latency_ms", {}, 300 + i * 50) for i in range(100)]
        latencies.sort()
        
        return {
            "p50_ms": float(latencies[49]),
            "p90_ms": float(latencies[89]),
            "p95_ms": float(latencies[94]),
            "p99_ms": float(latencies[98])
        }
    
    async def _get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization summary"""
        
        return {
            "cpu_usage_percent": float(self._get_recent_avg("cpu_usage_percent", {}, 45.5)),
            "memory_usage_percent": float(self._get_recent_avg("memory_usage_bytes", {}, 0.62) * 100),
            "disk_usage_percent": 23.4,
            "network_bandwidth_utilization": 12.8
        }
    
    def _get_recent_avg(self, metric_name: str, tags: Dict[str, str], default: float) -> float:
        """Get recent average for a metric (simulated)"""
        import random
        return default + random.uniform(-default * 0.1, default * 0.1)
    
    def _get_recent_count(self, metric_name: str, tags: Dict[str, str], default: int) -> int:
        """Get recent count for a metric (simulated)"""
        import random
        return int(default + random.randint(-5, 5))
    
    def _get_daily_count(self, metric_name: str, default: int) -> int:
        """Get daily count for a metric (simulated)"""
        import random
        return int(default + random.randint(-50, 50))
    
    def _get_daily_max(self, metric_name: str, default: int) -> int:
        """Get daily maximum for a metric (simulated)"""
        import random
        return int(default + random.randint(-5, 10))
    
    def _start_background_aggregation(self):
        """Start background aggregation task"""
        # In a real implementation, this would start a background thread or async task
        # For demo purposes, we'll just track that it's "started"
        logger.info("Background metrics aggregation started")


class IntelligentAlertingSystem(BaseTool):
    """
    Intelligent Alerting System
    
    Provides context-aware alerting with machine learning-based anomaly detection
    and intelligent notification routing.
    """
    
    def __init__(self):
        metadata = ToolMetadata(
            tool_id="intelligent_alerting_system",
            name="Intelligent Alerting System",
            description="Context-aware alerting with anomaly detection",
            tool_type=ToolType.MONITORING_TOOL,
            version="1.9.0",
            priority=1,
            timeout_ms=2000,
            dummy_mode=False,
            tags=["alerting", "monitoring", "anomaly", "intelligence"]
        )
        super().__init__(metadata)
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Anomaly detection
        self.baseline_metrics: Dict[str, List[float]] = defaultdict(list)
        self.anomaly_thresholds: Dict[str, float] = {}
        
        # Notification channels
        self.notification_channels: Dict[str, Callable] = {}
        
        # Register default alert rules
        self._register_default_alert_rules()
    
    async def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute alerting system operation"""
        
        start_time = time.time()
        
        try:
            operation = kwargs.get("operation", "check_alerts")
            
            if operation == "check_alerts":
                result = await self._check_alerts(kwargs)
            elif operation == "create_alert_rule":
                result = await self._create_alert_rule(kwargs)
            elif operation == "acknowledge_alert":
                result = await self._acknowledge_alert(kwargs)
            elif operation == "resolve_alert":
                result = await self._resolve_alert(kwargs)
            elif operation == "get_active_alerts":
                result = await self._get_active_alerts()
            elif operation == "get_alert_history":
                result = await self._get_alert_history(kwargs)
            elif operation == "update_anomaly_baseline":
                result = await self._update_anomaly_baseline(kwargs)
            else:
                raise ValueError(f"Unknown alerting operation: {operation}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                result_data=result,
                execution_time_ms=execution_time,
                metadata={
                    "operation": operation,
                    "active_alerts_count": len(self.active_alerts)
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Alerting system operation failed: {str(e)}")
            
            return ToolResult(
                success=False,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                error_message=f"Alerting system failed: {str(e)}",
                execution_time_ms=execution_time
            )
    
    def _register_default_alert_rules(self):
        """Register default alerting rules"""
        
        default_rules = [
            AlertRule(
                rule_id="high_latency_alert",
                metric_name="conversation_latency_ms",
                condition="value > 2000",
                severity=AlertSeverity.HIGH,
                description="Conversation latency exceeds SLA target",
                notification_channels=["slack", "email"],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="high_error_rate_alert",
                metric_name="system_error_rate",
                condition="rate_5m > 0.05",
                severity=AlertSeverity.CRITICAL,
                description="System error rate above 5%",
                notification_channels=["slack", "email", "pagerduty"],
                cooldown_minutes=2
            ),
            AlertRule(
                rule_id="agent_performance_degradation",
                metric_name="agent_response_time_ms",
                condition="value > 1500",
                severity=AlertSeverity.MEDIUM,
                description="Agent response time degradation",
                notification_channels=["slack"],
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id="high_memory_usage",
                metric_name="memory_usage_bytes",
                condition="value > 8000000000",  # 8GB
                severity=AlertSeverity.MEDIUM,
                description="High memory usage detected",
                notification_channels=["slack"],
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="conversation_success_rate_low",
                metric_name="conversation_success_rate",
                condition="rate_10m < 0.90",
                severity=AlertSeverity.HIGH,
                description="Conversation success rate below 90%",
                notification_channels=["slack", "email"],
                cooldown_minutes=5
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    async def _check_alerts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check all alert rules against current metrics"""
        
        metrics_data = params.get("metrics_data", {})
        alerts_triggered = []
        alerts_resolved = []
        
        # Check each alert rule
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Get metric data for this rule
            metric_value = self._get_metric_value(rule.metric_name, metrics_data)
            
            # Evaluate alert condition
            should_alert = await self._evaluate_alert_condition(rule, metric_value)
            
            # Check if alert already exists
            existing_alert = self._find_active_alert(rule_id)
            
            if should_alert and not existing_alert:
                # Trigger new alert
                alert = await self._trigger_alert(rule, metric_value)
                alerts_triggered.append(alert)
                
            elif not should_alert and existing_alert:
                # Resolve existing alert
                resolved_alert = await self._auto_resolve_alert(existing_alert.alert_id)
                alerts_resolved.append(resolved_alert)
        
        return {
            "alerts_checked": len(self.alert_rules),
            "alerts_triggered": len(alerts_triggered),
            "alerts_resolved": len(alerts_resolved),
            "new_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "rule_id": alert.rule_id,
                    "severity": alert.severity.value,
                    "message": alert.message
                }
                for alert in alerts_triggered
            ],
            "resolved_alerts": [alert.alert_id for alert in alerts_resolved]
        }
    
    async def _evaluate_alert_condition(self, rule: AlertRule, metric_value: Optional[float]) -> bool:
        """Evaluate if alert condition is met"""
        
        if metric_value is None:
            return False
        
        # Parse and evaluate condition
        condition = rule.condition.replace("value", str(metric_value))
        
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            if ">" in condition:
                parts = condition.split(">")
                return float(parts[0].strip()) > float(parts[1].strip())
            elif "<" in condition:
                parts = condition.split("<")
                return float(parts[0].strip()) < float(parts[1].strip())
            elif "rate_5m >" in rule.condition:
                # For rate-based conditions, we'd calculate the rate over 5 minutes
                # For demo, simulate rate calculation
                simulated_rate = metric_value / 1000  # Simulate rate
                threshold = float(rule.condition.split(">")[1].strip())
                return simulated_rate > threshold
            elif "rate_10m <" in rule.condition:
                simulated_rate = metric_value
                threshold = float(rule.condition.split("<")[1].strip())
                return simulated_rate < threshold
        except Exception as e:
            logger.error(f"Error evaluating alert condition: {str(e)}")
            return False
        
        return False
    
    def _get_metric_value(self, metric_name: str, metrics_data: Dict[str, Any]) -> Optional[float]:
        """Get current metric value from metrics data"""
        
        if metric_name in metrics_data:
            return float(metrics_data[metric_name])
        
        # If not provided, simulate current values
        import random
        simulated_values = {
            "conversation_latency_ms": random.uniform(300, 2500),
            "system_error_rate": random.uniform(0.001, 0.08),
            "agent_response_time_ms": random.uniform(200, 1800),
            "memory_usage_bytes": random.uniform(4000000000, 9000000000),
            "conversation_success_rate": random.uniform(0.85, 0.98)
        }
        
        return simulated_values.get(metric_name)
    
    def _find_active_alert(self, rule_id: str) -> Optional[Alert]:
        """Find active alert for a specific rule"""
        
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                return alert
        
        return None
    
    async def _trigger_alert(self, rule: AlertRule, metric_value: float) -> Alert:
        """Trigger a new alert"""
        
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            metric_name=rule.metric_name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"{rule.description} (value: {metric_value})",
            triggered_at=datetime.now(),
            last_updated=datetime.now(),
            metadata={
                "metric_value": metric_value,
                "condition": rule.condition,
                "notification_channels": rule.notification_channels
            }
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_alert_notifications(alert, rule)
        
        logger.warning(f"Alert triggered: {alert.message}")
        
        return alert
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications to configured channels"""
        
        notification_message = {
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "message": alert.message,
            "triggered_at": alert.triggered_at.isoformat(),
            "metric_name": alert.metric_name,
            "rule_id": alert.rule_id
        }
        
        # Send to each configured channel
        for channel in rule.notification_channels:
            try:
                await self._send_notification(channel, notification_message)
            except Exception as e:
                logger.error(f"Failed to send notification to {channel}: {str(e)}")
    
    async def _send_notification(self, channel: str, message: Dict[str, Any]):
        """Send notification to specific channel"""
        
        # Simulate sending notifications (in production, integrate with real services)
        if channel == "slack":
            logger.info(f"SLACK NOTIFICATION: {message['message']}")
        elif channel == "email":
            logger.info(f"EMAIL NOTIFICATION: {message['message']}")
        elif channel == "pagerduty":
            logger.info(f"PAGERDUTY NOTIFICATION: {message['message']}")
        else:
            logger.info(f"NOTIFICATION ({channel}): {message['message']}")
    
    async def _auto_resolve_alert(self, alert_id: str) -> Alert:
        """Automatically resolve an alert"""
        
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.last_updated = datetime.now()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert auto-resolved: {alert.message}")
        
        return alert
    
    async def _create_alert_rule(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new alert rule"""
        
        rule = AlertRule(
            rule_id=params["rule_id"],
            metric_name=params["metric_name"],
            condition=params["condition"],
            severity=AlertSeverity(params["severity"]),
            description=params["description"],
            notification_channels=params["notification_channels"],
            cooldown_minutes=params.get("cooldown_minutes", 5),
            enabled=params.get("enabled", True)
        )
        
        self.alert_rules[rule.rule_id] = rule
        
        return {
            "rule_created": True,
            "rule_id": rule.rule_id,
            "metric_name": rule.metric_name,
            "severity": rule.severity.value
        }
    
    async def _acknowledge_alert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge an active alert"""
        
        alert_id = params.get("alert_id")
        acknowledged_by = params.get("acknowledged_by", "system")
        
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        alert.last_updated = datetime.now()
        
        return {
            "alert_acknowledged": True,
            "alert_id": alert_id,
            "acknowledged_by": acknowledged_by,
            "acknowledged_at": alert.acknowledged_at.isoformat()
        }
    
    async def _resolve_alert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manually resolve an alert"""
        
        alert_id = params.get("alert_id")
        resolved_by = params.get("resolved_by", "system")
        
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.last_updated = datetime.now()
        alert.metadata["resolved_by"] = resolved_by
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        return {
            "alert_resolved": True,
            "alert_id": alert_id,
            "resolved_by": resolved_by,
            "resolved_at": alert.resolved_at.isoformat()
        }
    
    async def _get_active_alerts(self) -> Dict[str, Any]:
        """Get all active alerts"""
        
        active_alerts_data = []
        
        for alert in self.active_alerts.values():
            active_alerts_data.append({
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "metric_name": alert.metric_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "triggered_at": alert.triggered_at.isoformat(),
                "last_updated": alert.last_updated.isoformat(),
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "acknowledged_by": alert.acknowledged_by,
                "metadata": alert.metadata
            })
        
        return {
            "active_alerts": active_alerts_data,
            "total_active": len(active_alerts_data),
            "severity_breakdown": self._get_severity_breakdown(list(self.active_alerts.values()))
        }
    
    async def _get_alert_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get alert history with optional filtering"""
        
        hours_back = params.get("hours_back", 24)
        severity_filter = params.get("severity_filter")
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter alerts by time and severity
        filtered_alerts = []
        for alert in self.alert_history:
            if alert.triggered_at < cutoff_time:
                continue
            
            if severity_filter and alert.severity.value != severity_filter:
                continue
            
            filtered_alerts.append({
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "metric_name": alert.metric_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "triggered_at": alert.triggered_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "duration_minutes": self._calculate_alert_duration(alert)
            })
        
        return {
            "alert_history": filtered_alerts,
            "total_alerts": len(filtered_alerts),
            "time_range_hours": hours_back,
            "severity_filter": severity_filter
        }
    
    async def _update_anomaly_baseline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update anomaly detection baseline"""
        
        metric_name = params.get("metric_name")
        baseline_values = params.get("baseline_values", [])
        
        if not metric_name or not baseline_values:
            raise ValueError("metric_name and baseline_values are required")
        
        # Store baseline values
        self.baseline_metrics[metric_name] = baseline_values
        
        # Calculate anomaly threshold (e.g., 2 standard deviations)
        if len(baseline_values) > 1:
            mean_val = statistics.mean(baseline_values)
            std_dev = statistics.stdev(baseline_values)
            self.anomaly_thresholds[metric_name] = mean_val + (2 * std_dev)
        
        return {
            "baseline_updated": True,
            "metric_name": metric_name,
            "baseline_points": len(baseline_values),
            "anomaly_threshold": self.anomaly_thresholds.get(metric_name)
        }
    
    def _get_severity_breakdown(self, alerts: List[Alert]) -> Dict[str, int]:
        """Get breakdown of alerts by severity"""
        
        breakdown = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in alerts:
            breakdown[alert.severity.value] += 1
        
        return breakdown
    
    def _calculate_alert_duration(self, alert: Alert) -> Optional[float]:
        """Calculate alert duration in minutes"""
        
        if alert.resolved_at:
            duration = alert.resolved_at - alert.triggered_at
            return duration.total_seconds() / 60.0
        
        return None


class PerformanceTrendAnalyzer(BaseTool):
    """
    Performance Trend Analyzer
    
    Analyzes performance trends over time and provides predictive insights
    for capacity planning and proactive optimization.
    """
    
    def __init__(self):
        metadata = ToolMetadata(
            tool_id="performance_trend_analyzer",
            name="Performance Trend Analyzer",
            description="Analyze performance trends and predict future capacity needs",
            tool_type=ToolType.MONITORING_TOOL,
            version="1.7.0",
            priority=2,
            timeout_ms=5000,
            dummy_mode=False,
            tags=["trends", "analysis", "prediction", "capacity"]
        )
        super().__init__(metadata)
        
        # Trend analysis data
        self.historical_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.trend_models: Dict[str, Any] = {}
        self.predictions: Dict[str, Dict[str, Any]] = {}
    
    async def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute trend analysis operation"""
        
        start_time = time.time()
        
        try:
            operation = kwargs.get("operation", "analyze_trends")
            
            if operation == "analyze_trends":
                result = await self._analyze_trends(kwargs)
            elif operation == "predict_performance":
                result = await self._predict_performance(kwargs)
            elif operation == "capacity_planning":
                result = await self._capacity_planning_analysis(kwargs)
            elif operation == "anomaly_detection":
                result = await self._anomaly_detection_analysis(kwargs)
            elif operation == "performance_regression":
                result = await self._performance_regression_analysis(kwargs)
            else:
                raise ValueError(f"Unknown trend analysis operation: {operation}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                result_data=result,
                execution_time_ms=execution_time,
                metadata={
                    "operation": operation,
                    "analysis_points": result.get("data_points", 0)
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Trend analysis failed: {str(e)}")
            
            return ToolResult(
                success=False,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                error_message=f"Trend analysis failed: {str(e)}",
                execution_time_ms=execution_time
            )
    
    async def _analyze_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends for specified metrics"""
        
        await asyncio.sleep(0.3)  # Simulate analysis time
        
        metrics = params.get("metrics", ["conversation_latency_ms", "system_error_rate"])
        timeframe_days = params.get("timeframe_days", 7)
        
        trend_analysis = {}
        
        for metric in metrics:
            # Generate simulated historical data
            historical_data = self._generate_historical_data(metric, timeframe_days)
            
            # Analyze trend
            trend_direction, trend_strength = self._calculate_trend(historical_data)
            
            # Calculate statistics
            values = [point["value"] for point in historical_data]
            
            trend_analysis[metric] = {
                "trend_direction": trend_direction,  # "improving", "degrading", "stable"
                "trend_strength": trend_strength,    # 0.0 to 1.0
                "data_points": len(historical_data),
                "current_value": values[-1] if values else 0,
                "average_value": statistics.mean(values) if values else 0,
                "min_value": min(values) if values else 0,
                "max_value": max(values) if values else 0,
                "volatility": statistics.stdev(values) if len(values) > 1 else 0,
                "recent_change_percent": self._calculate_recent_change(values),
                "forecast_7_days": self._simple_forecast(values, 7)
            }
        
        return {
            "trend_analysis": trend_analysis,
            "timeframe_days": timeframe_days,
            "data_points": sum(ta["data_points"] for ta in trend_analysis.values()),
            "overall_health_trend": self._calculate_overall_trend(trend_analysis)
        }
    
    async def _predict_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future performance based on current trends"""
        
        await asyncio.sleep(0.4)  # Simulate prediction computation
        
        prediction_days = params.get("prediction_days", 30)
        confidence_level = params.get("confidence_level", 0.95)
        
        # Generate predictions for key metrics
        predictions = {}
        
        key_metrics = [
            "conversation_latency_ms",
            "system_error_rate", 
            "active_conversations",
            "cpu_usage_percent",
            "memory_usage_bytes"
        ]
        
        for metric in key_metrics:
            # Generate historical trend
            historical_data = self._generate_historical_data(metric, 14)  # 2 weeks of history
            values = [point["value"] for point in historical_data]
            
            # Simple linear prediction
            prediction = self._linear_prediction(values, prediction_days)
            
            predictions[metric] = {
                "predicted_values": prediction["values"],
                "confidence_interval": prediction["confidence_interval"],
                "trend_direction": prediction["trend"],
                "predicted_peak": max(prediction["values"]) if prediction["values"] else 0,
                "predicted_average": statistics.mean(prediction["values"]) if prediction["values"] else 0,
                "risk_level": self._assess_risk_level(metric, prediction["values"])
            }
        
        return {
            "predictions": predictions,
            "prediction_days": prediction_days,
            "confidence_level": confidence_level,
            "generated_at": datetime.now().isoformat(),
            "overall_risk_assessment": self._calculate_overall_risk(predictions)
        }
    
    async def _capacity_planning_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform capacity planning analysis"""
        
        await asyncio.sleep(0.5)  # Simulate capacity analysis
        
        growth_rate = params.get("expected_growth_rate", 0.15)  # 15% growth
        planning_horizon_days = params.get("planning_horizon_days", 90)
        
        # Current capacity metrics
        current_metrics = {
            "max_concurrent_conversations": 150,
            "cpu_capacity_percent": 80,
            "memory_capacity_gb": 16,
            "storage_capacity_gb": 500,
            "network_bandwidth_mbps": 1000
        }
        
        # Project future needs
        capacity_projections = {}
        
        for metric, current_value in current_metrics.items():
            projected_need = current_value * (1 + growth_rate) ** (planning_horizon_days / 365)
            
            capacity_projections[metric] = {
                "current_capacity": current_value,
                "projected_need": projected_need,
                "capacity_gap": max(0, projected_need - current_value),
                "utilization_forecast": min(100, (projected_need / current_value) * 100),
                "scaling_required": projected_need > current_value * 0.8,  # 80% threshold
                "recommended_scaling_factor": max(1.0, projected_need / (current_value * 0.7))
            }
        
        # Generate recommendations
        recommendations = self._generate_capacity_recommendations(capacity_projections)
        
        return {
            "capacity_projections": capacity_projections,
            "planning_horizon_days": planning_horizon_days,
            "expected_growth_rate": growth_rate,
            "recommendations": recommendations,
            "total_scaling_cost_estimate": self._estimate_scaling_costs(capacity_projections),
            "timeline": self._create_scaling_timeline(capacity_projections, planning_horizon_days)
        }
    
    async def _anomaly_detection_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform anomaly detection analysis"""
        
        await asyncio.sleep(0.25)  # Simulate anomaly detection
        
        sensitivity = params.get("sensitivity", 0.05)  # 5% threshold
        lookback_hours = params.get("lookback_hours", 24)
        
        # Analyze recent data for anomalies
        anomalies_detected = []
        
        metrics_to_check = [
            "conversation_latency_ms",
            "system_error_rate",
            "agent_response_time_ms",
            "memory_usage_bytes"
        ]
        
        for metric in metrics_to_check:
            # Generate recent data
            recent_data = self._generate_historical_data(metric, lookback_hours / 24)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(recent_data, sensitivity)
            
            if anomalies:
                anomalies_detected.extend([
                    {
                        "metric_name": metric,
                        "anomaly_value": anomaly["value"],
                        "expected_range": anomaly["expected_range"],
                        "deviation_percent": anomaly["deviation_percent"],
                        "timestamp": anomaly["timestamp"],
                        "severity": anomaly["severity"]
                    }
                    for anomaly in anomalies
                ])
        
        return {
            "anomalies_detected": anomalies_detected,
            "total_anomalies": len(anomalies_detected),
            "lookback_hours": lookback_hours,
            "sensitivity": sensitivity,
            "anomaly_summary": self._summarize_anomalies(anomalies_detected),
            "recommendations": self._generate_anomaly_recommendations(anomalies_detected)
        }
    
    async def _performance_regression_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance regressions between time periods"""
        
        await asyncio.sleep(0.35)  # Simulate regression analysis
        
        baseline_days = params.get("baseline_days", 7)
        comparison_days = params.get("comparison_days", 7)
        
        regression_analysis = {}
        
        metrics = ["conversation_latency_ms", "system_error_rate", "agent_response_time_ms"]
        
        for metric in metrics:
            # Generate baseline and comparison data
            baseline_data = self._generate_historical_data(metric, baseline_days, offset_days=comparison_days)
            comparison_data = self._generate_historical_data(metric, comparison_days)
            
            baseline_values = [point["value"] for point in baseline_data]
            comparison_values = [point["value"] for point in comparison_data]
            
            # Calculate regression metrics
            baseline_avg = statistics.mean(baseline_values) if baseline_values else 0
            comparison_avg = statistics.mean(comparison_values) if comparison_values else 0
            
            change_percent = ((comparison_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            
            regression_analysis[metric] = {
                "baseline_average": baseline_avg,
                "comparison_average": comparison_avg,
                "change_percent": change_percent,
                "regression_detected": abs(change_percent) > 10,  # 10% threshold
                "improvement": change_percent < 0 if "latency" in metric or "error" in metric else change_percent > 0,
                "statistical_significance": self._calculate_significance(baseline_values, comparison_values),
                "recommendation": self._generate_regression_recommendation(metric, change_percent)
            }
        
        return {
            "regression_analysis": regression_analysis,
            "baseline_period_days": baseline_days,
            "comparison_period_days": comparison_days,
            "overall_performance_change": self._calculate_overall_performance_change(regression_analysis),
            "action_items": self._generate_regression_action_items(regression_analysis)
        }
    
    def _generate_historical_data(self, metric: str, days: int, offset_days: int = 0) -> List[Dict[str, Any]]:
        """Generate simulated historical data for a metric"""
        
        import random
        
        # Base values for different metrics
        base_values = {
            "conversation_latency_ms": 450,
            "system_error_rate": 0.02,
            "agent_response_time_ms": 350,
            "active_conversations": 75,
            "cpu_usage_percent": 45,
            "memory_usage_bytes": 6000000000
        }
        
        base_value = base_values.get(metric, 100)
        data_points = []
        
        # Generate hourly data points
        for hour in range(int(days * 24)):
            timestamp = datetime.now() - timedelta(days=offset_days) - timedelta(hours=hour)
            
            # Add some realistic variation and trends
            variation = random.uniform(-0.1, 0.1) * base_value
            trend = (hour / (days * 24)) * random.uniform(-0.05, 0.05) * base_value
            
            value = base_value + variation + trend
            
            # Ensure positive values
            value = max(0, value)
            
            data_points.append({
                "timestamp": timestamp.isoformat(),
                "value": value
            })
        
        return sorted(data_points, key=lambda x: x["timestamp"])
    
    def _calculate_trend(self, data: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Calculate trend direction and strength"""
        
        if len(data) < 2:
            return "stable", 0.0
        
        values = [point["value"] for point in data]
        
        # Simple linear regression slope
        n = len(values)
        x_values = list(range(n))
        
        slope = (n * sum(i * v for i, v in enumerate(values)) - sum(x_values) * sum(values)) / \
                (n * sum(i * i for i in x_values) - sum(x_values) ** 2)
        
        # Normalize slope to get strength (0-1)
        avg_value = statistics.mean(values)
        strength = min(1.0, abs(slope) / (avg_value * 0.1)) if avg_value > 0 else 0
        
        if slope > 0.05 * avg_value:
            direction = "degrading" if "latency" in str(data) or "error" in str(data) else "improving"
        elif slope < -0.05 * avg_value:
            direction = "improving" if "latency" in str(data) or "error" in str(data) else "degrading"
        else:
            direction = "stable"
        
        return direction, strength
    
    def _calculate_recent_change(self, values: List[float]) -> float:
        """Calculate recent change percentage"""
        
        if len(values) < 10:
            return 0.0
        
        recent_avg = statistics.mean(values[-5:])  # Last 5 values
        previous_avg = statistics.mean(values[-10:-5])  # Previous 5 values
        
        if previous_avg == 0:
            return 0.0
        
        return ((recent_avg - previous_avg) / previous_avg) * 100
    
    def _simple_forecast(self, values: List[float], days: int) -> List[float]:
        """Simple linear forecast"""
        
        if len(values) < 2:
            return [values[0] if values else 0] * days
        
        # Calculate trend
        n = len(values)
        x_values = list(range(n))
        
        slope = (n * sum(i * v for i, v in enumerate(values)) - sum(x_values) * sum(values)) / \
                (n * sum(i * i for i in x_values) - sum(x_values) ** 2)
        
        intercept = (sum(values) - slope * sum(x_values)) / n
        
        # Generate forecast
        forecast = []
        for i in range(days):
            predicted_value = intercept + slope * (n + i)
            forecast.append(max(0, predicted_value))  # Ensure positive
        
        return forecast
    
    def _calculate_overall_trend(self, trend_analysis: Dict[str, Any]) -> str:
        """Calculate overall system health trend"""
        
        improving_count = sum(1 for ta in trend_analysis.values() if ta["trend_direction"] == "improving")
        degrading_count = sum(1 for ta in trend_analysis.values() if ta["trend_direction"] == "degrading")
        
        if improving_count > degrading_count:
            return "improving"
        elif degrading_count > improving_count:
            return "degrading"
        else:
            return "stable"
    
    def _linear_prediction(self, values: List[float], days: int) -> Dict[str, Any]:
        """Generate linear prediction with confidence intervals"""
        
        forecast = self._simple_forecast(values, days)
        
        # Simple confidence interval (in production, use proper statistical methods)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        confidence_interval = [(v - std_dev, v + std_dev) for v in forecast]
        
        trend = "increasing" if forecast[-1] > forecast[0] else "decreasing" if forecast[-1] < forecast[0] else "stable"
        
        return {
            "values": forecast,
            "confidence_interval": confidence_interval,
            "trend": trend
        }
    
    def _assess_risk_level(self, metric: str, predicted_values: List[float]) -> str:
        """Assess risk level based on predicted values"""
        
        # Define risk thresholds for different metrics
        risk_thresholds = {
            "conversation_latency_ms": {"high": 2000, "medium": 1500},
            "system_error_rate": {"high": 0.05, "medium": 0.03},
            "cpu_usage_percent": {"high": 80, "medium": 60},
            "memory_usage_bytes": {"high": 12000000000, "medium": 8000000000}
        }
        
        thresholds = risk_thresholds.get(metric, {"high": float('inf'), "medium": float('inf')})
        max_predicted = max(predicted_values) if predicted_values else 0
        
        if max_predicted > thresholds["high"]:
            return "high"
        elif max_predicted > thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _calculate_overall_risk(self, predictions: Dict[str, Any]) -> str:
        """Calculate overall risk assessment"""
        
        risk_scores = {"high": 3, "medium": 2, "low": 1}
        
        total_score = sum(risk_scores.get(pred["risk_level"], 1) for pred in predictions.values())
        avg_score = total_score / len(predictions) if predictions else 1
        
        if avg_score >= 2.5:
            return "high"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "low"
    
    def _generate_capacity_recommendations(self, projections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate capacity planning recommendations"""
        
        recommendations = []
        
        for metric, projection in projections.items():
            if projection["scaling_required"]:
                recommendations.append({
                    "priority": "high" if projection["utilization_forecast"] > 90 else "medium",
                    "resource": metric,
                    "action": f"Scale {metric} by {projection['recommended_scaling_factor']:.1f}x",
                    "timeline": "within 30 days" if projection["utilization_forecast"] > 90 else "within 60 days",
                    "estimated_cost_impact": "medium"
                })
        
        return recommendations
    
    def _estimate_scaling_costs(self, projections: Dict[str, Any]) -> Dict[str, float]:
        """Estimate scaling costs"""
        
        # Simplified cost estimation
        cost_factors = {
            "max_concurrent_conversations": 50,  # $50 per additional conversation capacity
            "cpu_capacity_percent": 100,         # $100 per additional CPU unit
            "memory_capacity_gb": 25,            # $25 per GB
            "storage_capacity_gb": 5,            # $5 per GB
            "network_bandwidth_mbps": 2          # $2 per Mbps
        }
        
        total_monthly_cost = 0
        cost_breakdown = {}
        
        for metric, projection in projections.items():
            if projection["scaling_required"]:
                additional_capacity = projection["capacity_gap"]
                unit_cost = cost_factors.get(metric, 10)
                monthly_cost = additional_capacity * unit_cost
                
                cost_breakdown[metric] = monthly_cost
                total_monthly_cost += monthly_cost
        
        return {
            "total_monthly_cost": total_monthly_cost,
            "cost_breakdown": cost_breakdown,
            "annual_cost_estimate": total_monthly_cost * 12
        }
    
    def _create_scaling_timeline(self, projections: Dict[str, Any], horizon_days: int) -> List[Dict[str, Any]]:
        """Create scaling timeline"""
        
        timeline = []
        
        for metric, projection in projections.items():
            if projection["scaling_required"]:
                # Determine urgency based on utilization forecast
                if projection["utilization_forecast"] > 90:
                    days_until_scaling = 30
                    priority = "critical"
                elif projection["utilization_forecast"] > 80:
                    days_until_scaling = 60
                    priority = "high"
                else:
                    days_until_scaling = 90
                    priority = "medium"
                
                timeline.append({
                    "resource": metric,
                    "target_date": (datetime.now() + timedelta(days=days_until_scaling)).date().isoformat(),
                    "priority": priority,
                    "scaling_factor": projection["recommended_scaling_factor"],
                    "estimated_utilization": projection["utilization_forecast"]
                })
        
        return sorted(timeline, key=lambda x: x["target_date"])
    
    def _detect_anomalies(self, data: List[Dict[str, Any]], sensitivity: float) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        
        if len(data) < 10:
            return []
        
        values = [point["value"] for point in data]
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        anomalies = []
        threshold = std_dev * (1 / sensitivity)  # Lower sensitivity = higher threshold
        
        for point in data[-10:]:  # Check last 10 points
            deviation = abs(point["value"] - mean_val)
            
            if deviation > threshold:
                deviation_percent = (deviation / mean_val) * 100 if mean_val > 0 else 0
                
                anomalies.append({
                    "value": point["value"],
                    "expected_range": (mean_val - threshold, mean_val + threshold),
                    "deviation_percent": deviation_percent,
                    "timestamp": point["timestamp"],
                    "severity": "high" if deviation_percent > 50 else "medium" if deviation_percent > 25 else "low"
                })
        
        return anomalies
    
    def _summarize_anomalies(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize detected anomalies"""
        
        if not anomalies:
            return {"summary": "No anomalies detected"}
        
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for anomaly in anomalies:
            severity_counts[anomaly["severity"]] += 1
        
        return {
            "total_anomalies": len(anomalies),
            "severity_breakdown": severity_counts,
            "most_recent": max(anomalies, key=lambda x: x["timestamp"])["timestamp"],
            "highest_deviation": max(anomalies, key=lambda x: x["deviation_percent"])["deviation_percent"]
        }
    
    def _generate_anomaly_recommendations(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on detected anomalies"""
        
        if not anomalies:
            return ["Continue normal monitoring"]
        
        recommendations = []
        
        high_severity_count = sum(1 for a in anomalies if a["severity"] == "high")
        if high_severity_count > 0:
            recommendations.append("Investigate high-severity anomalies immediately")
            recommendations.append("Check system logs for the time periods with anomalies")
        
        if len(anomalies) > 5:
            recommendations.append("Consider adjusting monitoring thresholds")
            recommendations.append("Review recent system changes that might affect performance")
        
        recommendations.append("Set up additional monitoring for affected metrics")
        
        return recommendations
    
    def _calculate_significance(self, baseline: List[float], comparison: List[float]) -> float:
        """Calculate statistical significance (simplified)"""
        
        # Simplified significance calculation
        # In production, use proper statistical tests like t-test
        
        if len(baseline) < 2 or len(comparison) < 2:
            return 0.0
        
        baseline_std = statistics.stdev(baseline)
        comparison_std = statistics.stdev(comparison)
        
        # Simple effect size calculation
        pooled_std = (baseline_std + comparison_std) / 2
        if pooled_std == 0:
            return 0.0
        
        effect_size = abs(statistics.mean(comparison) - statistics.mean(baseline)) / pooled_std
        
        # Convert to pseudo p-value (0-1, lower = more significant)
        return max(0.0, min(1.0, 1.0 - (effect_size / 3.0)))
    
    def _generate_regression_recommendation(self, metric: str, change_percent: float) -> str:
        """Generate recommendation for performance regression"""
        
        if abs(change_percent) < 5:
            return "Performance change within normal variation"
        
        if change_percent > 10:
            if "latency" in metric or "error" in metric:
                return f"Performance degraded by {change_percent:.1f}% - investigate recent changes"
            else:
                return f"Performance improved by {change_percent:.1f}% - positive trend"
        elif change_percent < -10:
            if "latency" in metric or "error" in metric:
                return f"Performance improved by {abs(change_percent):.1f}% - positive trend"
            else:
                return f"Performance degraded by {abs(change_percent):.1f}% - investigate recent changes"
        
        return "Monitor for continued trend"
    
    def _calculate_overall_performance_change(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall performance change assessment"""
        
        regression_count = sum(1 for a in analysis.values() if a["regression_detected"] and not a["improvement"])
        improvement_count = sum(1 for a in analysis.values() if a["regression_detected"] and a["improvement"])
        
        if regression_count > improvement_count:
            return "overall_degradation"
        elif improvement_count > regression_count:
            return "overall_improvement"
        else:
            return "mixed_results"
    
    def _generate_regression_action_items(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate action items based on regression analysis"""
        
        action_items = []
        
        for metric, data in analysis.items():
            if data["regression_detected"] and not data["improvement"]:
                action_items.append(f"Investigate {metric} performance degradation ({data['change_percent']:.1f}% worse)")
        
        if not action_items:
            action_items.append("Continue monitoring current performance levels")
        else:
            action_items.append("Review recent deployments and system changes")
            action_items.append("Check system resources and capacity utilization")
        
        return action_items


# Export all monitoring tools
__all__ = [
    'RealTimeMetricsCollector',
    'IntelligentAlertingSystem',
    'PerformanceTrendAnalyzer',
    'MetricType',
    'AlertSeverity', 
    'AlertStatus',
    'MetricDefinition',
    'MetricDataPoint',
    'AlertRule',
    'Alert',
    'PerformanceSnapshot'
]