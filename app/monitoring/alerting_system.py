"""
Alerting System for Multi-Agent Voice AI
Comprehensive monitoring, alerting, and incident management system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time
from collections import defaultdict, deque

import requests
import redis
import structlog
from jinja2 import Template

# Configure structured logging
logger = structlog.get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    WEBHOOK = "webhook"
    TEAMS = "teams"

@dataclass
class AlertCondition:
    """Alert condition definition"""
    name: str
    description: str
    metric_path: str  # e.g., "performance.latency.total"
    operator: str     # >, <, >=, <=, ==, !=
    threshold: float
    duration_minutes: int = 5  # How long condition must persist
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    
    # Grouping and filtering
    agent_filter: Optional[str] = None  # Specific agent or "all"
    component_filter: Optional[str] = None
    
    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_delay_minutes: int = 30
    auto_resolve: bool = True
    
    # Rate limiting
    cooldown_minutes: int = 15  # Minimum time between alerts
    max_alerts_per_hour: int = 4

@dataclass
class Alert:
    """Individual alert instance"""
    id: str
    condition_name: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    
    # Context information
    metric_value: float
    threshold: float
    agent_id: Optional[str] = None
    component: Optional[str] = None
    
    # Lifecycle tracking
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_minutes(self) -> float:
        """Get alert duration in minutes"""
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.created_at).total_seconds() / 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "condition_name": self.condition_name,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "agent_id": self.agent_id,
            "component": self.component,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "duration_minutes": self.duration_minutes,
            "metadata": self.metadata
        }

class NotificationProvider:
    """Base class for notification providers"""
    
    async def send_notification(self, alert: Alert, escalation_level: int = 1) -> bool:
        """Send notification for an alert"""
        raise NotImplementedError

class EmailNotificationProvider(NotificationProvider):
    """Email notification provider"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        
        # Email templates
        self.alert_template = Template("""
        <html>
        <body>
            <h2 style="color: {{ color }};">{{ severity.upper() }} ALERT: {{ title }}</h2>
            
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Alert ID:</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ alert_id }}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Severity:</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ severity }}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Agent:</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ agent_id or 'All Agents' }}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Component:</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ component or 'System-wide' }}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Current Value:</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ metric_value }}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Threshold:</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ threshold }}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Time:</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ created_at }}</td>
                </tr>
            </table>
            
            <p><strong>Description:</strong> {{ description }}</p>
            
            <p style="margin-top: 20px;">
                <a href="{{ dashboard_url }}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
                    View Dashboard
                </a>
            </p>
        </body>
        </html>
        """)
    
    async def send_notification(self, alert: Alert, escalation_level: int = 1) -> bool:
        """Send email notification"""
        try:
            # Determine email addresses based on escalation level
            recipients = self._get_recipients_by_escalation(alert.severity, escalation_level)
            
            if not recipients:
                return False
            
            # Choose color based on severity
            color_map = {
                AlertSeverity.INFO: "#2196F3",
                AlertSeverity.WARNING: "#FF9800", 
                AlertSeverity.CRITICAL: "#F44336",
                AlertSeverity.EMERGENCY: "#9C27B0"
            }
            
            # Render email content
            html_content = self.alert_template.render(
                alert_id=alert.id,
                title=alert.title,
                description=alert.description,
                severity=alert.severity.value,
                color=color_map.get(alert.severity, "#666666"),
                agent_id=alert.agent_id,
                component=alert.component,
                metric_value=alert.metric_value,
                threshold=alert.threshold,
                created_at=alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                dashboard_url="https://dashboard.voiceai.com/alerts"
            )
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.username
            msg['To'] = ", ".join(recipients)
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(
                "Email alert sent successfully",
                alert_id=alert.id,
                recipients=recipients,
                escalation_level=escalation_level
            )
            return True
            
        except Exception as e:
            logger.error(
                "Failed to send email alert",
                alert_id=alert.id,
                error=str(e)
            )
            return False
    
    def _get_recipients_by_escalation(self, severity: AlertSeverity, level: int) -> List[str]:
        """Get email recipients based on severity and escalation level"""
        # This would typically come from configuration
        escalation_matrix = {
            AlertSeverity.INFO: {
                1: ["devops@company.com"],
                2: ["devops@company.com", "team-lead@company.com"]
            },
            AlertSeverity.WARNING: {
                1: ["devops@company.com", "engineering@company.com"],
                2: ["devops@company.com", "engineering@company.com", "manager@company.com"]
            },
            AlertSeverity.CRITICAL: {
                1: ["devops@company.com", "engineering@company.com", "oncall@company.com"],
                2: ["devops@company.com", "engineering@company.com", "oncall@company.com", "manager@company.com"]
            },
            AlertSeverity.EMERGENCY: {
                1: ["devops@company.com", "engineering@company.com", "oncall@company.com", "manager@company.com"],
                2: ["devops@company.com", "engineering@company.com", "oncall@company.com", "manager@company.com", "director@company.com"]
            }
        }
        
        return escalation_matrix.get(severity, {}).get(level, [])

class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_notification(self, alert: Alert, escalation_level: int = 1) -> bool:
        """Send Slack notification"""
        try:
            # Color coding for Slack
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500",
                AlertSeverity.CRITICAL: "#ff0000", 
                AlertSeverity.EMERGENCY: "#800080"
            }
            
            # Create Slack message
            slack_message = {
                "text": f":rotating_light: {alert.severity.value.upper()} Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#666666"),
                        "fields": [
                            {
                                "title": "Alert ID",
                                "value": alert.id,
                                "short": True
                            },
                            {
                                "title": "Severity", 
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Agent",
                                "value": alert.agent_id or "All Agents",
                                "short": True
                            },
                            {
                                "title": "Component",
                                "value": alert.component or "System-wide",
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": str(alert.metric_value),
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": str(alert.threshold),
                                "short": True
                            }
                        ],
                        "text": alert.description,
                        "footer": "Voice AI Monitoring",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            # Add escalation indicator
            if escalation_level > 1:
                slack_message["text"] += f" (Escalation Level {escalation_level})"
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=slack_message,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(
                "Slack alert sent successfully",
                alert_id=alert.id,
                escalation_level=escalation_level
            )
            return True
            
        except Exception as e:
            logger.error(
                "Failed to send Slack alert",
                alert_id=alert.id,
                error=str(e)
            )
            return False

class PagerDutyNotificationProvider(NotificationProvider):
    """PagerDuty notification provider"""
    
    def __init__(self, integration_key: str):
        self.integration_key = integration_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
    
    async def send_notification(self, alert: Alert, escalation_level: int = 1) -> bool:
        """Send PagerDuty notification"""
        try:
            # Map severity to PagerDuty severity
            severity_map = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "error",
                AlertSeverity.EMERGENCY: "critical"
            }
            
            # Create PagerDuty event
            event_data = {
                "routing_key": self.integration_key,
                "event_action": "trigger",
                "dedup_key": f"voice-ai-{alert.condition_name}-{alert.agent_id or 'global'}",
                "payload": {
                    "summary": alert.title,
                    "source": "voice-ai-monitoring",
                    "severity": severity_map.get(alert.severity, "warning"),
                    "component": alert.component or "system",
                    "group": alert.agent_id or "global",
                    "class": "performance",
                    "custom_details": {
                        "alert_id": alert.id,
                        "condition": alert.condition_name,
                        "current_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "agent_id": alert.agent_id,
                        "escalation_level": escalation_level,
                        "dashboard_url": "https://dashboard.voiceai.com/alerts"
                    }
                }
            }
            
            # Send to PagerDuty
            response = requests.post(
                self.api_url,
                json=event_data,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(
                "PagerDuty alert sent successfully",
                alert_id=alert.id,
                escalation_level=escalation_level
            )
            return True
            
        except Exception as e:
            logger.error(
                "Failed to send PagerDuty alert",
                alert_id=alert.id,
                error=str(e)
            )
            return False

class AlertEvaluator:
    """Evaluates metrics against alert conditions"""
    
    def __init__(self):
        self.condition_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def evaluate_condition(self, 
                         condition: AlertCondition, 
                         current_metrics: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate a condition against current metrics"""
        try:
            # Extract metric value
            metric_value = self._get_metric_value(current_metrics, condition.metric_path)
            if metric_value is None:
                return None
            
            # Store metric in history
            metric_key = f"{condition.name}:{condition.agent_filter or 'all'}"
            self.metric_history[metric_key].append({
                "timestamp": datetime.utcnow(),
                "value": metric_value
            })
            
            # Check if condition is met
            condition_met = self._evaluate_threshold(
                metric_value, condition.operator, condition.threshold
            )
            
            # Get condition state
            state_key = f"{condition.name}:{condition.agent_filter or 'all'}"
            state = self.condition_states[state_key]
            
            if condition_met:
                # Condition is currently met
                if "first_breach" not in state:
                    state["first_breach"] = datetime.utcnow()
                    state["breach_count"] = 1
                else:
                    state["breach_count"] = state.get("breach_count", 0) + 1
                
                # Check if duration threshold is met
                duration_met = (
                    datetime.utcnow() - state["first_breach"]
                ).total_seconds() >= (condition.duration_minutes * 60)
                
                # Check if we should fire alert (considering cooldown)
                if duration_met and self._should_fire_alert(condition, state):
                    alert = self._create_alert(condition, metric_value)
                    state["last_alert"] = datetime.utcnow()
                    state["alert_count"] = state.get("alert_count", 0) + 1
                    return alert
            
            else:
                # Condition is not met - reset state
                if "first_breach" in state:
                    state.clear()
            
            return None
            
        except Exception as e:
            logger.error(
                "Failed to evaluate alert condition",
                condition_name=condition.name,
                error=str(e)
            )
            return None
    
    def _get_metric_value(self, metrics: Dict[str, Any], path: str) -> Optional[float]:
        """Extract metric value using dot notation path"""
        try:
            current = metrics
            for part in path.split('.'):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            
            return float(current) if current is not None else None
            
        except (ValueError, TypeError):
            return None
    
    def _evaluate_threshold(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate threshold condition"""
        operators = {
            '>': lambda v, t: v > t,
            '<': lambda v, t: v < t, 
            '>=': lambda v, t: v >= t,
            '<=': lambda v, t: v <= t,
            '==': lambda v, t: v == t,
            '!=': lambda v, t: v != t
        }
        
        op_func = operators.get(operator)
        return op_func(value, threshold) if op_func else False
    
    def _should_fire_alert(self, condition: AlertCondition, state: Dict[str, Any]) -> bool:
        """Check if alert should be fired based on rate limiting"""
        now = datetime.utcnow()
        
        # Check cooldown period
        last_alert = state.get("last_alert")
        if last_alert:
            cooldown_elapsed = (now - last_alert).total_seconds() / 60
            if cooldown_elapsed < condition.cooldown_minutes:
                return False
        
        # Check hourly rate limit
        alert_count = state.get("alert_count", 0)
        first_alert_hour = state.get("first_alert_hour")
        
        if first_alert_hour is None or (now - first_alert_hour).total_seconds() >= 3600:
            # Reset hourly counter
            state["first_alert_hour"] = now
            state["alert_count"] = 0
            return True
        
        return alert_count < condition.max_alerts_per_hour
    
    def _create_alert(self, condition: AlertCondition, metric_value: float) -> Alert:
        """Create alert from condition and current metric value"""
        alert_id = f"alert-{int(datetime.utcnow().timestamp() * 1000)}"
        
        return Alert(
            id=alert_id,
            condition_name=condition.name,
            title=f"{condition.name}: {condition.description}",
            description=f"Metric {condition.metric_path} is {metric_value} (threshold: {condition.threshold})",
            severity=condition.severity,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metric_value=metric_value,
            threshold=condition.threshold,
            agent_id=condition.agent_filter,
            component=condition.component_filter,
            metadata={
                "condition_name": condition.name,
                "metric_path": condition.metric_path,
                "operator": condition.operator
            }
        )

class AlertingSystem:
    """Main alerting system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.conditions: Dict[str, AlertCondition] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.evaluator = AlertEvaluator()
        
        # Background processing
        self._running = False
        self._evaluation_thread: Optional[threading.Thread] = None
        self._escalation_thread: Optional[threading.Thread] = None
        
        # Default alert conditions
        self._initialize_default_conditions()
    
    def add_notification_provider(self, 
                                channel: NotificationChannel, 
                                provider: NotificationProvider):
        """Add notification provider"""
        self.notification_providers[channel] = provider
        logger.info(f"Added notification provider for {channel.value}")
    
    def add_alert_condition(self, condition: AlertCondition):
        """Add alert condition"""
        self.conditions[condition.name] = condition
        logger.info(
            "Added alert condition",
            name=condition.name,
            severity=condition.severity.value,
            threshold=condition.threshold
        )
    
    def remove_alert_condition(self, condition_name: str):
        """Remove alert condition"""
        if condition_name in self.conditions:
            del self.conditions[condition_name]
            logger.info("Removed alert condition", name=condition_name)
    
    def start_monitoring(self):
        """Start background monitoring threads"""
        if self._running:
            return
        
        self._running = True
        
        # Start evaluation thread
        self._evaluation_thread = threading.Thread(
            target=self._evaluation_loop,
            daemon=True
        )
        self._evaluation_thread.start()
        
        # Start escalation thread
        self._escalation_thread = threading.Thread(
            target=self._escalation_loop,
            daemon=True
        )
        self._escalation_thread.start()
        
        logger.info("Alerting system started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._running = False
        
        if self._evaluation_thread:
            self._evaluation_thread.join(timeout=5.0)
        
        if self._escalation_thread:
            self._escalation_thread.join(timeout=5.0)
        
        logger.info("Alerting system stopped")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.utcnow()
            
            # Store in Redis
            if self.redis_client:
                await self._store_alert_redis(alert)
            
            logger.info(
                "Alert acknowledged",
                alert_id=alert_id,
                acknowledged_by=acknowledged_by
            )
    
    async def resolve_alert(self, alert_id: str, resolved_by: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            alert.updated_at = datetime.utcnow()
            
            # Store in Redis
            if self.redis_client:
                await self._store_alert_redis(alert)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(
                "Alert resolved",
                alert_id=alert_id,
                resolved_by=resolved_by,
                duration_minutes=alert.duration_minutes
            )
    
    def get_active_alerts(self, 
                         severity: Optional[AlertSeverity] = None,
                         agent_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        active_alerts = list(self.active_alerts.values())
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_active_alerts": len(active_alerts),
            "severity_breakdown": dict(severity_counts),
            "total_conditions": len(self.conditions),
            "enabled_conditions": sum(1 for c in self.conditions.values() if c.enabled),
            "notification_providers": list(self.notification_providers.keys())
        }
    
    def _evaluation_loop(self):
        """Background thread for evaluating alert conditions"""
        while self._running:
            try:
                # Get current metrics (this would integrate with performance tracker)
                current_metrics = self._get_current_metrics()
                
                if current_metrics:
                    # Evaluate each enabled condition
                    for condition in self.conditions.values():
                        if not condition.enabled:
                            continue
                        
                        alert = self.evaluator.evaluate_condition(condition, current_metrics)
                        if alert:
                            asyncio.create_task(self._handle_new_alert(alert))
                
                time.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error("Error in alert evaluation loop", error=str(e))
                time.sleep(60)  # Wait longer on error
    
    def _escalation_loop(self):
        """Background thread for handling alert escalations"""
        while self._running:
            try:
                now = datetime.utcnow()
                
                for alert in list(self.active_alerts.values()):
                    if alert.status == AlertStatus.ACTIVE:
                        # Check if escalation is needed
                        time_since_created = (now - alert.created_at).total_seconds() / 60
                        condition = self.conditions.get(alert.condition_name)
                        
                        if condition and time_since_created >= condition.escalation_delay_minutes:
                            asyncio.create_task(self._escalate_alert(alert))
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("Error in escalation loop", error=str(e))
                time.sleep(120)  # Wait longer on error
    
    async def _handle_new_alert(self, alert: Alert):
        """Handle a new alert"""
        try:
            # Store alert
            self.active_alerts[alert.id] = alert
            
            # Store in Redis
            if self.redis_client:
                await self._store_alert_redis(alert)
            
            # Send notifications
            condition = self.conditions.get(alert.condition_name)
            if condition:
                await self._send_notifications(alert, condition.notification_channels)
            
            logger.warning(
                "New alert fired",
                alert_id=alert.id,
                condition=alert.condition_name,
                severity=alert.severity.value,
                metric_value=alert.metric_value,
                threshold=alert.threshold
            )
            
        except Exception as e:
            logger.error("Failed to handle new alert", alert_id=alert.id, error=str(e))
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate an alert to next level"""
        try:
            condition = self.conditions.get(alert.condition_name)
            if not condition:
                return
            
            # Send escalated notifications
            await self._send_notifications(alert, condition.notification_channels, escalation_level=2)
            
            logger.warning(
                "Alert escalated",
                alert_id=alert.id,
                condition=alert.condition_name,
                duration_minutes=alert.duration_minutes
            )
            
        except Exception as e:
            logger.error("Failed to escalate alert", alert_id=alert.id, error=str(e))
    
    async def _send_notifications(self, 
                                alert: Alert, 
                                channels: List[NotificationChannel],
                                escalation_level: int = 1):
        """Send notifications through specified channels"""
        for channel in channels:
            provider = self.notification_providers.get(channel)
            if provider:
                try:
                    success = await provider.send_notification(alert, escalation_level)
                    if success:
                        logger.info(
                            "Notification sent",
                            alert_id=alert.id,
                            channel=channel.value,
                            escalation_level=escalation_level
                        )
                except Exception as e:
                    logger.error(
                        "Failed to send notification",
                        alert_id=alert.id,
                        channel=channel.value,
                        error=str(e)
                    )
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for evaluation"""
        # This would integrate with the performance tracker
        # For now, return mock data structure
        return {
            "performance": {
                "latency": {
                    "total": 850,  # ms
                    "stt": 120,
                    "routing": 15,
                    "vector": 8,
                    "llm": 280,
                    "tts": 150,
                    "network": 45
                },
                "error_rate": {
                    "total": 0.02,  # 2%
                    "stt": 0.001,
                    "routing": 0.005,
                    "vector": 0.001,
                    "llm": 0.01,
                    "tts": 0.002,
                    "network": 0.001
                }
            },
            "system": {
                "cpu_usage": 75.5,  # %
                "memory_usage": 82.3,  # %
                "disk_usage": 45.2  # %
            },
            "business": {
                "active_sessions": 125,
                "revenue_per_hour": 2450.75,
                "customer_satisfaction": 4.2
            }
        }
    
    async def _store_alert_redis(self, alert: Alert):
        """Store alert in Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"alerts:active:{alert.id}"
            value = json.dumps(alert.to_dict())
            
            # Store with TTL of 24 hours
            await self.redis_client.setex(key, 86400, value)
            
            # Add to alerts list
            await self.redis_client.zadd(
                "alerts:timeline",
                {alert.id: int(alert.created_at.timestamp())}
            )
            
        except Exception as e:
            logger.error("Failed to store alert in Redis", error=str(e))
    
    def _initialize_default_conditions(self):
        """Initialize default alert conditions"""
        default_conditions = [
            # Performance alerts
            AlertCondition(
                name="high_total_latency",
                description="Total response latency is too high",
                metric_path="performance.latency.total",
                operator=">",
                threshold=1000.0,  # 1 second
                duration_minutes=2,
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
            ),
            
            AlertCondition(
                name="critical_total_latency", 
                description="Total response latency is critically high",
                metric_path="performance.latency.total",
                operator=">",
                threshold=2000.0,  # 2 seconds
                duration_minutes=1,
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY]
            ),
            
            # Error rate alerts
            AlertCondition(
                name="high_error_rate",
                description="System error rate is elevated",
                metric_path="performance.error_rate.total",
                operator=">",
                threshold=0.05,  # 5%
                duration_minutes=3,
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.SLACK]
            ),
            
            # System resource alerts
            AlertCondition(
                name="high_cpu_usage",
                description="CPU usage is very high",
                metric_path="system.cpu_usage",
                operator=">",
                threshold=90.0,  # 90%
                duration_minutes=5,
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
            ),
            
            AlertCondition(
                name="high_memory_usage",
                description="Memory usage is critically high",
                metric_path="system.memory_usage",
                operator=">",
                threshold=95.0,  # 95%
                duration_minutes=2,
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY]
            ),
            
            # Business alerts
            AlertCondition(
                name="low_customer_satisfaction",
                description="Customer satisfaction score is below acceptable threshold",
                metric_path="business.customer_satisfaction",
                operator="<",
                threshold=3.5,  # Below 3.5/5
                duration_minutes=10,
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            
            AlertCondition(
                name="high_session_load",
                description="Active session count is approaching capacity",
                metric_path="business.active_sessions",
                operator=">",
                threshold=900,  # 90% of 1000 capacity
                duration_minutes=5,
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
            )
        ]
        
        for condition in default_conditions:
            self.add_alert_condition(condition)

class IncidentManager:
    """Manages incidents and their lifecycle"""
    
    def __init__(self, alerting_system: AlertingSystem):
        self.alerting_system = alerting_system
        self.incidents: Dict[str, Dict[str, Any]] = {}
    
    async def create_incident(self, 
                            title: str, 
                            description: str, 
                            severity: AlertSeverity,
                            alerts: List[str] = None) -> str:
        """Create a new incident"""
        incident_id = f"inc-{int(datetime.utcnow().timestamp() * 1000)}"
        
        incident = {
            "id": incident_id,
            "title": title,
            "description": description,
            "severity": severity.value,
            "status": "open",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "alerts": alerts or [],
            "timeline": [
                {
                    "timestamp": datetime.utcnow(),
                    "event": "incident_created",
                    "description": f"Incident created: {title}"
                }
            ],
            "assignee": None,
            "resolution": None
        }
        
        self.incidents[incident_id] = incident
        
        logger.info(
            "Incident created",
            incident_id=incident_id,
            title=title,
            severity=severity.value
        )
        
        return incident_id
    
    async def update_incident(self, 
                            incident_id: str, 
                            status: str = None,
                            assignee: str = None,
                            notes: str = None):
        """Update an incident"""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        incident["updated_at"] = datetime.utcnow()
        
        if status:
            incident["status"] = status
            incident["timeline"].append({
                "timestamp": datetime.utcnow(),
                "event": "status_changed",
                "description": f"Status changed to {status}"
            })
        
        if assignee:
            incident["assignee"] = assignee
            incident["timeline"].append({
                "timestamp": datetime.utcnow(),
                "event": "assigned",
                "description": f"Assigned to {assignee}"
            })
        
        if notes:
            incident["timeline"].append({
                "timestamp": datetime.utcnow(),
                "event": "notes_added",
                "description": notes
            })
        
        logger.info(
            "Incident updated",
            incident_id=incident_id,
            status=status,
            assignee=assignee
        )
        
        return True
    
    async def resolve_incident(self, 
                             incident_id: str, 
                             resolution: str,
                             resolved_by: str):
        """Resolve an incident"""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        incident["status"] = "resolved"
        incident["resolution"] = resolution
        incident["resolved_at"] = datetime.utcnow()
        incident["resolved_by"] = resolved_by
        incident["updated_at"] = datetime.utcnow()
        
        incident["timeline"].append({
            "timestamp": datetime.utcnow(),
            "event": "resolved",
            "description": f"Resolved by {resolved_by}: {resolution}"
        })
        
        # Auto-resolve associated alerts
        for alert_id in incident["alerts"]:
            await self.alerting_system.resolve_alert(alert_id, f"incident-{incident_id}")
        
        logger.info(
            "Incident resolved",
            incident_id=incident_id,
            resolved_by=resolved_by,
            resolution=resolution
        )
        
        return True
    
    def get_open_incidents(self) -> List[Dict[str, Any]]:
        """Get all open incidents"""
        return [
            incident for incident in self.incidents.values()
            if incident["status"] != "resolved"
        ]
    
    def get_incident_statistics(self) -> Dict[str, Any]:
        """Get incident statistics"""
        incidents = list(self.incidents.values())
        
        total_incidents = len(incidents)
        open_incidents = len([i for i in incidents if i["status"] != "resolved"])
        
        severity_counts = defaultdict(int)
        for incident in incidents:
            severity_counts[incident["severity"]] += 1
        
        # Calculate MTTR (Mean Time To Resolution)
        resolved_incidents = [
            i for i in incidents 
            if i["status"] == "resolved" and "resolved_at" in i
        ]
        
        if resolved_incidents:
            resolution_times = [
                (i["resolved_at"] - i["created_at"]).total_seconds() / 60
                for i in resolved_incidents
            ]
            mttr_minutes = sum(resolution_times) / len(resolution_times)
        else:
            mttr_minutes = 0
        
        return {
            "total_incidents": total_incidents,
            "open_incidents": open_incidents,
            "resolved_incidents": len(resolved_incidents),
            "severity_breakdown": dict(severity_counts),
            "mttr_minutes": round(mttr_minutes, 2)
        }

# Global alerting system instance
alerting_system: Optional[AlertingSystem] = None

def initialize_alerting_system(redis_client: Optional[redis.Redis] = None,
                             email_config: Optional[Dict[str, str]] = None,
                             slack_webhook: Optional[str] = None,
                             pagerduty_key: Optional[str] = None):
    """Initialize the global alerting system"""
    global alerting_system
    
    if alerting_system is None:
        alerting_system = AlertingSystem(redis_client)
        
        # Add notification providers
        if email_config:
            email_provider = EmailNotificationProvider(
                smtp_host=email_config["smtp_host"],
                smtp_port=int(email_config["smtp_port"]),
                username=email_config["username"],
                password=email_config["password"]
            )
            alerting_system.add_notification_provider(NotificationChannel.EMAIL, email_provider)
        
        if slack_webhook:
            slack_provider = SlackNotificationProvider(slack_webhook)
            alerting_system.add_notification_provider(NotificationChannel.SLACK, slack_provider)
        
        if pagerduty_key:
            pagerduty_provider = PagerDutyNotificationProvider(pagerduty_key)
            alerting_system.add_notification_provider(NotificationChannel.PAGERDUTY, pagerduty_provider)
        
        # Start monitoring
        alerting_system.start_monitoring()
        
        logger.info("Alerting system initialized and started")
    
    return alerting_system

def get_alerting_system() -> AlertingSystem:
    """Get the global alerting system instance"""
    if alerting_system is None:
        raise RuntimeError("Alerting system not initialized. Call initialize_alerting_system() first.")
    return alerting_system

# Integration functions for easy use
async def fire_custom_alert(title: str, 
                          description: str, 
                          severity: AlertSeverity,
                          metric_value: float = 0.0,
                          threshold: float = 0.0,
                          agent_id: Optional[str] = None,
                          component: Optional[str] = None):
    """Fire a custom alert manually"""
    system = get_alerting_system()
    
    alert_id = f"custom-{int(datetime.utcnow().timestamp() * 1000)}"
    
    alert = Alert(
        id=alert_id,
        condition_name="custom_alert",
        title=title,
        description=description,
        severity=severity,
        status=AlertStatus.ACTIVE,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        metric_value=metric_value,
        threshold=threshold,
        agent_id=agent_id,
        component=component,
        metadata={"custom": True}
    )
    
    await system._handle_new_alert(alert)
    return alert_id

def create_custom_condition(name: str,
                          description: str,
                          metric_path: str,
                          operator: str,
                          threshold: float,
                          severity: AlertSeverity = AlertSeverity.WARNING,
                          duration_minutes: int = 5,
                          channels: List[NotificationChannel] = None) -> AlertCondition:
    """Create a custom alert condition"""
    if channels is None:
        channels = [NotificationChannel.SLACK]
    
    condition = AlertCondition(
        name=name,
        description=description,
        metric_path=metric_path,
        operator=operator,
        threshold=threshold,
        severity=severity,
        duration_minutes=duration_minutes,
        notification_channels=channels
    )
    
    system = get_alerting_system()
    system.add_alert_condition(condition)
    
    return condition

# Health check for the alerting system
def alerting_system_health_check() -> Dict[str, Any]:
    """Check the health of the alerting system"""
    try:
        system = get_alerting_system()
        stats = system.get_alert_statistics()
        
        return {
            "status": "healthy",
            "monitoring_active": system._running,
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Example usage and testing functions
async def test_alerting_system():
    """Test the alerting system with sample alerts"""
    try:
        # Test custom alert
        await fire_custom_alert(
            title="Test Alert",
            description="This is a test alert to verify the system is working",
            severity=AlertSeverity.INFO,
            metric_value=100.0,
            threshold=50.0,
            component="test"
        )
        
        logger.info("Test alert fired successfully")
        
        # Test high latency alert
        await fire_custom_alert(
            title="High Latency Detected",
            description="System latency exceeded acceptable thresholds",
            severity=AlertSeverity.WARNING,
            metric_value=1200.0,
            threshold=1000.0,
            agent_id="roadside-assistance",
            component="total_latency"
        )
        
        logger.info("High latency test alert fired successfully")
        
    except Exception as e:
        logger.error("Failed to test alerting system", error=str(e))
