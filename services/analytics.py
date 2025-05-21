# services/analytics.py

"""
Analytics service for tracking system performance and metrics.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path

from agents.base_agent import AgentType
from core.state_manager import ConversationState

logger = logging.getLogger(__name__)

@dataclass
class ConversationMetrics:
    """Metrics for a single conversation."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    agent_type: AgentType
    total_turns: int
    response_times: List[float]
    state_transitions: List[Dict[str, Any]]
    handoff_occurred: bool
    completion_status: str
    customer_info: Dict[str, Any]

@dataclass
class ServiceMetrics:
    """Metrics for a service request."""
    request_id: str
    service_type: str
    creation_time: float
    completion_time: Optional[float]
    wait_time: float
    service_duration: Optional[float]
    customer_rating: Optional[int]
    service_successful: bool

class AnalyticsService:
    """
    Service for tracking and analyzing system performance.
    
    This service handles:
    1. Conversation analytics
    2. Service performance metrics
    3. System health monitoring
    4. Customer satisfaction tracking
    """
    
    def __init__(self, storage_dir: str = "./analytics"):
        """
        Initialize analytics service.
        
        Args:
            storage_dir: Directory for storing analytics data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.conversation_metrics: Dict[str, ConversationMetrics] = {}
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        
        # Performance tracking
        self.response_times: List[float] = []
        self.state_transitions: Dict[str, int] = {}
        self.handoff_counts: Dict[str, int] = {}
        
        # System health
        self.error_counts: Dict[str, int] = {}
        self.component_health: Dict[str, Dict[str, Any]] = {}
        
        # Initialize metric files
        self.metrics_file = self.storage_dir / "metrics.json"
        self.load_metrics()
        
        logger.info("Initialized analytics service")
    
    def load_metrics(self):
        """Load metrics from storage."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file) as f:
                    data = json.load(f)
                    
                    # Load conversation metrics
                    for session_id, metrics in data.get("conversations", {}).items():
                        self.conversation_metrics[session_id] = ConversationMetrics(**metrics)
                    
                    # Load service metrics
                    for request_id, metrics in data.get("services", {}).items():
                        self.service_metrics[request_id] = ServiceMetrics(**metrics)
                    
                    logger.info(f"Loaded metrics for {len(self.conversation_metrics)} conversations "
                              f"and {len(self.service_metrics)} services")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    async def save_metrics(self):
        """Save metrics to storage."""
        try:
            # Prepare data for serialization
            data = {
                "conversations": {
                    session_id: metrics.__dict__
                    for session_id, metrics in self.conversation_metrics.items()
                },
                "services": {
                    request_id: metrics.__dict__
                    for request_id, metrics in self.service_metrics.items()
                },
                "last_updated": time.time()
            }
            
            # Save to file
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Saved metrics to storage")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    async def track_conversation(
        self,
        session_id: str,
        agent_type: AgentType,
        response_time: float,
        state_transition: Optional[Dict[str, Any]] = None,
        customer_info: Optional[Dict[str, Any]] = None
    ):
        """
        Track conversation metrics.
        
        Args:
            session_id: Session identifier
            agent_type: Type of agent handling conversation
            response_time: Response generation time
            state_transition: Optional state transition info
            customer_info: Optional customer information
        """
        # Get or create metrics for session
        if session_id not in self.conversation_metrics:
            self.conversation_metrics[session_id] = ConversationMetrics(
                session_id=session_id,
                start_time=time.time(),
                end_time=None,
                agent_type=agent_type,
                total_turns=0,
                response_times=[],
                state_transitions=[],
                handoff_occurred=False,
                completion_status="in_progress",
                customer_info={}
            )
        
        metrics = self.conversation_metrics[session_id]
        
        # Update metrics
        metrics.total_turns += 1
        metrics.response_times.append(response_time)
        
        if state_transition:
            metrics.state_transitions.append(state_transition)
            
            # Track state transition counts
            from_state = state_transition["from_state"]
            to_state = state_transition["to_state"]
            transition_key = f"{from_state}->{to_state}"
            
            if transition_key not in self.state_transitions:
                self.state_transitions[transition_key] = 0
            self.state_transitions[transition_key] += 1
        
        if customer_info:
            metrics.customer_info.update(customer_info)
        
        # Track response time
        self.response_times.append(response_time)
        
        # Schedule periodic save
        asyncio.create_task(self.save_metrics())
    
    # services/analytics.py (continued)

    async def track_service(
        self,
        request_id: str,
        service_type: str,
        wait_time: float,
        service_duration: Optional[float] = None,
        customer_rating: Optional[int] = None,
        service_successful: bool = True
    ):
        """
        Track service metrics.
        
        Args:
            request_id: Service request identifier
            service_type: Type of service provided
            wait_time: Time customer waited for service
            service_duration: Optional duration of service
            customer_rating: Optional customer satisfaction rating (1-5)
            service_successful: Whether service was completed successfully
        """
        # Create service metrics
        metrics = ServiceMetrics(
            request_id=request_id,
            service_type=service_type,
            creation_time=time.time(),
            completion_time=time.time() + service_duration if service_duration else None,
            wait_time=wait_time,
            service_duration=service_duration,
            customer_rating=customer_rating,
            service_successful=service_successful
        )
        
        # Store metrics
        self.service_metrics[request_id] = metrics
        
        # Schedule save
        asyncio.create_task(self.save_metrics())
    
    async def track_error(
        self,
        component: str,
        error_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Track system errors.
        
        Args:
            component: Component where error occurred
            error_type: Type of error
            details: Optional error details
        """
        error_key = f"{component}:{error_type}"
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        self.error_counts[error_key] += 1
        
        # Update component health
        if component not in self.component_health:
            self.component_health[component] = {
                "errors": 0,
                "last_error": None,
                "status": "healthy"
            }
            
        self.component_health[component]["errors"] += 1
        self.component_health[component]["last_error"] = {
            "type": error_type,
            "timestamp": time.time(),
            "details": details
        }
        
        # Update status if too many errors
        if self.component_health[component]["errors"] > 5:
            self.component_health[component]["status"] = "degraded"
    
    def get_conversation_stats(
        self,
        time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Args:
            time_window: Optional time window in seconds
            
        Returns:
            Dictionary of conversation statistics
        """
        stats = {
            "total_conversations": len(self.conversation_metrics),
            "active_conversations": 0,
            "completed_conversations": 0,
            "handoff_rate": 0.0,
            "avg_turns": 0.0,
            "avg_response_time": 0.0,
            "completion_rate": 0.0
        }
        
        # Filter by time window if specified
        current_time = time.time()
        metrics = self.conversation_metrics.values()
        if time_window:
            cutoff_time = current_time - time_window
            metrics = [m for m in metrics if m.start_time >= cutoff_time]
        
        if not metrics:
            return stats
        
        # Calculate statistics
        total = len(metrics)
        completed = sum(1 for m in metrics if m.completion_status == "completed")
        handoffs = sum(1 for m in metrics if m.handoff_occurred)
        active = sum(1 for m in metrics if not m.end_time)
        total_turns = sum(m.total_turns for m in metrics)
        total_response_time = sum(sum(m.response_times) for m in metrics)
        
        stats.update({
            "total_conversations": total,
            "active_conversations": active,
            "completed_conversations": completed,
            "handoff_rate": handoffs / total if total > 0 else 0.0,
            "avg_turns": total_turns / total if total > 0 else 0.0,
            "avg_response_time": total_response_time / total_turns if total_turns > 0 else 0.0,
            "completion_rate": completed / total if total > 0 else 0.0
        })
        
        return stats
    
    def get_service_stats(
        self,
        time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Args:
            time_window: Optional time window in seconds
            
        Returns:
            Dictionary of service statistics
        """
        stats = {
            "total_services": len(self.service_metrics),
            "active_services": 0,
            "completed_services": 0,
            "avg_wait_time": 0.0,
            "avg_duration": 0.0,
            "success_rate": 0.0,
            "avg_rating": 0.0,
            "services_by_type": {}
        }
        
        # Filter by time window if specified
        current_time = time.time()
        metrics = self.service_metrics.values()
        if time_window:
            cutoff_time = current_time - time_window
            metrics = [m for m in metrics if m.creation_time >= cutoff_time]
        
        if not metrics:
            return stats
        
        # Calculate statistics
        total = len(metrics)
        completed = sum(1 for m in metrics if m.completion_time)
        active = sum(1 for m in metrics if not m.completion_time)
        successful = sum(1 for m in metrics if m.service_successful)
        total_wait = sum(m.wait_time for m in metrics)
        total_duration = sum(m.service_duration for m in metrics if m.service_duration)
        total_ratings = sum(m.customer_rating for m in metrics if m.customer_rating)
        rated_services = sum(1 for m in metrics if m.customer_rating)
        
        # Group by service type
        by_type = {}
        for metric in metrics:
            if metric.service_type not in by_type:
                by_type[metric.service_type] = 0
            by_type[metric.service_type] += 1
        
        stats.update({
            "total_services": total,
            "active_services": active,
            "completed_services": completed,
            "avg_wait_time": total_wait / total if total > 0 else 0.0,
            "avg_duration": total_duration / completed if completed > 0 else 0.0,
            "success_rate": successful / completed if completed > 0 else 0.0,
            "avg_rating": total_ratings / rated_services if rated_services > 0 else 0.0,
            "services_by_type": by_type
        })
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        return {
            "components": self.component_health,
            "error_counts": self.error_counts,
            "response_time_percentiles": {
                "p50": self._calculate_percentile(self.response_times, 50),
                "p90": self._calculate_percentile(self.response_times, 90),
                "p99": self._calculate_percentile(self.response_times, 99)
            },
            "state_transitions": self.state_transitions
        }
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[index]
    
    def get_customer_stats(self) -> Dict[str, Any]:
        """Get customer-related statistics."""
        stats = {
            "total_customers": 0,
            "returning_customers": 0,
            "avg_services_per_customer": 0.0,
            "customer_satisfaction": {
                "total_ratings": 0,
                "avg_rating": 0.0,
                "rating_distribution": {str(i): 0 for i in range(1, 6)}
            }
        }
        
        # Track unique customers and their services
        customer_services = {}
        total_ratings = 0
        rating_sum = 0
        
        for metrics in self.conversation_metrics.values():
            customer_id = metrics.customer_info.get("customer_id")
            if not customer_id:
                continue
                
            if customer_id not in customer_services:
                customer_services[customer_id] = 0
            customer_services[customer_id] += 1
        
        for metrics in self.service_metrics.values():
            if metrics.customer_rating:
                total_ratings += 1
                rating_sum += metrics.customer_rating
                rating_str = str(metrics.customer_rating)
                stats["customer_satisfaction"]["rating_distribution"][rating_str] += 1
        
        # Calculate statistics
        total_customers = len(customer_services)
        if total_customers > 0:
            stats.update({
                "total_customers": total_customers,
                "returning_customers": sum(1 for services in customer_services.values() if services > 1),
                "avg_services_per_customer": sum(customer_services.values()) / total_customers
            })
        
        if total_ratings > 0:
            stats["customer_satisfaction"].update({
                "total_ratings": total_ratings,
                "avg_rating": rating_sum / total_ratings
            })
        
        return stats