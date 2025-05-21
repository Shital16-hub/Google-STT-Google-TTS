# services/dispatcher.py

"""
Dispatcher service for managing human handoffs and service coordination.
"""
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import asyncio

from agents.base_agent import AgentType
from core.state_manager import ConversationState

logger = logging.getLogger(__name__)

class ServicePriority(str, Enum):
    """Service priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ServiceRequest:
    """Service request details."""
    request_id: str
    session_id: str
    agent_type: AgentType
    customer_info: Dict[str, Any]
    location_info: Dict[str, Any]
    service_requirements: Dict[str, Any]
    priority: ServicePriority
    timestamp: float
    estimated_duration: int  # minutes
    handoff_reason: str

class DispatcherService:
    """
    Manages service requests and coordinates with human dispatchers.
    
    This service handles:
    1. Queuing and prioritizing service requests
    2. Coordinating with human dispatchers
    3. Tracking service status
    4. Managing customer callbacks
    """
    
    def __init__(self):
        """Initialize the dispatcher service."""
        # Active service requests
        self.active_requests: Dict[str, ServiceRequest] = {}
        
        # Queue by priority
        self.request_queues: Dict[ServicePriority, List[str]] = {
            ServicePriority.HIGH: [],
            ServicePriority.MEDIUM: [],
            ServicePriority.LOW: []
        }
        
        # Track dispatcher availability
        self.available_dispatchers: int = 0
        self.dispatcher_assignments: Dict[str, str] = {}  # dispatcher_id -> request_id
        
        # Service status tracking
        self.service_status: Dict[str, Dict[str, Any]] = {}
        
        # Customer callbacks
        self.pending_callbacks: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized dispatcher service")
    
    def determine_priority(
        self,
        agent_type: AgentType,
        service_requirements: Dict[str, Any]
    ) -> ServicePriority:
        """
        Determine request priority based on type and requirements.
        
        Args:
            agent_type: Type of service needed
            service_requirements: Service requirements
            
        Returns:
            Service priority level
        """
        # High priority conditions
        if (
            "accident" in service_requirements.get("vehicle_condition", "").lower() or
            service_requirements.get("safety_concern", False) or
            service_requirements.get("children_present", False)
        ):
            return ServicePriority.HIGH
        
        # Medium priority conditions
        if (
            agent_type == AgentType.TOWING and
            service_requirements.get("vehicle_condition") == "not_driveable"
        ) or (
            service_requirements.get("weather_concern", False)
        ):
            return ServicePriority.MEDIUM
        
        # Default to low priority
        return ServicePriority.LOW
    
    async def create_service_request(
        self,
        session_id: str,
        agent_type: AgentType,
        customer_info: Dict[str, Any],
        service_requirements: Dict[str, Any],
        handoff_reason: str
    ) -> str:
        """
        Create a new service request.
        
        Args:
            session_id: Session identifier
            agent_type: Type of service needed
            customer_info: Customer information
            service_requirements: Service requirements
            handoff_reason: Reason for handoff
            
        Returns:
            Request ID
        """
        # Generate request ID
        import uuid
        request_id = str(uuid.uuid4())
        
        # Determine priority
        priority = self.determine_priority(agent_type, service_requirements)
        
        # Create request
        request = ServiceRequest(
            request_id=request_id,
            session_id=session_id,
            agent_type=agent_type,
            customer_info=customer_info,
            location_info=self._extract_location_info(customer_info),
            service_requirements=service_requirements,
            priority=priority,
            timestamp=time.time(),
            estimated_duration=self._estimate_service_duration(agent_type, service_requirements),
            handoff_reason=handoff_reason
        )
        
        # Add to active requests
        self.active_requests[request_id] = request
        
        # Add to priority queue
        self.request_queues[priority].append(request_id)
        
        # Initialize service status
        self.service_status[request_id] = {
            "status": "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
            "assigned_dispatcher": None,
            "estimated_arrival": None
        }
        
        logger.info(f"Created {priority.value} priority request {request_id} for {agent_type.value} service")
        
        # Try to assign to dispatcher
        await self._try_assign_dispatcher(request_id)
        
        return request_id
    
    def _extract_location_info(self, customer_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure location information."""
        return {
            "address": customer_info.get("location", ""),
            "landmarks": customer_info.get("landmarks", ""),
            "coordinates": customer_info.get("coordinates"),
            "accessibility": customer_info.get("location_accessibility", "unknown")
        }
    
    def _estimate_service_duration(
        self,
        agent_type: AgentType,
        requirements: Dict[str, Any]
    ) -> int:
        """Estimate service duration in minutes."""
        base_duration = {
            AgentType.TOWING: 45,
            AgentType.TIRE: 30,
            AgentType.JUMP_START: 20
        }.get(agent_type, 30)
        
        # Add time for complications
        if requirements.get("difficult_access", False):
            base_duration += 15
        if requirements.get("needs_winch", False):
            base_duration += 20
        if requirements.get("special_equipment", False):
            base_duration += 15
        
        return base_duration
    
    async def _try_assign_dispatcher(self, request_id: str):
        """Try to assign a dispatcher to the request."""
        if self.available_dispatchers > 0:
            # In a real implementation, this would involve dispatcher selection logic
            dispatcher_id = "DISP001"  # Example
            
            # Assign request
            self.dispatcher_assignments[dispatcher_id] = request_id
            self.available_dispatchers -= 1
            
            # Update status
            self.service_status[request_id].update({
                "status": "assigned",
                "assigned_dispatcher": dispatcher_id,
                "updated_at": time.time()
            })
            
            # Estimate arrival time
            arrival_estimate = await self._calculate_arrival_estimate(request_id)
            self.service_status[request_id]["estimated_arrival"] = arrival_estimate
            
            logger.info(f"Assigned request {request_id} to dispatcher {dispatcher_id}")
            
            # Schedule customer notification
            await self._notify_customer(request_id)
    
    async def _calculate_arrival_estimate(self, request_id: str) -> int:
        """Calculate estimated arrival time in minutes."""
        request = self.active_requests[request_id]
        
        # In a real implementation, this would use maps API
        # For now, use basic estimation
        base_time = 20  # minutes
        
        if request.priority == ServicePriority.HIGH:
            base_time = 15
        elif request.priority == ServicePriority.LOW:
            base_time = 30
        
        # Add time for current conditions
        if request.service_requirements.get("weather_concern", False):
            base_time += 10
        if request.service_requirements.get("difficult_access", False):
            base_time += 5
        
        return base_time
    
    async def _notify_customer(self, request_id: str):
        """Send notification to customer about service status."""
        status = self.service_status[request_id]
        request = self.active_requests[request_id]
        
        # Prepare notification
        notification = {
            "type": "service_update",
            "request_id": request_id,
            "status": status["status"],
            "estimated_arrival": status["estimated_arrival"],
            "service_type": request.agent_type.value,
            "timestamp": time.time()
        }
        
        # In a real implementation, send through proper channels
        logger.info(f"Would send notification to customer: {notification}")
    
    async def update_service_status(
        self,
        request_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Update service request status.
        
        Args:
            request_id: Request identifier
            status: New status
            details: Optional status details
        """
        if request_id not in self.service_status:
            logger.error(f"Request {request_id} not found")
            return
        
        # Update status
        self.service_status[request_id].update({
            "status": status,
            "updated_at": time.time(),
            **(details or {})
        })
        
        # Handle completion
        if status == "completed":
            # Clean up
            request = self.active_requests.pop(request_id, None)
            if request:
                self.request_queues[request.priority].remove(request_id)
            
            # Free up dispatcher
            for dispatcher_id, assigned_request in self.dispatcher_assignments.items():
                if assigned_request == request_id:
                    self.dispatcher_assignments.pop(dispatcher_id)
                    self.available_dispatchers += 1
                    break
            
            logger.info(f"Completed service request {request_id}")
        
        # Notify customer of update
        await self._notify_customer(request_id)
    
    def get_service_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a service request."""
        return self.service_status.get(request_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher service statistics."""
        stats = {
            "active_requests": len(self.active_requests),
            "available_dispatchers": self.available_dispatchers,
            "requests_by_priority": {
                priority.value: len(queue)
                for priority, queue in self.request_queues.items()
            },
            "requests_by_type": {},
            "average_wait_time": 0,
            "completed_requests": 0
        }
        
        # Count requests by type
        for request in self.active_requests.values():
            agent_type = request.agent_type.value
            if agent_type not in stats["requests_by_type"]:
                stats["requests_by_type"][agent_type] = 0
            stats["requests_by_type"][agent_type] += 1
        
        # Calculate average wait time
        current_time = time.time()
        wait_times = [
            current_time - status["created_at"]
            for status in self.service_status.values()
            if status["status"] != "completed"
        ]
        if wait_times:
            stats["average_wait_time"] = sum(wait_times) / len(wait_times)
        
        # Count completed requests
        stats["completed_requests"] = len([
            1 for status in self.service_status.values()
            if status["status"] == "completed"
        ])
        
        return stats