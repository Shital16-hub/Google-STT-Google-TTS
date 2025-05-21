# agents/dispatcher_agent.py

"""
Specialized agent for handling dispatcher operations and human handoffs.
"""
import logging
from typing import Dict, Any, List, Optional
import time
import asyncio

from agents.base_agent import BaseAgent, AgentType
from core.state_manager import ConversationState
from services.dispatcher import DispatcherService

logger = logging.getLogger(__name__)

class DispatcherAgent(BaseAgent):
    """
    Specialized agent for dispatcher operations.
    
    This agent handles:
    1. Human handoff coordination
    2. Service dispatch operations
    3. Status updates and notifications
    4. Complex service requirements
    """
    
    def __init__(self, dispatcher_service: Optional[DispatcherService] = None, **kwargs):
        """
        Initialize dispatcher agent.
        
        Args:
            dispatcher_service: Optional dispatcher service instance
            **kwargs: Additional arguments
        """
        super().__init__(agent_type=AgentType.DISPATCHER, **kwargs)
        
        # Get or create dispatcher service
        self.dispatcher_service = dispatcher_service or DispatcherService()
        
        # Dispatcher-specific fields
        self.required_fields.extend([
            "callback_number",
            "alternate_contact",
            "preferred_contact_method"
        ])
        
        # Handoff tracking
        self.handoff_info: Optional[Dict[str, Any]] = None
        self.service_request_id: Optional[str] = None
        self.estimated_wait_time: Optional[int] = None
        
        # Status tracking
        self.status_updates: List[Dict[str, Any]] = []
        self.last_status_check = time.time()
        
        logger.info("Initialized dispatcher agent")
    
    async def handle_handoff(
        self,
        previous_agent: BaseAgent,
        collected_info: Dict[str, Any],
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle handoff from another agent.
        
        Args:
            previous_agent: Agent handing off the conversation
            collected_info: Information collected so far
            conversation_history: Conversation history
            
        Returns:
            Handoff response
        """
        try:
            # Store handoff information
            self.handoff_info = {
                "previous_agent_type": previous_agent.agent_type,
                "collected_info": collected_info,
                "conversation_history": conversation_history,
                "timestamp": time.time()
            }
            
            # Create service request
            self.service_request_id = await self.dispatcher_service.create_service_request(
                session_id=self.conversation_manager.session_id,
                agent_type=previous_agent.agent_type,
                customer_info=collected_info,
                service_requirements=previous_agent.service_requirements,
                handoff_reason="Complex service requirements"
            )
            
            # Get initial status
            status = self.dispatcher_service.get_service_status(self.service_request_id)
            if status:
                self.estimated_wait_time = status.get("estimated_arrival")
                self.status_updates.append(status)
            
            # Prepare handoff response
            response = {
                "response": self._create_handoff_message(),
                "service_request_id": self.service_request_id,
                "estimated_wait": self.estimated_wait_time,
                "requires_additional_info": self._needs_additional_info()
            }
            
            logger.info(f"Handled handoff for service request {self.service_request_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error handling handoff: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing the handoff. Let me connect you with a supervisor.",
                "error": str(e),
                "requires_supervisor": True
            }
    
    def _create_handoff_message(self) -> str:
        """Create appropriate handoff message."""
        if not self.service_request_id or not self.estimated_wait_time:
            return (
                "I'm transferring you to our dispatch team for assistance. "
                "They'll be with you shortly."
            )
        
        # Create detailed message
        message_parts = [
            "I'm connecting you with our dispatch team.",
            f"Your service request ID is {self.service_request_id}.",
            f"Estimated arrival time is {self.estimated_wait_time} minutes."
        ]
        
        if self._needs_additional_info():
            message_parts.append(
                "I'll need a few additional details to help coordinate your service."
            )
        
        return " ".join(message_parts)
    
    def _needs_additional_info(self) -> bool:
        """Check if we need additional information."""
        if not self.handoff_info:
            return True
            
        collected_info = self.handoff_info.get("collected_info", {})
        
        # Check required fields
        missing_fields = [
            field for field in self.required_fields
            if field not in collected_info or not collected_info[field]
        ]
        
        return bool(missing_fields)
    
    async def _extract_information(
        self,
        message: str,
        response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract dispatcher-specific information from messages.
        
        Args:
            message: User message
            response: Generated response
            
        Returns:
            Dictionary of extracted information
        """
        extracted_info = {}
        
        # Convert to lowercase for pattern matching
        message_lower = message.lower()
        
        # Extract callback number
        if "callback_number" not in self.collected_info:
            # Look for phone numbers
            import re
            phone_match = re.search(
                r'(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})',
                message
            )
            if phone_match:
                phone = f"{phone_match.group(1)}{phone_match.group(2)}{phone_match.group(3)}"
                extracted_info["callback_number"] = phone
        
        # Extract contact preferences
        if "preferred_contact_method" not in self.collected_info:
            if "text" in message_lower or "sms" in message_lower:
                extracted_info["preferred_contact_method"] = "sms"
            elif "call" in message_lower or "phone" in message_lower:
                extracted_info["preferred_contact_method"] = "phone"
        
        # Extract alternate contact
        if "alternate_contact" not in self.collected_info:
            # Look for mentions of alternate contact
            alt_contact_patterns = [
                r'(?:alternate|other|another|secondary)\s+(?:number|contact|phone)?\s*[is:]?\s*([0-9-]+)',
                r'(?:can also reach|also call)\s+(?:me at|on)?\s*([0-9-]+)'
            ]
            
            for pattern in alt_contact_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    extracted_info["alternate_contact"] = match.group(1)
                    break
        
        logger.info(f"Extracted dispatcher info: {extracted_info}")
        return extracted_info
    
    async def check_status(self) -> Dict[str, Any]:
        """Check current service request status."""
        if not self.service_request_id:
            return {
                "error": "No active service request",
                "status": "unknown"
            }
        
        try:
            # Get latest status
            status = self.dispatcher_service.get_service_status(self.service_request_id)
            
            if status:
                # Update tracking
                self.status_updates.append(status)
                self.last_status_check = time.time()
                
                return {
                    "status": status["status"],
                    "estimated_arrival": status.get("estimated_arrival"),
                    "assigned_dispatcher": status.get("assigned_dispatcher"),
                    "last_update": status.get("updated_at")
                }
            
            return {
                "error": "Status not found",
                "status": "unknown"
            }
            
        except Exception as e:
            logger.error(f"Error checking status: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    async def update_status(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update service request status.
        
        Args:
            status: New status
            details: Optional status details
            
        Returns:
            True if update was successful
        """
        if not self.service_request_id:
            return False
            
        try:
            await self.dispatcher_service.update_service_status(
                self.service_request_id,
                status,
                details
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher-specific statistics."""
        stats = super().get_stats()
        
        # Add dispatcher-specific stats
        stats.update({
            "service_request_id": self.service_request_id,
            "estimated_wait_time": self.estimated_wait_time,
            "status_updates": len(self.status_updates),
            "last_status_check": self.last_status_check,
            "handoff_info": {
                "previous_agent": self.handoff_info["previous_agent_type"].value
                if self.handoff_info else None,
                "timestamp": self.handoff_info["timestamp"]
                if self.handoff_info else None
            }
        })
        
        return stats