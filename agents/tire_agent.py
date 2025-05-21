# agents/tire_agent.py

"""
Specialized agent for handling tire service requests.
"""
import logging
from typing import Dict, Any, List, Optional
import re

from agents.base_agent import BaseAgent, AgentType
from core.state_manager import ConversationState

logger = logging.getLogger(__name__)

class TireAgent(BaseAgent):
    """Specialized agent for tire service requests."""
    
    def __init__(self, **kwargs):
        """Initialize tire agent."""
        super().__init__(agent_type=AgentType.TIRE, **kwargs)
        
        # Tire-specific fields
        self.required_fields.extend([
            "tire_location",
            "has_spare",
            "tire_size",
            "vehicle_access"
        ])
        
        # Service pricing
        self.base_price = 65.00
        self.additional_fees = {
            "no_spare": 85.00,  # New tire cost
            "special_tools": 25.00,
            "difficult_access": 35.00
        }
        
        # Service requirements tracking
        self.service_requirements = {
            "needs_new_tire": False,
            "needs_special_tools": False,
            "difficult_access": False
        }
        
        logger.info("Initialized tire agent with specialized fields")
    
    async def _extract_information(
        self,
        message: str,
        response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract tire-specific information from messages.
        
        Args:
            message: User message
            response: Generated response
            
        Returns:
            Dictionary of extracted information
        """
        extracted_info = {}
        
        # Convert to lowercase for pattern matching
        message_lower = message.lower()
        
        # Extract tire location
        if "tire_location" not in self.collected_info:
            location_patterns = {
                "front left": ["front left", "driver front", "front driver"],
                "front right": ["front right", "passenger front", "front passenger"],
                "rear left": ["rear left", "back left", "driver rear", "rear driver"],
                "rear right": ["rear right", "back right", "passenger rear", "rear passenger"]
            }
            
            for location, patterns in location_patterns.items():
                if any(pattern in message_lower for pattern in patterns):
                    extracted_info["tire_location"] = location
                    break
        
        # Check spare tire availability
        if "has_spare" not in self.collected_info:
            if "spare" in message_lower:
                if any(neg in message_lower for neg in ["no", "don't have", "doesn't have"]):
                    extracted_info["has_spare"] = False
                    self.service_requirements["needs_new_tire"] = True
                else:
                    extracted_info["has_spare"] = True
        
        # Extract tire size
        if "tire_size" not in self.collected_info:
            # Look for common tire size patterns (e.g., 215/55R17)
            size_match = re.search(r'\d{3}/\d{2}[Rr]\d{2}', message)
            if size_match:
                extracted_info["tire_size"] = size_match.group(0)
        
        # Check vehicle access
        if "vehicle_access" not in self.collected_info:
            if any(term in message_lower for term in ["stuck", "ditch", "shoulder", "mud"]):
                extracted_info["vehicle_access"] = "difficult"
                self.service_requirements["difficult_access"] = True
            elif "parking" in message_lower:
                extracted_info["vehicle_access"] = "normal"
        
        # Check for special tools requirement
        if any(term in message_lower for term in ["locking", "security", "special wrench"]):
            self.service_requirements["needs_special_tools"] = True
        
        logger.info(f"Extracted tire info: {extracted_info}")
        return extracted_info
    
    def _calculate_price_estimate(self) -> float:
        """Calculate estimated price for tire service."""
        total = self.base_price
        
        # Add fees based on service requirements
        if self.service_requirements["needs_new_tire"]:
            total += self.additional_fees["no_spare"]
        if self.service_requirements["needs_special_tools"]:
            total += self.additional_fees["special_tools"]
        if self.service_requirements["difficult_access"]:
            total += self.additional_fees["difficult_access"]
        
        return total
    
    def _should_handoff(self) -> bool:
        """Determine if we should hand off to a dispatcher."""
        # Check base conditions first
        if super()._should_handoff():
            return True
            
        # Add tire-specific conditions
        return (
            self.service_requirements["difficult_access"] and
            self.service_requirements["needs_special_tools"]
        ) or (
            not self.collected_info.get("has_spare", True) and
            self.service_requirements["needs_new_tire"]
        )
    
    async def _prepare_handoff(self) -> Dict[str, Any]:
        """Prepare handoff with tire-specific information."""
        handoff_info = await super()._prepare_handoff()
        
        # Add tire-specific details
        handoff_info.update({
            "service_type": "tire",
            "price_estimate": self._calculate_price_estimate(),
            "service_requirements": self.service_requirements,
            "tire_location": self.collected_info.get("tire_location"),
            "tire_size": self.collected_info.get("tire_size")
        })
        
        return handoff_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tire-specific statistics."""
        stats = super().get_stats()
        
        # Add tire-specific stats
        stats.update({
            "service_requirements": self.service_requirements,
            "price_estimate": self._calculate_price_estimate(),
            "tire_info": {
                "location": self.collected_info.get("tire_location"),
                "has_spare": self.collected_info.get("has_spare"),
                "size": self.collected_info.get("tire_size")
            }
        })
        
        return stats