# agents/towing_agent.py

"""
Specialized agent for handling towing service requests.
"""
import logging
from typing import Dict, Any, List, Optional
import re

from agents.base_agent import BaseAgent, AgentType
from core.state_manager import ConversationState

logger = logging.getLogger(__name__)

class TowingAgent(BaseAgent):
    """Specialized agent for towing service requests."""
    
    def __init__(self, **kwargs):
        """Initialize towing agent."""
        super().__init__(agent_type=AgentType.TOWING, **kwargs)
        
        # Towing-specific fields
        self.required_fields.extend([
            "vehicle_make",
            "vehicle_model",
            "vehicle_year",
            "vehicle_condition",
            "destination"
        ])
        
        # Price calculation factors
        self.base_price = 75.00
        self.per_mile_rate = 3.50
        self.additional_fees = {
            "winch_out": 50.00,
            "after_hours": 25.00,
            "difficult_access": 35.00
        }
        
        # Service requirements tracking
        self.service_requirements = {
            "needs_winch": False,
            "difficult_access": False,
            "after_hours": False
        }
        
        logger.info("Initialized towing agent with specialized fields")
    
    async def _extract_information(
        self,
        message: str,
        response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract towing-specific information from messages.
        
        Args:
            message: User message
            response: Generated response
            
        Returns:
            Dictionary of extracted information
        """
        extracted_info = {}
        
        # Convert to lowercase for pattern matching
        message_lower = message.lower()
        
        # Extract vehicle information
        if "vehicle_make" not in self.collected_info:
            # Look for common car makes
            car_makes = ["toyota", "honda", "ford", "chevrolet", "bmw", "mercedes"]
            for make in car_makes:
                if make in message_lower:
                    extracted_info["vehicle_make"] = make.title()
                    break
        
        # Extract vehicle year
        if "vehicle_year" not in self.collected_info:
            # Look for 4-digit years
            year_match = re.search(r'\b(19|20)\d{2}\b', message)
            if year_match:
                extracted_info["vehicle_year"] = year_match.group(0)
        
        # Extract location information
        if "location" not in self.collected_info:
            # Look for address patterns
            address_match = re.search(r'\b\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|highway|hwy|lane|ln|drive|dr)\b', message_lower)
            if address_match:
                extracted_info["location"] = address_match.group(0).title()
        
        # Check for service requirements
        if "winch" in message_lower or "stuck" in message_lower:
            self.service_requirements["needs_winch"] = True
        if "mud" in message_lower or "ditch" in message_lower:
            self.service_requirements["difficult_access"] = True
        
        # Extract destination if provided
        if "destination" not in self.collected_info:
            if "to the" in message_lower and "shop" in message_lower:
                # Look for repair shop destination
                shop_match = re.search(r'to the\s+([\w\s]+shop)', message_lower)
                if shop_match:
                    extracted_info["destination"] = shop_match.group(1).title() + " Shop"
        
        logger.info(f"Extracted towing info: {extracted_info}")
        return extracted_info
    
    def _calculate_price_estimate(self) -> float:
        """Calculate estimated price for towing service."""
        total = self.base_price
        
        # Add per-mile charge if we have both location and destination
        if "location" in self.collected_info and "destination" in self.collected_info:
            # In a real implementation, calculate actual distance
            estimated_miles = 10  # Example distance
            total += estimated_miles * self.per_mile_rate
        
        # Add additional fees based on service requirements
        if self.service_requirements["needs_winch"]:
            total += self.additional_fees["winch_out"]
        if self.service_requirements["difficult_access"]:
            total += self.additional_fees["difficult_access"]
        if self.service_requirements["after_hours"]:
            total += self.additional_fees["after_hours"]
        
        return total
    
    def _should_handoff(self) -> bool:
        """Determine if we should hand off to a dispatcher."""
        # Check base conditions first
        if super()._should_handoff():
            return True
            
        # Add towing-specific conditions
        return (
            self.service_requirements["difficult_access"] and 
            self.service_requirements["needs_winch"]
        ) or (
            "vehicle_condition" in self.collected_info and
            "accident" in self.collected_info["vehicle_condition"].lower()
        )
    
    async def _prepare_handoff(self) -> Dict[str, Any]:
        """Prepare handoff with towing-specific information."""
        handoff_info = await super()._prepare_handoff()
        
        # Add towing-specific details
        handoff_info.update({
            "service_type": "towing",
            "price_estimate": self._calculate_price_estimate(),
            "service_requirements": self.service_requirements
        })
        
        return handoff_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get towing-specific statistics."""
        stats = super().get_stats()
        
        # Add towing-specific stats
        stats.update({
            "service_requirements": self.service_requirements,
            "price_estimate": self._calculate_price_estimate()
        })
        
        return stats