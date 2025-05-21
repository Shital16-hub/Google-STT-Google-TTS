# agents/jump_start_agent.py

"""
Specialized agent for handling jump start service requests.
"""
import logging
from typing import Dict, Any, List, Optional
import re

from agents.base_agent import BaseAgent, AgentType
from core.state_manager import ConversationState

logger = logging.getLogger(__name__)

class JumpStartAgent(BaseAgent):
    """Specialized agent for jump start service requests."""
    
    def __init__(self, **kwargs):
        """Initialize jump start agent."""
        super().__init__(agent_type=AgentType.JUMP_START, **kwargs)
        
        # Jump start-specific fields
        self.required_fields.extend([
            "battery_age",
            "previous_attempts",
            "vehicle_symptoms",
            "vehicle_access"
        ])
        
        # Service pricing
        self.base_price = 45.00
        self.additional_fees = {
            "difficult_access": 25.00,
            "battery_test": 15.00,
            "after_hours": 25.00
        }
        
        # Service requirements tracking
        self.service_requirements = {
            "needs_battery_test": False,
            "difficult_access": False,
            "after_hours": False,
            "potential_alternator": False
        }
        
        # Battery symptoms tracking
        self.battery_symptoms = []
        
        logger.info("Initialized jump start agent with specialized fields")
    
    async def _extract_information(
        self,
        message: str,
        response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract jump start-specific information from messages.
        
        Args:
            message: User message
            response: Generated response
            
        Returns:
            Dictionary of extracted information
        """
        extracted_info = {}
        
        # Convert to lowercase for pattern matching
        message_lower = message.lower()
        
        # Extract battery age if mentioned
        if "battery_age" not in self.collected_info:
            age_patterns = [
                r'(\d+)\s*(?:year|yr)s?\s*old',
                r'bought (?:the battery|it)\s*(\d+)\s*(?:year|yr)s?\s*ago'
            ]
            
            for pattern in age_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    extracted_info["battery_age"] = int(match.group(1))
                    break
        
        # Check for previous jump start attempts
        if "previous_attempts" not in self.collected_info:
            if "tried" in message_lower or "attempt" in message_lower:
                if any(neg in message_lower for neg in ["no", "haven't", "didn't"]):
                    extracted_info["previous_attempts"] = "none"
                else:
                    extracted_info["previous_attempts"] = "tried"
        
        # Extract vehicle symptoms
        symptoms = []
        symptom_patterns = {
            "clicking": ["clicking", "click sound", "clicks"],
            "no_crank": ["won't crank", "doesn't crank", "no crank"],
            "slow_crank": ["slow crank", "barely cranks", "weak crank"],
            "dashboard_lights": ["dashboard", "lights flickering", "dim lights"],
            "radio_reset": ["radio reset", "clock reset", "lost settings"]
        }
        
        for symptom, patterns in symptom_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                symptoms.append(symptom)
        
        if symptoms:
            extracted_info["vehicle_symptoms"] = symptoms
            self.battery_symptoms.extend(symptoms)
            
            # Check for alternator issues
            if "dashboard_lights" in symptoms or "radio_reset" in symptoms:
                self.service_requirements["potential_alternator"] = True
                self.service_requirements["needs_battery_test"] = True
        
        # Check vehicle access
        if "vehicle_access" not in self.collected_info:
            if any(term in message_lower for term in ["garage", "parking structure", "underground"]):
                extracted_info["vehicle_access"] = "difficult"
                self.service_requirements["difficult_access"] = True
            elif "parking" in message_lower:
                extracted_info["vehicle_access"] = "normal"
        
        logger.info(f"Extracted jump start info: {extracted_info}")
        return extracted_info
    
    def _calculate_price_estimate(self) -> float:
        """Calculate estimated price for jump start service."""
        total = self.base_price
        
        # Add fees based on service requirements
        if self.service_requirements["difficult_access"]:
            total += self.additional_fees["difficult_access"]
        if self.service_requirements["needs_battery_test"]:
            total += self.additional_fees["battery_test"]
        if self.service_requirements["after_hours"]:
            total += self.additional_fees["after_hours"]
        
        return total
    
    def _should_handoff(self) -> bool:
        """Determine if we should hand off to a dispatcher."""
        # Check base conditions first
        if super()._should_handoff():
            return True
            
        # Add jump start-specific conditions
        return (
            self.service_requirements["potential_alternator"] or
            (len(self.battery_symptoms) >= 3) or
            (self.collected_info.get("previous_attempts") == "tried" and
             self.service_requirements["needs_battery_test"])
        )
    
    async def _prepare_handoff(self) -> Dict[str, Any]:
        """Prepare handoff with jump start-specific information."""
        handoff_info = await super()._prepare_handoff()
        
        # Add jump start-specific details
        handoff_info.update({
            "service_type": "jump_start",
            "price_estimate": self._calculate_price_estimate(),
            "service_requirements": self.service_requirements,
            "battery_symptoms": self.battery_symptoms,
            "battery_age": self.collected_info.get("battery_age"),
            "previous_attempts": self.collected_info.get("previous_attempts")
        })
        
        return handoff_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get jump start-specific statistics."""
        stats = super().get_stats()
        
        # Add jump start-specific stats
        stats.update({
            "service_requirements": self.service_requirements,
            "price_estimate": self._calculate_price_estimate(),
            "battery_info": {
                "age": self.collected_info.get("battery_age"),
                "symptoms": self.battery_symptoms,
                "previous_attempts": self.collected_info.get("previous_attempts")
            }
        })
        
        return stats