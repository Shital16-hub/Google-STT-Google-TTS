"""
Emergency-Specialized Roadside Assistance Agent
Handles emergency roadside situations with urgency detection and safety protocols.
Optimized for emergency response with <150ms processing time.
"""
import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from app.agents.base_agent import (
    BaseAgent, AgentResponse, AgentConfiguration, UrgencyLevel,
    AgentCapability, ToolResult
)
from app.vector_db.hybrid_vector_system import HybridVectorSystem
from app.tools.orchestrator import ComprehensiveToolOrchestrator

logger = logging.getLogger(__name__)

@dataclass
class EmergencyAssessment:
    """Assessment of emergency situation severity."""
    severity_level: UrgencyLevel
    safety_risk: float  # 0.0 to 1.0
    response_priority: int  # 1 (highest) to 5 (lowest)
    recommended_actions: List[str]
    escalation_needed: bool
    eta_minutes: Optional[int] = None

class RoadsideEmergencyDetector:
    """Detects and classifies emergency situations."""
    
    def __init__(self):
        self.emergency_patterns = {
            UrgencyLevel.EMERGENCY: [
                "accident", "crash", "collision", "injured", "bleeding",
                "unconscious", "trapped", "fire", "smoke", "gas leak",
                "highway", "freeway", "traffic", "blocking traffic"
            ],
            UrgencyLevel.CRITICAL: [
                "stuck on highway", "dead battery highway", "flat tire highway",
                "breakdown highway", "won't start highway", "stranded highway",
                "urgent", "asap", "emergency", "dangerous location"
            ],
            UrgencyLevel.HIGH: [
                "stuck", "stranded", "won't start", "dead battery",
                "flat tire", "need tow", "breakdown", "help"
            ],
            UrgencyLevel.NORMAL: [
                "jump start", "tire change", "minor issue", "maintenance"
            ]
        }
        
        self.location_risk_indicators = [
            "highway", "freeway", "interstate", "bridge", "tunnel",
            "busy street", "intersection", "traffic", "dark area"
        ]
        
        self.safety_keywords = [
            "safe", "safety", "dangerous", "risk", "hazard",
            "visibility", "shoulder", "pulled over"
        ]
    
    def assess_emergency(self, query: str, context: Dict[str, Any]) -> EmergencyAssessment:
        """Assess emergency level and safety requirements."""
        query_lower = query.lower()
        
        # Determine urgency level
        urgency = self._detect_urgency_level(query_lower, context)
        
        # Assess safety risk
        safety_risk = self._calculate_safety_risk(query_lower, context)
        
        # Set response priority
        priority_map = {
            UrgencyLevel.EMERGENCY: 1,
            UrgencyLevel.CRITICAL: 2,
            UrgencyLevel.HIGH: 3,
            UrgencyLevel.NORMAL: 4,
            UrgencyLevel.LOW: 5
        }
        priority = priority_map.get(urgency, 4)
        
        # Generate recommended actions
        actions = self._generate_safety_actions(urgency, safety_risk, query_lower)
        
        # Determine if escalation needed
        escalation_needed = (
            urgency == UrgencyLevel.EMERGENCY or
            safety_risk > 0.8 or
            "injured" in query_lower or
            "emergency services" in query_lower
        )
        
        # Estimate ETA based on urgency
        eta_map = {
            UrgencyLevel.EMERGENCY: 15,
            UrgencyLevel.CRITICAL: 25,
            UrgencyLevel.HIGH: 35,
            UrgencyLevel.NORMAL: 45,
            UrgencyLevel.LOW: 60
        }
        eta = eta_map.get(urgency, 45)
        
        return EmergencyAssessment(
            severity_level=urgency,
            safety_risk=safety_risk,
            response_priority=priority,
            recommended_actions=actions,
            escalation_needed=escalation_needed,
            eta_minutes=eta
        )
    
    def _detect_urgency_level(self, query_lower: str, context: Dict[str, Any]) -> UrgencyLevel:
        """Detect urgency level from query and context."""
        # Check for emergency patterns
        for urgency_level in [UrgencyLevel.EMERGENCY, UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.NORMAL]:
            patterns = self.emergency_patterns.get(urgency_level, [])
            for pattern in patterns:
                if pattern in query_lower:
                    return urgency_level
        
        # Check time context for urgency boost
        current_hour = time.localtime().tm_hour
        if current_hour < 6 or current_hour > 22:  # Night time
            if any(word in query_lower for word in ["stuck", "stranded", "breakdown"]):
                return UrgencyLevel.HIGH
        
        return UrgencyLevel.NORMAL
    
    def _calculate_safety_risk(self, query_lower: str, context: Dict[str, Any]) -> float:
        """Calculate safety risk score."""
        risk_score = 0.0
        
        # Location-based risk
        for indicator in self.location_risk_indicators:
            if indicator in query_lower:
                risk_score += 0.3
        
        # Weather/time factors
        weather = context.get("weather", {})
        if weather.get("conditions") in ["rain", "snow", "fog"]:
            risk_score += 0.2
        
        current_hour = time.localtime().tm_hour
        if current_hour < 6 or current_hour > 22:  # Night time
            risk_score += 0.2
        
        # Traffic indicators
        if any(word in query_lower for word in ["blocking traffic", "busy road", "intersection"]):
            risk_score += 0.4
        
        # Injury/emergency indicators
        if any(word in query_lower for word in ["injured", "hurt", "bleeding", "unconscious"]):
            risk_score += 0.8
        
        return min(1.0, risk_score)
    
    def _generate_safety_actions(self, urgency: UrgencyLevel, safety_risk: float, query_lower: str) -> List[str]:
        """Generate safety action recommendations."""
        actions = []
        
        # Universal safety actions
        actions.append("Ensure your vehicle is in a safe location away from traffic")
        
        if safety_risk > 0.5:
            actions.append("Turn on hazard lights immediately")
            actions.append("If possible, move vehicle to shoulder or safe area")
        
        if urgency in [UrgencyLevel.EMERGENCY, UrgencyLevel.CRITICAL]:
            actions.append("Call 911 if there are injuries or immediate danger")
            actions.append("Exit vehicle on the side away from traffic if safe")
        
        if "highway" in query_lower or "freeway" in query_lower:
            actions.append("Stand behind a barrier or away from your vehicle")
            actions.append("Be visible to oncoming traffic")
        
        if safety_risk > 0.7:
            actions.append("Consider calling emergency services")
        
        return actions

class LocationExtractor:
    """Extracts and validates location information."""
    
    def __init__(self):
        self.address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)',
            r'[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)',
            r'(?:Highway|Hwy|Route|Rt)\s*\d+',
            r'Interstate\s*\d+|I-\d+',
            r'Mile\s*(?:marker|post)\s*\d+'
        ]
        
        self.city_state_pattern = r'([A-Za-z\s]+),\s*([A-Z]{2})\s*(\d{5})?'
        self.coordinates_pattern = r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)'
    
    def extract_location(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location information from query and context."""
        location_info = {
            "address": None,
            "city": None,
            "state": None,
            "zip_code": None,
            "coordinates": None,
            "landmarks": [],
            "confidence": 0.0
        }
        
        # Check context first
        if context.get("location"):
            location_info.update(context["location"])
            location_info["confidence"] = 0.9
            return location_info
        
        # Extract from query
        query_upper = query.upper()
        
        # Try to find street address
        for pattern in self.address_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location_info["address"] = match.group(0)
                location_info["confidence"] = 0.8
                break
        
        # Try to find city, state, zip
        city_match = re.search(self.city_state_pattern, query, re.IGNORECASE)
        if city_match:
            location_info["city"] = city_match.group(1).strip()
            location_info["state"] = city_match.group(2)
            if city_match.group(3):
                location_info["zip_code"] = city_match.group(3)
            location_info["confidence"] = max(location_info["confidence"], 0.7)
        
        # Try to find coordinates
        coord_match = re.search(self.coordinates_pattern, query)
        if coord_match:
            location_info["coordinates"] = {
                "lat": float(coord_match.group(1)),
                "lng": float(coord_match.group(2))
            }
            location_info["confidence"] = 0.9
        
        # Extract landmarks
        landmark_keywords = [
            "near", "by", "close to", "next to", "at", "on"
        ]
        
        for keyword in landmark_keywords:
            if keyword in query.lower():
                # Simple landmark extraction
                parts = query.lower().split(keyword)
                if len(parts) > 1:
                    potential_landmark = parts[1].split('.')[0].split(',')[0].strip()
                    if len(potential_landmark) > 3:
                        location_info["landmarks"].append(potential_landmark)
        
        return location_info

class VehicleInfoExtractor:
    """Extracts vehicle information from queries."""
    
    def __init__(self):
        self.vehicle_makes = [
            "toyota", "honda", "ford", "chevrolet", "chevy", "nissan",
            "hyundai", "volkswagen", "vw", "bmw", "mercedes", "audi",
            "jeep", "dodge", "chrysler", "subaru", "mazda", "lexus"
        ]
        
        self.vehicle_types = [
            "car", "truck", "suv", "van", "motorcycle", "bike",
            "sedan", "coupe", "hatchback", "wagon", "pickup"
        ]
        
        self.year_pattern = r'(19|20)\d{2}'
    
    def extract_vehicle_info(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract vehicle information from query."""
        vehicle_info = {
            "year": None,
            "make": None,
            "model": None,
            "type": None,
            "color": None,
            "confidence": 0.0
        }
        
        query_lower = query.lower()
        
        # Extract year
        year_match = re.search(self.year_pattern, query)
        if year_match:
            vehicle_info["year"] = int(year_match.group(0))
            vehicle_info["confidence"] += 0.2
        
        # Extract make
        for make in self.vehicle_makes:
            if make in query_lower:
                vehicle_info["make"] = make.title()
                vehicle_info["confidence"] += 0.3
                break
        
        # Extract vehicle type
        for vtype in self.vehicle_types:
            if vtype in query_lower:
                vehicle_info["type"] = vtype
                vehicle_info["confidence"] += 0.2
                break
        
        # Extract color (simple approach)
        colors = ["red", "blue", "black", "white", "silver", "gray", "green", "yellow"]
        for color in colors:
            if color in query_lower:
                vehicle_info["color"] = color
                vehicle_info["confidence"] += 0.1
                break
        
        return vehicle_info

class RoadsideAssistanceAgent(BaseAgent):
    """
    Emergency-specialized roadside assistance agent with safety protocols and urgency detection.
    Optimized for rapid emergency response with comprehensive tool integration.
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfiguration,
        hybrid_vector_system: HybridVectorSystem,
        tool_orchestrator: Optional[ComprehensiveToolOrchestrator] = None,
        target_response_time_ms: int = 150  # Faster for emergencies
    ):
        """Initialize roadside assistance agent with emergency capabilities."""
        super().__init__(
            agent_id=agent_id,
            config=config,
            hybrid_vector_system=hybrid_vector_system,
            tool_orchestrator=tool_orchestrator,
            target_response_time_ms=target_response_time_ms
        )
        
        # Specialized components
        self.emergency_detector = RoadsideEmergencyDetector()
        self.location_extractor = LocationExtractor()
        self.vehicle_extractor = VehicleInfoExtractor()
        
        # Domain-specific capabilities
        self.capabilities.extend([
            AgentCapability.EMERGENCY_RESPONSE,
            AgentCapability.LOCATION_SERVICES,
            AgentCapability.SAFETY_PROTOCOLS
        ])
        
        # Emergency response templates
        self.emergency_templates = {
            UrgencyLevel.EMERGENCY: "I'm dispatching emergency assistance immediately. Please ensure your safety first.",
            UrgencyLevel.CRITICAL: "I understand this is urgent. I'm arranging immediate roadside assistance.",
            UrgencyLevel.HIGH: "I'll get help to you as quickly as possible. Let me arrange assistance.",
            UrgencyLevel.NORMAL: "I can help you with that. Let me find the best solution for you."
        }
        
        # Service area knowledge
        self.service_areas = {
            "coverage_radius_miles": 50,
            "emergency_response_time_minutes": 15,
            "standard_response_time_minutes": 45
        }
        
        logger.info(f"RoadsideAssistanceAgent initialized with emergency response capabilities")
    
    async def _detect_intent(self, query: str, context: Dict[str, Any]) -> str:
        """Detect roadside assistance intent."""
        query_lower = query.lower()
        
        # Emergency intents
        if any(word in query_lower for word in ["accident", "crash", "emergency", "injured"]):
            return "emergency_response"
        
        # Service intents
        if any(word in query_lower for word in ["tow", "towing"]):
            return "towing_service"
        elif any(word in query_lower for word in ["jump", "battery", "dead battery"]):
            return "jump_start"
        elif any(word in query_lower for word in ["tire", "flat tire"]):
            return "tire_service"
        elif any(word in query_lower for word in ["lockout", "locked out", "keys"]):
            return "lockout_service"
        elif any(word in query_lower for word in ["fuel", "gas", "empty tank"]):
            return "fuel_delivery"
        else:
            return "general_assistance"
    
    async def _requires_tools(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if tools are required for roadside assistance."""
        # Most roadside queries require dispatching or workflow tools
        intent = await self._detect_intent(query, context)
        
        tool_required_intents = [
            "emergency_response", "towing_service", "jump_start",
            "tire_service", "lockout_service", "fuel_delivery"
        ]
        
        return intent in tool_required_intents
    
    async def _suggest_tools(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Suggest appropriate tools based on query analysis."""
        intent = await self._detect_intent(query, context)
        emergency_assessment = self.emergency_detector.assess_emergency(query, context)
        
        suggested_tools = []
        
        # Emergency tools
        if emergency_assessment.escalation_needed:
            suggested_tools.append("emergency_escalation_workflow")
        
        # Service-specific tools
        if intent == "towing_service":
            suggested_tools.append("dispatch_tow_truck_workflow")
        elif intent == "jump_start":
            suggested_tools.append("dispatch_service_technician")
        elif intent in ["tire_service", "lockout_service", "fuel_delivery"]:
            suggested_tools.append("dispatch_service_technician")
        
        # Communication tools
        location_info = self.location_extractor.extract_location(query, context)
        if location_info["confidence"] > 0.5:
            suggested_tools.append("send_location_sms")
        
        # Always consider service coverage check
        suggested_tools.append("search_service_coverage")
        
        return suggested_tools
    
    async def _generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        knowledge_context: List[Dict[str, Any]],
        tool_results: List[ToolResult],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate specialized roadside assistance response."""
        
        # Assess emergency situation
        emergency_assessment = self.emergency_detector.assess_emergency(query, context)
        
        # Extract key information
        location_info = self.location_extractor.extract_location(query, context)
        vehicle_info = self.vehicle_extractor.extract_vehicle_info(query, context)
        intent = analysis.get("intent", "general_assistance")
        
        # Start with appropriate urgency template
        urgency_template = self.emergency_templates.get(
            emergency_assessment.severity_level,
            self.emergency_templates[UrgencyLevel.NORMAL]
        )
        
        response_parts = [urgency_template]
        
        # Add safety instructions for high-risk situations
        if emergency_assessment.safety_risk > 0.6:
            safety_actions = emergency_assessment.recommended_actions[:2]  # Top 2 actions
            if safety_actions:
                response_parts.append(f"For your safety: {' and '.join(safety_actions).lower()}")
        
        # Process tool results
        dispatch_info = self._process_tool_results(tool_results, intent)
        if dispatch_info:
            response_parts.append(dispatch_info)
        
        # Add service details based on intent
        service_details = self._generate_service_details(
            intent, location_info, vehicle_info, emergency_assessment
        )
        if service_details:
            response_parts.append(service_details)
        
        # Add ETA information
        if emergency_assessment.eta_minutes:
            eta_text = f"Estimated arrival time: {emergency_assessment.eta_minutes} minutes"
            response_parts.append(eta_text)
        
        # Add next steps
        next_steps = self._generate_next_steps(intent, emergency_assessment, location_info)
        if next_steps:
            response_parts.append(next_steps)
        
        # Combine response parts
        full_response = ". ".join(response_parts)
        
        # Ensure response is concise for voice
        if len(full_response) > 200:  # Max words for voice optimization
            # Prioritize safety and dispatch information
            priority_parts = [urgency_template]
            if emergency_assessment.safety_risk > 0.6 and emergency_assessment.recommended_actions:
                priority_parts.append(f"Please {emergency_assessment.recommended_actions[0].lower()}")
            if dispatch_info:
                priority_parts.append(dispatch_info)
            if emergency_assessment.eta_minutes:
                priority_parts.append(f"ETA: {emergency_assessment.eta_minutes} minutes")
            
            full_response = ". ".join(priority_parts)
        
        return full_response
    
    def _process_tool_results(self, tool_results: List[ToolResult], intent: str) -> Optional[str]:
        """Process tool execution results into response text."""
        successful_tools = [r for r in tool_results if r.success]
        
        if not successful_tools:
            return None
        
        dispatch_messages = []
        
        for result in successful_tools:
            if result.tool_name == "dispatch_tow_truck_workflow":
                if result.output and isinstance(result.output, dict):
                    truck_id = result.output.get("truck_assigned", "a truck")
                    driver_name = result.output.get("driver_name", "your driver")
                    dispatch_messages.append(f"I've dispatched {truck_id} with {driver_name}")
            
            elif result.tool_name == "emergency_escalation_workflow":
                dispatch_messages.append("Emergency services have been notified")
            
            elif result.tool_name == "send_location_sms":
                dispatch_messages.append("I've sent your location to our dispatch team")
        
        return ". ".join(dispatch_messages) if dispatch_messages else None
    
    def _generate_service_details(
        self,
        intent: str,
        location_info: Dict[str, Any],
        vehicle_info: Dict[str, Any],
        emergency_assessment: EmergencyAssessment
    ) -> Optional[str]:
        """Generate service-specific details."""
        
        if intent == "towing_service":
            details = "Our tow truck will safely transport your vehicle"
            if vehicle_info.get("type"):
                details += f" ({vehicle_info['type']})"
            return details
        
        elif intent == "jump_start":
            return "A technician will arrive with equipment to jump-start your battery"
        
        elif intent == "tire_service":
            return "We'll either change your tire or provide roadside tire repair"
        
        elif intent == "lockout_service":
            return "A locksmith will help you regain access to your vehicle"
        
        elif intent == "fuel_delivery":
            return "We'll deliver enough fuel to get you to the nearest gas station"
        
        elif intent == "emergency_response":
            if emergency_assessment.escalation_needed:
                return "Emergency services are being contacted for immediate assistance"
        
        return None
    
    def _generate_next_steps(
        self,
        intent: str,
        emergency_assessment: EmergencyAssessment,
        location_info: Dict[str, Any]
    ) -> Optional[str]:
        """Generate next steps for the customer."""
        
        if emergency_assessment.severity_level == UrgencyLevel.EMERGENCY:
            return "Please remain safe and wait for emergency responders"
        
        if emergency_assessment.severity_level in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]:
            return "Please stay with your vehicle in a safe location"
        
        if location_info["confidence"] < 0.5:
            return "Our driver will call you to confirm your exact location"
        
        return "You'll receive an SMS with tracking information"
    
    async def _needs_escalation(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if human escalation is needed."""
        emergency_assessment = self.emergency_detector.assess_emergency(query, context)
        
        # Escalate for emergencies or safety risks
        if emergency_assessment.escalation_needed:
            return True
        
        # Escalate if customer explicitly requests
        escalation_keywords = ["speak to human", "manager", "supervisor", "not satisfied"]
        query_lower = query.lower()
        
        return any(keyword in query_lower for keyword in escalation_keywords)
    
    async def _load_knowledge_base(self):
        """Load roadside assistance specific knowledge."""
        # This would load roadside assistance procedures, safety protocols,
        # service area information, pricing, and emergency response guidelines
        logger.info(f"Loading roadside assistance knowledge base for {self.agent_id}")
        
        # Sample knowledge areas that would be loaded:
        knowledge_areas = [
            "emergency_response_procedures",
            "towing_protocols",
            "safety_guidelines",
            "service_area_coverage",
            "pricing_information",
            "vehicle_compatibility",
            "weather_considerations",
            "traffic_safety_protocols"
        ]
        
        # In a real implementation, this would load documents from these areas
        # into the hybrid vector system under the agent's namespace
        
    async def _initialize_specialized_components(self):
        """Initialize roadside-specific components."""
        # Initialize emergency response capabilities
        logger.info("Initializing emergency response capabilities")
        
        # Set up real-time traffic and weather monitoring
        # This would integrate with external APIs for current conditions
        
        # Initialize service provider network
        # This would load available service providers, their capabilities, and locations
        
        # Set up emergency escalation pathways
        # This would configure connections to emergency services when needed
        
        logger.info("Roadside assistance specialized components initialized")
    
    def get_emergency_capabilities(self) -> Dict[str, Any]:
        """Get emergency response capabilities."""
        return {
            "emergency_response": True,
            "safety_protocols": True,
            "24_7_availability": True,
            "emergency_escalation": True,
            "multi_language_support": False,  # Would be enhanced
            "weather_monitoring": True,
            "traffic_awareness": True,
            "gps_tracking": True,
            "service_areas": self.service_areas,
            "response_times": {
                "emergency": "15 minutes",
                "critical": "25 minutes", 
                "high": "35 minutes",
                "normal": "45 minutes"
            }
        }
    
    def get_service_coverage(self, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get service coverage information."""
        coverage_info = {
            "covered": True,  # Would be calculated based on actual location
            "radius_miles": self.service_areas["coverage_radius_miles"],
            "emergency_response_available": True,
            "estimated_response_time_minutes": self.service_areas["standard_response_time_minutes"],
            "available_services": [
                "Towing", "Jump Start", "Tire Change", "Lockout Service",
                "Fuel Delivery", "Emergency Response"
            ]
        }
        
        if location:
            # In a real implementation, this would check actual service coverage
            # based on coordinates, service provider locations, etc.
            pass
        
        return coverage_info