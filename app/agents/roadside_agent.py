"""
Roadside Assistance Specialized Agent.
Handles emergency situations, towing requests, and roadside services with urgency optimization.
"""
import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.agents.base_agent import BaseAgent, ToolExecutionResult, AgentCapability
from app.config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

class RoadsideAssistanceAgent(BaseAgent):
    """
    Specialized agent for roadside assistance and emergency services.
    
    Capabilities:
    - Emergency situation assessment and prioritization
    - Tow truck dispatch and coordination
    - Roadside service recommendations
    - ETA calculations and location services
    - Customer safety guidance
    - Insurance and billing coordination
    """
    
    def __init__(self, agent_id: str, agent_config: Dict[str, Any], vector_store=None, performance_tracker=None):
        """Initialize roadside assistance agent."""
        super().__init__(agent_id, agent_config, vector_store, performance_tracker)
        
        # Add specific capabilities
        self.capabilities.update({
            AgentCapability.EMERGENCY_RESPONSE,
            AgentCapability.AFTER_HOURS
        })
        
        # Roadside-specific settings
        self.emergency_keywords = {
            "high_priority": {"accident", "crash", "collision", "injury", "dangerous", "highway", "freeway"},
            "medium_priority": {"breakdown", "stuck", "stranded", "flat tire", "battery"},
            "service_types": {"tow", "jump start", "tire change", "lockout", "fuel delivery"}
        }
        
        # Service area and dispatch info (would be loaded from config in production)
        self.service_areas = {
            "metro": {"coverage_radius": 25, "avg_response_time": 30},
            "suburban": {"coverage_radius": 15, "avg_response_time": 45},
            "rural": {"coverage_radius": 50, "avg_response_time": 60}
        }
        
        logger.info(f"RoadsideAssistanceAgent initialized: {agent_id}")
    
    async def _analyze_tool_requirements(
        self, 
        user_input: str, 
        context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze if tools are needed for roadside assistance request."""
        user_input_lower = user_input.lower()
        tool_calls = []
        
        # Check for location-based services
        location_patterns = [
            r'\b(on|at|near)\s+([A-Za-z\s]+(?:street|st|avenue|ave|road|rd|highway|hwy|freeway|blvd|boulevard))\b',
            r'\b(mile marker|mm)\s*(\d+)\b',
            r'\b(exit\s*\d+|interstate|i-\d+)\b',
            r'\b(zip code|zipcode)\s*(\d{5})\b'
        ]
        
        has_location = any(re.search(pattern, user_input_lower) for pattern in location_patterns)
        
        # Tow truck dispatch
        if any(word in user_input_lower for word in ["tow", "towing", "tow truck", "stuck", "breakdown"]):
            tool_calls.append({
                "tool": "dispatch_tow_truck",
                "input": {
                    "request_type": "towing",
                    "user_input": user_input,
                    "priority": self._assess_priority(user_input)
                }
            })
        
        # ETA calculation
        if has_location or any(word in user_input_lower for word in ["how long", "eta", "when", "time"]):
            tool_calls.append({
                "tool": "calculate_eta",
                "input": {
                    "location": self._extract_location(user_input),
                    "service_type": self._determine_service_type(user_input)
                }
            })
        
        # Service recommendation
        if any(word in user_input_lower for word in ["what should", "what do", "help", "advice"]):
            tool_calls.append({
                "tool": "service_recommendation",
                "input": {
                    "situation": user_input,
                    "safety_first": self._is_safety_critical(user_input)
                }
            })
        
        return tool_calls
    
    def _assess_priority(self, user_input: str) -> str:
        """Assess priority level of roadside request."""
        user_input_lower = user_input.lower()
        
        # High priority - emergency situations
        for keyword in self.emergency_keywords["high_priority"]:
            if keyword in user_input_lower:
                return "high"
        
        # Check for safety indicators
        safety_indicators = ["dark", "night", "rain", "snow", "dangerous", "unsafe", "scared"]
        if any(indicator in user_input_lower for indicator in safety_indicators):
            return "high"
        
        # Medium priority - standard breakdowns
        for keyword in self.emergency_keywords["medium_priority"]:
            if keyword in user_input_lower:
                return "medium"
        
        return "low"
    
    def _extract_location(self, user_input: str) -> str:
        """Extract location information from user input."""
        # Simple location extraction - in production would use NLP/NER
        location_patterns = [
            r'\b(on|at|near)\s+([A-Za-z\s]+(?:street|st|avenue|ave|road|rd|highway|hwy|freeway|blvd|boulevard))\b',
            r'\b(mile marker|mm)\s*(\d+)\b',
            r'\b(exit\s*\d+)\b',
            r'\b(interstate|i-\d+)\b'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "location not specified"
    
    def _determine_service_type(self, user_input: str) -> str:
        """Determine type of roadside service needed."""
        user_input_lower = user_input.lower()
        
        service_mapping = {
            "towing": ["tow", "towing", "tow truck", "stuck", "breakdown", "won't start"],
            "jump_start": ["battery", "jump", "jump start", "dead battery", "won't start"],
            "tire_change": ["flat tire", "tire", "puncture", "blowout"],
            "lockout": ["locked out", "keys", "locked in", "key in car"],
            "fuel_delivery": ["gas", "fuel", "empty", "ran out"]
        }
        
        for service_type, keywords in service_mapping.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return service_type
        
        return "general_assistance"
    
    def _is_safety_critical(self, user_input: str) -> bool:
        """Determine if situation is safety critical."""
        safety_keywords = [
            "accident", "crash", "collision", "injury", "hurt", "bleeding",
            "highway", "freeway", "interstate", "dangerous", "unsafe",
            "dark", "night", "rain", "snow", "ice", "storm"
        ]
        
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in safety_keywords)
    
    async def _generate_response(
        self,
        user_input: str,
        context: List[Dict[str, Any]],
        conversation_history: List[BaseMessage],
        tool_results: List[ToolExecutionResult]
    ) -> Dict[str, Any]:
        """Generate specialized roadside assistance response."""
        
        # Assess urgency and priority
        priority = self._assess_priority(user_input)
        is_safety_critical = self._is_safety_critical(user_input)
        
        # Build context-aware system prompt
        system_prompt = self._build_system_prompt(priority, is_safety_critical, tool_results)
        
        # Prepare conversation messages
        messages = [
            SystemMessage(content=system_prompt),
        ]
        
        # Add relevant context
        if context:
            context_text = self._format_context_for_roadside(context)
            if context_text:
                messages.append(SystemMessage(content=f"Relevant information:\n{context_text}"))
        
        # Add tool results
        if tool_results:
            tool_summary = self._format_tool_results_for_roadside(tool_results)
            messages.append(SystemMessage(content=f"Service information:\n{tool_summary}"))
        
        # Add conversation history (keep recent)
        if conversation_history:
            messages.extend(conversation_history[-4:])  # Last 4 messages for context
        
        # Add user input
        messages.append(HumanMessage(content=user_input))
        
        # Generate response with appropriate urgency
        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content
            
            # Post-process response for roadside context
            response_text = self._post_process_roadside_response(
                response_text, priority, is_safety_critical, tool_results
            )
            
            # Calculate confidence based on context and tools
            confidence = self._calculate_confidence(user_input, context, tool_results)
            
            return {
                "text": response_text,
                "confidence": confidence,
                "priority": priority,
                "safety_critical": is_safety_critical,
                "needs_human": priority == "high" and confidence < 0.7
            }
            
        except Exception as e:
            logger.error(f"Error generating roadside response: {e}")
            
            # Fallback response for errors
            fallback_response = self._generate_fallback_response(priority, is_safety_critical)
            
            return {
                "text": fallback_response,
                "confidence": 0.3,
                "priority": priority,
                "safety_critical": is_safety_critical,
                "needs_human": True
            }
    
    def _build_system_prompt(
        self, 
        priority: str, 
        is_safety_critical: bool, 
        tool_results: List[ToolExecutionResult]
    ) -> str:
        """Build context-aware system prompt for roadside assistance."""
        
        base_prompt = """You are a professional roadside assistance coordinator with expertise in emergency response and vehicle services. Your primary goals are:

1. SAFETY FIRST - Always prioritize customer safety and well-being
2. Quick, efficient service dispatch and coordination  
3. Clear, calm communication, especially in stressful situations
4. Accurate information about services, timing, and costs

Response Guidelines:
- Keep responses concise and action-oriented (1-2 sentences for voice)
- Use reassuring, professional tone
- Provide specific next steps when possible
- Include relevant timing information (ETAs, response times)
- Mention safety precautions when appropriate"""

        if is_safety_critical:
            base_prompt += """

SAFETY CRITICAL SITUATION DETECTED:
- Prioritize immediate safety instructions
- Recommend contacting emergency services if needed
- Keep customer calm and informed
- Expedite service dispatch"""

        if priority == "high":
            base_prompt += """

HIGH PRIORITY REQUEST:
- Treat as urgent - expedited response required
-