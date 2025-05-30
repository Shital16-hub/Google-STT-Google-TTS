"""
LLM-Based Intelligent Agent Router
==================================

Uses OpenAI to intelligently understand user intent and route to the correct agent.
No hardcoded patterns - pure LLM-based routing.
"""
import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

@dataclass
class RoutingResult:
    """Result from LLM-based routing decision."""
    agent_id: str
    confidence: float
    reasoning: str
    intent: str
    urgency: str
    entities: List[Dict[str, Any]]
    requires_tools: bool
    fallback_used: bool = False

class LLMIntelligentRouter:
    """
    LLM-based intelligent agent router using OpenAI for intent recognition.
    Replaces hardcoded keyword matching with intelligent understanding.
    """
    
    def __init__(self, agent_registry=None):
        """Initialize LLM router with proper OpenAI client setup."""
        self.agent_registry = agent_registry
        
        # FIXED: Create OpenAI client with clean configuration
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found for LLM router")
            self.client = None
        else:
            # Create client with ONLY API key - no organization
            self.client = AsyncOpenAI(api_key=api_key)
        
        # Available agents (loaded from registry)
        self.available_agents = {}
        
        # Performance tracking
        self.stats = {
            "total_routings": 0,
            "successful_routings": 0,
            "fallback_routings": 0,
            "average_confidence": 0.0,
            "average_response_time_ms": 0.0,
            "intent_distribution": {},
            "agent_usage": {}
        }
        
        logger.info("LLM Intelligent Router initialized")

    
    async def initialize(self):
        """Initialize router with agent information."""
        logger.info("🚀 Initializing LLM Intelligent Router...")
        
        try:
            # Load available agents from registry
            if self.agent_registry:
                agents = await self.agent_registry.list_active_agents()
                for agent in agents:
                    self.available_agents[agent.agent_id] = {
                        "id": agent.agent_id,
                        "name": agent.config.get("name", agent.agent_id),
                        "description": agent.config.get("description", ""),
                        "specialization": agent.config.get("specialization", {}),
                        "tools": agent.config.get("tools", []),
                        "domain": agent.config.get("specialization", {}).get("domain_expertise", "")
                    }
                
                logger.info(f"✅ Loaded {len(self.available_agents)} agents for routing")
            else:
                # Fallback agent definitions
                self.available_agents = {
                    "roadside-assistance-v2": {
                        "id": "roadside-assistance-v2",
                        "name": "Roadside Assistance",
                        "description": "Emergency roadside assistance for vehicles",
                        "domain": "emergency_vehicle_assistance",
                        "tools": ["tow_truck_dispatch", "emergency_escalation"]
                    },
                    "billing-support-v2": {
                        "id": "billing-support-v2", 
                        "name": "Billing Support",
                        "description": "Help with billing, payments, and account issues",
                        "domain": "financial_support",
                        "tools": ["payment_processing", "refund_workflow"]
                    },
                    "technical-support-v2": {
                        "id": "technical-support-v2",
                        "name": "Technical Support", 
                        "description": "Technical troubleshooting and setup assistance",
                        "domain": "technical_troubleshooting",
                        "tools": ["diagnostic_tools", "setup_assistance"]
                    }
                }
                
                logger.warning("⚠️ Using fallback agent definitions")
            
            logger.info("✅ LLM Intelligent Router initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Error initializing LLM router: {e}")
            raise
    
    async def route_intelligently(
        self,
        user_input: str,
        context: Dict[str, Any] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> RoutingResult:
        """
        Use LLM to intelligently determine the best agent for the user's request.
        """
        routing_start = time.time()
        self.stats["total_routings"] += 1
        
        try:
            logger.info(f"🎯 LLM routing for: '{user_input}'")
            
            # Build routing prompt
            routing_prompt = self._build_routing_prompt(user_input, context, conversation_history)
            
            # Get LLM routing decision
            llm_response = await self._get_llm_routing_decision(routing_prompt)
            
            # Parse LLM response
            routing_result = self._parse_llm_response(llm_response, user_input)
            
            # Update statistics
            routing_time = (time.time() - routing_start) * 1000
            self._update_routing_stats(routing_result, routing_time)
            
            logger.info(f"✅ LLM routing: {routing_result.agent_id} (confidence: {routing_result.confidence:.2f})")
            return routing_result
            
        except Exception as e:
            logger.error(f"❌ LLM routing error: {e}")
            
            # Fallback to simple routing
            fallback_result = self._fallback_routing(user_input)
            routing_time = (time.time() - routing_start) * 1000
            self._update_routing_stats(fallback_result, routing_time)
            
            return fallback_result
    
    def _build_routing_prompt(
        self,
        user_input: str,
        context: Dict[str, Any] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """FIXED: Pure LLM routing - ZERO hardcoded patterns."""
        
        # Agent descriptions for LLM
        agent_descriptions = []
        for agent_id, agent_info in self.available_agents.items():
            description = f"""
    Agent: {agent_info['name']} (ID: {agent_id})
    - Description: {agent_info['description']}
    - Specialization: {agent_info.get('domain', 'General assistance')}
    """
            agent_descriptions.append(description)
        
        agents_text = "\n".join(agent_descriptions)
        
        # Build conversation context
        context_text = ""
        if conversation_history:
            context_text = "\nConversation History:\n"
            for msg in conversation_history[-3:]:
                context_text += f"- {msg.get('role', 'user')}: {msg.get('content', '')}\n"
        
        prompt = f"""You are an intelligent customer service router. Analyze the user's request and determine the most appropriate agent.
    
                Available Agents:
                {agents_text}
                
                User's Request: "{user_input}"
                {context_text}
                
                Your task is to understand the user's intent and match it to the agent who can best help them.
                
                Respond in JSON format:
                {{
                    "agent_id": "exact_agent_id_from_list_or_base_agent",
                    "confidence": 0.0-1.0,
                    "reasoning": "why this agent is the best choice",
                    "intent": "what the user needs",
                    "urgency": "low|normal|high|critical",
                    "needs_clarification": false
                }}
                
                IMPORTANT: 
                - If confidence is below 0.7, set "needs_clarification": true
                - If unsure, choose "base_agent" for general assistance
                - Do NOT guess - analyze the actual request"""
        
        return prompt
    
    async def _get_llm_routing_decision(self, prompt: str) -> str:
        """FIXED: Get routing decision from LLM with error handling."""
        if not self.client:
            raise Exception("OpenAI client not available")
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model for routing
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert customer service routing assistant. Always respond with valid JSON as requested."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent routing
                timeout=10  # Add timeout
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"❌ LLM API call failed: {e}")
            raise
    
    def _parse_llm_response(self, llm_response: str, user_input: str) -> RoutingResult:
        """FIXED: Parse LLM response with proper fallback strategy."""
        try:
            # Try to extract JSON from response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = llm_response[json_start:json_end]
                routing_data = json.loads(json_text)
            else:
                raise ValueError("No JSON found in LLM response")
            
            # Get routing decision
            agent_id = routing_data.get("agent_id", "base_agent")
            confidence = float(routing_data.get("confidence", 0.5))
            needs_clarification = routing_data.get("needs_clarification", False)
            
            # Validate agent_id - including base_agent
            valid_agents = list(self.available_agents.keys()) + ["base_agent"]
            if agent_id not in valid_agents:
                logger.warning(f"⚠️ Invalid agent_id from LLM: {agent_id}")
                agent_id = "base_agent"
                confidence = 0.4
                needs_clarification = True
            
            # Create routing result
            result = RoutingResult(
                agent_id=agent_id,
                confidence=confidence,
                reasoning=routing_data.get("reasoning", "LLM routing decision"),
                intent=routing_data.get("intent", "general_inquiry"),
                urgency=routing_data.get("urgency", "normal"),
                entities=routing_data.get("entities", []),
                requires_tools=routing_data.get("requires_tools", False),
                fallback_used=False
            )
            
            # If needs clarification, modify response
            if needs_clarification or confidence < 0.7:
                result.agent_id = "base_agent"  # Use base agent for clarification
                result.needs_clarification = True
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing LLM response: {e}")
            logger.debug(f"LLM response was: {llm_response}")
            
            # Return base agent fallback - NOT technical support
            return self._intelligent_fallback(user_input)

    def _intelligent_fallback(self, user_input: str) -> RoutingResult:
        """FIXED: Intelligent fallback using BaseAgent for clarification."""
        
        return RoutingResult(
            agent_id="base_agent",  # Use BaseAgent, not technical support
            confidence=0.3,
            reasoning="LLM routing failed - using base agent for clarification",
            intent="needs_clarification",
            urgency="normal",
            entities=[],
            requires_tools=False,
            fallback_used=True,
            needs_clarification=True
        )
    
    def _fallback_routing(self, user_input: str) -> RoutingResult:
        """Fallback routing when LLM fails."""
        self.stats["fallback_routings"] += 1
        
        agent_id = self._get_fallback_agent(user_input)
        
        return RoutingResult(
            agent_id=agent_id,
            confidence=0.6,
            reasoning="Fallback routing due to LLM error",
            intent="unknown",
            urgency="normal",
            entities=[],
            requires_tools=False,
            fallback_used=True
        )
    
    
    
    def _update_routing_stats(self, result: RoutingResult, response_time_ms: float):
        """Update routing statistics."""
        if not result.fallback_used:
            self.stats["successful_routings"] += 1
        
        # Update averages
        total = self.stats["total_routings"]
        
        # Average confidence
        current_avg_conf = self.stats["average_confidence"]
        self.stats["average_confidence"] = (
            (current_avg_conf * (total - 1) + result.confidence) / total
        )
        
        # Average response time
        current_avg_time = self.stats["average_response_time_ms"]
        self.stats["average_response_time_ms"] = (
            (current_avg_time * (total - 1) + response_time_ms) / total
        )
        
        # Intent distribution
        intent = result.intent
        self.stats["intent_distribution"][intent] = (
            self.stats["intent_distribution"].get(intent, 0) + 1
        )
        
        # Agent usage
        agent_id = result.agent_id
        self.stats["agent_usage"][agent_id] = (
            self.stats["agent_usage"].get(agent_id, 0) + 1
        )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total = self.stats["total_routings"]
        successful = self.stats["successful_routings"]
        success_rate = successful / total if total > 0 else 0.0
        
        return {
            "performance": {
                "total_routings": total,
                "successful_routings": successful,
                "fallback_routings": self.stats["fallback_routings"],
                "success_rate": round(success_rate, 3),
                "average_confidence": round(self.stats["average_confidence"], 3),
                "average_response_time_ms": round(self.stats["average_response_time_ms"], 2)
            },
            "usage_patterns": {
                "intent_distribution": self.stats["intent_distribution"],
                "agent_usage": self.stats["agent_usage"]
            },
            "available_agents": list(self.available_agents.keys()),
            "health": {
                "status": "healthy" if success_rate > 0.8 else "degraded",
                "llm_connectivity": "operational"
            }
        }
    
    async def explain_routing_decision(self, user_input: str) -> Dict[str, Any]:
        """Get detailed explanation of routing decision for debugging."""
        try:
            result = await self.route_intelligently(user_input)
            
            return {
                "user_input": user_input,
                "routing_result": {
                    "agent_id": result.agent_id,
                    "agent_name": self.available_agents.get(result.agent_id, {}).get("name", "Unknown"),
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "intent": result.intent,
                    "urgency": result.urgency,
                    "entities": result.entities,
                    "requires_tools": result.requires_tools,
                    "fallback_used": result.fallback_used
                },
                "available_agents": {
                    agent_id: agent_info["name"] 
                    for agent_id, agent_info in self.available_agents.items()
                }
            }
            
        except Exception as e:
            return {
                "user_input": user_input,
                "error": str(e),
                "available_agents": list(self.available_agents.keys())
            }