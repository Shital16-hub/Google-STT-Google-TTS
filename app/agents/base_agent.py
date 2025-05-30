"""
Enhanced Base Agent for Multi-Agent Voice AI System
Handles general assistance and clarification conversations.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class AgentStatus(str, Enum):
    """Agent status enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class UrgencyLevel(str, Enum):
    """Urgency level enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AgentCapability(str, Enum):
    """Agent capability enumeration."""
    CONVERSATION = "conversation"
    CLARIFICATION = "clarification"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    TOOL_INTEGRATION = "tool_integration"
    ESCALATION = "escalation"
    EMERGENCY_RESPONSE = "emergency_response"
    LOCATION_SERVICES = "location_services"
    SAFETY_PROTOCOLS = "safety_protocols"

@dataclass
class AgentConfiguration:
    """Agent configuration dataclass."""
    agent_id: str
    version: str
    specialization: Dict[str, Any]
    voice_settings: Dict[str, Any]
    tools: List[str]
    routing: Dict[str, Any]
    performance_monitoring: Dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.INITIALIZING

@dataclass
class AgentStats:
    """Agent statistics dataclass."""
    total_queries: int = 0
    successful_responses: int = 0
    failed_responses: int = 0
    average_response_time_ms: float = 0.0
    average_confidence: float = 0.0
    tools_executed: int = 0
    escalations: int = 0
    uptime_seconds: float = 0.0
    last_activity_timestamp: float = 0.0

@dataclass
class AgentResponse:
    """Agent response dataclass."""
    success: bool
    response: str
    confidence: float
    processing_time_ms: float
    tools_used: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    escalation_needed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResult:
    """Tool execution result."""
    tool_name: str
    success: bool
    output: Any
    execution_time_ms: float
    error: Optional[str] = None

class BaseAgent:
    """
    Enhanced Base Agent for general assistance and clarification conversations.
    Serves as the foundation for specialized agents and handles routing uncertainties.
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfiguration,
        hybrid_vector_system,
        tool_orchestrator=None,
        target_response_time_ms: int = 200
    ):
        """Initialize BaseAgent."""
        self.agent_id = agent_id
        self.config = config
        self.version = config.version
        self.hybrid_vector_system = hybrid_vector_system
        self.tool_orchestrator = tool_orchestrator
        self.target_response_time_ms = target_response_time_ms
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.initialized = False
        self.start_time = time.time()
        
        # Performance tracking
        self.stats = AgentStats()
        self.response_cache = {}
        self.max_cache_size = 100
        
        # Capabilities
        self.capabilities = [
            AgentCapability.CONVERSATION,
            AgentCapability.CLARIFICATION,
            AgentCapability.KNOWLEDGE_RETRIEVAL
        ]
        
        # Voice optimization
        self.max_voice_response_words = 30  # Keep responses short for voice
        
        logger.info(f"BaseAgent initialized: {agent_id} v{config.version}")
    
    async def initialize(self):
        """Initialize the agent."""
        if self.initialized:
            return
        
        try:
            self.status = AgentStatus.INITIALIZING
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            # Initialize specialized components
            await self._initialize_specialized_components()
            
            self.status = AgentStatus.ACTIVE
            self.initialized = True
            self.stats.last_activity_timestamp = time.time()
            
            logger.info(f"✅ Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"❌ Agent {self.agent_id} initialization failed: {e}")
            raise
    
    async def process_query(
        self,
        query: str,
        context: Dict[str, Any] = None,
        session_id: Optional[str] = None
    ) -> AgentResponse:
        """
        Main method to process user queries with clarification support.
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        context = context or {}
        
        try:
            self.stats.total_queries += 1
            self.stats.last_activity_timestamp = time.time()
            
            # Analyze the query
            analysis = await self._analyze_query(query, context)
            
            # Check if this is a clarification conversation
            is_clarification = self._is_clarification_needed(analysis, context)
            
            # Get knowledge context
            knowledge_context = await self._retrieve_knowledge(query, context, analysis)
            
            # Execute tools if needed
            tool_results = []
            if await self._requires_tools(query, context):
                suggested_tools = await self._suggest_tools(query, context)
                tool_results = await self._execute_tools(suggested_tools, query, context)
            
            # Generate response
            if is_clarification:
                response_text = await self._generate_clarification_response(
                    query, context, analysis
                )
            else:
                response_text = await self._generate_response(
                    query, context, knowledge_context, tool_results, analysis
                )
            
            # Optimize for voice
            response_text = self._optimize_for_voice(response_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(analysis, knowledge_context, tool_results)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.stats.successful_responses += 1
            self._update_response_time(processing_time)
            self._update_confidence(confidence)
            
            return AgentResponse(
                success=True,
                response=response_text,
                confidence=confidence,
                processing_time_ms=processing_time,
                tools_used=[r.tool_name for r in tool_results if r.success],
                sources=self._extract_sources(knowledge_context),
                escalation_needed=await self._needs_escalation(query, context),
                metadata={
                    "agent_id": self.agent_id,
                    "analysis": analysis,
                    "is_clarification": is_clarification
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.stats.failed_responses += 1
            
            logger.error(f"Error processing query: {e}")
            
            return AgentResponse(
                success=False,
                response=self._get_error_response(str(e)),
                confidence=0.3,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
    
    def _is_clarification_needed(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Determine if this is a clarification conversation."""
        
        # Check routing factors
        routing_factors = context.get("routing_factors", {})
        if routing_factors.get("needs_clarification"):
            return True
        
        # Check if intent suggests clarification needed
        intent = analysis.get("intent", "")
        if intent in ["needs_clarification", "unclear_request", "general_inquiry"]:
            return True
        
        # Check if this is a follow-up clarification
        if "routing_strategy" in context and context["routing_strategy"] == "clarification_needed":
            return True
        
        return False
    
    async def _generate_clarification_response(
        self,
        query: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """
        Generate clarification questions when routing is uncertain.
        This method helps users provide clearer information for better routing.
        """
        
        # Get routing context
        routing_factors = context.get("routing_factors", {})
        original_intent = routing_factors.get("original_intent", "")
        llm_reasoning = routing_factors.get("llm_reasoning", "")
        
        # Analyze query for contextual clues (without hardcoded patterns)
        query_lower = query.lower()
        
        # Use LLM to generate contextual clarification
        try:
            clarification_prompt = f"""
            A user said: "{query}"
            
            The routing system is uncertain about which specialist to connect them with.
            
            Generate a brief, conversational clarification question to help determine what kind of help they need.
            
            Keep the response:
            - Under 25 words
            - Conversational and friendly
            - Focused on understanding their specific need
            
            Example responses:
            - "I want to help you with that. Could you tell me a bit more about what's happening?"
            - "I can connect you with the right specialist. What specific issue are you experiencing?"
            """
            
            import openai
            client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": clarification_prompt}],
                max_tokens=50,
                temperature=0.7
            )
            
            clarification = response.choices[0].message.content.strip()
            
            # Clean up any quotes
            clarification = clarification.strip('"').strip("'")
            
            logger.info(f"Generated clarification: {clarification}")
            return clarification
            
        except Exception as e:
            logger.error(f"LLM clarification generation failed: {e}")
            
            # Fallback to contextual clarification
            if "vehicle" in query_lower or "car" in query_lower:
                return "I understand you're having a vehicle issue. Are you looking for roadside assistance?"
            
            elif "bill" in query_lower or "payment" in query_lower:
                return "I can help with billing questions. What specific issue are you experiencing?"
            
            elif "not working" in query_lower or "error" in query_lower:
                return "I can help with technical issues. What exactly isn't working for you?"
            
            else:
                # General clarification
                return "I want to make sure I connect you with the right specialist. Could you tell me more about what you need help with?"
    
    async def _analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query to understand intent and requirements."""
        
        analysis = {
            "intent": "general_inquiry",
            "urgency": UrgencyLevel.NORMAL,
            "complexity": "simple",
            "requires_tools": False,
            "confidence": 0.7
        }
        
        # Check for clarification indicators
        routing_factors = context.get("routing_factors", {})
        if routing_factors.get("needs_clarification"):
            analysis["intent"] = "needs_clarification"
        
        # Simple urgency detection
        query_lower = query.lower()
        if any(word in query_lower for word in ["emergency", "urgent", "help", "critical"]):
            analysis["urgency"] = UrgencyLevel.HIGH
        
        return analysis
    
    async def _retrieve_knowledge(
        self, 
        query: str, 
        context: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge from vector database."""
        
        knowledge_context = []
        
        try:
            if self.hybrid_vector_system and self.hybrid_vector_system.initialized:
                # Simple knowledge retrieval for BaseAgent
                results = await self.hybrid_vector_system.search(
                    query=query,
                    namespace=f"agent_{self.agent_id}",
                    top_k=3
                )
                
                if results:
                    knowledge_context = [
                        {
                            "content": result.get("content", ""),
                            "source": result.get("source", "knowledge_base"),
                            "relevance": result.get("score", 0.0)
                        }
                        for result in results
                    ]
        
        except Exception as e:
            logger.error(f"Knowledge retrieval error: {e}")
        
        return knowledge_context
    
    async def _requires_tools(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if tools are required."""
        # BaseAgent typically doesn't require tools for clarification
        return False
    
    async def _suggest_tools(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Suggest appropriate tools."""
        return []
    
    async def _execute_tools(
        self, 
        tools: List[str], 
        query: str, 
        context: Dict[str, Any]
    ) -> List[ToolResult]:
        """Execute suggested tools."""
        return []
    
    async def _generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        knowledge_context: List[Dict[str, Any]],
        tool_results: List[ToolResult],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate general assistance response."""
        
        # For BaseAgent, provide helpful general response
        if analysis.get("intent") == "general_inquiry":
            return "I'm here to help you today. Let me make sure I understand what you need assistance with."
        
        # Default helpful response
        return "I understand you need assistance. I'm here to help you get connected with the right support."
    
    def _optimize_for_voice(self, response: str) -> str:
        """Optimize response for voice interaction."""
        
        # Remove markdown formatting
        response = response.replace("**", "").replace("*", "")
        response = response.replace("###", "").replace("##", "").replace("#", "")
        
        # Remove bullet points and formatting
        lines = response.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-') and not line.startswith('*'):
                clean_lines.append(line)
        
        # Join and limit length for voice
        response = ' '.join(clean_lines)
        
        # Limit to max words for voice
        words = response.split()
        if len(words) > self.max_voice_response_words:
            response = ' '.join(words[:self.max_voice_response_words]) + "."
        
        return response.strip()
    
    def _calculate_confidence(
        self, 
        analysis: Dict[str, Any], 
        knowledge_context: List[Dict[str, Any]], 
        tool_results: List[ToolResult]
    ) -> float:
        """Calculate response confidence."""
        
        base_confidence = 0.7
        
        # Adjust based on knowledge context
        if knowledge_context:
            avg_relevance = sum(k.get("relevance", 0) for k in knowledge_context) / len(knowledge_context)
            base_confidence += avg_relevance * 0.2
        
        # Adjust for successful tool execution
        if tool_results:
            success_rate = sum(1 for r in tool_results if r.success) / len(tool_results)
            base_confidence += success_rate * 0.1
        
        return min(1.0, base_confidence)
    
    async def _needs_escalation(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if human escalation is needed."""
        
        # Check for explicit escalation requests
        escalation_keywords = ["speak to human", "manager", "supervisor", "representative"]
        query_lower = query.lower()
        
        return any(keyword in query_lower for keyword in escalation_keywords)
    
    def _extract_sources(self, knowledge_context: List[Dict[str, Any]]) -> List[str]:
        """Extract sources from knowledge context."""
        return [k.get("source", "unknown") for k in knowledge_context if k.get("source")]
    
    def _get_error_response(self, error: str) -> str:
        """Generate user-friendly error response."""
        return "I apologize, but I'm having a technical issue right now. Could you please repeat your request?"
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time."""
        if self.stats.total_queries > 0:
            current_avg = self.stats.average_response_time_ms
            total_queries = self.stats.total_queries
            self.stats.average_response_time_ms = (
                (current_avg * (total_queries - 1) + response_time_ms) / total_queries
            )
        else:
            self.stats.average_response_time_ms = response_time_ms
    
    def _update_confidence(self, confidence: float):
        """Update average confidence."""
        if self.stats.successful_responses > 0:
            current_avg = self.stats.average_confidence
            successful = self.stats.successful_responses
            self.stats.average_confidence = (
                (current_avg * (successful - 1) + confidence) / successful
            )
        else:
            self.stats.average_confidence = confidence
    
    async def _load_knowledge_base(self):
        """Load agent-specific knowledge base."""
        # BaseAgent has general knowledge
        logger.info(f"Loading general knowledge base for {self.agent_id}")
    
    async def _initialize_specialized_components(self):
        """Initialize specialized components."""
        # BaseAgent uses default components
        logger.info(f"BaseAgent specialized components initialized for {self.agent_id}")
    
    def get_stats(self) -> AgentStats:
        """Get current agent statistics."""
        self.stats.uptime_seconds = time.time() - self.start_time
        return self.stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        return {
            "healthy": self.status == AgentStatus.ACTIVE,
            "status": self.status.value,
            "initialized": self.initialized,
            "uptime_seconds": time.time() - self.start_time,
            "last_activity": self.stats.last_activity_timestamp,
            "response_time_ms": self.stats.average_response_time_ms,
            "success_rate": (
                self.stats.successful_responses / max(self.stats.total_queries, 1)
            ) * 100
        }
    
    async def shutdown(self):
        """Shutdown the agent."""
        logger.info(f"Shutting down agent: {self.agent_id}")
        
        self.status = AgentStatus.INACTIVE
        self.initialized = False
        
        # Clear caches
        self.response_cache.clear()
        
        logger.info(f"✅ Agent {self.agent_id} shutdown complete")