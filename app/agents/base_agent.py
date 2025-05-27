"""
Advanced Agent Framework for Revolutionary Multi-Agent Voice AI System
Provides the foundation for specialized agents with domain expertise and tool integration.
Target: <200ms agent processing time with intelligent caching and optimization.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from app.vector_db.hybrid_vector_system import HybridVectorSystem, SearchResult
from app.tools.tool_orchestrator import ComprehensiveToolOrchestrator, ToolResult

logger = logging.getLogger(__name__)

class AgentStatus(str, Enum):
    """Agent deployment and operational status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"

class AgentCapability(str, Enum):
    """Core agent capabilities."""
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    TOOL_EXECUTION = "tool_execution"
    CONVERSATION_MANAGEMENT = "conversation_management"
    CONTEXT_UNDERSTANDING = "context_understanding"
    MULTILINGUAL_SUPPORT = "multilingual_support"
    EMOTION_DETECTION = "emotion_detection"
    ESCALATION_MANAGEMENT = "escalation_management"

class UrgencyLevel(str, Enum):
    """Urgency levels for agent processing."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class AgentResponse:
    """Response from an agent with metadata and context."""
    response: str
    confidence: float
    agent_id: str
    processing_time_ms: float
    sources_used: List[str] = field(default_factory=list)
    tools_executed: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    requires_tools: bool = False
    suggested_tools: List[str] = field(default_factory=list)
    escalation_needed: bool = False
    urgency_level: UrgencyLevel = UrgencyLevel.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentConfiguration:
    """Comprehensive agent configuration."""
    agent_id: str
    version: str
    specialization: Dict[str, Any]
    voice_settings: Dict[str, Any]
    tools: List[str]
    routing: Dict[str, Any]
    performance_monitoring: Dict[str, Any]
    status: AgentStatus = AgentStatus.ACTIVE
    capabilities: List[AgentCapability] = field(default_factory=list)
    priority: int = 1
    deployment_strategy: str = "blue_green"

@dataclass
class AgentStats:
    """Agent performance statistics."""
    total_queries: int = 0
    successful_responses: int = 0
    failed_responses: int = 0
    average_response_time_ms: float = 0.0
    average_confidence: float = 0.0
    tools_executed: int = 0
    escalations: int = 0
    cache_hits: int = 0
    last_used: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0
    error_rate: float = 0.0

class BaseAgent(ABC):
    """
    Advanced base agent class providing core functionality for specialized agents.
    Implements caching, performance monitoring, and intelligent response generation.
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfiguration,
        hybrid_vector_system: HybridVectorSystem,
        tool_orchestrator: Optional[ComprehensiveToolOrchestrator] = None,
        target_response_time_ms: int = 200
    ):
        """Initialize base agent with advanced capabilities."""
        self.agent_id = agent_id
        self.config = config
        self.hybrid_vector_system = hybrid_vector_system
        self.tool_orchestrator = tool_orchestrator
        self.target_response_time_ms = target_response_time_ms
        
        # Agent metadata
        self.version = config.version
        self.specialization = config.specialization
        self.capabilities = config.capabilities or []
        self.status = config.status
        
        # Performance tracking
        self.stats = AgentStats()
        self.deployment_time = time.time()
        self.last_optimization = time.time()
        
        # Caching and optimization
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 1000
        
        # Knowledge and context
        self.knowledge_namespace = f"agent_{agent_id}"
        self.context_window_size = config.specialization.get("context_window_size", 2048)
        
        # Voice and personality settings
        self.voice_settings = config.voice_settings or {}
        self.personality_profile = config.specialization.get("personality_profile", "professional")
        
        # Tool integration
        self.available_tools = config.tools or []
        self.tool_execution_timeout = 5.0
        
        # System prompts and templates
        self.system_prompt = self._build_system_prompt()
        self.response_templates = self._load_response_templates()
        
        # Performance optimization
        self.optimization_enabled = True
        self.adaptive_caching = True
        
        self.initialized = False
        logger.info(f"BaseAgent initialized: {agent_id} v{self.version}")
    
    async def initialize(self):
        """Initialize agent with vector system and tool orchestrator."""
        if self.initialized:
            return
        
        try:
            # Ensure hybrid vector system is initialized
            if not self.hybrid_vector_system.initialized:
                await self.hybrid_vector_system.initialize()
            
            # Ensure tool orchestrator is initialized
            if self.tool_orchestrator and not self.tool_orchestrator.initialized:
                await self.tool_orchestrator.initialize()
            
            # Load agent-specific knowledge
            await self._load_knowledge_base()
            
            # Initialize agent-specific components
            await self._initialize_specialized_components()
            
            # Update stats
            self.stats.uptime_seconds = time.time() - self.deployment_time
            
            self.initialized = True
            logger.info(f"✅ Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            raise
    
    async def process_query(
        self,
        query: str,
        context: Dict[str, Any],
        urgency: UrgencyLevel = UrgencyLevel.NORMAL
    ) -> AgentResponse:
        """
        Process a query with intelligent caching and optimization.
        Target: <200ms processing time.
        """
        if not self.initialized:
            await self.initialize()
        
        processing_start = time.time()
        query_id = str(uuid.uuid4())
        
        logger.debug(f"Agent {self.agent_id} processing query: {query[:100]}...")
        
        try:
            # Update stats
            self.stats.total_queries += 1
            self.stats.last_used = time.time()
            
            # Check cache first for performance
            cache_key = self._generate_cache_key(query, context)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response and urgency != UrgencyLevel.EMERGENCY:
                self.stats.cache_hits += 1
                cached_response.processing_time_ms = (time.time() - processing_start) * 1000
                logger.debug(f"Cache hit for agent {self.agent_id}: {cached_response.processing_time_ms:.2f}ms")
                return cached_response
            
            # Analyze query for routing and tool requirements
            query_analysis = await self._analyze_query(query, context, urgency)
            
            # Retrieve relevant knowledge
            knowledge_context = await self._retrieve_knowledge(query, context, query_analysis)
            
            # Execute tools if needed
            tool_results = []
            if query_analysis.get("requires_tools", False):
                tool_results = await self._execute_tools(query, context, query_analysis)
            
            # Generate response using specialized logic
            response_text = await self._generate_response(
                query, context, knowledge_context, tool_results, query_analysis
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(response_text, knowledge_context, tool_results)
            
            # Create agent response
            processing_time = (time.time() - processing_start) * 1000
            
            agent_response = AgentResponse(
                response=response_text,
                confidence=confidence,
                agent_id=self.agent_id,
                processing_time_ms=processing_time,
                sources_used=[src.get("source", "") for src in knowledge_context],
                tools_executed=[result.tool_name for result in tool_results if result.success],
                context_data=query_analysis,
                requires_tools=query_analysis.get("requires_tools", False),
                suggested_tools=query_analysis.get("suggested_tools", []),
                escalation_needed=query_analysis.get("escalation_needed", False),
                urgency_level=urgency,
                metadata={
                    "query_id": query_id,
                    "knowledge_sources": len(knowledge_context),
                    "tool_executions": len(tool_results),
                    "cache_used": False,
                    "specialization": self.specialization.get("domain_expertise", "general")
                }
            )
            
            # Cache response for future use
            if self.adaptive_caching and confidence > 0.7:
                self._cache_response(cache_key, agent_response)
            
            # Update performance statistics
            self._update_stats(agent_response)
            
            # Log performance
            if processing_time > self.target_response_time_ms:
                logger.warning(f"⚠️ Agent {self.agent_id} exceeded target: {processing_time:.2f}ms > {self.target_response_time_ms}ms")
            else:
                logger.debug(f"✅ Agent {self.agent_id} processed query in {processing_time:.2f}ms")
            
            return agent_response
            
        except Exception as e:
            # Handle errors gracefully
            processing_time = (time.time() - processing_start) * 1000
            logger.error(f"❌ Error in agent {self.agent_id}: {e}", exc_info=True)
            
            self.stats.failed_responses += 1
            error_response = await self._generate_error_response(query, str(e), processing_time)
            return error_response
    
    async def _analyze_query(
        self,
        query: str,
        context: Dict[str, Any],
        urgency: UrgencyLevel
    ) -> Dict[str, Any]:
        """Analyze query for intent, complexity, and requirements."""
        analysis = {
            "intent": await self._detect_intent(query, context),
            "complexity": self._calculate_complexity(query),
            "urgency": urgency,
            "requires_tools": await self._requires_tools(query, context),
            "suggested_tools": await self._suggest_tools(query, context),
            "escalation_needed": await self._needs_escalation(query, context),
            "entities": self._extract_entities(query),
            "keywords": self._extract_keywords(query),
            "sentiment": self._analyze_sentiment(query),
            "language": self._detect_language(query)
        }
        
        return analysis
    
    async def _retrieve_knowledge(
        self,
        query: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge using hybrid vector system."""
        try:
            # Create query vector (this would use your embedding model)
            query_vector = await self._create_query_vector(query)
            
            # Search with agent-specific parameters
            search_result = await self.hybrid_vector_system.hybrid_search(
                query_vector=query_vector,
                agent_id=self.agent_id,
                top_k=self._get_retrieval_count(analysis),
                similarity_threshold=self._get_similarity_threshold(analysis),
                filters=self._build_search_filters(context, analysis)
            )
            
            if search_result and search_result.vectors:
                logger.debug(f"Retrieved {len(search_result.vectors)} knowledge sources "
                           f"in {search_result.search_time_ms:.2f}ms from {search_result.tier_used}")
                return search_result.vectors
            
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []
    
    async def _execute_tools(
        self,
        query: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[ToolResult]:
        """Execute required tools based on query analysis."""
        if not self.tool_orchestrator:
            return []
        
        tool_results = []
        suggested_tools = analysis.get("suggested_tools", [])
        
        for tool_name in suggested_tools:
            if tool_name in self.available_tools:
                try:
                    # Prepare tool parameters
                    tool_params = self._prepare_tool_parameters(tool_name, query, context, analysis)
                    
                    # Execute tool with timeout
                    result = await asyncio.wait_for(
                        self.tool_orchestrator.execute_tool(tool_name, tool_params),
                        timeout=self.tool_execution_timeout
                    )
                    
                    tool_results.append(result)
                    
                    if result.success:
                        logger.debug(f"Tool {tool_name} executed successfully")
                    else:
                        logger.warning(f"Tool {tool_name} failed: {result.error}")
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Tool {tool_name} timed out after {self.tool_execution_timeout}s")
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
        
        return tool_results
    
    @abstractmethod
    async def _generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        knowledge_context: List[Dict[str, Any]],
        tool_results: List[ToolResult],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate agent-specific response. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _detect_intent(self, query: str, context: Dict[str, Any]) -> str:
        """Detect query intent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _requires_tools(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if tools are required. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _suggest_tools(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Suggest appropriate tools. Must be implemented by subclasses."""
        pass
    
    async def _load_knowledge_base(self):
        """Load agent-specific knowledge base."""
        # This would be implemented to load agent-specific documents
        # into the hybrid vector system under the agent's namespace
        logger.info(f"Loading knowledge base for agent {self.agent_id}")
    
    async def _initialize_specialized_components(self):
        """Initialize agent-specific components. Override in subclasses."""
        pass
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on agent configuration."""
        specialization = self.specialization
        personality = specialization.get("personality_profile", "professional")
        domain = specialization.get("domain_expertise", "general")
        
        base_prompt = f"""You are a specialized {domain} assistant with a {personality} personality.
        
Your capabilities include:
- Expert knowledge in {domain}
- Access to specialized tools and workflows
- Voice-optimized responses for clear communication
- Context-aware conversation management

Guidelines:
- Provide accurate, helpful responses based on your expertise
- Keep responses concise and natural for voice interaction
- Use your specialized tools when appropriate
- Maintain your {personality} tone throughout conversations
- Escalate complex issues when necessary"""
        
        return base_prompt
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates for common scenarios."""
        return {
            "greeting": "Hello! I'm here to help with {domain} questions.",
            "clarification": "Could you please provide more details about {specific_aspect}?",
            "tool_execution": "I'm {action} for you. Please wait a moment...",
            "escalation": "This requires specialized attention. Let me connect you with an expert.",
            "error": "I apologize, but I encountered an issue. Let me try a different approach.",
            "completion": "I've {completed_action}. Is there anything else I can help you with?"
        }
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key for query and context."""
        import hashlib
        
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Include relevant context elements
        context_key = str(sorted(context.items())) if context else ""
        
        # Create hash
        cache_string = f"{self.agent_id}:{normalized_query}:{context_key}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[AgentResponse]:
        """Get cached response if available and not expired."""
        if cache_key not in self.response_cache:
            return None
        
        cached_entry = self.response_cache[cache_key]
        
        # Check expiration
        if time.time() - cached_entry["timestamp"] > self.cache_ttl:
            del self.response_cache[cache_key]
            return None
        
        return cached_entry["response"]
    
    def _cache_response(self, cache_key: str, response: AgentResponse):
        """Cache response for future use."""
        # Manage cache size
        if len(self.response_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k]["timestamp"]
            )
            del self.response_cache[oldest_key]
        
        # Cache the response
        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
    
    def _calculate_confidence(
        self,
        response: str,
        knowledge_context: List[Dict[str, Any]],
        tool_results: List[ToolResult]
    ) -> float:
        """Calculate confidence score for the response."""
        confidence = 0.5  # Base confidence
        
        # Add confidence based on knowledge sources
        if knowledge_context:
            knowledge_scores = [ctx.get("score", 0.0) for ctx in knowledge_context]
            if knowledge_scores:
                avg_knowledge_score = sum(knowledge_scores) / len(knowledge_scores)
                confidence += 0.3 * avg_knowledge_score
        
        # Add confidence based on successful tool executions
        successful_tools = [r for r in tool_results if r.success]
        if tool_results:
            tool_success_rate = len(successful_tools) / len(tool_results)
            confidence += 0.2 * tool_success_rate
        
        # Response quality factors
        if len(response) > 10:  # Not too short
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _update_stats(self, response: AgentResponse):
        """Update agent performance statistics."""
        self.stats.successful_responses += 1
        
        # Update average response time
        if self.stats.average_response_time_ms == 0:
            self.stats.average_response_time_ms = response.processing_time_ms
        else:
            total_responses = self.stats.successful_responses
            self.stats.average_response_time_ms = (
                (self.stats.average_response_time_ms * (total_responses - 1) + response.processing_time_ms) /
                total_responses
            )
        
        # Update average confidence
        if self.stats.average_confidence == 0:
            self.stats.average_confidence = response.confidence
        else:
            self.stats.average_confidence = (
                (self.stats.average_confidence * (total_responses - 1) + response.confidence) /
                total_responses
            )
        
        # Update tool execution count
        self.stats.tools_executed += len(response.tools_executed)
        
        # Update escalation count
        if response.escalation_needed:
            self.stats.escalations += 1
        
        # Calculate error rate
        total_queries = self.stats.total_queries
        self.stats.error_rate = self.stats.failed_responses / total_queries if total_queries > 0 else 0
    
    async def _generate_error_response(
        self,
        query: str,
        error_message: str,
        processing_time: float
    ) -> AgentResponse:
        """Generate error response for failed queries."""
        domain = self.specialization.get("domain_expertise", "general")
        
        error_responses = [
            f"I apologize, but I encountered an issue processing your {domain} request.",
            f"Let me try to help you with your {domain} question in a different way.",
            "I'm having some technical difficulties. Could you please rephrase your question?"
        ]
        
        # Choose response based on error type
        if "timeout" in error_message.lower():
            response_text = "I'm sorry, but that request is taking longer than expected. Please try again."
        elif "not found" in error_message.lower():
            response_text = f"I don't have information about that specific {domain} topic in my knowledge base."
        else:
            response_text = error_responses[0]
        
        return AgentResponse(
            response=response_text,
            confidence=0.3,
            agent_id=self.agent_id,
            processing_time_ms=processing_time,
            escalation_needed=True,
            metadata={"error": error_message}
        )
    
    # Helper methods for query analysis
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        # Simple complexity metrics
        word_count = len(query.split())
        question_words = sum(1 for word in query.lower().split() 
                           if word in ["what", "how", "why", "when", "where", "which", "who"])
        
        complexity = min(1.0, (word_count / 20) + (question_words * 0.2))
        return complexity
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        # Simple entity extraction (could be enhanced with NLP models)
        import re
        
        # Extract capitalized words as potential entities
        entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _analyze_sentiment(self, query: str) -> str:
        """Analyze sentiment of query."""
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'love', 'like', 'happy', 'pleased']
        negative_words = ['bad', 'terrible', 'hate', 'dislike', 'angry', 'frustrated', 'problem']
        
        query_lower = query.lower()
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        
        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"
    
    def _detect_language(self, query: str) -> str:
        """Detect query language."""
        # Simple language detection (could be enhanced)
        return "en"  # Default to English
    
    async def _create_query_vector(self, query: str) -> np.ndarray:
        """Create vector representation of query."""
        # This would use your embedding model
        # For now, returning a placeholder vector
        return np.random.rand(1536).astype(np.float32)
    
    def _get_retrieval_count(self, analysis: Dict[str, Any]) -> int:
        """Get number of documents to retrieve based on query analysis."""
        complexity = analysis.get("complexity", 0.5)
        urgency = analysis.get("urgency", UrgencyLevel.NORMAL)
        
        if urgency == UrgencyLevel.EMERGENCY:
            return 2  # Fewer documents for faster response
        elif complexity > 0.7:
            return 5  # More documents for complex queries
        else:
            return 3  # Default
    
    def _get_similarity_threshold(self, analysis: Dict[str, Any]) -> float:
        """Get similarity threshold based on query analysis."""
        complexity = analysis.get("complexity", 0.5)
        
        if complexity > 0.8:
            return 0.6  # Lower threshold for complex queries
        else:
            return 0.7  # Default threshold
    
    def _build_search_filters(
        self,
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build search filters based on context and analysis."""
        filters = {}
        
        # Add domain-specific filters
        domain = self.specialization.get("domain_expertise")
        if domain:
            filters["domain"] = domain
        
        # Add urgency-based filters
        urgency = analysis.get("urgency")
        if urgency == UrgencyLevel.EMERGENCY:
            filters["priority"] = "high"
        
        return filters if filters else None
    
    def _prepare_tool_parameters(
        self,
        tool_name: str,
        query: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare parameters for tool execution."""
        return {
            "query": query,
            "context": context,
            "agent_id": self.agent_id,
            "urgency": analysis.get("urgency", UrgencyLevel.NORMAL),
            "entities": analysis.get("entities", []),
            "intent": analysis.get("intent", "unknown")
        }
    
    async def _needs_escalation(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if query needs human escalation."""
        # Check for escalation keywords
        escalation_keywords = ["speak to human", "manager", "supervisor", "complaint", "legal"]
        query_lower = query.lower()
        
        return any(keyword in query_lower for keyword in escalation_keywords)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        uptime = time.time() - self.deployment_time
        
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "uptime_seconds": uptime,
            "initialized": self.initialized,
            "avg_response_time_ms": self.stats.average_response_time_ms,
            "success_rate": (
                self.stats.successful_responses / max(self.stats.total_queries, 1)
            ) * 100,
            "error_rate": self.stats.error_rate * 100,
            "cache_hit_rate": (
                self.stats.cache_hits / max(self.stats.total_queries, 1)
            ) * 100,
            "performance_target_met": self.stats.average_response_time_ms <= self.target_response_time_ms
        }
    
    def get_stats(self) -> AgentStats:
        """Get comprehensive agent statistics."""
        self.stats.uptime_seconds = time.time() - self.deployment_time
        return self.stats
    
    async def shutdown(self):
        """Shutdown agent and cleanup resources."""
        logger.info(f"Shutting down agent {self.agent_id}")
        
        self.status = AgentStatus.INACTIVE
        self.initialized = False
        
        # Clear cache
        self.response_cache.clear()
        
        logger.info(f"✅ Agent {self.agent_id} shutdown complete")