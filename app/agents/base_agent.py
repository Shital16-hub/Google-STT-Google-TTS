"""
Base Agent Framework for Multi-Agent Voice AI System.
Provides foundation for specialized agents with tool integration and performance optimization.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from langchain_core.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent deployment and operational status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

class AgentCapability(Enum):
    """Agent capabilities for routing decisions."""
    EMERGENCY_RESPONSE = "emergency_response"
    PAYMENT_PROCESSING = "payment_processing"
    TECHNICAL_SUPPORT = "technical_support"
    CUSTOMER_SERVICE = "customer_service"
    MULTILINGUAL = "multilingual"
    AFTER_HOURS = "after_hours"

@dataclass
class AgentMetrics:
    """Performance metrics for agent tracking."""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    avg_confidence: float = 0.0
    tool_usage_count: Dict[str, int] = field(default_factory=dict)
    last_active: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    def update_request_metrics(self, success: bool, response_time: float, confidence: float = 0.0):
        """Update request metrics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update running averages
        if self.total_requests == 1:
            self.avg_response_time = response_time
            self.avg_confidence = confidence
        else:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + response_time) / 
                self.total_requests
            )
            if confidence > 0:
                self.avg_confidence = (
                    (self.avg_confidence * (self.total_requests - 1) + confidence) / 
                    self.total_requests
                )
        
        self.last_active = time.time()

@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    user_message: Optional[str] = None  # Message to show to user
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "user_message": self.user_message
        }

class BaseAgent(ABC):
    """
    Base class for all specialized agents in the multi-agent system.
    
    Provides common functionality:
    - LLM integration with conversation memory
    - Tool orchestration and execution
    - Performance monitoring and metrics
    - Hot deployment and configuration management
    - Vector store integration for context retrieval
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_config: Dict[str, Any],
        vector_store=None,
        performance_tracker=None
    ):
        """
        Initialize base agent with common functionality.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_config: Agent configuration dictionary
            vector_store: Vector store for context retrieval
            performance_tracker: Performance tracking system
        """
        self.agent_id = agent_id
        self.config = agent_config
        self.vector_store = vector_store
        self.performance_tracker = performance_tracker
        
        # Agent metadata
        self.version = self.config.get("version", "1.0.0")
        self.status = AgentStatus.INITIALIZING
        self.capabilities = set()
        
        # LLM and conversation components
        self.llm = None
        self.tools = {}
        self.system_prompt = self.config.get("specialization", {}).get("system_prompt", "")
        
        # Performance tracking
        self.metrics = AgentMetrics(agent_id=agent_id)
        self.conversation_history = []
        
        # Configuration
        self.max_context_length = self.config.get("specialization", {}).get("max_context", 2048)
        self.response_style = self.config.get("specialization", {}).get("response_style", "professional")
        
        logger.info(f"Base agent initialized: {agent_id} v{self.version}")
    
    async def init(self):
        """Initialize agent components."""
        try:
            # Initialize LLM
            await self._init_llm()
            
            # Initialize tools
            await self._init_tools()
            
            # Load capabilities
            self._load_capabilities()
            
            # Set up vector collection if specified
            if self.vector_store and "vector_config" in self.config:
                await self._init_vector_context()
            
            self.status = AgentStatus.ACTIVE
            logger.info(f"✅ Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"❌ Failed to initialize agent {self.agent_id}: {e}")
            raise
    
    async def _init_llm(self):
        """Initialize the language model for this agent."""
        # Use OpenAI with agent-specific settings
        model_config = self.config.get("llm_config", {})
        
        self.llm = ChatOpenAI(
            model=model_config.get("model", "gpt-4o-mini"),
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_tokens", 512),
            streaming=model_config.get("streaming", True)
        )
        
        logger.debug(f"LLM initialized for agent {self.agent_id}")
    
    async def _init_tools(self):
        """Initialize tools specific to this agent."""
        tools_config = self.config.get("tools", [])
        
        for tool_config in tools_config:
            try:
                tool = await self._create_tool(tool_config)
                self.tools[tool_config["name"]] = tool
                logger.debug(f"Tool initialized: {tool_config['name']}")
            except Exception as e:
                logger.error(f"Failed to initialize tool {tool_config.get('name')}: {e}")
    
    async def _create_tool(self, tool_config: Dict[str, Any]) -> Tool:
        """Create a tool from configuration."""
        tool_name = tool_config["name"]
        tool_type = tool_config["type"]
        
        if tool_type == "api_call":
            return await self._create_api_tool(tool_config)
        elif tool_type == "database_query":
            return await self._create_database_tool(tool_config)
        elif tool_type == "external_service":
            return await self._create_external_service_tool(tool_config)
        else:
            raise ValueError(f"Unknown tool type: {tool_type}")
    
    async def _create_api_tool(self, tool_config: Dict[str, Any]) -> Tool:
        """Create an API-based tool."""
        import aiohttp
        
        async def api_tool_func(query: str = "") -> str:
            """Execute API call tool."""
            try:
                endpoint = tool_config["endpoint"]
                method = tool_config.get("method", "POST")
                auth = tool_config.get("auth")
                timeout = tool_config.get("timeout", 5.0)
                
                headers = {}
                if auth == "bearer_token":
                    token = tool_config.get("token")
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                
                async with aiohttp.ClientSession() as session:
                    if method.upper() == "POST":
                        data = {"query": query, "agent_id": self.agent_id}
                        async with session.post(
                            endpoint, 
                            json=data, 
                            headers=headers,
                            timeout=timeout
                        ) as resp:
                            result = await resp.json()
                            return str(result)
                    else:
                        params = {"query": query}
                        async with session.get(
                            endpoint, 
                            params=params, 
                            headers=headers,
                            timeout=timeout
                        ) as resp:
                            result = await resp.json()
                            return str(result)
                            
            except Exception as e:
                return f"API call failed: {str(e)}"
        
        return Tool(
            name=tool_config["name"],
            description=tool_config.get("description", f"API tool for {tool_config['name']}"),
            func=api_tool_func
        )
    
    async def _create_database_tool(self, tool_config: Dict[str, Any]) -> Tool:
        """Create a database query tool."""
        async def db_tool_func(query: str) -> str:
            """Execute database query tool."""
            # This would integrate with your database
            # For now, return a placeholder
            return f"Database query executed for: {query}"
        
        return Tool(
            name=tool_config["name"],
            description=tool_config.get("description", f"Database tool for {tool_config['name']}"),
            func=db_tool_func
        )
    
    async def _create_external_service_tool(self, tool_config: Dict[str, Any]) -> Tool:
        """Create an external service integration tool."""
        service = tool_config.get("service")
        
        if service == "google_maps_api":
            return await self._create_maps_tool(tool_config)
        elif service == "weather_api":
            return await self._create_weather_tool(tool_config)
        else:
            # Generic external service tool
            async def external_tool_func(query: str) -> str:
                return f"External service call to {service}: {query}"
            
            return Tool(
                name=tool_config["name"],
                description=tool_config.get("description", f"External service: {service}"),
                func=external_tool_func
            )
    
    async def _create_maps_tool(self, tool_config: Dict[str, Any]) -> Tool:
        """Create Google Maps integration tool."""
        async def maps_tool_func(location: str) -> str:
            """Calculate ETA or get location information."""
            # This would integrate with Google Maps API
            # For demonstration, return estimated response
            return f"ETA to {location}: approximately 15-25 minutes"
        
        return Tool(
            name=tool_config["name"],
            description="Calculate ETA and provide location information",
            func=maps_tool_func
        )
    
    async def _create_weather_tool(self, tool_config: Dict[str, Any]) -> Tool:
        """Create weather API integration tool."""
        async def weather_tool_func(location: str) -> str:
            """Get weather information for location."""
            # This would integrate with weather API
            return f"Weather in {location}: partly cloudy, 72°F"
        
        return Tool(
            name=tool_config["name"],
            description="Get current weather information",
            func=weather_tool_func
        )
    
    def _load_capabilities(self):
        """Load agent capabilities from configuration."""
        capabilities_config = self.config.get("capabilities", [])
        
        for capability in capabilities_config:
            if hasattr(AgentCapability, capability.upper()):
                self.capabilities.add(AgentCapability[capability.upper()])
    
    async def _init_vector_context(self):
        """Initialize vector store context for this agent."""
        vector_config = self.config.get("vector_config", {})
        collection_name = vector_config.get("qdrant_collection")
        
        if collection_name and self.vector_store:
            # Ensure agent's vector collection is available in hot tier
            try:
                if hasattr(self.vector_store, 'faiss_hot_tier'):
                    await self.vector_store.faiss_hot_tier.create_agent_index(
                        agent_id=self.agent_id,
                        documents=[]  # Will be populated from Qdrant
                    )
                logger.debug(f"Vector context initialized for {self.agent_id}")
            except Exception as e:
                logger.warning(f"Could not initialize vector context: {e}")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user request through this agent.
        
        Args:
            request: Request containing user input and context
            
        Returns:
            Agent response with tools results and metadata
        """
        start_time = time.time()
        
        try:
            # Extract request components
            user_input = request.get("user_input", "")
            context = request.get("context", [])
            conversation_history = request.get("conversation_history", [])
            
            # Get relevant context from vector store
            if self.vector_store:
                relevant_context = await self._retrieve_context(user_input)
                context.extend(relevant_context)
            
            # Determine if tools are needed
            tool_calls = await self._analyze_tool_requirements(user_input, context)
            
            # Execute tools if needed
            tool_results = []
            if tool_calls:
                tool_results = await self._execute_tools(tool_calls, user_input)
            
            # Generate response
            response = await self._generate_response(
                user_input=user_input,
                context=context,
                conversation_history=conversation_history,
                tool_results=tool_results
            )
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.update_request_metrics(
                success=True,
                response_time=processing_time,
                confidence=response.get("confidence", 0.8)
            )
            
            # Track performance
            if self.performance_tracker:
                await self.performance_tracker.track_agent_request(
                    agent_id=self.agent_id,
                    processing_time=processing_time,
                    tool_count=len(tool_calls),
                    success=True
                )
            
            return {
                "response": response.get("text", ""),
                "confidence": response.get("confidence", 0.8),
                "tool_calls": [result.to_dict() for result in tool_results],
                "processing_time": processing_time,
                "agent_id": self.agent_id,
                "requires_escalation": await self._check_escalation_needed(user_input, response),
                "needs_human_handoff": response.get("needs_human", False)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update error metrics
            self.metrics.update_request_metrics(
                success=False,
                response_time=processing_time
            )
            
            logger.error(f"Error processing request in agent {self.agent_id}: {e}")
            
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again or let me transfer you to a human agent.",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": processing_time,
                "agent_id": self.agent_id,
                "needs_human_handoff": True
            }
    
    async def _retrieve_context(self, user_input: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector store."""
        try:
            if hasattr(self.vector_store, 'hybrid_search'):
                results = await self.vector_store.hybrid_search(
                    query=user_input,
                    agent_id=self.agent_id,
                    top_k=3,
                    hybrid_alpha=0.7
                )
                return [
                    {
                        "content": result.content,
                        "metadata": result.metadata,
                        "score": result.score
                    }
                    for result in results
                ]
        except Exception as e:
            logger.warning(f"Error retrieving context: {e}")
        
        return []
    
    @abstractmethod
    async def _analyze_tool_requirements(
        self, 
        user_input: str, 
        context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze user input to determine required tools.
        Must be implemented by each specialized agent.
        """
        pass
    
    async def _execute_tools(
        self, 
        tool_calls: List[Dict[str, Any]], 
        user_input: str
    ) -> List[ToolExecutionResult]:
        """Execute the required tools."""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            tool_input = tool_call.get("input", user_input)
            
            if tool_name in self.tools:
                result = await self.execute_tool(tool_name, tool_input)
                results.append(result)
            else:
                results.append(ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"Tool {tool_name} not available"
                ))
        
        return results
    
    async def execute_tool(
        self, 
        tool_name: str, 
        tool_input: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """
        Execute a specific tool.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input for the tool
            context: Additional context for tool execution
            
        Returns:
            Tool execution result
        """
        start_time = time.time()
        
        try:
            if tool_name not in self.tools:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"Tool {tool_name} not found"
                )
            
            tool = self.tools[tool_name]
            
            # Execute tool with timeout
            try:
                if isinstance(tool_input, dict):
                    result = await asyncio.wait_for(
                        tool.arun(**tool_input),
                        timeout=LatencyConfig.TARGET_TOOL_TIME
                    )
                else:
                    result = await asyncio.wait_for(
                        tool.arun(tool_input),
                        timeout=LatencyConfig.TARGET_TOOL_TIME
                    )
                
                execution_time = time.time() - start_time
                
                # Update tool usage metrics
                if tool_name not in self.metrics.tool_usage_count:
                    self.metrics.tool_usage_count[tool_name] = 0
                self.metrics.tool_usage_count[tool_name] += 1
                
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    user_message=self._format_tool_result_for_user(tool_name, result)
                )
                
            except asyncio.TimeoutError:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error="Tool execution timed out",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _format_tool_result_for_user(self, tool_name: str, result: Any) -> str:
        """Format tool result for user presentation."""
        # This can be overridden by specialized agents
        return f"Tool {tool_name} executed successfully: {str(result)[:100]}"
    
    @abstractmethod
    async def _generate_response(
        self,
        user_input: str,
        context: List[Dict[str, Any]],
        conversation_history: List[BaseMessage],
        tool_results: List[ToolExecutionResult]
    ) -> Dict[str, Any]:
        """
        Generate response using LLM with context and tool results.
        Must be implemented by each specialized agent.
        """
        pass
    
    async def _check_escalation_needed(
        self, 
        user_input: str, 
        response: Dict[str, Any]
    ) -> bool:
        """Check if request needs escalation to human or different agent."""
        # Check confidence threshold
        if response.get("confidence", 1.0) < 0.6:
            return True
        
        # Check for escalation keywords
        escalation_keywords = [
            "manager", "supervisor", "complaint", "angry", "frustrated",
            "legal", "lawsuit", "emergency", "urgent"
        ]
        
        user_input_lower = user_input.lower()
        if any(keyword in user_input_lower for keyword in escalation_keywords):
            return True
        
        return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "total_requests": self.metrics.total_requests,
            "success_rate": self.metrics.success_rate,
            "avg_response_time": self.metrics.avg_response_time,
            "avg_confidence": self.metrics.avg_confidence,
            "tool_usage": self.metrics.tool_usage_count,
            "last_active": self.metrics.last_active,
            "tools_available": list(self.tools.keys())
        }
    
    async def health_check(self) -> bool:
        """Perform agent health check."""
        try:
            # Check if LLM is responsive
            if self.llm:
                test_response = await self.llm.ainvoke([HumanMessage(content="test")])
                if not test_response:
                    return False
            
            # Check tool availability
            for tool_name, tool in self.tools.items():
                if not tool:
                    logger.warning(f"Tool {tool_name} is not available")
                    return False
            
            # Check vector store connection if configured
            if self.vector_store and hasattr(self.vector_store, 'health_check'):
                if not await self.vector_store.health_check():
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for agent {self.agent_id}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown agent gracefully."""
        logger.info(f"🛑 Shutting down agent {self.agent_id}")
        
        self.status = AgentStatus.SHUTTING_DOWN
        
        # Clean up resources
        try:
            # Close any open connections
            if self.tools:
                for tool_name, tool in self.tools.items():
                    if hasattr(tool, 'cleanup'):
                        await tool.cleanup()
            
            # Clear conversation history
            self.conversation_history.clear()
            
            logger.info(f"✅ Agent {self.agent_id} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")