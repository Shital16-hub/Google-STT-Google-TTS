"""
Multi-Agent Orchestrator using LangGraph for intelligent agent coordination
Implements hot deployment, intelligent routing, and performance optimization
"""
import os
import yaml
import json
import time
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from pathlib import Path
from dataclasses import dataclass, field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

# Custom imports
from app.core.conversation_manager import EnhancedConversationManager
from app.core.latency_optimizer import LatencyOptimizer
from app.vector_db.hybrid_vector_store import HybridVectorStore
from app.agents.intelligent_router import IntelligentRouter
from app.agents.base_agent import BaseAgent
from app.tools.tool_orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """Enhanced conversation state for LangGraph"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_agent: Optional[str] = None
    agent_history: List[str] = field(default_factory=list)
    session_id: str = ""
    call_sid: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    latency_budget: float = 650.0  # ms target
    error_count: int = 0
    conversation_turn: int = 0

@dataclass
class AgentDeploymentResult:
    """Result of agent deployment"""
    success: bool
    agent_id: str
    deployment_time: float
    error: Optional[str] = None
    health_check: bool = False

class MultiAgentOrchestrator:
    """
    LangGraph-based multi-agent orchestrator with hot deployment and intelligent routing
    """
    
    # Performance targets from transformation plan
    TARGET_ROUTING_LATENCY = 20  # ms
    TARGET_RETRIEVAL_LATENCY = 50  # ms
    TARGET_TOTAL_LATENCY = 650  # ms
    
    def __init__(
        self,
        latency_optimizer: Optional[LatencyOptimizer] = None,
        credentials_file: Optional[str] = None
    ):
        """Initialize the multi-agent orchestrator"""
        self.latency_optimizer = latency_optimizer
        self.credentials_file = credentials_file
        
        # Core components
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.routing_engine: Optional[IntelligentRouter] = None
        self.tool_orchestrator: Optional[ToolOrchestrator] = None
        self.hybrid_vector_store: Optional[HybridVectorStore] = None
        
        # LangGraph workflow
        self.workflow: Optional[StateGraph] = None
        self.compiled_graph = None
        self.memory_saver = MemorySaver()
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_latency": 0.0,
            "agent_usage": {},
            "routing_accuracy": 0.0,
            "tool_usage": {}
        }
        
        # Session management
        self.active_sessions: Dict[str, EnhancedConversationManager] = {}
        
        self.initialized = False
        
        logger.info("🎯 Multi-Agent Orchestrator initialized")
    
    async def init(self):
        """Initialize all orchestrator components"""
        logger.info("🔄 Initializing Multi-Agent Orchestrator...")
        
        try:
            # Initialize hybrid vector store
            self.hybrid_vector_store = HybridVectorStore()
            await self.hybrid_vector_store.init()
            
            # Initialize intelligent router
            self.routing_engine = IntelligentRouter(
                vector_store=self.hybrid_vector_store,
                target_latency_ms=self.TARGET_ROUTING_LATENCY
            )
            await self.routing_engine.init()
            
            # Initialize tool orchestrator
            self.tool_orchestrator = ToolOrchestrator(
                credentials_file=self.credentials_file
            )
            await self.tool_orchestrator.init()
            
            # Build LangGraph workflow
            await self._build_orchestration_graph()
            
            self.initialized = True
            logger.info("✅ Multi-Agent Orchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            raise
    
    async def _build_orchestration_graph(self):
        """Build the LangGraph orchestration workflow"""
        logger.info("🔧 Building LangGraph orchestration workflow...")
        
        # Create the state graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes for each step in the pipeline
        workflow.add_node("route_agent", self._route_to_agent)
        workflow.add_node("execute_agent", self._execute_with_agent)
        workflow.add_node("tool_orchestration", self._orchestrate_tools)
        workflow.add_node("response_synthesis", self._synthesize_response)
        workflow.add_node("error_handling", self._handle_errors)
        
        # Define the flow
        workflow.add_edge("route_agent", "execute_agent")
        workflow.add_edge("execute_agent", "tool_orchestration")
        workflow.add_edge("tool_orchestration", "response_synthesis")
        workflow.add_edge("response_synthesis", END)
        workflow.add_edge("error_handling", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "route_agent",
            self._should_handle_error,
            {
                "continue": "execute_agent",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_agent",
            self._should_handle_error,
            {
                "continue": "tool_orchestration",
                "error": "error_handling"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("route_agent")
        
        # Compile the workflow
        self.workflow = workflow
        self.compiled_graph = workflow.compile(checkpointer=self.memory_saver)
        
        logger.info("✅ LangGraph workflow compiled successfully")
    
    async def _route_to_agent(self, state: ConversationState) -> ConversationState:
        """Route user input to the appropriate agent"""
        start_time = time.time()
        
        try:
            # Get latest user message
            if not state.messages:
                raise ValueError("No messages to route")
            
            user_message = state.messages[-1]["content"]
            
            # Intelligent routing with performance tracking
            routing_result = await self.routing_engine.route_query(
                query=user_message,
                conversation_history=state.messages[-5:],  # Last 5 messages for context
                current_agent=state.current_agent
            )
            
            # Update state
            selected_agent = routing_result["agent_id"]
            confidence = routing_result["confidence"]
            
            # Agent transition logic
            if state.current_agent != selected_agent:
                state.agent_history.append(state.current_agent or "none")
                state.current_agent = selected_agent
                
                logger.info(f"🔄 Agent transition: {state.current_agent} → {selected_agent} (confidence: {confidence:.2f})")
            
            # Update context
            state.context.update({
                "routing_confidence": confidence,
                "routing_reasoning": routing_result.get("reasoning", ""),
                "agent_capabilities": self.active_agents.get(selected_agent, {}).get("capabilities", [])
            })
            
            # Track performance
            routing_time = (time.time() - start_time) * 1000
            if self.latency_optimizer:
                await self.latency_optimizer.record_processing_time(
                    state.session_id, "agent_routing", routing_time
                )
            
            # Validate routing latency
            if routing_time > self.TARGET_ROUTING_LATENCY:
                logger.warning(f"⚠️ Routing latency exceeded target: {routing_time:.1f}ms > {self.TARGET_ROUTING_LATENCY}ms")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Agent routing error: {e}")
            state.error_count += 1
            state.context["routing_error"] = str(e)
            return state
    
    async def _execute_with_agent(self, state: ConversationState) -> ConversationState:
        """Execute query with the selected agent"""
        start_time = time.time()
        
        try:
            agent_id = state.current_agent
            if not agent_id or agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not available")
            
            agent_config = self.active_agents[agent_id]
            
            # Get user message
            user_message = state.messages[-1]["content"]
            
            # Retrieve relevant context with hybrid vector store
            retrieval_start = time.time()
            context_results = await self.hybrid_vector_store.query_agent(
                agent_id=agent_id,
                query=user_message,
                top_k=3
            )
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            # Validate retrieval latency
            if retrieval_time > self.TARGET_RETRIEVAL_LATENCY:
                logger.warning(f"⚠️ Retrieval latency exceeded target: {retrieval_time:.1f}ms > {self.TARGET_RETRIEVAL_LATENCY}ms")
            
            # Prepare agent prompt with context
            system_prompt = agent_config.get("system_prompt", "You are a helpful assistant.")
            formatted_context = self._format_context(context_results)
            
            # Create agent-specific LLM with optimized settings
            llm = ChatOpenAI(
                model=agent_config.get("model", "gpt-4o-mini"),
                temperature=agent_config.get("temperature", 0.7),
                max_tokens=agent_config.get("max_tokens", 256),  # Smaller for voice
                streaming=True
            )
            
            # Build message history
            messages = [
                SystemMessage(content=f"{system_prompt}\n\nContext: {formatted_context}")
            ]
            
            # Add conversation history (last 5 messages)
            for msg in state.messages[-5:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # Generate response
            response = await llm.ainvoke(messages)
            agent_response = response.content
            
            # Update state
            state.messages.append({
                "role": "assistant",
                "content": agent_response,
                "agent_id": agent_id,
                "timestamp": time.time(),
                "context_used": len(context_results)
            })
            
            # Track agent usage
            self.performance_stats["agent_usage"][agent_id] = \
                self.performance_stats["agent_usage"].get(agent_id, 0) + 1
            
            # Update agent performance
            agent_config["total_queries"] = agent_config.get("total_queries", 0) + 1
            agent_config["last_used"] = time.time()
            
            # Track execution time
            execution_time = (time.time() - start_time) * 1000
            if self.latency_optimizer:
                await self.latency_optimizer.record_processing_time(
                    state.session_id, "agent_execution", execution_time
                )
            
            # Store response for tool orchestration
            state.context["agent_response"] = agent_response
            state.context["execution_time"] = execution_time
            state.context["retrieval_time"] = retrieval_time
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Agent execution error: {e}")
            state.error_count += 1
            state.context["execution_error"] = str(e)
            return state
    
    async def _orchestrate_tools(self, state: ConversationState) -> ConversationState:
        """Orchestrate tool usage if needed"""
        start_time = time.time()
        
        try:
            agent_id = state.current_agent
            agent_response = state.context.get("agent_response", "")
            
            # Check if tools are needed based on agent response and configuration
            agent_config = self.active_agents.get(agent_id, {})
            available_tools = agent_config.get("tools", {})
            
            if not available_tools:
                # No tools available, skip orchestration
                state.context["tools_executed"] = []
                return state
            
            # Analyze if tools should be used
            tool_analysis = await self.tool_orchestrator.analyze_tool_needs(
                agent_response=agent_response,
                available_tools=list(available_tools.keys()),
                conversation_context=state.messages[-3:]
            )
            
            if tool_analysis["should_use_tools"]:
                # Execute tools
                tool_results = await self.tool_orchestrator.execute_tools(
                    tools_to_execute=tool_analysis["recommended_tools"],
                    context=state.context,
                    agent_config=agent_config
                )
                
                # Update state with tool results
                state.context["tool_results"] = tool_results
                state.tools_used.extend(tool_analysis["recommended_tools"])
                
                # Track tool usage
                for tool_name in tool_analysis["recommended_tools"]:
                    self.performance_stats["tool_usage"][tool_name] = \
                        self.performance_stats["tool_usage"].get(tool_name, 0) + 1
                
                logger.info(f"🔧 Executed {len(tool_results)} tools for agent {agent_id}")
            else:
                state.context["tool_results"] = []
            
            # Track orchestration time
            orchestration_time = (time.time() - start_time) * 1000
            if self.latency_optimizer:
                await self.latency_optimizer.record_processing_time(
                    state.session_id, "tool_orchestration", orchestration_time
                )
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Tool orchestration error: {e}")
            state.error_count += 1
            state.context["tool_orchestration_error"] = str(e)
            return state
    
    async def _synthesize_response(self, state: ConversationState) -> ConversationState:
        """Synthesize final response incorporating tool results"""
        start_time = time.time()
        
        try:
            agent_response = state.context.get("agent_response", "")
            tool_results = state.context.get("tool_results", [])
            
            if tool_results:
                # Enhance response with tool results
                enhanced_response = await self._enhance_response_with_tools(
                    original_response=agent_response,
                    tool_results=tool_results,
                    agent_id=state.current_agent
                )
                
                # Update the last message with enhanced response
                if state.messages and state.messages[-1]["role"] == "assistant":
                    state.messages[-1]["content"] = enhanced_response
                    state.messages[-1]["tools_used"] = len(tool_results)
            
            # Track synthesis time
            synthesis_time = (time.time() - start_time) * 1000
            if self.latency_optimizer:
                await self.latency_optimizer.record_processing_time(
                    state.session_id, "response_synthesis", synthesis_time
                )
            
            # Update conversation turn
            state.conversation_turn += 1
            
            # Update performance stats
            self.performance_stats["total_queries"] += 1
            if state.error_count == 0:
                self.performance_stats["successful_queries"] += 1
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Response synthesis error: {e}")
            state.error_count += 1
            state.context["synthesis_error"] = str(e)
            return state
    
    async def _handle_errors(self, state: ConversationState) -> ConversationState:
        """Handle errors gracefully"""
        logger.warning(f"⚠️ Handling error for session {state.session_id}")
        
        # Generate fallback response
        fallback_response = "I apologize, but I encountered an issue processing your request. Could you please try rephrasing your question?"
        
        state.messages.append({
            "role": "assistant",
            "content": fallback_response,
            "agent_id": "error_handler",
            "timestamp": time.time(),
            "error_recovery": True
        })
        
        return state
    
    def _should_handle_error(self, state: ConversationState) -> str:
        """Determine if we should handle errors"""
        if state.error_count > 0:
            return "error"
        return "continue"
    
    def _format_context(self, context_results: List[Dict[str, Any]]) -> str:
        """Format context from vector search results"""
        if not context_results:
            return "No specific context available."
        
        formatted_parts = []
        for i, result in enumerate(context_results[:3]):  # Top 3 results
            text = result.get("text", "")
            score = result.get("score", 0)
            source = result.get("source", f"Document {i+1}")
            
            formatted_parts.append(f"[{source}] {text[:200]}...")
        
        return "\n".join(formatted_parts)
    
    async def _enhance_response_with_tools(
        self, 
        original_response: str, 
        tool_results: List[Dict[str, Any]], 
        agent_id: str
    ) -> str:
        """Enhance response with tool execution results"""
        if not tool_results:
            return original_response
        
        # Create enhancement prompt
        enhancement_prompt = f"""
        Original response: {original_response}
        
        Tool execution results:
        {json.dumps(tool_results, indent=2)}
        
        Please enhance the original response by incorporating the tool results naturally.
        Keep the response concise and conversational for voice interaction.
        """
        
        # Use a small model for enhancement to maintain latency
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=200
        )
        
        messages = [SystemMessage(content=enhancement_prompt)]
        response = await llm.ainvoke(messages)
        
        return response.content
    
    async def process_conversation_turn(
        self, 
        session_id: str, 
        user_input: str, 
        call_sid: str = ""
    ) -> Dict[str, Any]:
        """Process a complete conversation turn through the orchestration graph"""
        start_time = time.time()
        
        try:
            # Create or update conversation state
            state = ConversationState(
                messages=[{"role": "user", "content": user_input, "timestamp": time.time()}],
                session_id=session_id,
                call_sid=call_sid
            )
            
            # Get existing state if available
            config = {"configurable": {"thread_id": session_id}}
            
            try:
                # Try to get existing state
                existing_state = await self.compiled_graph.aget_state(config)
                if existing_state and existing_state.values:
                    # Merge with existing state
                    current_state = existing_state.values
                    current_state.messages.append(state.messages[0])
                    current_state.conversation_turn += 1
                    state = current_state
            except:
                # New conversation
                pass
            
            # Execute the workflow
            result = await self.compiled_graph.ainvoke(state, config)
            
            # Extract final response
            final_response = ""
            if result.messages:
                for msg in reversed(result.messages):
                    if msg.get("role") == "assistant":
                        final_response = msg.get("content", "")
                        break
            
            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000
            
            # Check against target
            if total_latency > self.TARGET_TOTAL_LATENCY:
                logger.warning(f"⚠️ Total latency exceeded target: {total_latency:.1f}ms > {self.TARGET_TOTAL_LATENCY}ms")
            
            # Update performance stats
            self._update_average_latency(total_latency)
            
            return {
                "response": final_response,
                "agent_used": result.current_agent,
                "tools_used": result.tools_used,
                "total_latency": total_latency,
                "context": result.context,
                "success": result.error_count == 0
            }
            
        except Exception as e:
            logger.error(f"❌ Conversation processing error: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "error": str(e),
                "success": False,
                "total_latency": (time.time() - start_time) * 1000
            }
    
    def _update_average_latency(self, latency: float):
        """Update rolling average latency"""
        current_avg = self.performance_stats["average_latency"]
        total_queries = self.performance_stats["total_queries"]
        
        if total_queries == 0:
            self.performance_stats["average_latency"] = latency
        else:
            # Weighted average with more weight on recent measurements
            alpha = 0.1  # Smoothing factor
            self.performance_stats["average_latency"] = \
                (1 - alpha) * current_avg + alpha * latency
    
    async def load_agents_from_config(self, config_dir: str):
        """Load and deploy agents from configuration directory"""
        logger.info(f"📂 Loading agents from {config_dir}")
        
        config_path = Path(config_dir)
        if not config_path.exists():
            logger.warning(f"⚠️ Config directory {config_dir} does not exist")
            return
        
        # Load YAML configuration files
        for config_file in config_path.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    agent_config = yaml.safe_load(f)
                
                agent_id = agent_config.get("agent_id")
                if not agent_id:
                    logger.error(f"❌ No agent_id in {config_file}")
                    continue
                
                # Deploy the agent
                await self.deploy_agent(agent_id, agent_config)
                
            except Exception as e:
                logger.error(f"❌ Error loading {config_file}: {e}")
    
    async def deploy_agent(self, agent_id: str, config: Dict[str, Any]) -> AgentDeploymentResult:
        """Hot deploy a new agent without system restart"""
        start_time = time.time()
        
        logger.info(f"🚀 Deploying agent: {agent_id}")
        
        try:
            # Validate configuration
            required_fields = ["agent_id", "system_prompt"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Initialize vector collection for agent if needed
            if "knowledge_sources" in config:
                await self.hybrid_vector_store.create_agent_collection(
                    agent_id=agent_id,
                    knowledge_sources=config["knowledge_sources"]
                )
            
            # Store agent configuration
            self.agent_configs[agent_id] = config
            
            # Initialize agent with performance tracking
            agent_instance = {
                "config": config,
                "status": "active",
                "version": config.get("version", "1.0.0"),
                "deployment_time": start_time,
                "total_queries": 0,
                "successful_queries": 0,
                "average_latency": 0.0,
                "last_used": None
            }
            
            # Add to active agents
            self.active_agents[agent_id] = agent_instance
            
            # Update routing engine
            await self.routing_engine.add_agent(agent_id, config)
            
            # Health check
            health_check = await self._health_check_agent(agent_id)
            
            deployment_time = time.time() - start_time
            
            logger.info(f"✅ Agent {agent_id} deployed successfully in {deployment_time:.2f}s")
            
            return AgentDeploymentResult(
                success=True,
                agent_id=agent_id,
                deployment_time=deployment_time,
                health_check=health_check
            )
            
        except Exception as e:
            logger.error(f"❌ Agent deployment failed for {agent_id}: {e}")
            return AgentDeploymentResult(
                success=False,
                agent_id=agent_id,
                deployment_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _health_check_agent(self, agent_id: str) -> bool:
        """Perform health check on deployed agent"""
        try:
            # Test query
            test_result = await self.process_conversation_turn(
                session_id=f"health_check_{agent_id}_{int(time.time())}",
                user_input="Hello, this is a health check."
            )
            
            return test_result.get("success", False)
            
        except Exception as e:
            logger.error(f"❌ Health check failed for {agent_id}: {e}")
            return False
    
    async def create_conversation_handler(
        self, 
        call_sid: str, 
        session_id: str, 
        websocket
    ) -> EnhancedConversationManager:
        """Create an enhanced conversation handler for a session"""
        handler = EnhancedConversationManager(
            orchestrator=self,
            call_sid=call_sid,
            session_id=session_id,
            websocket=websocket,
            latency_optimizer=self.latency_optimizer
        )
        
        await handler.init()
        self.active_sessions[session_id] = handler
        
        return handler
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "orchestrator": self.performance_stats.copy(),
            "active_agents": len(self.active_agents),
            "active_sessions": len(self.active_sessions),
            "agent_performance": {
                agent_id: {
                    "total_queries": agent.get("total_queries", 0),
                    "success_rate": agent.get("successful_queries", 0) / max(agent.get("total_queries", 1), 1) * 100,
                    "average_latency": agent.get("average_latency", 0),
                    "last_used": agent.get("last_used")
                }
                for agent_id, agent in self.active_agents.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("🔄 Shutting down Multi-Agent Orchestrator...")
        
        # Cleanup active sessions
        for session_id, handler in list(self.active_sessions.items()):
            try:
                await handler.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
        
        # Cleanup components
        if self.hybrid_vector_store:
            await self.hybrid_vector_store.cleanup()
        
        if self.tool_orchestrator:
            await self.tool_orchestrator.cleanup()
        
        logger.info("✅ Multi-Agent Orchestrator shutdown complete")