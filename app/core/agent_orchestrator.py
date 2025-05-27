"""
LangGraph-based Multi-Agent Orchestrator for Revolutionary Voice AI System
Implements sophisticated graph-based workflows with persistent conversation state.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, AsyncIterator, Union
from dataclasses import dataclass, field
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Core system imports
from app.core.state_manager import ConversationState, ConversationStateManager
from app.agents.registry import AgentRegistry
from app.agents.router import IntelligentAgentRouter, RoutingResult
from app.vector_db.hybrid_vector_system import HybridVectorSystem
from app.tools.tool_orchestrator import ComprehensiveToolOrchestrator, ToolResult
from app.voice.enhanced_stt import EnhancedSTTSystem
from app.voice.dual_streaming_tts import DualStreamingTTSEngine

logger = logging.getLogger(__name__)

class WorkflowState(str, Enum):
    """Workflow execution states."""
    INIT = "initialization"
    INPUT_ANALYSIS = "input_analysis"
    AGENT_ROUTING = "agent_routing"
    CONTEXT_ENRICHMENT = "context_enrichment"
    AGENT_EXECUTION = "agent_execution"
    TOOL_ORCHESTRATION = "tool_orchestration"
    RESPONSE_SYNTHESIS = "response_synthesis"
    QUALITY_VALIDATION = "quality_validation"
    STREAMING_DELIVERY = "streaming_delivery"
    CONVERSATION_UPDATE = "conversation_update"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ConversationWorkflowState:
    """State object for LangGraph workflow execution."""
    session_id: str
    input_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Processing state
    current_state: WorkflowState = WorkflowState.INIT
    analysis_result: Optional[Dict[str, Any]] = None
    routing_result: Optional[RoutingResult] = None
    selected_agent_id: Optional[str] = None
    enriched_context: Optional[Dict[str, Any]] = None
    
    # Agent execution
    agent_response: Optional[str] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    
    # Response generation
    final_response: Optional[str] = None
    response_quality_score: float = 0.0
    confidence_score: float = 0.0
    
    # Metadata
    start_time: float = field(default_factory=time.time)
    processing_steps: List[Dict[str, Any]] = field(default_factory=list)
    error_info: Optional[Dict[str, Any]] = None
    
    # Performance tracking
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)

@dataclass
class OrchestrationResult:
    """Result of multi-agent orchestration."""
    success: bool
    session_id: str
    response: str
    agent_id: str
    confidence: float
    latency_ms: float
    tools_used: List[str]
    sources: List[str]
    quality_score: float
    error: Optional[str] = None

class MultiAgentOrchestrator:
    """
    Advanced multi-agent orchestrator using LangGraph for stateful workflow execution.
    Coordinates specialized agents with intelligent routing and tool orchestration.
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        agent_router: IntelligentAgentRouter,
        state_manager: ConversationStateManager,
        hybrid_vector_system: HybridVectorSystem,
        stt_system: EnhancedSTTSystem,
        tts_engine: DualStreamingTTSEngine,
        tool_orchestrator: ComprehensiveToolOrchestrator,
        target_latency_ms: int = 377
    ):
        """Initialize the multi-agent orchestrator."""
        self.agent_registry = agent_registry
        self.agent_router = agent_router
        self.state_manager = state_manager
        self.hybrid_vector_system = hybrid_vector_system
        self.stt_system = stt_system
        self.tts_engine = tts_engine
        self.tool_orchestrator = tool_orchestrator
        self.target_latency_ms = target_latency_ms
        
        # LangGraph components
        self.workflow_graph = None
        self.memory_saver = MemorySaver()  # For persistent conversation state
        
        # Performance tracking
        self.performance_stats = {
            "total_conversations": 0,
            "avg_latency_ms": 0.0,
            "success_rate": 0.0,
            "agent_usage": {},
            "tool_usage": {},
            "quality_scores": []
        }
        
        # Error and retry tracking
        self.error_metrics = {}
        self.timeout_metrics = {}
        self.response_cache = {}
        
        # Quality gates
        self.quality_thresholds = {
            "min_confidence": 0.7,
            "min_quality_score": 0.8,
            "max_retries": 2
        }
        
        self.initialized = False
        logger.info("Multi-Agent Orchestrator initialized with LangGraph")
    
    async def initialize(self):
        """Initialize the orchestrator and build the workflow graph."""
        logger.info("Initializing LangGraph multi-agent workflow...")
        
        # Build the comprehensive workflow graph
        self.workflow_graph = self._build_advanced_workflow()
        
        self.initialized = True
        logger.info("✅ LangGraph orchestrator initialized successfully")
    
    def _build_advanced_workflow(self) -> StateGraph:
        """Build comprehensive multi-agent workflow with quality gates."""
        workflow = StateGraph(ConversationWorkflowState)
        
        # Core workflow nodes
        workflow.add_node("session_init", self.initialize_session)
        workflow.add_node("input_analysis", self.analyze_user_input)
        workflow.add_node("intelligent_routing", self.route_to_agent)
        workflow.add_node("context_enrichment", self.enrich_context)
        workflow.add_node("agent_execution", self.execute_agent)
        workflow.add_node("tool_orchestration", self.orchestrate_tools)
        workflow.add_node("response_synthesis", self.synthesize_response)
        workflow.add_node("quality_validation", self.validate_quality)
        workflow.add_node("streaming_delivery", self.stream_response)
        workflow.add_node("conversation_update", self.update_conversation_memory)
        workflow.add_node("error_handler", self.handle_error)
        workflow.add_node("retry_handler", self.handle_retry)
        
        # Entry point
        workflow.set_entry_point("session_init")
        
        # Linear flow with conditional routing
        workflow.add_edge("session_init", "input_analysis")
        
        # Conditional routing after input analysis
        workflow.add_conditional_edges(
            "input_analysis",
            self.determine_flow,
            {
                "direct_response": "response_synthesis",
                "agent_required": "intelligent_routing",
                "clarification_needed": "response_synthesis",
                "error": "error_handler"
            }
        )
        
        # Agent workflow
        workflow.add_edge("intelligent_routing", "context_enrichment")
        workflow.add_edge("context_enrichment", "agent_execution")
        
        # Tool orchestration (conditional)
        workflow.add_conditional_edges(
            "agent_execution",
            self.check_tools_needed,
            {
                "tools_needed": "tool_orchestration",
                "no_tools": "response_synthesis",
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("tool_orchestration", "response_synthesis")
        
        # Quality validation with retry logic
        workflow.add_conditional_edges(
            "response_synthesis",
            self.check_response_quality,
            {
                "quality_passed": "quality_validation",
                "quality_failed": "retry_handler",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "quality_validation",
            self.quality_check,
            {
                "approved": "streaming_delivery",
                "retry": "agent_execution",
                "escalate": "error_handler",
                "regenerate": "response_synthesis"
            }
        )
        
        # Final steps
        workflow.add_edge("streaming_delivery", "conversation_update")
        workflow.add_edge("conversation_update", END)
        
        # Error handling
        workflow.add_conditional_edges(
            "error_handler",
            self.handle_error_decision,
            {
                "retry": "retry_handler",
                "fallback": "response_synthesis",
                "terminate": END
            }
        )
        
        workflow.add_conditional_edges(
            "retry_handler",
            self.check_retry_limit,
            {
                "retry_allowed": "agent_execution",
                "max_retries": "response_synthesis",
                "terminate": END
            }
        )
        
        # Compile with memory checkpoint
        compiled_workflow = workflow.compile(checkpointer=self.memory_saver)
        
        logger.info("✅ Advanced LangGraph workflow compiled successfully")
        return compiled_workflow
    
    async def process_conversation(
        self,
        session_id: str,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        preferred_agent_id: Optional[str] = None
    ) -> OrchestrationResult:
        """Process a conversation through the complete multi-agent workflow."""
        if not self.initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        start_time = time.time()
        
        # Create initial workflow state
        initial_state = ConversationWorkflowState(
            session_id=session_id,
            input_text=input_text,
            context=context or {},
            start_time=start_time
        )
        
        # Add preferred agent if specified
        if preferred_agent_id:
            initial_state.context["preferred_agent_id"] = preferred_agent_id
        
        try:
            # Execute workflow with persistent state
            config = {"configurable": {"thread_id": session_id}}
            
            result = await self.workflow_graph.ainvoke(
                initial_state,
                config=config
            )
            
            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000
            
            # Update performance stats
            await self._update_performance_stats(result, total_latency)
            
            # Log performance
            if total_latency > self.target_latency_ms:
                logger.warning(f"⚠️ Latency exceeded target: {total_latency:.2f}ms > {self.target_latency_ms}ms")
            else:
                logger.info(f"✅ Conversation processed in {total_latency:.2f}ms (target: {self.target_latency_ms}ms)")
            
            # Create orchestration result
            return OrchestrationResult(
                success=result.current_state != WorkflowState.ERROR,
                session_id=session_id,
                response=result.final_response or "I apologize, but I couldn't process your request.",
                agent_id=result.selected_agent_id or "fallback",
                confidence=result.confidence_score,
                latency_ms=total_latency,
                tools_used=result.tools_used,
                sources=result.sources_used,
                quality_score=result.response_quality_score,
                error=result.error_info.get("message") if result.error_info else None
            )
            
        except Exception as e:
            logger.error(f"❌ Orchestration error for session {session_id}: {e}", exc_info=True)
            
            return OrchestrationResult(
                success=False,
                session_id=session_id,
                response="I'm sorry, I encountered an error processing your request. Please try again.",
                agent_id="error",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                tools_used=[],
                sources=[],
                quality_score=0.0,
                error=str(e)
            )
    
    # Workflow node implementations
    async def initialize_session(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Initialize conversation session with context loading."""
        step_start = time.time()
        
        try:
            # Load existing conversation state
            conversation_state = await self.state_manager.get_conversation_state(state.session_id)
            
            if conversation_state:
                # Enrich context with conversation history
                state.context.update({
                    "conversation_history": conversation_state.message_history[-5:],  # Last 5 messages
                    "user_preferences": conversation_state.user_preferences,
                    "conversation_metadata": conversation_state.metadata
                })
            
            state.current_state = WorkflowState.INPUT_ANALYSIS
            state.latency_breakdown["session_init"] = (time.time() - step_start) * 1000
            
            logger.debug(f"Session initialized for {state.session_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "session_init", "message": str(e)}
            return state
    
    async def analyze_user_input(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Analyze user input for intent, urgency, and complexity."""
        step_start = time.time()
        
        try:
            # Use hybrid vector system for fast intent analysis
            analysis_result = await self.hybrid_vector_system.analyze_query_intent(
                query=state.input_text,
                context=state.context
            )
            
            state.analysis_result = {
                "intent": analysis_result.get("intent", "general"),
                "urgency_level": analysis_result.get("urgency", "normal"),
                "complexity_score": analysis_result.get("complexity", 0.5),
                "entities": analysis_result.get("entities", []),
                "keywords": analysis_result.get("keywords", []),
                "requires_tools": analysis_result.get("requires_tools", False),
                "confidence": analysis_result.get("confidence", 0.8)
            }
            
            state.current_state = WorkflowState.AGENT_ROUTING
            state.latency_breakdown["input_analysis"] = (time.time() - step_start) * 1000
            
            logger.debug(f"Input analyzed - Intent: {state.analysis_result['intent']}, "
                       f"Urgency: {state.analysis_result['urgency_level']}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing input: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "input_analysis", "message": str(e)}
            return state
    
    async def route_to_agent(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Route to the most appropriate specialized agent."""
        step_start = time.time()
        
        try:
            # Check for preferred agent
            preferred_agent = state.context.get("preferred_agent_id")
            
            # Use intelligent router
            routing_result = await self.agent_router.route_query(
                query=state.input_text,
                context=state.context,
                analysis_result=state.analysis_result,
                preferred_agent_id=preferred_agent
            )
            
            state.routing_result = routing_result
            state.selected_agent_id = routing_result.selected_agent_id
            state.confidence_score = routing_result.confidence
            
            # Update agent usage stats
            if routing_result.selected_agent_id:
                self.performance_stats["agent_usage"][routing_result.selected_agent_id] = \
                    self.performance_stats["agent_usage"].get(routing_result.selected_agent_id, 0) + 1
            
            state.current_state = WorkflowState.CONTEXT_ENRICHMENT
            state.latency_breakdown["agent_routing"] = (time.time() - step_start) * 1000
            
            logger.info(f"Routed to agent: {state.selected_agent_id} "
                       f"(confidence: {state.confidence_score:.2f})")
            
            return state
            
        except Exception as e:
            logger.error(f"Error routing to agent: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "agent_routing", "message": str(e)}
            return state
    
    async def enrich_context(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Enrich context with relevant information for the selected agent."""
        step_start = time.time()
        
        try:
            # Get agent-specific context from hybrid vector system
            enriched_context = await self.hybrid_vector_system.get_agent_context(
                agent_id=state.selected_agent_id,
                query=state.input_text,
                base_context=state.context
            )
            
            state.enriched_context = {
                **state.context,
                **enriched_context,
                "agent_id": state.selected_agent_id,
                "routing_confidence": state.confidence_score,
                "analysis": state.analysis_result
            }
            
            state.current_state = WorkflowState.AGENT_EXECUTION
            state.latency_breakdown["context_enrichment"] = (time.time() - step_start) * 1000
            
            logger.debug(f"Context enriched for agent {state.selected_agent_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error enriching context: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "context_enrichment", "message": str(e)}
            return state
    
    async def execute_agent(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Execute the selected specialized agent."""
        step_start = time.time()
        
        try:
            # Get agent from registry
            agent = await self.agent_registry.get_agent(state.selected_agent_id)
            if not agent:
                raise ValueError(f"Agent {state.selected_agent_id} not found")
            
            # Execute agent with enriched context
            agent_response = await agent.process_query(
                query=state.input_text,
                context=state.enriched_context
            )
            
            state.agent_response = agent_response.get("response", "")
            state.sources_used.extend(agent_response.get("sources", []))
            
            # Check if tools are needed
            tools_needed = (
                state.analysis_result.get("requires_tools", False) or
                agent_response.get("requires_tools", False) or
                len(agent_response.get("suggested_tools", [])) > 0
            )
            
            if tools_needed:
                state.current_state = WorkflowState.TOOL_ORCHESTRATION
            else:
                state.current_state = WorkflowState.RESPONSE_SYNTHESIS
            
            state.latency_breakdown["agent_execution"] = (time.time() - step_start) * 1000
            
            logger.debug(f"Agent {state.selected_agent_id} executed "
                       f"(tools needed: {tools_needed})")
            
            return state
            
        except Exception as e:
            logger.error(f"Error executing agent: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "agent_execution", "message": str(e)}
            return state
    
    async def orchestrate_tools(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Orchestrate tool execution based on agent requirements."""
        step_start = time.time()
        
        try:
            # Determine required tools
            required_tools = []
            
            # From analysis
            if state.analysis_result.get("requires_tools"):
                required_tools.extend(state.analysis_result.get("suggested_tools", []))
            
            # From agent response
            agent_tools = []  # This would come from agent response
            required_tools.extend(agent_tools)
            
            # Execute tools through orchestrator
            tool_results = []
            for tool_name in required_tools[:3]:  # Limit to 3 tools for latency
                try:
                    tool_result = await self.tool_orchestrator.execute_tool(
                        tool_name=tool_name,
                        parameters={
                            "query": state.input_text,
                            "context": state.enriched_context,
                            "agent_id": state.selected_agent_id
                        }
                    )
                    tool_results.append(tool_result)
                    state.tools_used.append(tool_name)
                    
                    # Update tool usage stats
                    self.performance_stats["tool_usage"][tool_name] = \
                        self.performance_stats["tool_usage"].get(tool_name, 0) + 1
                        
                except Exception as tool_error:
                    logger.warning(f"Tool {tool_name} execution failed: {tool_error}")
            
            state.tool_results = tool_results
            state.current_state = WorkflowState.RESPONSE_SYNTHESIS
            state.latency_breakdown["tool_orchestration"] = (time.time() - step_start) * 1000
            
            logger.debug(f"Executed {len(tool_results)} tools successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error orchestrating tools: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "tool_orchestration", "message": str(e)}
            return state
    
    async def synthesize_response(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Synthesize final response incorporating agent output and tool results."""
        step_start = time.time()
        
        try:
            # Combine agent response with tool results
            response_components = []
            
            if state.agent_response:
                response_components.append(state.agent_response)
            
            # Add tool results
            for tool_result in state.tool_results:
                if tool_result.success and tool_result.result_data:
                    response_components.append(f"Tool result: {tool_result.result_data}")
            
            # Synthesize final response (could use LLM for better integration)
            if response_components:
                state.final_response = " ".join(response_components)
            else:
                state.final_response = "I'm sorry, I couldn't find a suitable response to your query."
            
            # Calculate quality score
            state.response_quality_score = self._calculate_quality_score(state)
            
            state.current_state = WorkflowState.QUALITY_VALIDATION
            state.latency_breakdown["response_synthesis"] = (time.time() - step_start) * 1000
            
            logger.debug(f"Response synthesized (quality: {state.response_quality_score:.2f})")
            return state
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "response_synthesis", "message": str(e)}
            return state
    
    async def validate_quality(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Validate response quality and decide on next action."""
        step_start = time.time()
        
        try:
            # Quality checks
            quality_passed = (
                state.response_quality_score >= self.quality_thresholds["min_quality_score"] and
                state.confidence_score >= self.quality_thresholds["min_confidence"] and
                len(state.final_response) > 10  # Basic length check
            )
            
            if quality_passed:
                state.current_state = WorkflowState.STREAMING_DELIVERY
            else:
                # Check retry count
                retry_count = state.context.get("retry_count", 0)
                if retry_count < self.quality_thresholds["max_retries"]:
                    state.context["retry_count"] = retry_count + 1
                    state.current_state = WorkflowState.AGENT_EXECUTION  # Retry
                    logger.warning(f"Quality validation failed, retrying (attempt {retry_count + 1})")
                else:
                    # Use fallback response
                    state.final_response = "I apologize, but I'm having difficulty providing a complete answer. Could you please rephrase your question?"
                    state.current_state = WorkflowState.STREAMING_DELIVERY
            
            state.latency_breakdown["quality_validation"] = (time.time() - step_start) * 1000
            return state
            
        except Exception as e:
            logger.error(f"Error validating quality: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "quality_validation", "message": str(e)}
            return state
    
    async def stream_response(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Prepare response for streaming delivery."""
        step_start = time.time()
        
        try:
            # Prepare response for streaming (could implement actual streaming here)
            # For now, just mark as ready for delivery
            state.current_state = WorkflowState.CONVERSATION_UPDATE
            state.latency_breakdown["streaming_delivery"] = (time.time() - step_start) * 1000
            
            logger.debug("Response prepared for streaming delivery")
            return state
            
        except Exception as e:
            logger.error(f"Error preparing streaming response: {e}")
            state.current_state = WorkflowState.ERROR
            state.error_info = {"step": "streaming_delivery", "message": str(e)}
            return state
    
    async def update_conversation_memory(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Update conversation memory with the interaction."""
        step_start = time.time()
        
        try:
            # Update conversation state
            await self.state_manager.update_conversation(
                session_id=state.session_id,
                user_message=state.input_text,
                assistant_message=state.final_response,
                agent_id=state.selected_agent_id,
                tools_used=state.tools_used,
                metadata={
                    "confidence": state.confidence_score,
                    "quality_score": state.response_quality_score,
                    "latency_breakdown": state.latency_breakdown,
                    "sources": state.sources_used
                }
            )
            
            state.current_state = WorkflowState.COMPLETED
            state.latency_breakdown["conversation_update"] = (time.time() - step_start) * 1000
            
            logger.debug(f"Conversation memory updated for session {state.session_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error updating conversation memory: {e}")
            # Don't fail the entire workflow for memory update issues
            state.current_state = WorkflowState.COMPLETED
            state.latency_breakdown["conversation_update"] = (time.time() - step_start) * 1000
            return state
    
    async def handle_error(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Handle errors in the workflow."""
        try:
            error = state.error_info.get("message", "Unknown error") if state.error_info else "Unknown error"
            error_context = state.error_info or {}
            
            logger.error(f"Handling workflow error: {error}")
            logger.error(f"Error context: {error_context}")
            
            # Categorize error type
            error_type = "unknown"
            if isinstance(error, str):
                if "timeout" in error.lower():
                    error_type = "timeout"
                elif "connection" in error.lower():
                    error_type = "connection"
                elif "validation" in error.lower():
                    error_type = "validation"
                elif "rate limit" in error.lower():
                    error_type = "rate_limit"
                elif "authentication" in error.lower():
                    error_type = "auth"
            
            # Store error information in context
            if "error_type" not in state.context:
                state.context["error_type"] = error_type
            if "error_timestamp" not in state.context:
                state.context["error_timestamp"] = time.time()
            
            # Determine if error is retryable
            retryable_errors = ["timeout", "connection", "rate_limit"]
            state.context["is_retryable"] = error_type in retryable_errors
            
            # Store error for potential retry
            state.context["last_error"] = str(error)
            
            # Update error metrics
            self.error_metrics["total_errors"] = self.error_metrics.get("total_errors", 0) + 1
            self.error_metrics[f"{error_type}_errors"] = self.error_metrics.get(f"{error_type}_errors", 0) + 1
            
            # Provide fallback response
            state.final_response = "I apologize, but I encountered an error processing your request. Please try again."
            state.response_quality_score = 0.5  # Low quality due to error
            state.current_state = WorkflowState.COMPLETED
            
            logger.info(f"Error classified as: {error_type} (retryable: {state.context.get('is_retryable', False)})")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
            state.context["error_handler_error"] = str(e)
            state.final_response = "I apologize, but I encountered a system error. Please try again."
            state.current_state = WorkflowState.COMPLETED
            return state
    
    async def handle_retry(self, state: ConversationWorkflowState) -> ConversationWorkflowState:
        """Handle retry logic for failed operations."""
        try:
            # Get retry information from context
            retry_count = state.context.get("retry_count", 0)
            max_retries = state.context.get("max_retries", 3)
            last_error = state.context.get("last_error")
            
            logger.info(f"Handling retry attempt {retry_count + 1}/{max_retries}")
            
            if retry_count >= max_retries:
                logger.error(f"Maximum retries ({max_retries}) exceeded. Final error: {last_error}")
                state.context["retry_exhausted"] = True
                state.context["final_error"] = last_error
                state.final_response = "I apologize, but I'm unable to process your request after multiple attempts. Please try again later or contact support."
                state.current_state = WorkflowState.COMPLETED
                return state
            
            # Increment retry count
            state.context["retry_count"] = retry_count + 1
            
            # Calculate exponential backoff delay
            import random
            base_delay = 1.0  # 1 second base delay
            max_delay = 60.0  # Maximum 60 seconds
            delay = min(base_delay * (2 ** retry_count), max_delay)
            
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0.1, 0.3)
            delay = delay * (1 + jitter)
            
            logger.info(f"Waiting {delay:.2f} seconds before retry...")
            await asyncio.sleep(delay)
            
            # Clear error state for retry
            state.context["last_error"] = None
            state.context["retry_ready"] = True
            state.error_info = None
            
            # Reset to agent execution for retry
            state.current_state = WorkflowState.AGENT_EXECUTION
            
            return state
            
        except Exception as e:
            logger.error(f"Error in retry handler: {str(e)}")
            state.context["retry_handler_error"] = str(e)
            state.final_response = "I apologize, but I encountered an error during retry. Please try again."
            state.current_state = WorkflowState.COMPLETED
            return state
    
    # Conditional edge functions
    def determine_flow(self, state: ConversationWorkflowState) -> str:
        """Determine flow after input analysis."""
        if state.error_info:
            return "error"
        
        analysis = state.analysis_result
        if not analysis:
            return "error"
        
        # Simple flow determination
        if analysis.get("confidence", 0) < 0.5:
            return "clarification_needed"
        elif analysis.get("intent") == "direct":
            return "direct_response"
        else:
            return "agent_required"
    
    def check_tools_needed(self, state: ConversationWorkflowState) -> str:
        """Check if tools are needed after agent execution."""
        if state.error_info:
            return "error"
        
        if (state.analysis_result.get("requires_tools", False) or
            len(state.tool_results) == 0 and state.analysis_result.get("complexity_score", 0) > 0.7):
            return "tools_needed"
        else:
            return "no_tools"
    
    def check_response_quality(self, state: ConversationWorkflowState) -> str:
        """Check response quality."""
        if state.error_info:
            return "error"
        
        if state.response_quality_score >= self.quality_thresholds["min_quality_score"]:
            return "quality_passed"
        else:
            return "quality_failed"
    
    def quality_check(self, state: ConversationWorkflowState) -> str:
        """Final quality check."""
        if (state.response_quality_score >= 0.8 and
            state.confidence_score >= 0.7):
            return "approved"
        elif state.context.get("retry_count", 0) < 2:
            return "retry"
        else:
            return "regenerate"
    
    def handle_error_decision(self, state: ConversationWorkflowState) -> str:
        """Decide how to handle errors."""
        retry_count = state.context.get("error_retry_count", 0)
        if retry_count < 1:
            return "retry"
        else:
            return "fallback"
    
    def check_retry_limit(self, state: ConversationWorkflowState) -> str:
        """Check if retry limit reached."""
        retry_count = state.context.get("retry_count", 0)
        if retry_count < self.quality_thresholds["max_retries"]:
            return "retry_allowed"
        else:
            return "max_retries"
    
    def _calculate_quality_score(self, state: ConversationWorkflowState) -> float:
        """Calculate response quality score."""
        score = 0.5  # Base score
        
        # Add points for various factors
        if state.agent_response:
            score += 0.2
        
        if state.tool_results:
            score += 0.1 * min(len(state.tool_results), 3) / 3
        
        if state.confidence_score > 0.8:
            score += 0.2
        
        if len(state.final_response) > 20:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _update_performance_stats(self, result: ConversationWorkflowState, latency_ms: float):
        """Update performance statistics."""
        self.performance_stats["total_conversations"] += 1
        
        # Update average latency
        total = self.performance_stats["total_conversations"]
        current_avg = self.performance_stats["avg_latency_ms"]
        self.performance_stats["avg_latency_ms"] = (current_avg * (total - 1) + latency_ms) / total
        
        # Update success rate
        success = result.current_state == WorkflowState.COMPLETED
        current_success_rate = self.performance_stats["success_rate"]
        self.performance_stats["success_rate"] = (current_success_rate * (total - 1) + (1 if success else 0)) / total
        
        # Add quality score
        if result.response_quality_score > 0:
            self.performance_stats["quality_scores"].append(result.response_quality_score)
            # Keep only last 100 scores
            if len(self.performance_stats["quality_scores"]) > 100:
                self.performance_stats["quality_scores"].pop(0)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            **self.performance_stats,
            "avg_quality_score": sum(self.performance_stats["quality_scores"]) / len(self.performance_stats["quality_scores"]) if self.performance_stats["quality_scores"] else 0.0,
            "target_latency_ms": self.target_latency_ms,
            "latency_performance": {
                "meets_target": self.performance_stats["avg_latency_ms"] <= self.target_latency_ms,
                "improvement_needed_ms": max(0, self.performance_stats["avg_latency_ms"] - self.target_latency_ms)
            }
        }
    
    async def shutdown(self):
        """Shutdown the orchestrator."""
        logger.info("Shutting down Multi-Agent Orchestrator...")
        self.initialized = False
        logger.info("✅ Multi-Agent Orchestrator shutdown complete")