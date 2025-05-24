"""
Multi-Agent Orchestrator using LangGraph for intelligent agent coordination.
Optimized for <2-second response times with state-aware routing.
"""
import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from enum import Enum
from dataclasses import dataclass, field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# System imports
from app.vector_db.hybrid_vector_store import HybridVectorStore
from app.agents.agent_registry import AgentRegistry
from app.agents.intelligent_router import IntelligentRouter, RoutingDecision
from app.agents.base_agent import BaseAgent
from app.config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

class ConversationState(TypedDict):
    """Enhanced conversation state for LangGraph workflow."""
    # Input
    user_input: str
    session_id: str
    call_sid: Optional[str]
    audio_data: Optional[bytes]
    
    # Routing
    routing_decision: Optional[RoutingDecision]
    selected_agent: Optional[str]
    confidence_score: float
    
    # Context
    conversation_history: List[BaseMessage]
    retrieved_context: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    session_metadata: Dict[str, Any]
    
    # Processing
    agent_response: Optional[str]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    
    # Output
    final_response: str
    response_audio: Optional[bytes]
    sources: List[Dict[str, Any]]
    
    # Performance tracking
    processing_start_time: float
    routing_time: Optional[float]
    retrieval_time: Optional[float]
    agent_processing_time: Optional[float]
    tts_time: Optional[float]
    total_time: Optional[float]
    
    # Flags
    requires_escalation: bool
    needs_human_handoff: bool
    conversation_ended: bool
    error_occurred: bool
    error_message: Optional[str]

class ProcessingStage(Enum):
    """Processing stages for performance tracking."""
    INITIALIZED = "initialized"
    ROUTING = "routing"  
    CONTEXT_RETRIEVAL = "context_retrieval"
    AGENT_PROCESSING = "agent_processing"
    TOOL_EXECUTION = "tool_execution"
    RESPONSE_GENERATION = "response_generation"
    TTS_SYNTHESIS = "tts_synthesis"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class PerformanceMetrics:
    """Track performance metrics for each conversation."""
    session_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    routing_time: float = 0.0
    retrieval_time: float = 0.0
    agent_time: float = 0.0
    tool_time: float = 0.0
    tts_time: float = 0.0
    
    total_time: float = 0.0
    
    agent_used: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    context_retrieved: int = 0
    
    def complete(self):
        """Mark metrics as complete and calculate total time."""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/analytics."""
        return {
            "session_id": self.session_id,
            "total_time": self.total_time,
            "routing_time": self.routing_time,
            "retrieval_time": self.retrieval_time,
            "agent_time": self.agent_time,
            "tool_time": self.tool_time,
            "tts_time": self.tts_time,
            "agent_used": self.agent_used,
            "tools_called": self.tools_called,
            "context_retrieved": self.context_retrieved,
            "latency_target_met": self.total_time < LatencyConfig.TARGET_TOTAL_LATENCY
        }

class MultiAgentOrchestrator:
    """
    LangGraph-based multi-agent orchestrator for intelligent conversation routing.
    
    Manages the complete conversation workflow with sub-2-second latency optimization:
    1. Intelligent agent routing
    2. Context retrieval from hybrid vector store
    3. Agent execution with tool integration
    4. Response synthesis and audio generation
    """
    
    def __init__(
        self,
        vector_store: HybridVectorStore,
        agent_registry: AgentRegistry,
        router: IntelligentRouter,
        stt=None,
        tts=None
    ):
        """Initialize the orchestrator with all required components."""
        self.vector_store = vector_store
        self.agent_registry = agent_registry
        self.router = router
        self.stt = stt
        self.tts = tts
        
        # LangGraph components
        self.workflow_graph = None
        self.checkpointer = MemorySaver()
        
        # Performance tracking
        self.active_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Agent instances cache
        self.agent_instances: Dict[str, BaseAgent] = {}
        
        # System prompts
        self.system_prompts = {
            "router": """You are an intelligent routing system. Analyze the user's input and determine the most appropriate specialized agent to handle their request. Consider:
            - Intent and urgency
            - Domain expertise required
            - Tool requirements
            - User context and history""",
            
            "synthesizer": """You are a response synthesizer. Take the agent's response and tool results, then create a natural, conversational response optimized for voice delivery. Keep responses:
            - Concise and clear
            - Naturally spoken
            - Under 2 sentences when possible
            - Emotionally appropriate to the context"""
        }
        
        logger.info("MultiAgentOrchestrator initialized")
    
    async def init(self):
        """Initialize the orchestrator and build the LangGraph workflow."""
        logger.info("Initializing multi-agent orchestrator...")
        
        # Build the conversation workflow graph
        await self._build_workflow_graph()
        
        # Pre-load agent instances for faster access
        await self._preload_agents()
        
        logger.info("✅ Multi-agent orchestrator ready")
    
    async def _build_workflow_graph(self):
        """Build the LangGraph workflow for conversation processing."""
        # Create the state graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes for each processing stage
        workflow.add_node("initialize_conversation", self._initialize_conversation)
        workflow.add_node("route_to_agent", self._route_to_agent) 
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("execute_agent", self._execute_agent)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("synthesize_response", self._synthesize_response)
        workflow.add_node("generate_audio", self._generate_audio)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the workflow edges
        workflow.set_entry_point("initialize_conversation")
        
        workflow.add_edge("initialize_conversation", "route_to_agent")
        workflow.add_edge("route_to_agent", "retrieve_context")
        workflow.add_edge("retrieve_context", "execute_agent")
        
        # Conditional edge for tool execution
        workflow.add_conditional_edges(
            "execute_agent",
            self._should_execute_tools,
            {
                "execute_tools": "execute_tools",
                "synthesize": "synthesize_response"
            }
        )
        
        workflow.add_edge("execute_tools", "synthesize_response")
        workflow.add_edge("synthesize_response", "generate_audio")
        workflow.add_edge("generate_audio", END)
        
        # Error handling edges
        workflow.add_edge("handle_error", END)
        
        # Compile the graph with checkpointing for conversation memory
        self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("✅ LangGraph workflow compiled")
    
    async def _preload_agents(self):
        """Pre-load agent instances for faster response times."""
        active_agents = await self.agent_registry.get_active_agents()
        
        for agent_id in active_agents:
            try:
                agent_config = await self.agent_registry.get_agent_config(agent_id)
                agent_instance = await self.agent_registry.create_agent_instance(agent_id, agent_config)
                self.agent_instances[agent_id] = agent_instance
                logger.info(f"✅ Pre-loaded agent: {agent_id}")
            except Exception as e:
                logger.error(f"❌ Failed to pre-load agent {agent_id}: {e}")
    
    async def process_conversation(
        self,
        user_input: str,
        session_id: str,
        call_sid: Optional[str] = None,
        audio_data: Optional[bytes] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete conversation turn through the multi-agent workflow.
        
        Args:
            user_input: User's text input
            session_id: Unique session identifier
            call_sid: Twilio call SID (if applicable)
            audio_data: Raw audio data (if available)
            user_context: Additional user context
            
        Returns:
            Complete response with audio, sources, and metadata
        """
        # Start performance tracking
        metrics = PerformanceMetrics(session_id=session_id)
        self.active_metrics[session_id] = metrics
        
        try:
            # Initialize conversation state
            initial_state = ConversationState(
                user_input=user_input,
                session_id=session_id,
                call_sid=call_sid,
                audio_data=audio_data,
                routing_decision=None,
                selected_agent=None,
                confidence_score=0.0,
                conversation_history=await self._get_conversation_history(session_id),
                retrieved_context=[],
                user_profile=user_context or {},
                session_metadata=self.active_sessions.get(session_id, {}),
                agent_response=None,
                tool_calls=[],
                tool_results=[],
                final_response="",
                response_audio=None,
                sources=[],
                processing_start_time=time.time(),
                routing_time=None,
                retrieval_time=None,
                agent_processing_time=None,
                tts_time=None,
                total_time=None,
                requires_escalation=False,
                needs_human_handoff=False,
                conversation_ended=False,
                error_occurred=False,
                error_message=None
            )
            
            # Process through LangGraph workflow
            config = {"configurable": {"thread_id": session_id}}
            
            final_state = await self.workflow_graph.ainvoke(initial_state, config)
            
            # Update performance metrics
            metrics.agent_used = final_state.get("selected_agent")
            metrics.context_retrieved = len(final_state.get("retrieved_context", []))
            metrics.tools_called = [call.get("tool") for call in final_state.get("tool_calls", [])]
            metrics.complete()
            
            # Prepare response
            response = {
                "response": final_state["final_response"],
                "audio": final_state.get("response_audio"),
                "sources": final_state.get("sources", []),
                "agent_used": final_state.get("selected_agent"),
                "confidence": final_state.get("confidence_score", 0.0),
                "requires_escalation": final_state.get("requires_escalation", False),
                "conversation_ended": final_state.get("conversation_ended", False),
                "performance_metrics": metrics.to_dict(),
                "session_id": session_id
            }
            
            # Log performance if over target
            if metrics.total_time > LatencyConfig.TARGET_TOTAL_LATENCY:
                logger.warning(f"⚠️ Latency target exceeded: {metrics.total_time:.3f}s > {LatencyConfig.TARGET_TOTAL_LATENCY}s")
                await self._analyze_latency_bottleneck(metrics)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in conversation processing: {e}", exc_info=True)
            
            # Update metrics with error
            metrics.complete()
            
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "error": str(e),
                "performance_metrics": metrics.to_dict(),
                "session_id": session_id
            }
        finally:
            # Clean up metrics after processing
            if session_id in self.active_metrics:
                del self.active_metrics[session_id]
    
    async def _initialize_conversation(self, state: ConversationState) -> ConversationState:
        """Initialize conversation with session management."""
        session_id = state["session_id"]
        
        # Initialize or update session
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "start_time": time.time(),
                "message_count": 0,
                "current_agent": None,
                "user_profile": state["user_profile"]
            }
        
        # Update session
        session = self.active_sessions[session_id]
        session["message_count"] += 1
        session["last_activity"] = time.time()
        
        # Update state
        state["session_metadata"] = session
        
        logger.info(f"🔄 Initialized conversation for session {session_id}")
        return state
    
    async def _route_to_agent(self, state: ConversationState) -> ConversationState:
        """Route user input to the most appropriate agent."""
        routing_start = time.time()
        
        try:
            # Get routing decision from intelligent router
            routing_decision = await self.router.route_request(
                user_input=state["user_input"],
                conversation_history=state["conversation_history"],
                user_context=state["user_profile"],
                session_metadata=state["session_metadata"]
            )
            
            state["routing_decision"] = routing_decision
            state["selected_agent"] = routing_decision.agent_id
            state["confidence_score"] = routing_decision.confidence
            
            # Update session with current agent
            if state["session_id"] in self.active_sessions:
                self.active_sessions[state["session_id"]]["current_agent"] = routing_decision.agent_id
            
            # Track routing time
            routing_time = time.time() - routing_start
            state["routing_time"] = routing_time
            
            if state["session_id"] in self.active_metrics:
                self.active_metrics[state["session_id"]].routing_time = routing_time
            
            logger.info(f"🎯 Routed to agent: {routing_decision.agent_id} (confidence: {routing_decision.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Routing error: {e}")
            # Fallback to technical support agent
            state["selected_agent"] = "technical-support"
            state["confidence_score"] = 0.5
            state["routing_time"] = time.time() - routing_start
        
        return state
    
    async def _retrieve_context(self, state: ConversationState) -> ConversationState:
        """Retrieve relevant context from hybrid vector store."""
        retrieval_start = time.time()
        
        try:
            # Get agent-specific collection/namespace
            agent_id = state["selected_agent"]
            
            # Retrieve context with agent-specific optimization
            retrieved_docs = await self.vector_store.hybrid_search(
                query=state["user_input"],
                agent_id=agent_id,
                top_k=LatencyConfig.MAX_CONTEXT_DOCS,
                hybrid_alpha=0.7  # Favor semantic search
            )
            
            state["retrieved_context"] = retrieved_docs
            
            # Track retrieval time
            retrieval_time = time.time() - retrieval_start
            state["retrieval_time"] = retrieval_time
            
            if state["session_id"] in self.active_metrics:
                self.active_metrics[state["session_id"]].retrieval_time = retrieval_time
                self.active_metrics[state["session_id"]].context_retrieved = len(retrieved_docs)
            
            logger.info(f"📚 Retrieved {len(retrieved_docs)} context documents in {retrieval_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Context retrieval error: {e}")
            state["retrieved_context"] = []
            state["retrieval_time"] = time.time() - retrieval_start
        
        return state
    
    async def _execute_agent(self, state: ConversationState) -> ConversationState:
        """Execute the selected agent with retrieved context."""
        agent_start = time.time()
        
        try:
            agent_id = state["selected_agent"]
            
            # Get or create agent instance
            if agent_id not in self.agent_instances:
                agent_config = await self.agent_registry.get_agent_config(agent_id)
                agent_instance = await self.agent_registry.create_agent_instance(agent_id, agent_config)
                self.agent_instances[agent_id] = agent_instance
            
            agent = self.agent_instances[agent_id]
            
            # Prepare agent input
            agent_input = {
                "user_input": state["user_input"],
                "context": state["retrieved_context"],
                "conversation_history": state["conversation_history"],
                "user_profile": state["user_profile"],
                "session_metadata": state["session_metadata"]
            }
            
            # Execute agent
            agent_response = await agent.process_request(agent_input)
            
            state["agent_response"] = agent_response.get("response", "")
            state["tool_calls"] = agent_response.get("tool_calls", [])
            state["requires_escalation"] = agent_response.get("requires_escalation", False)
            state["needs_human_handoff"] = agent_response.get("needs_human_handoff", False)
            
            # Track agent processing time
            agent_time = time.time() - agent_start
            state["agent_processing_time"] = agent_time
            
            if state["session_id"] in self.active_metrics:
                self.active_metrics[state["session_id"]].agent_time = agent_time
            
            logger.info(f"🤖 Agent {agent_id} processed request in {agent_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Agent execution error: {e}")
            state["agent_response"] = "I apologize, but I'm having trouble processing your request right now."
            state["error_occurred"] = True
            state["error_message"] = str(e)
            state["agent_processing_time"] = time.time() - agent_start
        
        return state
    
    def _should_execute_tools(self, state: ConversationState) -> str:
        """Determine if tools need to be executed."""
        if state.get("tool_calls") and len(state["tool_calls"]) > 0:
            return "execute_tools"
        return "synthesize"
    
    async def _execute_tools(self, state: ConversationState) -> ConversationState:
        """Execute tools required by the agent."""
        tool_start = time.time()
        
        try:
            tool_results = []
            agent_id = state["selected_agent"]
            
            # Get agent instance for tool execution
            agent = self.agent_instances.get(agent_id)
            if not agent:
                logger.error(f"Agent {agent_id} not found for tool execution")
                state["tool_results"] = []
                return state
            
            # Execute each tool call
            for tool_call in state["tool_calls"]:
                try:
                    result = await agent.execute_tool(
                        tool_name=tool_call.get("tool"),
                        tool_input=tool_call.get("input", {}),
                        context=state
                    )
                    tool_results.append(result)
                    
                    # Track tool usage
                    if state["session_id"] in self.active_metrics:
                        self.active_metrics[state["session_id"]].tools_called.append(tool_call.get("tool"))
                        
                except Exception as e:
                    logger.error(f"❌ Tool execution error for {tool_call.get('tool')}: {e}")
                    tool_results.append({
                        "tool": tool_call.get("tool"),
                        "error": str(e),
                        "success": False
                    })
            
            state["tool_results"] = tool_results
            
            # Track tool execution time
            tool_time = time.time() - tool_start
            
            if state["session_id"] in self.active_metrics:
                self.active_metrics[state["session_id"]].tool_time = tool_time
            
            logger.info(f"🔧 Executed {len(tool_results)} tools in {tool_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Tool execution error: {e}")
            state["tool_results"] = []
        
        return state
    
    async def _synthesize_response(self, state: ConversationState) -> ConversationState:
        """Synthesize the final response from agent output and tool results."""
        try:
            # Combine agent response with tool results
            response_parts = []
            
            if state.get("agent_response"):
                response_parts.append(state["agent_response"])
            
            # Add tool results to response if available
            for tool_result in state.get("tool_results", []):
                if tool_result.get("success") and tool_result.get("user_message"):
                    response_parts.append(tool_result["user_message"])
            
            # Create final response
            final_response = " ".join(response_parts) if response_parts else "I understand your request, but I need more information to help you properly."
            
            # Optimize for voice delivery (keep under 2 sentences when possible)
            if len(final_response.split('.')) > 2:
                sentences = final_response.split('.')
                final_response = '. '.join(sentences[:2]) + '.'
            
            state["final_response"] = final_response
            
            # Prepare sources from retrieved context
            sources = []
            for doc in state.get("retrieved_context", []):
                if doc.get("metadata"):
                    sources.append({
                        "source": doc["metadata"].get("source", "Knowledge Base"),
                        "score": doc.get("score", 0.0),
                        "content_preview": doc.get("content", "")[:100] + "..."
                    })
            
            state["sources"] = sources
            
            logger.info(f"✨ Synthesized response: {final_response[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Response synthesis error: {e}")
            state["final_response"] = "I apologize, but I'm having trouble formulating a response right now."
            state["sources"] = []
        
        return state
    
    async def _generate_audio(self, state: ConversationState) -> ConversationState:
        """Generate audio response using TTS."""
        if not self.tts:
            state["response_audio"] = None
            return state
            
        tts_start = time.time()
        
        try:
            # Generate audio with telephony optimization
            audio_data = await self.tts.synthesize(
                text=state["final_response"],
                optimize_for="telephony",
                voice_config={
                    "agent": state.get("selected_agent"),
                    "context": "conversation"
                }
            )
            
            state["response_audio"] = audio_data
            
            # Track TTS time
            tts_time = time.time() - tts_start
            state["tts_time"] = tts_time
            
            if state["session_id"] in self.active_metrics:
                self.active_metrics[state["session_id"]].tts_time = tts_time
            
            logger.info(f"🔊 Generated audio response in {tts_time:.3f}s ({len(audio_data) if audio_data else 0} bytes)")
            
        except Exception as e:
            logger.error(f"❌ TTS generation error: {e}")
            state["response_audio"] = None
            state["tts_time"] = time.time() - tts_start
        
        # Calculate total processing time
        total_time = time.time() - state["processing_start_time"]
        state["total_time"] = total_time
        
        return state
    
    async def _handle_error(self, state: ConversationState) -> ConversationState:
        """Handle errors in the conversation workflow."""
        logger.error(f"❌ Error in conversation workflow: {state.get('error_message', 'Unknown error')}")
        
        state["final_response"] = "I apologize, but I encountered an error. Please try rephrasing your request."
        state["response_audio"] = None
        state["sources"] = []
        state["error_occurred"] = True
        
        return state
    
    async def _get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """Retrieve conversation history for a session."""
        try:
            # Get conversation history from checkpointer
            config = {"configurable": {"thread_id": session_id}}
            
            # This would retrieve the conversation history from the checkpointer
            # For now, return empty list as a placeholder
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    async def _analyze_latency_bottleneck(self, metrics: PerformanceMetrics):
        """Analyze where latency bottlenecks are occurring."""
        bottlenecks = []
        
        if metrics.routing_time > LatencyConfig.TARGET_ROUTING_TIME:
            bottlenecks.append(f"Routing: {metrics.routing_time:.3f}s")
        
        if metrics.retrieval_time > LatencyConfig.TARGET_RETRIEVAL_TIME:
            bottlenecks.append(f"Retrieval: {metrics.retrieval_time:.3f}s")
        
        if metrics.agent_time > LatencyConfig.TARGET_AGENT_TIME:
            bottlenecks.append(f"Agent: {metrics.agent_time:.3f}s")
        
        if metrics.tts_time > LatencyConfig.TARGET_TTS_TIME:
            bottlenecks.append(f"TTS: {metrics.tts_time:.3f}s")
        
        if bottlenecks:
            logger.warning(f"🐌 Latency bottlenecks detected: {', '.join(bottlenecks)}")
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        return {
            "session_id": session_id,
            "start_time": session["start_time"],
            "duration": current_time - session["start_time"],
            "message_count": session["message_count"],
            "current_agent": session.get("current_agent"),
            "last_activity": session.get("last_activity", session["start_time"]),
            "time_since_activity": current_time - session.get("last_activity", session["start_time"])
        }
    
    async def cleanup_session(self, session_id: str):
        """Clean up a completed session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if session_id in self.active_metrics:
            del self.active_metrics[session_id]
        
        logger.info(f"🧹 Cleaned up session: {session_id}")
    
    async def health_check(self) -> bool:
        """Perform health check on the orchestrator."""
        try:
            # Check core components
            if not self.workflow_graph:
                return False
            
            if not self.vector_store:
                return False
            
            if not self.agent_registry:
                return False
            
            # Check if agents are loaded
            active_agents = await self.agent_registry.get_active_agents()
            if not active_agents:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all recent sessions."""
        # This would aggregate performance data across sessions
        return {
            "active_sessions": len(self.active_sessions),
            "total_agents": len(self.agent_instances),
            "average_latency": 0.0,  # Calculate from recent metrics
            "latency_target_met_percentage": 0.0,  # Calculate from recent metrics
        }
    
    async def shutdown(self):
        """Shutdown the orchestrator gracefully."""
        logger.info("🛑 Shutting down multi-agent orchestrator...")
        
        # Clean up all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.cleanup_session(session_id)
        
        # Shutdown agent instances
        for agent_id, agent in self.agent_instances.items():
            try:
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent {agent_id}: {e}")
        
        self.agent_instances.clear()
        
        logger.info("✅ Multi-agent orchestrator shutdown complete")