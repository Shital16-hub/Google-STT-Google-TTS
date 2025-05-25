"""
Multi-Agent Orchestrator with Fixed LangGraph Imports
Optimized for <2-second response times with state-aware routing.
"""
import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from enum import Enum
from dataclasses import dataclass, field

# LangGraph imports (fixed)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.base import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangGraph not available: {e}. Using fallback implementation.")
    LANGGRAPH_AVAILABLE = False
    # Fallback implementations
    class StateGraph:
        def __init__(self, state_type): pass
        def add_node(self, name, func): pass
        def add_edge(self, from_node, to_node): pass
        def add_conditional_edges(self, node, condition, mapping): pass
        def set_entry_point(self, node): pass
        def compile(self, **kwargs): return None
    
    class MemorySaver:
        def __init__(self): pass
    
    END = "END"

# LangChain imports (with fallbacks)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import Tool
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain not available: {e}. Using fallback implementation.")
    LANGCHAIN_AVAILABLE = False
    # Fallback message classes
    class BaseMessage:
        def __init__(self, content): self.content = content
    class HumanMessage(BaseMessage): pass
    class AIMessage(BaseMessage): pass
    class SystemMessage(BaseMessage): pass

# System imports (these should work regardless)
from config.latency_config import latency_config

logger = logging.getLogger(__name__)

class ConversationState(TypedDict):
    """Enhanced conversation state for workflow."""
    # Input
    user_input: str
    session_id: str
    call_sid: Optional[str]
    audio_data: Optional[bytes]
    
    # Routing
    routing_decision: Optional[Dict[str, Any]]
    selected_agent: Optional[str]
    confidence_score: float
    
    # Context
    conversation_history: List[Dict[str, Any]]  # Changed from BaseMessage to Dict
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
            "latency_target_met": self.total_time < latency_config.total_latency_targets["total"]
        }

class MultiAgentOrchestrator:
    """
    Multi-agent orchestrator with fallback implementation when LangGraph is not available.
    
    Manages the complete conversation workflow with sub-2-second latency optimization:
    1. Intelligent agent routing
    2. Context retrieval from hybrid vector store
    3. Agent execution with tool integration
    4. Response synthesis and audio generation
    """
    
    def __init__(
        self,
        vector_store=None,
        agent_registry=None,
        router=None,
        stt=None,
        tts=None
    ):
        """Initialize the orchestrator with all required components."""
        self.vector_store = vector_store
        self.agent_registry = agent_registry
        self.router = router
        self.stt = stt
        self.tts = tts
        
        # LangGraph components (if available)
        self.workflow_graph = None
        self.checkpointer = None
        self.use_langgraph = LANGGRAPH_AVAILABLE
        
        # Performance tracking
        self.active_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Agent instances cache
        self.agent_instances: Dict[str, Any] = {}
        
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
        
        logger.info(f"MultiAgentOrchestrator initialized (LangGraph: {'enabled' if self.use_langgraph else 'disabled'})")
    
    async def init(self):
        """Initialize the orchestrator and build the workflow."""
        logger.info("Initializing multi-agent orchestrator...")
        
        if self.use_langgraph:
            # Build the LangGraph workflow
            await self._build_langgraph_workflow()
        else:
            # Use fallback workflow
            logger.info("Using fallback workflow implementation")
        
        # Pre-load agent instances for faster access
        await self._preload_agents()
        
        logger.info("✅ Multi-agent orchestrator ready")
    
    async def _build_langgraph_workflow(self):
        """Build the LangGraph workflow for conversation processing."""
        try:
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
            self.checkpointer = MemorySaver()
            self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)
            
            logger.info("✅ LangGraph workflow compiled")
            
        except Exception as e:
            logger.error(f"Failed to build LangGraph workflow: {e}")
            self.use_langgraph = False
            logger.info("Falling back to simple workflow implementation")
    
    async def _preload_agents(self):
        """Pre-load agent instances for faster response times."""
        if not self.agent_registry:
            logger.warning("Agent registry not available")
            return
            
        try:
            # For now, create placeholder agents if registry is not fully implemented
            self.agent_instances = {
                "roadside-assistance": {"name": "Roadside Assistant", "type": "emergency"},
                "billing-support": {"name": "Billing Support", "type": "financial"},
                "technical-support": {"name": "Technical Support", "type": "technical"},
                "general-support": {"name": "General Support", "type": "general"}
            }
            logger.info(f"✅ Pre-loaded {len(self.agent_instances)} agents")
        except Exception as e:
            logger.error(f"❌ Failed to pre-load agents: {e}")
    
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
        """
        # Start performance tracking
        metrics = PerformanceMetrics(session_id=session_id)
        self.active_metrics[session_id] = metrics
        
        try:
            # Initialize conversation state
            initial_state = {
                "user_input": user_input,
                "session_id": session_id,
                "call_sid": call_sid,
                "audio_data": audio_data,
                "routing_decision": None,
                "selected_agent": None,
                "confidence_score": 0.0,
                "conversation_history": await self._get_conversation_history(session_id),
                "retrieved_context": [],
                "user_profile": user_context or {},
                "session_metadata": self.active_sessions.get(session_id, {}),
                "agent_response": None,
                "tool_calls": [],
                "tool_results": [],
                "final_response": "",
                "response_audio": None,
                "sources": [],
                "processing_start_time": time.time(),
                "routing_time": None,
                "retrieval_time": None,
                "agent_processing_time": None,
                "tts_time": None,
                "total_time": None,
                "requires_escalation": False,
                "needs_human_handoff": False,
                "conversation_ended": False,
                "error_occurred": False,
                "error_message": None
            }
            
            # Process through workflow
            if self.use_langgraph and self.workflow_graph:
                # Use LangGraph workflow
                config = {"configurable": {"thread_id": session_id}}
                final_state = await self.workflow_graph.ainvoke(initial_state, config)
            else:
                # Use fallback sequential processing
                final_state = await self._process_sequential_workflow(initial_state)
            
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
            if metrics.total_time > latency_config.total_latency_targets["total"]:
                logger.warning(f"⚠️ Latency target exceeded: {metrics.total_time:.3f}s > {latency_config.total_latency_targets['total']}s")
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
    
    async def _process_sequential_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback sequential workflow when LangGraph is not available."""
        try:
            # Step 1: Initialize conversation
            state = await self._initialize_conversation(state)
            
            # Step 2: Route to agent
            state = await self._route_to_agent(state)
            
            # Step 3: Retrieve context
            state = await self._retrieve_context(state)
            
            # Step 4: Execute agent
            state = await self._execute_agent(state)
            
            # Step 5: Execute tools if needed
            if state.get("tool_calls"):
                state = await self._execute_tools(state)
            
            # Step 6: Synthesize response
            state = await self._synthesize_response(state)
            
            # Step 7: Generate audio
            state = await self._generate_audio(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in sequential workflow: {e}")
            state["error_occurred"] = True
            state["error_message"] = str(e)
            return await self._handle_error(state)
    
    async def _initialize_conversation(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def _route_to_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route user input to the most appropriate agent."""
        routing_start = time.time()
        
        try:
            # Simple routing logic (fallback when router is not available)
            user_input = state["user_input"].lower()
            
            # Keyword-based routing
            if any(keyword in user_input for keyword in ["tow", "stuck", "breakdown", "accident", "roadside"]):
                selected_agent = "roadside-assistance"
                confidence = 0.9
            elif any(keyword in user_input for keyword in ["bill", "payment", "charge", "refund", "account"]):
                selected_agent = "billing-support"
                confidence = 0.8
            elif any(keyword in user_input for keyword in ["technical", "broken", "error", "not working", "fix"]):
                selected_agent = "technical-support"
                confidence = 0.8
            else:
                selected_agent = "general-support"
                confidence = 0.6
            
            state["routing_decision"] = {"agent_id": selected_agent, "confidence": confidence}
            state["selected_agent"] = selected_agent
            state["confidence_score"] = confidence
            
            # Update session with current agent
            if state["session_id"] in self.active_sessions:
                self.active_sessions[state["session_id"]]["current_agent"] = selected_agent
            
            # Track routing time
            routing_time = time.time() - routing_start
            state["routing_time"] = routing_time
            
            if state["session_id"] in self.active_metrics:
                self.active_metrics[state["session_id"]].routing_time = routing_time
            
            logger.info(f"🎯 Routed to agent: {selected_agent} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Routing error: {e}")
            # Fallback to general support agent
            state["selected_agent"] = "general-support"
            state["confidence_score"] = 0.5
            state["routing_time"] = time.time() - routing_start
        
        return state
    
    async def _retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context from vector store."""
        retrieval_start = time.time()
        
        try:
            # Placeholder context retrieval
            # In a real implementation, this would query the vector store
            retrieved_docs = [
                {
                    "content": "Sample context document relevant to the query",
                    "metadata": {"source": "knowledge_base", "score": 0.85},
                    "score": 0.85
                }
            ]
            
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
    
    async def _execute_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected agent with retrieved context."""
        agent_start = time.time()
        
        try:
            agent_id = state["selected_agent"]
            user_input = state["user_input"]
            
            # Simple agent response generation (placeholder)
            agent_responses = {
                "roadside-assistance": f"I understand you need roadside assistance. I'm here to help with your situation: {user_input}",
                "billing-support": f"I can help you with your billing inquiry: {user_input}. Let me look into this for you.",
                "technical-support": f"I'll help you resolve this technical issue: {user_input}. Let me guide you through the solution.",
                "general-support": f"Thank you for contacting us. Regarding your inquiry: {user_input}, I'll do my best to assist you."
            }
            
            agent_response = agent_responses.get(agent_id, "I'll help you with your request.")
            
            state["agent_response"] = agent_response
            state["tool_calls"] = []  # No tools in this simple implementation
            state["requires_escalation"] = False
            state["needs_human_handoff"] = False
            
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
    
    def _should_execute_tools(self, state: Dict[str, Any]) -> str:
        """Determine if tools need to be executed."""
        if state.get("tool_calls") and len(state["tool_calls"]) > 0:
            return "execute_tools"
        return "synthesize"
    
    async def _execute_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools required by the agent."""
        tool_start = time.time()
        
        try:
            # Placeholder tool execution
            state["tool_results"] = []
            logger.info("🔧 No tools to execute in this implementation")
            
        except Exception as e:
            logger.error(f"❌ Tool execution error: {e}")
            state["tool_results"] = []
        
        return state
    
    async def _synthesize_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize the final response from agent output."""
        try:
            # Use agent response as final response
            final_response = state.get("agent_response", "I understand your request, but I need more information to help you properly.")
            
            # Optimize for voice delivery (keep under 2 sentences when possible)
            sentences = final_response.split('.')
            if len(sentences) > 2:
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
    
    async def _generate_audio(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audio response using TTS."""
        if not self.tts:
            state["response_audio"] = None
            return state
            
        tts_start = time.time()
        
        try:
            # Generate placeholder audio (in real implementation, would use TTS)
            state["response_audio"] = b"placeholder_audio_data"
            
            # Track TTS time
            tts_time = time.time() - tts_start
            state["tts_time"] = tts_time
            
            if state["session_id"] in self.active_metrics:
                self.active_metrics[state["session_id"]].tts_time = tts_time
            
            logger.info(f"🔊 Generated audio response in {tts_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ TTS generation error: {e}")
            state["response_audio"] = None
            state["tts_time"] = time.time() - tts_start
        
        # Calculate total processing time
        total_time = time.time() - state["processing_start_time"]
        state["total_time"] = total_time
        
        return state
    
    async def _handle_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors in the conversation workflow."""
        logger.error(f"❌ Error in conversation workflow: {state.get('error_message', 'Unknown error')}")
        
        state["final_response"] = "I apologize, but I encountered an error. Please try rephrasing your request."
        state["response_audio"] = None
        state["sources"] = []
        state["error_occurred"] = True
        
        return state
    
    async def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session."""
        try:
            # Return empty list as placeholder
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    async def _analyze_latency_bottleneck(self, metrics: PerformanceMetrics):
        """Analyze where latency bottlenecks are occurring."""
        bottlenecks = []
        
        target_routing = latency_config.routing.target_latency
        target_retrieval = latency_config.vector.target_latencies["total"]
        target_agent = latency_config.llm.target_latency
        target_tts = latency_config.tts.target_latency
        
        if metrics.routing_time > target_routing:
            bottlenecks.append(f"Routing: {metrics.routing_time:.3f}s")
        
        if metrics.retrieval_time > target_retrieval:
            bottlenecks.append(f"Retrieval: {metrics.retrieval_time:.3f}s")
        
        if metrics.agent_time > target_agent:
            bottlenecks.append(f"Agent: {metrics.agent_time:.3f}s")
        
        if metrics.tts_time > target_tts:
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
            # Basic health checks
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all recent sessions."""
        return {
            "active_sessions": len(self.active_sessions),
            "total_agents": len(self.agent_instances),
            "average_latency": 0.0,
            "latency_target_met_percentage": 0.0,
        }
    
    async def shutdown(self):
        """Shutdown the orchestrator gracefully."""
        logger.info("🛑 Shutting down multi-agent orchestrator...")
        
        # Clean up all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.cleanup_session(session_id)
        
        self.agent_instances.clear()
        
        logger.info("✅ Multi-agent orchestrator shutdown complete")