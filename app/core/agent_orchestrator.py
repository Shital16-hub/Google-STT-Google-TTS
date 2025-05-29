# 1. Enhanced Agent Orchestrator with LLM Integration
# File: app/core/agent_orchestrator.py (UPDATED)

"""
FIXED: Enhanced Multi-Agent Orchestrator with Integrated LLM System
Combines existing orchestration with advanced LLM context management and routing
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

# Core system imports
from app.core.state_manager import ConversationStateManager
from app.agents.registry import AgentRegistry
from app.agents.router import IntelligentAgentRouter, RoutingResult
from app.vector_db.hybrid_vector_system import HybridVectorSystem
from app.tools.tool_orchestrator import ComprehensiveToolOrchestrator

# NEW: LLM System Integration
from app.llm.context_manager import LLMContextManager, ContextType
from app.llm.intelligent_router import IntelligentLLMRouter, RoutingStrategy
from app.llm.streaming_handler import LLMStreamingHandler, StreamingConfig, StreamingMode

logger = logging.getLogger(__name__)

@dataclass
class EnhancedOrchestrationResult:
    """Enhanced result with LLM integration"""
    success: bool
    session_id: str
    response: str
    agent_id: str
    confidence: float
    latency_ms: float
    tools_used: List[str]
    sources: List[str]
    quality_score: float
    
    # NEW: LLM Integration fields
    llm_model_used: str
    context_tokens_used: int
    llm_latency_ms: float
    streaming_enabled: bool
    error: Optional[str] = None

class EnhancedMultiAgentOrchestrator:
    """
    FIXED: Enhanced orchestrator with integrated LLM system
    Resolves agent routing override issues and adds intelligent LLM processing
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        agent_router: IntelligentAgentRouter,
        state_manager: ConversationStateManager,
        hybrid_vector_system: HybridVectorSystem,
        stt_system=None,
        tts_engine=None,
        tool_orchestrator: ComprehensiveToolOrchestrator = None,
        target_latency_ms: int = 377
    ):
        """Initialize enhanced orchestrator with LLM integration"""
        self.agent_registry = agent_registry
        self.agent_router = agent_router
        self.state_manager = state_manager
        self.hybrid_vector_system = hybrid_vector_system
        self.stt_system = stt_system
        self.tts_engine = tts_engine
        self.tool_orchestrator = tool_orchestrator
        self.target_latency_ms = target_latency_ms
        
        # NEW: Initialize LLM System Components
        self.llm_context_manager = LLMContextManager(
            enable_persistence=True,
            default_max_tokens=4000,
            compression_threshold=0.9
        )
        
        self.llm_router = IntelligentLLMRouter(
            default_strategy=RoutingStrategy.AGENT_SPECIFIC,
            enable_caching=True,
            cache_ttl_seconds=300
        )
        
        self.llm_streaming_handler = LLMStreamingHandler(
            max_concurrent_streams=50
        )
        
        # Performance tracking
        self.performance_stats = {
            "total_conversations": 0,
            "successful_conversations": 0,
            "avg_latency_ms": 0.0,
            "agent_routing_accuracy": 0.0,
            "llm_integration_success_rate": 0.0,
            "context_compression_ratio": 0.0
        }
        
        self.initialized = False
        logger.info("Enhanced Multi-Agent Orchestrator with LLM integration initialized")
    
    async def initialize(self):
        """Initialize all orchestrator components"""
        logger.info("🚀 Initializing Enhanced Multi-Agent Orchestrator...")
        
        try:
            # Initialize LLM components
            await self.llm_context_manager.initialize()
            await self.llm_router.initialize()
            await self.llm_streaming_handler.initialize()
            
            self.initialized = True
            logger.info("✅ Enhanced orchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Enhanced orchestrator initialization failed: {e}")
            # Continue with degraded functionality
            self.initialized = True
            logger.warning("⚠️ Running with degraded LLM functionality")
    
    async def process_conversation(
        self,
        session_id: str,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        preferred_agent_id: Optional[str] = None
    ) -> EnhancedOrchestrationResult:
        """
        FIXED: Main conversation processing with proper agent routing
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        context = context or {}
        
        logger.info(f"🎯 Processing conversation: {input_text[:100]}...")
        
        try:
            # Step 1: Intelligent Agent Routing (FIXED)
            routing_result = await self._route_to_agent(
                input_text, context, preferred_agent_id
            )
            
            if not routing_result.selected_agent_id:
                raise Exception("No suitable agent found")
            
            # CRITICAL FIX: Respect routing decision
            selected_agent_id = routing_result.selected_agent_id
            logger.info(f"✅ Agent routing decision: {selected_agent_id}")
            
            # Step 2: Get Agent and Validate
            agent = await self.agent_registry.get_agent(selected_agent_id)
            if not agent:
                # FALLBACK: Try to get any available agent
                available_agents = await self.agent_registry.list_active_agents()
                if available_agents:
                    agent = available_agents[0]
                    selected_agent_id = agent.agent_id
                    logger.warning(f"⚠️ Using fallback agent: {selected_agent_id}")
                else:
                    raise Exception("No active agents available")
            
            # Step 3: Prepare Enhanced Context with LLM Integration
            enhanced_context = await self._prepare_enhanced_context(
                session_id, input_text, context, selected_agent_id
            )
            
            # Step 4: Generate Response with LLM Integration
            response_result = await self._generate_llm_enhanced_response(
                session_id, input_text, enhanced_context, selected_agent_id, agent
            )
            
            # Step 5: Update Context and State
            await self._update_conversation_state(
                session_id, input_text, response_result['response'], 
                selected_agent_id, response_result.get('tools_used', [])
            )
            
            # Calculate final metrics
            total_latency = (time.time() - start_time) * 1000
            
            # Update performance stats
            self._update_performance_stats(total_latency, True, selected_agent_id)
            
            return EnhancedOrchestrationResult(
                success=True,
                session_id=session_id,
                response=response_result['response'],
                agent_id=selected_agent_id,  # FIXED: Return actual selected agent
                confidence=routing_result.confidence,
                latency_ms=total_latency,
                tools_used=response_result.get('tools_used', []),
                sources=response_result.get('sources', []),
                quality_score=response_result.get('quality_score', 0.8),
                llm_model_used=response_result.get('llm_model', 'gpt-4o-mini'),
                context_tokens_used=response_result.get('context_tokens', 0),
                llm_latency_ms=response_result.get('llm_latency_ms', 0),
                streaming_enabled=response_result.get('streaming_enabled', False)
            )
            
        except Exception as e:
            logger.error(f"❌ Enhanced orchestration error: {e}", exc_info=True)
            
            # Enhanced fallback with LLM
            fallback_response = await self._generate_fallback_response(
                input_text, context, str(e)
            )
            
            total_latency = (time.time() - start_time) * 1000
            self._update_performance_stats(total_latency, False, "fallback")
            
            return EnhancedOrchestrationResult(
                success=False,
                session_id=session_id,
                response=fallback_response,
                agent_id="fallback",
                confidence=0.3,
                latency_ms=total_latency,
                tools_used=[],
                sources=[],
                quality_score=0.5,
                llm_model_used="fallback",
                context_tokens_used=0,
                llm_latency_ms=0,
                streaming_enabled=False,
                error=str(e)
            )
    
    async def _route_to_agent(
        self, 
        input_text: str, 
        context: Dict[str, Any], 
        preferred_agent_id: Optional[str]
    ) -> RoutingResult:
        """FIXED: Proper agent routing with fallback"""
        
        try:
            # Use preferred agent if specified and valid
            if preferred_agent_id:
                agent = await self.agent_registry.get_agent(preferred_agent_id)
                if agent and agent.status.value == "active":
                    return RoutingResult(
                        selected_agent_id=preferred_agent_id,
                        confidence=1.0,
                        routing_time_ms=5.0,
                        strategy_used="preferred",
                        decision_type="direct_match",
                        alternatives=[],
                        routing_factors={"preferred": True}
                    )
            
            # Use intelligent agent router
            if self.agent_router and self.agent_router.initialized:
                routing_result = await self.agent_router.route_query(
                    query=input_text,
                    context=context,
                    preferred_agent_id=preferred_agent_id
                )
                
                # Validate the routing result
                if routing_result.selected_agent_id:
                    agent = await self.agent_registry.get_agent(routing_result.selected_agent_id)
                    if agent and agent.status.value == "active":
                        logger.info(f"🎯 Intelligent routing selected: {routing_result.selected_agent_id}")
                        return routing_result
            
            # Fallback to simple keyword-based routing
            return await self._simple_keyword_routing(input_text, context)
            
        except Exception as e:
            logger.error(f"Agent routing error: {e}")
            return await self._simple_keyword_routing(input_text, context)
    
    async def _simple_keyword_routing(self, input_text: str, context: Dict[str, Any]) -> RoutingResult:
        """Simple keyword-based routing as fallback"""
        
        input_lower = input_text.lower()
        
        # Roadside assistance keywords
        if any(word in input_lower for word in [
            'tow', 'stuck', 'breakdown', 'accident', 'emergency', 'help', 
            'roadside', 'car', 'vehicle', 'stranded'
        ]):
            selected_agent = "roadside-assistance-v2"
            confidence = 0.9
        
        # Billing support keywords  
        elif any(word in input_lower for word in [
            'bill', 'payment', 'charge', 'refund', 'invoice', 'money', 
            'cost', 'subscription', 'account'
        ]):
            selected_agent = "billing-support-v2"
            confidence = 0.8
        
        # Technical support keywords
        elif any(word in input_lower for word in [
            'error', 'problem', 'not working', 'bug', 'install', 'setup', 
            'login', 'technical', 'support'
        ]):
            selected_agent = "technical-support-v2"
            confidence = 0.8
        
        else:
            # Default to technical support for unknown queries
            selected_agent = "technical-support-v2"
            confidence = 0.5
        
        logger.info(f"🔄 Simple routing selected: {selected_agent} (confidence: {confidence})")
        
        return RoutingResult(
            selected_agent_id=selected_agent,
            confidence=confidence,
            routing_time_ms=2.0,
            strategy_used="keyword_fallback",
            decision_type="fallback",
            alternatives=[],
            routing_factors={"keywords": True}
        )
    
    async def _prepare_enhanced_context(
        self,
        session_id: str,
        input_text: str,
        context: Dict[str, Any],
        agent_id: str
    ) -> Dict[str, Any]:
        """Prepare enhanced context with LLM integration"""
        
        enhanced_context = context.copy()
        
        try:
            # Add current input to context
            await self.llm_context_manager.add_context(
                conversation_id=session_id,
                content=f"User: {input_text}",
                context_type=ContextType.CONVERSATION,
                agent_id=agent_id
            )
            
            # Get optimized context for LLM
            llm_messages = await self.llm_context_manager.get_context_for_llm(
                conversation_id=session_id,
                agent_id=agent_id,
                current_query=input_text,
                model_id="gpt-4o-mini"
            )
            
            enhanced_context.update({
                'llm_messages': llm_messages,
                'agent_id': agent_id,
                'conversation_id': session_id,
                'context_optimized': True
            })
            
        except Exception as e:
            logger.error(f"Context preparation error: {e}")
            # Fallback to basic context
            enhanced_context.update({
                'llm_messages': [
                    {"role": "system", "content": f"You are a helpful {agent_id} assistant."},
                    {"role": "user", "content": input_text}
                ],
                'agent_id': agent_id,
                'conversation_id': session_id,
                'context_optimized': False
            })
        
        return enhanced_context
    
    async def _generate_llm_enhanced_response(
        self,
        session_id: str,
        input_text: str,
        enhanced_context: Dict[str, Any],
        agent_id: str,
        agent
    ) -> Dict[str, Any]:
        """Generate response using LLM integration"""
        
        llm_start = time.time()
        
        try:
            # Get LLM messages
            messages = enhanced_context.get('llm_messages', [])
            
            # Route to optimal LLM model
            llm_response = await self.llm_router.route_and_generate(
                query=input_text,
                context=enhanced_context,
                agent_id=agent_id,
                streaming=False,  # For now, use non-streaming
                max_tokens=300
            )
            
            llm_latency = (time.time() - llm_start) * 1000
            
            # Add response to context
            await self.llm_context_manager.add_context(
                conversation_id=session_id,
                content=f"Assistant: {llm_response.content}",
                context_type=ContextType.CONVERSATION,
                agent_id=agent_id
            )
            
            return {
                'response': llm_response.content,
                'llm_model': llm_response.model_id,
                'llm_latency_ms': llm_latency,
                'context_tokens': llm_response.token_count,
                'quality_score': llm_response.quality_score,
                'tools_used': [],
                'sources': [],
                'streaming_enabled': llm_response.is_streaming
            }
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            
            # Fallback to agent's default response or simple response
            try:
                if agent and hasattr(agent, 'generate_response'):
                    agent_response = await agent.generate_response(input_text, enhanced_context)
                    return {
                        'response': agent_response,
                        'llm_model': 'agent_fallback',
                        'llm_latency_ms': (time.time() - llm_start) * 1000,
                        'context_tokens': 0,
                        'quality_score': 0.7,
                        'tools_used': [],
                        'sources': [],
                        'streaming_enabled': False
                    }
            except:
                pass
            
            # Ultimate fallback
            return await self._get_agent_specific_fallback(agent_id, input_text)
    
    async def _get_agent_specific_fallback(self, agent_id: str, input_text: str) -> Dict[str, Any]:
        """Get agent-specific fallback responses"""
        
        fallback_responses = {
            'roadside-assistance-v2': "I understand you need roadside assistance. I'm here to help with your emergency. Can you tell me your location and what type of assistance you need?",
            'billing-support-v2': "I'm here to help with your billing questions. I can assist with payments, refunds, and account issues. What specific billing matter can I help you with?",
            'technical-support-v2': "I'm here to help with your technical issue. I'll guide you through each step clearly. Let me provide step-by-step guidance for your technical issue. I'll explain each step in simple terms. Let me know if you need help with any of these steps."
        }
        
        response = fallback_responses.get(
            agent_id, 
            "I understand you need assistance. How can I help you today?"
        )
        
        return {
            'response': response,
            'llm_model': 'fallback',
            'llm_latency_ms': 5.0,
            'context_tokens': 0,
            'quality_score': 0.6,
            'tools_used': [],
            'sources': [],
            'streaming_enabled': False
        }
    
    async def _generate_fallback_response(
        self, 
        input_text: str, 
        context: Dict[str, Any], 
        error: str
    ) -> str:
        """Generate intelligent fallback response"""
        
        # Analyze input for context clues
        input_lower = input_text.lower()
        
        if any(word in input_lower for word in ['emergency', 'urgent', 'help', 'stuck']):
            return "I understand this is urgent. I'm here to help you right away. Can you tell me more about your situation?"
        
        elif any(word in input_lower for word in ['payment', 'bill', 'money', 'charge']):
            return "I can help you with your billing or payment question. What specific issue are you experiencing?"
        
        elif any(word in input_lower for word in ['problem', 'error', 'not working']):
            return "I can help you resolve this technical issue. Can you describe what's happening in more detail?"
        
        else:
            return "I apologize for the technical difficulty. I'm here to help you. Can you tell me what you need assistance with?"
    
    async def _update_conversation_state(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        agent_id: str,
        tools_used: List[str]
    ):
        """Update conversation state in state manager"""
        
        try:
            if self.state_manager:
                await self.state_manager.update_conversation(
                    session_id=session_id,
                    user_message=user_input,
                    assistant_message=assistant_response,
                    agent_id=agent_id,
                    tools_used=tools_used
                )
        except Exception as e:
            logger.error(f"State update error: {e}")
    
    def _update_performance_stats(self, latency_ms: float, success: bool, agent_id: str):
        """Update performance statistics"""
        
        self.performance_stats["total_conversations"] += 1
        
        if success:
            self.performance_stats["successful_conversations"] += 1
        
        # Update average latency
        total = self.performance_stats["total_conversations"]
        current_avg = self.performance_stats["avg_latency_ms"]
        self.performance_stats["avg_latency_ms"] = (
            (current_avg * (total - 1) + latency_ms) / total
        )
        
        # Update success rates
        success_rate = (
            self.performance_stats["successful_conversations"] / total
        ) * 100
        self.performance_stats["llm_integration_success_rate"] = success_rate
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        stats = self.performance_stats.copy()
        
        # Add LLM system stats if available
        try:
            if self.llm_context_manager:
                context_metrics = self.llm_context_manager.get_performance_metrics()
                stats.update({
                    'context_manager': context_metrics,
                    'context_compression_ratio': context_metrics.get('compression_effectiveness', 0)
                })
            
            if self.llm_router:
                routing_stats = self.llm_router.get_routing_stats()
                stats.update({
                    'llm_router': routing_stats
                })
            
            if self.llm_streaming_handler:
                streaming_stats = self.llm_streaming_handler.get_performance_metrics()
                stats.update({
                    'streaming_handler': streaming_stats
                })
                
        except Exception as e:
            logger.error(f"Error getting LLM stats: {e}")
        
        return stats
    
    async def shutdown(self):
        """Shutdown orchestrator and all components"""
        logger.info("Shutting down Enhanced Multi-Agent Orchestrator...")
        
        try:
            # Shutdown LLM components
            if self.llm_context_manager:
                await self.llm_context_manager.shutdown()
            
            if self.llm_router:
                await self.llm_router.shutdown()
            
            if self.llm_streaming_handler:
                await self.llm_streaming_handler.shutdown()
            
            self.initialized = False
            logger.info("✅ Enhanced orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")