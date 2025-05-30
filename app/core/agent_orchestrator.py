# 1. Enhanced Agent Orchestrator with LLM Integration
# File: app/core/agent_orchestrator.py (UPDATED)

"""
FIXED: Enhanced Multi-Agent Orchestrator with Integrated LLM System
Combines existing orchestration with advanced LLM context management and routing
"""
import os
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
            if selected_agent_id == "base_agent":
                # Handle BaseAgent specially
                agent = await self._get_or_create_base_agent()
                if not agent:
                    # If we can't create BaseAgent, use any available agent
                    available_agents = await self.agent_registry.list_active_agents()
                    if available_agents:
                        agent = available_agents[0]
                        selected_agent_id = agent.agent_id
                        logger.warning(f"⚠️ BaseAgent unavailable, using fallback: {selected_agent_id}")
                    else:
                        raise Exception("No agents available including BaseAgent")
            else:
                # Handle regular specialized agents
                agent = await self.agent_registry.get_agent(selected_agent_id)
                if not agent:
                    # FALLBACK: Try to get any available agent or BaseAgent
                    available_agents = await self.agent_registry.list_active_agents()
                    if available_agents:
                        agent = available_agents[0]
                        selected_agent_id = agent.agent_id
                        logger.warning(f"⚠️ Using fallback agent: {selected_agent_id}")
                    else:
                        # Last resort - try BaseAgent
                        agent = await self._get_or_create_base_agent()
                        selected_agent_id = "base_agent"
                        if not agent:
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
        """FIXED: Proper routing with BaseAgent support."""
        
        try:
            # Use preferred agent if specified and valid
            if preferred_agent_id:
                if preferred_agent_id == "base_agent":
                    return RoutingResult(
                        selected_agent_id="base_agent",
                        confidence=1.0,
                        routing_time_ms=5.0,
                        strategy_used="preferred_base_agent",
                        decision_type="direct_match",
                        alternatives=[],
                        routing_factors={"preferred": True}
                    )
                elif await self._agent_exists(preferred_agent_id):
                    return RoutingResult(
                        selected_agent_id=preferred_agent_id,
                        confidence=1.0,
                        routing_time_ms=5.0,
                        strategy_used="preferred",
                        decision_type="direct_match",
                        alternatives=[],
                        routing_factors={"preferred": True}
                    )
            
            # Use LLM router if available
            if self.llm_router and self.llm_router.initialized:
                try:
                    routing_result = await self.llm_router.route_intelligently(
                        user_input=input_text,
                        context=context
                    )
                    
                    # Check if routing suggests clarification
                    if (hasattr(routing_result, 'needs_clarification') and routing_result.needs_clarification) or routing_result.confidence < 0.7:
                        return RoutingResult(
                            selected_agent_id="base_agent",
                            confidence=0.6,
                            routing_time_ms=10.0,
                            strategy_used="clarification_needed",
                            decision_type="llm_uncertain",
                            alternatives=[],
                            routing_factors={
                                "needs_clarification": True, 
                                "original_intent": getattr(routing_result, 'intent', 'unclear'),
                                "llm_confidence": routing_result.confidence
                            }
                        )
                    
                    # Return the LLM routing decision
                    return RoutingResult(
                        selected_agent_id=routing_result.agent_id,
                        confidence=routing_result.confidence,
                        routing_time_ms=10.0,
                        strategy_used="llm_intelligent",  
                        decision_type="ai_analysis",
                        alternatives=[],
                        routing_factors={"llm_reasoning": routing_result.reasoning}
                    )
                    
                except Exception as llm_error:
                    logger.error(f"LLM routing failed: {llm_error}")
            
            # Fallback to BaseAgent when other routing fails
            logger.info("Using BaseAgent fallback for clarification")
            return RoutingResult(
                selected_agent_id="base_agent",
                confidence=0.5,
                routing_time_ms=2.0,
                strategy_used="base_agent_fallback",
                decision_type="system_fallback",
                alternatives=[],
                routing_factors={"reason": "routing_system_fallback"}
            )
            
        except Exception as e:
            logger.error(f"Agent routing error: {e}")
            return RoutingResult(
                selected_agent_id="base_agent",
                confidence=0.4,
                routing_time_ms=1.0,
                strategy_used="error_fallback",
                decision_type="error_recovery",
                alternatives=[],
                routing_factors={"error": str(e)}
            )

    async def _agent_exists(self, agent_id: str) -> bool:
        """Check if agent exists in registry."""
        if agent_id == "base_agent":
            return True  # Base agent always exists
        
        try:
            agent = await self.agent_registry.get_agent(agent_id)
            return agent is not None and agent.status.value == "active"
        except:
            return False
    
    async def _get_or_create_base_agent(self) -> 'BaseAgent':
        """Get or create BaseAgent for clarification."""
        
        try:
            # Try to get BaseAgent from registry first
            base_agent = await self.agent_registry.get_agent("base_agent")
            if base_agent:
                return base_agent
            
            # If not found, create a temporary BaseAgent
            logger.info("Creating temporary BaseAgent for clarification")
            
            from app.agents.base_agent import BaseAgent, AgentConfiguration
            
            base_config = AgentConfiguration(
                agent_id="base_agent",
                version="1.0.0",
                specialization={
                    "domain_expertise": "general_assistance_and_clarification",
                    "personality_profile": {
                        "traits": ["helpful", "clarifying", "patient"],
                        "communication_style": "conversational and clarifying"
                    }
                },
                voice_settings={
                    "tts_voice": "en-US-Standard-A",
                    "speaking_rate": 1.0,
                    "pitch": 0.0
                },
                tools=[],
                routing={
                    "primary_keywords": [],
                    "confidence_threshold": 0.3,
                    "fallback_responses": True
                }
            )
            
            base_agent = BaseAgent(
                agent_id="base_agent",
                config=base_config,
                hybrid_vector_system=self.hybrid_vector_system,
                tool_orchestrator=self.tool_orchestrator
            )
            
            await base_agent.initialize()
            
            logger.info("✅ Temporary BaseAgent created")
            return base_agent
            
        except Exception as e:
            logger.error(f"Error creating BaseAgent: {e}")
            # Return None - we'll handle this in the calling code
            return None
    
    async def _simple_keyword_routing(self, input_text: str, context: Dict[str, Any]) -> RoutingResult:
        """FIXED: Pure LLM-based routing - NO hardcoded keywords."""
        
        try:
            # Use a simple, direct LLM call for routing when main LLM router fails
            routing_prompt = f"""
                            You are an intelligent customer service router. Route this customer query to the most appropriate agent.
                            
                            Customer Query: "{input_text}"
                            
                            Available Agents:
                            - roadside-assistance-v2: Vehicle emergencies, breakdowns, towing, accidents
                            - billing-support-v2: Billing, payments, refunds, account issues  
                            - technical-support-v2: Technical problems, setup, troubleshooting
                            
                            Respond with just the agent ID and confidence (0-1):
                            Format: agent_id|confidence
                            
                            Example: roadside-assistance-v2|0.95
                            """
            
            # Create a simple OpenAI client for this specific routing call
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": routing_prompt}],
                    max_tokens=50,
                    temperature=0.1,
                    timeout=5
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Parse the result
                if '|' in result_text:
                    agent_id, confidence_str = result_text.split('|', 1)
                    agent_id = agent_id.strip()
                    confidence = float(confidence_str.strip())
                else:
                    # Fallback parsing
                    agent_id = result_text.strip()
                    confidence = 0.8
                
                # Validate agent_id
                valid_agents = ["roadside-assistance-v2", "billing-support-v2", "technical-support-v2"]
                if agent_id not in valid_agents:
                    agent_id = "technical-support-v2"  # Safe default
                    confidence = 0.5
                
                logger.info(f"🤖 LLM-based routing selected: {agent_id} (confidence: {confidence})")
                
                return RoutingResult(
                    selected_agent_id=agent_id,
                    confidence=confidence,
                    routing_time_ms=2.0,
                    strategy_used="llm_based_routing",
                    decision_type="intelligent_analysis",
                    alternatives=[],
                    routing_factors={"method": "pure_llm"}
                )
                
            except Exception as llm_error:
                logger.error(f"LLM-based routing failed: {llm_error}")
                # Only as absolute last resort - use safest default
                return RoutingResult(
                    selected_agent_id="technical-support-v2",
                    confidence=0.4,
                    routing_time_ms=1.0,
                    strategy_used="emergency_fallback",
                    decision_type="safe_default",
                    alternatives=[],
                    routing_factors={"method": "safe_fallback", "reason": str(llm_error)}
                )
                
        except Exception as e:
            logger.error(f"Routing completely failed: {e}")
            return RoutingResult(
                selected_agent_id="technical-support-v2",
                confidence=0.3,
                routing_time_ms=1.0,
                strategy_used="error_fallback",
                decision_type="error_recovery",
                alternatives=[],
                routing_factors={"error": str(e)}
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
        """FIXED: Generate response with proper error handling for OpenAI failures"""
        
        llm_start = time.time()
        
        try:
            # Check if LLM router is available and working
            if self.llm_router and self.llm_router.initialized:
                try:
                    # Route to optimal LLM model
                    llm_response = await self.llm_router.route_and_generate(
                        query=input_text,
                        context=enhanced_context,
                        agent_id=agent_id,
                        streaming=False,
                        max_tokens=300
                    )
                    
                    llm_latency = (time.time() - llm_start) * 1000
                    
                    # Add response to context
                    if self.llm_context_manager:
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
                    
                except Exception as llm_error:
                    logger.error(f"LLM generation failed: {llm_error}")
                    # Fall through to agent fallback
            
            # Fallback to agent's direct response or simple response
            try:
                if agent and hasattr(agent, 'generate_response'):
                    agent_response = await agent.generate_response(input_text, enhanced_context)
                    return {
                        'response': agent_response,
                        'llm_model': 'agent_direct',
                        'llm_latency_ms': (time.time() - llm_start) * 1000,
                        'context_tokens': 0,
                        'quality_score': 0.7,
                        'tools_used': [],
                        'sources': [],
                        'streaming_enabled': False
                    }
            except Exception as agent_error:
                logger.error(f"Agent response failed: {agent_error}")
            
            # Ultimate fallback - agent-specific responses
            return await self._get_agent_specific_fallback(agent_id, input_text)
            
        except Exception as e:
            logger.error(f"❌ Response generation completely failed: {e}")
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