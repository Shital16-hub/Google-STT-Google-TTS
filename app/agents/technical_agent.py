"""
Technical Support Agent - Patience-Optimized for Complex Technical Issues
Part of the Revolutionary Multi-Agent Voice AI System with <377ms latency target.
Specialized for technical troubleshooting with step-by-step guidance and unlimited patience.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

from app.agents.base_agent import BaseAgent, AgentResponse, AgentCapability
from app.tools.orchestrator import ComprehensiveToolOrchestrator, ToolResult
from app.vector_db.hybrid_vector_system import HybridVectorSystem

logger = logging.getLogger(__name__)

class TechnicalComplexity(str, Enum):
    """Technical complexity levels for adaptive response."""
    BASIC = "basic"              # Simple setup/configuration
    INTERMEDIATE = "intermediate" # Multi-step procedures
    ADVANCED = "advanced"        # Complex troubleshooting
    EXPERT = "expert"           # System-level diagnostics

class TechnicalCategory(str, Enum):
    """Categories of technical issues."""
    SETUP = "setup"                    # Initial configuration
    TROUBLESHOOTING = "troubleshooting" # Problem diagnosis
    INSTALLATION = "installation"      # Software/hardware installation
    CONFIGURATION = "configuration"    # System configuration
    PERFORMANCE = "performance"        # Performance optimization
    INTEGRATION = "integration"        # Third-party integrations
    MAINTENANCE = "maintenance"        # Ongoing maintenance
    SECURITY = "security"             # Security-related issues

@dataclass
class TechnicalSession:
    """Tracks technical support session state."""
    session_id: str
    complexity_level: TechnicalComplexity = TechnicalComplexity.BASIC
    category: TechnicalCategory = TechnicalCategory.TROUBLESHOOTING
    current_step: int = 0
    total_steps: int = 0
    steps_completed: List[str] = field(default_factory=list)
    user_skill_level: str = "beginner"  # beginner, intermediate, advanced
    previous_attempts: List[str] = field(default_factory=list)
    session_start: float = field(default_factory=time.time)
    patience_level: str = "maximum"  # Always maximum for technical issues
    clarification_count: int = 0
    repetition_count: int = 0

@dataclass
class TechnicalResponse:
    """Enhanced response for technical support."""
    response: str
    step_by_step: bool = False
    current_step: Optional[str] = None
    next_step: Optional[str] = None
    visual_aid_needed: bool = False
    follow_up_required: bool = False
    estimated_time: Optional[str] = None
    difficulty_level: TechnicalComplexity = TechnicalComplexity.BASIC
    prerequisites: List[str] = field(default_factory=list)
    troubleshooting_tips: List[str] = field(default_factory=list)

class TechnicalSupportAgent(BaseAgent):
    """
    Patience-optimized technical support agent for complex technical issues.
    
    Key Features:
    - Unlimited patience with step-by-step guidance
    - Adaptive complexity based on user skill level
    - Comprehensive troubleshooting methodologies
    - Integration with technical knowledge base
    - Tool orchestration for diagnostics and ticket creation
    """
    
    def __init__(self,
                 agent_id: str = "technical-support-v2",
                 hybrid_vector_system: Optional[HybridVectorSystem] = None,
                 tool_orchestrator: Optional[ComprehensiveToolOrchestrator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the technical support agent with patience optimization."""
        
        # Initialize base agent with technical specialization
        super().__init__(
            agent_id=agent_id,
            agent_type="technical_support",
            specialization="technical_troubleshooting",
            capabilities=[
                AgentCapability.KNOWLEDGE_RETRIEVAL,
                AgentCapability.TOOL_INTEGRATION,
                AgentCapability.STEP_BY_STEP_GUIDANCE,
                AgentCapability.CONTEXT_AWARENESS,
                AgentCapability.PATIENCE_OPTIMIZATION
            ],
            hybrid_vector_system=hybrid_vector_system,
            tool_orchestrator=tool_orchestrator,
            config=config or {}
        )
        
        # Technical support specific configuration
        self.personality_profile = "patient_instructional_expert"
        self.response_style = "detailed_step_by_step"
        self.patience_level = "unlimited"
        self.clarification_tolerance = 999  # Unlimited clarifications
        self.repetition_tolerance = 999     # Unlimited repetitions
        
        # Active technical sessions
        self.active_sessions: Dict[str, TechnicalSession] = {}
        
        # Technical knowledge domains
        self.knowledge_domains = [
            "software_installation",
            "hardware_troubleshooting",
            "network_configuration",
            "system_optimization",
            "security_setup",
            "integration_support",
            "performance_tuning",
            "backup_recovery",
            "user_account_management",
            "application_configuration"
        ]
        
        # Voice settings optimized for technical explanations
        self.voice_settings = {
            "tts_voice": "en-US-Neural2-D",  # Patient, clear male voice
            "speaking_rate": 0.9,            # Slightly slower for technical content
            "pitch_adjustment": 0.0,
            "volume_gain": 2.0,
            "instructional_mode": True,      # Special mode for step-by-step
            "pause_between_steps": 1.5,     # Longer pauses between steps
            "emphasis_on_keywords": True,    # Emphasize important technical terms
            "confirmation_prompts": True     # Ask for confirmation after each step
        }
        
        # Technical tools available
        self.available_tools = [
            "create_support_ticket",
            "run_diagnostics_workflow",
            "schedule_callback_workflow",
            "technical_knowledge_search",
            "system_health_check",
            "generate_troubleshooting_guide",
            "escalate_to_specialist",
            "remote_assistance_setup"
        ]
        
        # Initialize technical patterns
        self._initialize_technical_patterns()
        
        logger.info(f"Technical Support Agent initialized with unlimited patience mode")
    
    def _initialize_technical_patterns(self):
        """Initialize patterns for technical issue recognition."""
        self.technical_patterns = {
            # Installation issues
            "installation": {
                "keywords": ["install", "setup", "download", "configure", "deployment"],
                "complexity_indicators": ["failed", "error", "won't work", "can't install"],
                "common_solutions": ["check_requirements", "verify_permissions", "clear_cache"]
            },
            
            # Configuration issues
            "configuration": {
                "keywords": ["config", "settings", "preferences", "options", "parameters"],
                "complexity_indicators": ["not working", "incorrect", "won't save", "reset"],
                "common_solutions": ["check_syntax", "verify_paths", "restart_service"]
            },
            
            # Performance issues
            "performance": {
                "keywords": ["slow", "fast", "optimization", "speed", "performance", "lag"],
                "complexity_indicators": ["very slow", "crashes", "freezes", "memory"],
                "common_solutions": ["optimize_settings", "check_resources", "update_drivers"]
            },
            
            # Integration issues
            "integration": {
                "keywords": ["connect", "integrate", "sync", "api", "webhook", "plugin"],
                "complexity_indicators": ["won't connect", "authentication", "timeout", "failed"],
                "common_solutions": ["check_credentials", "verify_endpoints", "test_connection"]
            }
        }
    
    async def process_query(self,
                          query: str,
                          context: Optional[Dict[str, Any]] = None,
                          session_id: Optional[str] = None) -> AgentResponse:
        """
        Process technical support query with patience and step-by-step guidance.
        
        Args:
            query: User's technical question or issue
            context: Conversation context and user information
            session_id: Session identifier for tracking multi-step interactions
            
        Returns:
            AgentResponse with technical guidance and next steps
        """
        start_time = time.time()
        
        try:
            # Initialize or get existing session
            session_id = session_id or str(uuid.uuid4())
            tech_session = self._get_or_create_session(session_id, context)
            
            # Analyze technical query complexity and category
            analysis = await self._analyze_technical_query(query, context)
            
            # Update session with analysis
            tech_session.complexity_level = analysis["complexity"]
            tech_session.category = analysis["category"]
            tech_session.user_skill_level = analysis.get("user_skill_level", "beginner")
            
            # Search technical knowledge base
            knowledge_results = await self._search_technical_knowledge(
                query, 
                tech_session.category,
                tech_session.complexity_level
            )
            
            # Determine if this is a continuation of previous steps
            is_continuation = self._is_step_continuation(query, tech_session)
            
            if is_continuation:
                response = await self._handle_step_continuation(query, tech_session, knowledge_results)
            else:
                response = await self._generate_initial_technical_response(
                    query, tech_session, knowledge_results, analysis
                )
            
            # Update session state
            self._update_session_state(tech_session, response)
            
            # Determine tools needed
            tools_needed = self._determine_tools_needed(analysis, tech_session)
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                response=response.response,
                confidence=0.95,  # High confidence for technical expertise
                sources=knowledge_results.get("sources", []),
                processing_time_ms=processing_time,
                requires_followup=response.follow_up_required,
                context_used=knowledge_results.get("context", {}),
                tools_needed=tools_needed,
                metadata={
                    "session_id": session_id,
                    "complexity_level": tech_session.complexity_level.value,
                    "category": tech_session.category.value,
                    "current_step": tech_session.current_step,
                    "total_steps": tech_session.total_steps,
                    "user_skill_level": tech_session.user_skill_level,
                    "step_by_step": response.step_by_step,
                    "estimated_time": response.estimated_time,
                    "prerequisites": response.prerequisites,
                    "visual_aid_needed": response.visual_aid_needed
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing technical query: {e}", exc_info=True)
            
            # Return helpful error response
            return AgentResponse(
                response="I apologize, but I encountered an issue processing your technical question. "
                       "Let me help you step by step. Can you please describe the specific problem "
                       "you're experiencing in simple terms?",
                confidence=0.8,
                sources=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                requires_followup=True,
                error=str(e)
            )
    
    def _get_or_create_session(self, session_id: str, context: Optional[Dict[str, Any]]) -> TechnicalSession:
        """Get existing session or create new one."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = TechnicalSession(
                session_id=session_id,
                user_skill_level=context.get("user_skill_level", "beginner") if context else "beginner"
            )
        
        return self.active_sessions[session_id]
    
    async def _analyze_technical_query(self, 
                                     query: str, 
                                     context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze technical query to determine complexity and category."""
        query_lower = query.lower()
        
        # Determine category
        category = TechnicalCategory.TROUBLESHOOTING  # Default
        max_score = 0
        
        for cat, patterns in self.technical_patterns.items():
            score = sum(1 for keyword in patterns["keywords"] if keyword in query_lower)
            if score > max_score:
                max_score = score
                category = TechnicalCategory(cat)
        
        # Determine complexity
        complexity = TechnicalComplexity.BASIC
        
        # Check for complexity indicators
        complexity_indicators = []
        for patterns in self.technical_patterns.values():
            complexity_indicators.extend(patterns["complexity_indicators"])
        
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        error_keywords = ["error", "failed", "won't work", "broken", "crash", "freeze"]
        error_count = sum(1 for keyword in error_keywords if keyword in query_lower)
        
        if indicator_count >= 3 or error_count >= 2:
            complexity = TechnicalComplexity.ADVANCED
        elif indicator_count >= 2 or error_count >= 1:
            complexity = TechnicalComplexity.INTERMEDIATE
        elif "setup" in query_lower or "install" in query_lower:
            complexity = TechnicalComplexity.BASIC
        
        # Determine user skill level from context or query
        user_skill_level = "beginner"
        if context and "user_preferences" in context:
            user_skill_level = context["user_preferences"].get("technical_skill", "beginner")
        
        # Adjust based on query language complexity
        technical_terms = ["api", "configuration", "parameters", "integration", "authentication", 
                          "debugging", "optimization", "deployment", "architecture"]
        tech_term_count = sum(1 for term in technical_terms if term in query_lower)
        
        if tech_term_count >= 3:
            user_skill_level = "advanced"
        elif tech_term_count >= 1:
            user_skill_level = "intermediate"
        
        return {
            "category": category,
            "complexity": complexity,
            "user_skill_level": user_skill_level,
            "technical_terms": tech_term_count,
            "error_indicators": error_count,
            "requires_step_by_step": complexity in [TechnicalComplexity.INTERMEDIATE, TechnicalComplexity.ADVANCED]
        }
    
    async def _search_technical_knowledge(self, 
                                        query: str, 
                                        category: TechnicalCategory,
                                        complexity: TechnicalComplexity) -> Dict[str, Any]:
        """Search technical knowledge base with category and complexity filtering."""
        if not self.hybrid_vector_system:
            return {"sources": [], "context": {}}
        
        try:
            # Create enhanced query with technical context
            enhanced_query = f"{category.value} {complexity.value} {query}"
            
            # Search with agent-specific namespace
            search_result = await self.hybrid_vector_system.hybrid_search(
                query_vector=await self._vectorize_query(enhanced_query),
                agent_id=self.agent_id,
                top_k=5,
                similarity_threshold=0.7,
                filters={
                    "category": category.value,
                    "complexity": complexity.value,
                    "agent_type": "technical_support"
                }
            )
            
            if search_result and search_result.vectors:
                return {
                    "sources": [
                        {
                            "text": vector.get("text", ""),
                            "metadata": vector.get("metadata", {}),
                            "score": score
                        }
                        for vector, score in zip(search_result.vectors, search_result.scores)
                    ],
                    "context": {
                        "search_time_ms": search_result.search_time_ms,
                        "tier_used": search_result.tier_used.value,
                        "total_results": search_result.total_results
                    }
                }
            
            return {"sources": [], "context": {}}
            
        except Exception as e:
            logger.error(f"Error searching technical knowledge: {e}")
            return {"sources": [], "context": {}}
    
    def _is_step_continuation(self, query: str, session: TechnicalSession) -> bool:
        """Determine if this query is continuing a previous step-by-step process."""
        if session.current_step == 0:
            return False
        
        continuation_indicators = [
            "done", "completed", "finished", "next", "what's next", "then what",
            "didn't work", "still not working", "same problem", "no change",
            "yes", "okay", "ok", "ready", "continue"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in continuation_indicators)
    
    async def _handle_step_continuation(self,
                                      query: str,
                                      session: TechnicalSession,
                                      knowledge_results: Dict[str, Any]) -> TechnicalResponse:
        """Handle continuation of step-by-step process."""
        query_lower = query.lower()
        
        # Check if previous step was successful
        if any(word in query_lower for word in ["done", "completed", "finished", "works", "working"]):
            # Move to next step
            session.current_step += 1
            session.steps_completed.append(f"Step {session.current_step - 1}")
            
            if session.current_step > session.total_steps:
                # Process completed
                return TechnicalResponse(
                    response=f"Excellent! You've successfully completed all {session.total_steps} steps. "
                           f"Your technical issue should now be resolved. Is everything working as expected? "
                           f"If you encounter any other issues, I'm here to help.",
                    step_by_step=False,
                    follow_up_required=True,
                    difficulty_level=session.complexity_level
                )
            else:
                # Provide next step
                next_step = await self._get_next_step(session, knowledge_results)
                return TechnicalResponse(
                    response=f"Perfect! Now let's move to step {session.current_step} of {session.total_steps}. "
                           f"{next_step}",
                    step_by_step=True,
                    current_step=f"Step {session.current_step}",
                    follow_up_required=True,
                    difficulty_level=session.complexity_level
                )
        
        elif any(word in query_lower for word in ["didn't work", "not working", "problem", "error", "failed"]):
            # Previous step failed, provide troubleshooting
            session.clarification_count += 1
            
            troubleshooting_response = await self._provide_troubleshooting(session, query, knowledge_results)
            return TechnicalResponse(
                response=f"I understand step {session.current_step} isn't working as expected. Let me help you troubleshoot this. "
                       f"{troubleshooting_response} Take your time, and let me know when you're ready to try again.",
                step_by_step=True,
                current_step=f"Step {session.current_step} (Troubleshooting)",
                follow_up_required=True,
                difficulty_level=session.complexity_level,
                troubleshooting_tips=await self._get_troubleshooting_tips(session.category)
            )
        
        else:
            # User needs clarification or repetition
            session.repetition_count += 1
            
            clarification = await self._provide_step_clarification(session, query, knowledge_results)
            return TechnicalResponse(
                response=f"Of course! Let me explain step {session.current_step} in more detail. {clarification} "
                       f"Please take your time, and let me know if you need me to break this down further.",
                step_by_step=True,
                current_step=f"Step {session.current_step} (Clarification)",
                follow_up_required=True,
                difficulty_level=session.complexity_level
            )
    
    async def _generate_initial_technical_response(self,
                                                 query: str,
                                                 session: TechnicalSession,
                                                 knowledge_results: Dict[str, Any],
                                                 analysis: Dict[str, Any]) -> TechnicalResponse:
        """Generate initial technical response with step-by-step approach."""
        
        # Determine if step-by-step approach is needed
        requires_steps = analysis.get("requires_step_by_step", False)
        
        if requires_steps:
            # Generate step-by-step solution
            steps = await self._generate_solution_steps(query, session, knowledge_results)
            session.total_steps = len(steps)
            session.current_step = 1
            
            # Provide introduction and first step
            intro = self._generate_patient_introduction(session)
            first_step = steps[0] if steps else "Let's start by identifying the exact issue you're experiencing."
            
            response = f"{intro} {first_step}"
            
            return TechnicalResponse(
                response=response,
                step_by_step=True,
                current_step="Step 1",
                next_step=steps[1] if len(steps) > 1 else None,
                follow_up_required=True,
                estimated_time=self._estimate_completion_time(session.total_steps, session.complexity_level),
                difficulty_level=session.complexity_level,
                prerequisites=await self._get_prerequisites(session.category)
            )
        else:
            # Simple direct response
            response = await self._generate_direct_response(query, knowledge_results, session)
            
            return TechnicalResponse(
                response=response,
                step_by_step=False,
                follow_up_required=True,
                difficulty_level=session.complexity_level
            )
    
    def _generate_patient_introduction(self, session: TechnicalSession) -> str:
        """Generate patient, encouraging introduction."""
        intros = {
            TechnicalComplexity.BASIC: "I'll help you with this step by step. Don't worry, this is straightforward and we'll get it working perfectly.",
            TechnicalComplexity.INTERMEDIATE: "I'm here to guide you through this process patiently. We'll take it one step at a time, and I'll make sure you understand each part.",
            TechnicalComplexity.ADVANCED: "This is a bit more complex, but don't worry - I'll walk you through it carefully. We'll break it down into manageable steps, and you can ask questions anytime.",
            TechnicalComplexity.EXPERT: "This is quite technical, but I'll explain everything clearly and we'll work through it together. Please don't hesitate to ask for clarification at any point."
        }
        
        return intros.get(session.complexity_level, intros[TechnicalComplexity.BASIC])
    
    async def _generate_solution_steps(self,
                                     query: str,
                                     session: TechnicalSession,
                                     knowledge_results: Dict[str, Any]) -> List[str]:
        """Generate detailed solution steps based on the technical issue."""
        
        # This would integrate with your knowledge base to get specific steps
        # For now, providing a framework that would be populated from your vector database
        
        category_steps = {
            TechnicalCategory.INSTALLATION: [
                "First, let's verify your system meets the minimum requirements.",
                "Next, we'll download the correct version for your system.",
                "Now we'll run the installer with administrator privileges.",
                "Finally, we'll verify the installation was successful."
            ],
            TechnicalCategory.CONFIGURATION: [
                "Let's start by accessing the configuration settings.",
                "We'll back up your current settings for safety.",
                "Now we'll make the necessary changes step by step.",
                "Finally, we'll test the new configuration."
            ],
            TechnicalCategory.TROUBLESHOOTING: [
                "First, let's identify the exact symptoms of the problem.",
                "We'll check the most common causes one by one.",
                "Let's try the first solution method.",
                "If needed, we'll move to more advanced troubleshooting."
            ]
        }
        
        base_steps = category_steps.get(session.category, category_steps[TechnicalCategory.TROUBLESHOOTING])
        
        # Adapt steps based on complexity and user skill level
        if session.complexity_level == TechnicalComplexity.ADVANCED:
            # Add more detailed steps for complex issues
            enhanced_steps = []
            for step in base_steps:
                enhanced_steps.append(step)
                if "verify" in step.lower() or "check" in step.lower():
                    enhanced_steps.append("Let me know what you see, and I'll help interpret the results.")
            return enhanced_steps
        
        return base_steps
    
    def _estimate_completion_time(self, total_steps: int, complexity: TechnicalComplexity) -> str:
        """Estimate time to complete the technical process."""
        base_time_per_step = {
            TechnicalComplexity.BASIC: 2,        # 2 minutes per step
            TechnicalComplexity.INTERMEDIATE: 3,  # 3 minutes per step
            TechnicalComplexity.ADVANCED: 5,     # 5 minutes per step
            TechnicalComplexity.EXPERT: 8        # 8 minutes per step
        }
        
        time_per_step = base_time_per_step.get(complexity, 3)
        total_minutes = total_steps * time_per_step
        
        if total_minutes < 5:
            return "about 5 minutes"
        elif total_minutes < 15:
            return f"about {total_minutes} minutes"
        elif total_minutes < 60:
            return f"about {total_minutes // 5 * 5} to {(total_minutes // 5 + 1) * 5} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            if minutes == 0:
                return f"about {hours} hour{'s' if hours > 1 else ''}"
            else:
                return f"about {hours} hour{'s' if hours > 1 else ''} and {minutes} minutes"
    
    async def _get_prerequisites(self, category: TechnicalCategory) -> List[str]:
        """Get prerequisites for the technical category."""
        prerequisites = {
            TechnicalCategory.INSTALLATION: [
                "Administrator access to your computer",
                "Stable internet connection",
                "At least 1GB free disk space"
            ],
            TechnicalCategory.CONFIGURATION: [
                "Backup of current settings",
                "Basic understanding of the system being configured"
            ],
            TechnicalCategory.INTEGRATION: [
                "API credentials or access keys",
                "Understanding of both systems being integrated"
            ],
            TechnicalCategory.TROUBLESHOOTING: [
                "Description of when the problem started",
                "Any error messages you've seen"
            ]
        }
        
        return prerequisites.get(category, [])
    
    async def _get_troubleshooting_tips(self, category: TechnicalCategory) -> List[str]:
        """Get category-specific troubleshooting tips."""
        tips = {
            TechnicalCategory.INSTALLATION: [
                "Try running as administrator",
                "Temporarily disable antivirus",
                "Check for conflicting software",
                "Verify file download integrity"
            ],
            TechnicalCategory.CONFIGURATION: [
                "Double-check syntax and formatting",
                "Verify file paths are correct",
                "Restart the service after changes",
                "Check permission settings"
            ],
            TechnicalCategory.PERFORMANCE: [
                "Monitor system resources",
                "Check for background processes",
                "Clear temporary files and cache",
                "Update drivers and software"
            ]
        }
        
        return tips.get(category, ["Try restarting the application", "Check for updates"])
    
    def _determine_tools_needed(self, analysis: Dict[str, Any], session: TechnicalSession) -> List[str]:
        """Determine which tools might be needed for this technical issue."""
        tools = []
        
        # Always offer to create a support ticket for complex issues
        if session.complexity_level in [TechnicalComplexity.ADVANCED, TechnicalComplexity.EXPERT]:
            tools.append("create_support_ticket")
        
        # Offer diagnostics for troubleshooting
        if session.category == TechnicalCategory.TROUBLESHOOTING:
            tools.append("run_diagnostics_workflow")
        
        # Offer callback for complex multi-step processes
        if session.total_steps > 5:
            tools.append("schedule_callback_workflow")
        
        # Always offer knowledge search
        tools.append("technical_knowledge_search")
        
        return tools
    
    def _update_session_state(self, session: TechnicalSession, response: TechnicalResponse):
        """Update session state based on the response."""
        session.last_interaction = time.time()
        
        if response.step_by_step and response.current_step:
            # Update step tracking
            pass  # Current step already updated in the response generation
        
        # Track clarifications and repetitions for patience metrics
        # These are already tracked in the continuation handler
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive technical agent statistics."""
        base_stats = await super().get_agent_stats()
        
        # Add technical-specific stats
        total_sessions = len(self.active_sessions)
        avg_complexity = "basic"
        avg_steps = 0
        patience_metrics = {
            "total_clarifications": 0,
            "total_repetitions": 0,
            "avg_clarifications_per_session": 0,
            "avg_repetitions_per_session": 0
        }
        
        if total_sessions > 0:
            complexity_scores = {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}
            avg_complexity_score = sum(
                complexity_scores.get(session.complexity_level.value, 1) 
                for session in self.active_sessions.values()
            ) / total_sessions
            
            if avg_complexity_score >= 3.5:
                avg_complexity = "expert"
            elif avg_complexity_score >= 2.5:
                avg_complexity = "advanced"
            elif avg_complexity_score >= 1.5:
                avg_complexity = "intermediate"
            
            avg_steps = sum(session.total_steps for session in self.active_sessions.values()) / total_sessions
            
            patience_metrics["total_clarifications"] = sum(
                session.clarification_count for session in self.active_sessions.values()
            )
            patience_metrics["total_repetitions"] = sum(
                session.repetition_count for session in self.active_sessions.values()
            )
            patience_metrics["avg_clarifications_per_session"] = patience_metrics["total_clarifications"] / total_sessions
            patience_metrics["avg_repetitions_per_session"] = patience_metrics["total_repetitions"] / total_sessions
        
        technical_stats = {
            "total_technical_sessions": total_sessions,
            "average_complexity_level": avg_complexity,
            "average_steps_per_session": round(avg_steps, 1),
            "patience_metrics": patience_metrics,
            "category_distribution": {
                category.value: sum(
                    1 for session in self.active_sessions.values() 
                    if session.category == category
                ) for category in TechnicalCategory
            },
            "tools_available": len(self.available_tools),
            "knowledge_domains": len(self.knowledge_domains),
            "voice_optimization": {
                "instructional_mode": self.voice_settings["instructional_mode"],
                "speaking_rate": self.voice_settings["speaking_rate"],
                "pause_between_steps": self.voice_settings["pause_between_steps"]
            }
        }
        
        return {**base_stats, **technical_stats}
    
    async def cleanup_session(self, session_id: str):
        """Clean up completed technical support session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session_duration = time.time() - session.session_start
            
            logger.info(f"Technical session completed: {session_id}, "
                       f"Duration: {session_duration:.1f}s, "
                       f"Steps: {session.current_step}/{session.total_steps}, "
                       f"Clarifications: {session.clarification_count}, "
                       f"Repetitions: {session.repetition_count}")
            
            del self.active_sessions[session_id]
    
    async def shutdown(self):
        """Shutdown technical agent and cleanup resources."""
        logger.info(f"Shutting down Technical Support Agent {self.agent_id}")
        
        # Log final session statistics
        if self.active_sessions:
            total_patience_events = sum(
                session.clarification_count + session.repetition_count 
                for session in self.active_sessions.values()
            )
            logger.info(f"Final stats - Active sessions: {len(self.active_sessions)}, "
                       f"Total patience events: {total_patience_events}")
        
        # Clear active sessions
        self.active_sessions.clear()
        
        # Call parent shutdown
        await super().shutdown()
        
        logger.info("✅ Technical Support Agent shutdown complete")