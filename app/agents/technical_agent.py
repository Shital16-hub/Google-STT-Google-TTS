"""
Technical Support Agent - Patience-Optimized Implementation
==========================================================

Advanced technical support agent specialized in complex troubleshooting scenarios
with patience-optimized conversation patterns and step-by-step guidance.

Features:
- Advanced technical knowledge base with hierarchical troubleshooting
- Step-by-step guidance with verification checkpoints
- Patience-optimized conversation patterns for frustrated users
- Complex diagnostic workflows with adaptive complexity
- Screen sharing integration and remote assistance capabilities
- Comprehensive technical documentation and manual access
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, AsyncIterator
from dataclasses import dataclass
from enum import Enum

from app.agents.base_agent import BaseAgent, AgentResponse, ConversationContext
from app.tools.orchestrator import ComprehensiveToolOrchestrator, WorkflowResult, APIResult
from app.vector_db.hybrid_vector_system import HybridVectorArchitecture, SearchResult
from app.core.state_manager import ConversationStateManager
from app.voice.dual_streaming_tts import DualStreamingTTSEngine

logger = logging.getLogger(__name__)


class TechnicalComplexityLevel(Enum):
    """Technical complexity levels for adaptive support"""
    BASIC = "basic"           # Simple setup, basic configuration
    INTERMEDIATE = "intermediate"  # Moderate troubleshooting
    ADVANCED = "advanced"     # Complex system integration
    EXPERT = "expert"         # Deep technical diagnosis


class TechnicalUrgencyLevel(Enum):
    """Technical urgency levels for priority handling"""
    LOW = "low"               # General questions, optimization
    MEDIUM = "medium"         # Functionality issues, performance
    HIGH = "high"             # Service disruption, critical bugs
    CRITICAL = "critical"     # System down, security breach


@dataclass
class TechnicalDiagnosticStep:
    """Individual diagnostic step in troubleshooting workflow"""
    step_id: str
    title: str
    description: str
    instructions: List[str]
    expected_outcome: str
    verification_method: str
    complexity_level: TechnicalComplexityLevel
    estimated_time_minutes: int
    prerequisites: List[str] = None
    common_issues: List[str] = None
    next_steps_on_success: List[str] = None
    next_steps_on_failure: List[str] = None


@dataclass
class TechnicalSession:
    """Technical support session state management"""
    session_id: str
    user_technical_level: TechnicalComplexityLevel
    current_issue_category: str
    diagnostic_workflow_id: str
    current_step: Optional[TechnicalDiagnosticStep]
    completed_steps: List[str]
    failed_steps: List[str]
    session_notes: List[str]
    start_time: datetime
    estimated_resolution_time: Optional[int]
    escalation_triggers: List[str]
    patience_indicators: Dict[str, Any]
    adaptive_explanations_enabled: bool = True


class AdvancedTechnicalSupportAgent(BaseAgent):
    """
    Advanced Technical Support Agent with Patience-Optimized Interactions
    
    Specialized for complex technical troubleshooting with:
    - Adaptive complexity based on user technical level
    - Step-by-step guidance with verification checkpoints
    - Patience-optimized conversation patterns
    - Comprehensive diagnostic workflows
    - Advanced tool integration for technical operations
    """
    
    def __init__(self):
        super().__init__(
            agent_id="technical-support-v2",
            qdrant_collection="agent-technical-v2",
            specialization_config={
                "domain": "technical_support_troubleshooting",
                "patience_optimization": True,
                "step_by_step_guidance": True,
                "complexity_adaptation": True,
                "diagnostic_workflows": True,
                "time_sensitivity": "flexible_patient"
            }
        )
        
        # Technical Support Specialized Tools
        self.tools = ComprehensiveToolOrchestrator()
        self._initialize_technical_tools()
        
        # Patience and Guidance Systems
        self.patience_tracker = PatienceTracker()
        self.complexity_adapter = ComplexityAdapter()
        self.diagnostic_engine = DiagnosticWorkflowEngine()
        self.explanation_generator = AdaptiveExplanationGenerator()
        
        # Technical Knowledge Systems
        self.knowledge_hierarchy = TechnicalKnowledgeHierarchy()
        self.solution_validator = SolutionValidator()
        self.escalation_manager = TechnicalEscalationManager()
        
        # Session Management
        self.active_sessions: Dict[str, TechnicalSession] = {}
        self.session_analytics = TechnicalSessionAnalytics()
        
        # Voice Optimization for Technical Support
        self.tts_engine = DualStreamingTTSEngine()
        self.technical_voice_config = {
            "speaking_rate": 0.9,  # Slightly slower for technical explanations
            "pause_duration": 1.2,  # Longer pauses between steps
            "emphasis_on_key_terms": True,
            "patient_tone_modulation": True,
            "step_confirmation_prompts": True
        }
        
        logger.info("Advanced Technical Support Agent initialized with patience optimization")

    def _initialize_technical_tools(self):
        """Initialize technical support specific tools"""
        # Advanced diagnostic tools
        self.tools.register_tool("system_diagnostic_workflow", self._run_system_diagnostic)
        self.tools.register_tool("step_by_step_troubleshoot", self._execute_guided_troubleshooting)
        self.tools.register_tool("remote_diagnostic_session", self._initiate_remote_session)
        self.tools.register_tool("knowledge_base_search", self._search_technical_knowledge)
        self.tools.register_tool("solution_validation", self._validate_proposed_solution)
        
        # Communication and documentation tools
        self.tools.register_tool("create_support_ticket", self._create_technical_ticket)
        self.tools.register_tool("escalate_to_specialist", self._escalate_to_technical_specialist)
        self.tools.register_tool("schedule_follow_up", self._schedule_technical_follow_up)
        self.tools.register_tool("generate_technical_summary", self._generate_session_summary)
        
        # User assistance tools
        self.tools.register_tool("assess_user_technical_level", self._assess_technical_competency)
        self.tools.register_tool("adapt_explanation_complexity", self._adapt_explanation_level)
        self.tools.register_tool("provide_visual_guidance", self._generate_visual_instructions)

    async def process_technical_request(self, 
                                      request: str, 
                                      context: ConversationContext,
                                      session_id: Optional[str] = None) -> AgentResponse:
        """
        Process technical support request with patience-optimized approach
        
        Args:
            request: User's technical support request
            context: Current conversation context
            session_id: Optional existing session ID
            
        Returns:
            AgentResponse with patient, step-by-step guidance
        """
        start_time = time.time()
        
        try:
            # Initialize or retrieve technical session
            tech_session = await self._get_or_create_technical_session(
                session_id, request, context
            )
            
            # Analyze request complexity and urgency
            complexity_analysis = await self._analyze_technical_complexity(request, context)
            urgency_analysis = await self._analyze_technical_urgency(request, context)
            
            # Assess user's technical level and patience indicators
            user_tech_level = await self._assess_user_technical_level(request, context)
            patience_score = await self.patience_tracker.assess_user_patience(request, context)
            
            # Update session with current analysis
            tech_session.user_technical_level = user_tech_level
            tech_session.patience_indicators = {
                "current_patience_score": patience_score,
                "frustration_indicators": await self._detect_frustration_indicators(request),
                "complexity_comfort_level": user_tech_level.value,
                "preferred_explanation_style": await self._determine_explanation_style(context)
            }
            
            # Determine optimal response strategy
            response_strategy = await self._determine_response_strategy(
                complexity_analysis, urgency_analysis, user_tech_level, patience_score
            )
            
            # Execute technical support workflow
            support_result = await self._execute_technical_support_workflow(
                request, tech_session, response_strategy, context
            )
            
            # Generate patience-optimized response
            response = await self._generate_patient_technical_response(
                support_result, tech_session, context
            )
            
            # Update session analytics
            await self.session_analytics.record_interaction(
                tech_session, response_strategy, support_result
            )
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Technical request processed in {processing_time:.2f}ms")
            
            return AgentResponse(
                response=response.text,
                confidence_score=response.confidence,
                response_time_ms=processing_time,
                agent_id=self.agent_id,
                context_updated=True,
                tools_used=support_result.tools_executed,
                session_id=tech_session.session_id,
                metadata={
                    "technical_complexity": complexity_analysis.level.value,
                    "user_technical_level": user_tech_level.value,
                    "patience_score": patience_score,
                    "response_strategy": response_strategy.name,
                    "diagnostic_steps_completed": len(tech_session.completed_steps),
                    "estimated_resolution_time": tech_session.estimated_resolution_time,
                    "escalation_recommended": support_result.escalation_recommended
                }
            )
            
        except Exception as e:
            logger.error(f"Technical support processing failed: {str(e)}")
            return await self._generate_error_response(str(e), context)

    async def _execute_technical_support_workflow(self, 
                                                request: str, 
                                                session: TechnicalSession,
                                                strategy: 'ResponseStrategy',
                                                context: ConversationContext) -> 'TechnicalSupportResult':
        """Execute comprehensive technical support workflow"""
        
        workflow_results = []
        tools_executed = []
        
        try:
            # Step 1: Enhanced knowledge base search with technical context
            knowledge_search = await self._search_technical_knowledge_enhanced(
                request, session, context
            )
            workflow_results.append(knowledge_search)
            tools_executed.append("enhanced_knowledge_search")
            
            # Step 2: Determine if diagnostic workflow is needed
            if strategy.requires_diagnostic_workflow:
                diagnostic_result = await self._execute_diagnostic_workflow(
                    request, session, knowledge_search
                )
                workflow_results.append(diagnostic_result)
                tools_executed.append("diagnostic_workflow")
                
            # Step 3: Generate step-by-step solution if complex issue
            if strategy.requires_step_by_step:
                step_by_step_solution = await self._generate_step_by_step_solution(
                    request, session, knowledge_search
                )
                workflow_results.append(step_by_step_solution)
                tools_executed.append("step_by_step_solution")
                
            # Step 4: Validate solution applicability
            if knowledge_search.solutions_found:
                validation_result = await self._validate_solution_applicability(
                    knowledge_search.solutions, session, context
                )
                workflow_results.append(validation_result)
                tools_executed.append("solution_validation")
                
            # Step 5: Check for escalation needs
            escalation_check = await self._evaluate_escalation_needs(
                session, workflow_results, strategy
            )
            
            if escalation_check.escalation_needed:
                escalation_result = await self._prepare_escalation(
                    session, escalation_check, context
                )
                workflow_results.append(escalation_result)
                tools_executed.append("escalation_preparation")
            
            return TechnicalSupportResult(
                success=True,
                workflow_results=workflow_results,
                tools_executed=tools_executed,
                session_updated=session,
                escalation_recommended=escalation_check.escalation_needed,
                estimated_resolution_time=self._calculate_estimated_resolution_time(workflow_results),
                confidence_score=self._calculate_workflow_confidence(workflow_results)
            )
            
        except Exception as e:
            logger.error(f"Technical support workflow failed: {str(e)}")
            return TechnicalSupportResult(
                success=False,
                error=str(e),
                workflow_results=workflow_results,
                tools_executed=tools_executed,
                session_updated=session
            )

    async def _search_technical_knowledge_enhanced(self, 
                                                 request: str, 
                                                 session: TechnicalSession,
                                                 context: ConversationContext) -> 'TechnicalKnowledgeResult':
        """Enhanced technical knowledge search with context awareness"""
        
        # Build comprehensive search context
        search_context = {
            "user_technical_level": session.user_technical_level.value,
            "issue_category": session.current_issue_category,
            "completed_steps": session.completed_steps,
            "failed_steps": session.failed_steps,
            "session_history": context.conversation_history[-5:] if context.conversation_history else [],
            "system_context": await self._extract_system_context(request),
            "urgency_level": await self._determine_urgency_context(request, context)
        }
        
        # Execute hybrid vector search with technical specialization
        search_results = await self.hybrid_vector_system.hybrid_search(
            query_vector=await self.embedding_engine.embed_query(request),
            agent_id=self.agent_id,
            top_k=10,
            search_context=search_context,
            filter_criteria={
                "technical_level": {"$lte": session.user_technical_level.value},
                "category": session.current_issue_category,
                "solution_verified": True
            }
        )
        
        # Enhance results with contextual ranking
        enhanced_results = await self._enhance_search_results_with_context(
            search_results, search_context, session
        )
        
        # Extract actionable solutions
        solutions = await self._extract_actionable_solutions(enhanced_results, session)
        
        return TechnicalKnowledgeResult(
            search_results=enhanced_results,
            solutions_found=len(solutions) > 0,
            solutions=solutions,
            confidence_score=self._calculate_search_confidence(enhanced_results),
            context_relevance_score=self._calculate_context_relevance(enhanced_results, search_context)
        )

    async def _generate_step_by_step_solution(self, 
                                            request: str, 
                                            session: TechnicalSession,
                                            knowledge_result: 'TechnicalKnowledgeResult') -> 'StepByStepSolution':
        """Generate detailed step-by-step solution with patience optimization"""
        
        if not knowledge_result.solutions:
            return StepByStepSolution(success=False, error="No solutions found")
        
        # Select best solution based on user technical level and context
        best_solution = await self._select_optimal_solution(
            knowledge_result.solutions, session
        )
        
        # Generate adaptive step-by-step instructions
        diagnostic_steps = await self._generate_adaptive_diagnostic_steps(
            best_solution, session
        )
        
        # Add patience-optimized elements
        enhanced_steps = await self._enhance_steps_with_patience_elements(
            diagnostic_steps, session
        )
        
        # Calculate estimated time and complexity
        estimated_time = sum(step.estimated_time_minutes for step in enhanced_steps)
        complexity_score = self._calculate_solution_complexity(enhanced_steps)
        
        return StepByStepSolution(
            success=True,
            solution_id=best_solution.id,
            title=best_solution.title,
            diagnostic_steps=enhanced_steps,
            estimated_total_time_minutes=estimated_time,
            complexity_score=complexity_score,
            user_technical_level_required=session.user_technical_level,
            patience_optimizations_applied=True,
            verification_checkpoints=len([s for s in enhanced_steps if s.verification_method]),
            fallback_options=await self._generate_fallback_options(best_solution, session)
        )

    async def _generate_patient_technical_response(self, 
                                                 support_result: 'TechnicalSupportResult',
                                                 session: TechnicalSession,
                                                 context: ConversationContext) -> 'PatientTechnicalResponse':
        """Generate patience-optimized technical response"""
        
        # Assess current user state and adjust tone
        user_state_assessment = await self._assess_current_user_state(session, context)
        
        # Determine optimal response structure
        response_structure = await self._determine_optimal_response_structure(
            support_result, session, user_state_assessment
        )
        
        # Generate core response content
        core_content = await self._generate_core_technical_content(
            support_result, session, response_structure
        )
        
        # Apply patience optimizations
        patience_optimized_content = await self._apply_patience_optimizations(
            core_content, session, user_state_assessment
        )
        
        # Add supportive elements
        supportive_elements = await self._add_supportive_communication_elements(
            patience_optimized_content, session, user_state_assessment
        )
        
        # Optimize for voice delivery
        voice_optimized_response = await self._optimize_for_voice_delivery(
            supportive_elements, session
        )
        
        return PatientTechnicalResponse(
            text=voice_optimized_response.text,
            confidence=support_result.confidence_score,
            patience_score=user_state_assessment.patience_score,
            technical_accuracy=support_result.technical_accuracy,
            user_comprehension_optimized=True,
            voice_delivery_optimized=True,
            follow_up_scheduled=voice_optimized_response.follow_up_required,
            escalation_prepared=support_result.escalation_recommended
        )

    async def _apply_patience_optimizations(self, 
                                          content: str, 
                                          session: TechnicalSession,
                                          user_state: 'UserStateAssessment') -> str:
        """Apply comprehensive patience optimizations to technical content"""
        
        optimizations = []
        
        # Add empathy and acknowledgment for frustrated users
        if user_state.frustration_level > 0.6:
            optimizations.append(
                "I understand this technical issue can be really frustrating. "
                "Let's work through this step by step, and I'll make sure we get it resolved. "
            )
        
        # Simplify language based on technical level
        if session.user_technical_level in [TechnicalComplexityLevel.BASIC, TechnicalComplexityLevel.INTERMEDIATE]:
            content = await self._simplify_technical_language(content, session.user_technical_level)
        
        # Add reassurance for complex procedures
        if session.current_step and session.current_step.complexity_level in [TechnicalComplexityLevel.ADVANCED, TechnicalComplexityLevel.EXPERT]:
            optimizations.append(
                "This next step might seem complex, but I'll guide you through each part carefully. "
                "Take your time, and let me know if you need me to explain anything differently. "
            )
        
        # Add progress indicators for multi-step processes
        if len(session.completed_steps) > 0:
            progress_percentage = (len(session.completed_steps) / (len(session.completed_steps) + 3)) * 100
            optimizations.append(
                f"Great progress! We're about {progress_percentage:.0f}% through the troubleshooting process. "
            )
        
        # Add time expectations
        if session.estimated_resolution_time:
            optimizations.append(
                f"Based on what we're working on, this should take about {session.estimated_resolution_time} more minutes. "
            )
        
        # Combine optimizations with content
        optimized_content = "".join(optimizations) + content
        
        # Add patient delivery cues for voice synthesis
        optimized_content = await self._add_voice_patience_cues(optimized_content)
        
        return optimized_content

    async def _add_voice_patience_cues(self, content: str) -> str:
        """Add voice delivery cues for patient, clear technical communication"""
        
        # Add pauses after technical terms
        technical_terms = await self._identify_technical_terms(content)
        for term in technical_terms:
            content = content.replace(term, f"{term} <pause=short>")
        
        # Add longer pauses before steps
        step_indicators = ["Step ", "First, ", "Next, ", "Then, ", "Finally, "]
        for indicator in step_indicators:
            content = content.replace(indicator, f"<pause=medium> {indicator}")
        
        # Add emphasis on important instructions
        important_phrases = ["important", "crucial", "make sure", "be careful", "don't forget"]
        for phrase in important_phrases:
            content = content.replace(phrase, f"<emphasis>{phrase}</emphasis>")
        
        # Add confirmation prompts at key points
        content = await self._add_confirmation_prompts(content)
        
        return content

    async def _assess_user_technical_level(self, 
                                         request: str, 
                                         context: ConversationContext) -> TechnicalComplexityLevel:
        """Assess user's technical competency level"""
        
        # Technical indicators in language
        technical_indicators = {
            TechnicalComplexityLevel.EXPERT: [
                "API", "JSON", "XML", "SQL", "regex", "configuration file", 
                "command line", "terminal", "bash", "PowerShell", "registry",
                "service", "daemon", "port", "protocol", "SSL", "TLS"
            ],
            TechnicalComplexityLevel.ADVANCED: [
                "settings", "preferences", "install", "uninstall", "update",
                "browser", "cache", "cookies", "plugin", "extension",
                "network", "wifi", "ethernet", "router", "firewall"
            ],
            TechnicalComplexityLevel.INTERMEDIATE: [
                "computer", "laptop", "desktop", "application", "program",
                "file", "folder", "document", "save", "open", "click",
                "menu", "button", "screen", "window", "tab"
            ],
            TechnicalComplexityLevel.BASIC: [
                "help", "problem", "issue", "not working", "broken",
                "can't", "unable", "confused", "don't know how",
                "please help", "simple", "easy", "basic"
            ]
        }
        
        # Score based on technical language use
        scores = {}
        request_lower = request.lower()
        
        for level, indicators in technical_indicators.items():
            score = sum(1 for indicator in indicators if indicator.lower() in request_lower)
            scores[level] = score
        
        # Determine level based on highest score
        if scores[TechnicalComplexityLevel.EXPERT] >= 2:
            return TechnicalComplexityLevel.EXPERT
        elif scores[TechnicalComplexityLevel.ADVANCED] >= 2:
            return TechnicalComplexityLevel.ADVANCED
        elif scores[TechnicalComplexityLevel.INTERMEDIATE] >= 1:
            return TechnicalComplexityLevel.INTERMEDIATE
        else:
            return TechnicalComplexityLevel.BASIC

    async def _detect_frustration_indicators(self, request: str) -> List[str]:
        """Detect frustration indicators in user communication"""
        
        frustration_indicators = []
        request_lower = request.lower()
        
        # Emotional indicators
        emotional_keywords = [
            "frustrated", "annoyed", "angry", "upset", "mad",
            "hate", "terrible", "awful", "horrible", "worst",
            "stupid", "ridiculous", "crazy", "insane"
        ]
        
        for keyword in emotional_keywords:
            if keyword in request_lower:
                frustration_indicators.append(f"emotional_keyword: {keyword}")
        
        # Urgency indicators
        urgency_keywords = [
            "urgent", "asap", "immediately", "right now", "quickly",
            "emergency", "critical", "important", "deadline",
            "hurry", "fast", "soon"
        ]
        
        for keyword in urgency_keywords:
            if keyword in request_lower:
                frustration_indicators.append(f"urgency_keyword: {keyword}")
        
        # Repetition indicators
        if "again" in request_lower or "still" in request_lower:
            frustration_indicators.append("repetition_indicator")
        
        # Multiple exclamation marks or caps
        if "!!!" in request or request.isupper():
            frustration_indicators.append("emphasis_indicators")
        
        return frustration_indicators

    # Additional helper methods for technical operations
    async def _run_system_diagnostic(self, system_info: Dict[str, Any]) -> APIResult:
        """DUMMY: Run comprehensive system diagnostic"""
        logger.info(f"DUMMY: Running system diagnostic for {system_info.get('system_type', 'unknown')}")
        await asyncio.sleep(0.3)  # Simulate diagnostic time
        
        return APIResult(
            success=True,
            api_name="system_diagnostic",
            response_data={
                "diagnostic_id": f"diag_{uuid.uuid4().hex[:8]}",
                "system_status": "operational",
                "issues_found": 2,
                "recommendations": [
                    "Update system drivers",
                    "Clear temporary files",
                    "Restart affected services"
                ],
                "performance_score": 78
            },
            latency_ms=300
        )

    async def _create_technical_ticket(self, issue_data: Dict[str, Any]) -> APIResult:
        """DUMMY: Create technical support ticket"""
        logger.info(f"DUMMY: Creating technical support ticket for {issue_data.get('category', 'general')}")
        await asyncio.sleep(0.2)
        
        return APIResult(
            success=True,
            api_name="technical_ticketing",
            response_data={
                "ticket_id": f"TECH-{uuid.uuid4().hex[:6].upper()}",
                "priority": issue_data.get("priority", "medium"),
                "estimated_resolution": "2-4 hours",
                "assigned_specialist": "Senior Tech Support",
                "tracking_url": f"https://dummy-support.com/tickets/TECH-{uuid.uuid4().hex[:6].upper()}"
            },
            latency_ms=200
        )

    async def _schedule_technical_follow_up(self, follow_up_data: Dict[str, Any]) -> APIResult:
        """DUMMY: Schedule technical follow-up appointment"""
        logger.info(f"DUMMY: Scheduling follow-up in {follow_up_data.get('hours', 24)} hours")
        await asyncio.sleep(0.1)
        
        return APIResult(
            success=True,
            api_name="follow_up_scheduling",
            response_data={
                "appointment_id": f"followup_{uuid.uuid4().hex[:8]}",
                "scheduled_time": (datetime.now() + timedelta(hours=follow_up_data.get('hours', 24))).isoformat(),
                "reminder_sent": True,
                "calendar_invite": True
            },
            latency_ms=100
        )


# Supporting Classes and Data Structures

@dataclass
class TechnicalSupportResult:
    """Result of technical support workflow execution"""
    success: bool
    workflow_results: List[Any]
    tools_executed: List[str]
    session_updated: TechnicalSession
    escalation_recommended: bool = False
    estimated_resolution_time: Optional[int] = None
    confidence_score: float = 0.0
    error: Optional[str] = None
    technical_accuracy: float = 0.0


@dataclass 
class TechnicalKnowledgeResult:
    """Result of technical knowledge search"""
    search_results: SearchResult
    solutions_found: bool
    solutions: List[Any]
    confidence_score: float
    context_relevance_score: float


@dataclass
class StepByStepSolution:
    """Structured step-by-step technical solution"""
    success: bool
    solution_id: Optional[str] = None
    title: Optional[str] = None
    diagnostic_steps: List[TechnicalDiagnosticStep] = None
    estimated_total_time_minutes: int = 0
    complexity_score: float = 0.0
    user_technical_level_required: TechnicalComplexityLevel = TechnicalComplexityLevel.BASIC
    patience_optimizations_applied: bool = False
    verification_checkpoints: int = 0
    fallback_options: List[str] = None
    error: Optional[str] = None


@dataclass
class PatientTechnicalResponse:
    """Patience-optimized technical response"""
    text: str
    confidence: float
    patience_score: float
    technical_accuracy: float
    user_comprehension_optimized: bool
    voice_delivery_optimized: bool
    follow_up_scheduled: bool
    escalation_prepared: bool


# Supporting utility classes (simplified implementations)

class PatienceTracker:
    """Track and analyze user patience indicators"""
    
    async def assess_user_patience(self, request: str, context: ConversationContext) -> float:
        """Assess user patience level (0.0-1.0, higher = more patient)"""
        # Simplified implementation
        frustration_keywords = ["frustrated", "angry", "annoyed", "terrible", "awful"]
        request_lower = request.lower()
        
        frustration_count = sum(1 for keyword in frustration_keywords if keyword in request_lower)
        base_patience = 0.8  # Start with high patience assumption
        
        # Reduce patience score based on frustration indicators
        patience_score = max(0.1, base_patience - (frustration_count * 0.2))
        
        # Consider conversation length (longer = potentially less patient)
        if context.conversation_history and len(context.conversation_history) > 5:
            patience_score *= 0.9
        
        return patience_score


class ComplexityAdapter:
    """Adapt explanation complexity based on user technical level"""
    
    async def adapt_complexity(self, content: str, target_level: TechnicalComplexityLevel) -> str:
        """Adapt content complexity for target technical level"""
        # Simplified implementation - in production would use more sophisticated NLP
        if target_level == TechnicalComplexityLevel.BASIC:
            # Replace technical terms with simpler equivalents
            replacements = {
                "configuration": "settings",
                "initialize": "start up",
                "terminate": "close",
                "execute": "run",
                "parameters": "options"
            }
            for technical, simple in replacements.items():
                content = content.replace(technical, simple)
        
        return content


class DiagnosticWorkflowEngine:
    """Manage diagnostic workflows and step execution"""
    
    async def create_diagnostic_workflow(self, issue_type: str, user_level: TechnicalComplexityLevel) -> List[TechnicalDiagnosticStep]:
        """Create appropriate diagnostic workflow"""
        # Simplified example workflow
        return [
            TechnicalDiagnosticStep(
                step_id="initial_check",
                title="Initial System Check",
                description="Let's start with a basic system check",
                instructions=["Check if the system is powered on", "Verify all cables are connected"],
                expected_outcome="System shows signs of life",
                verification_method="Visual confirmation",
                complexity_level=TechnicalComplexityLevel.BASIC,
                estimated_time_minutes=2
            )
        ]


class AdaptiveExplanationGenerator:
    """Generate explanations adapted to user technical level"""
    
    async def generate_explanation(self, concept: str, target_level: TechnicalComplexityLevel) -> str:
        """Generate level-appropriate explanation"""
        # Simplified implementation
        if target_level == TechnicalComplexityLevel.BASIC:
            return f"In simple terms, {concept} means..."
        elif target_level == TechnicalComplexityLevel.EXPERT:
            return f"Technical details for {concept}:"
        else:
            return f"Here's how {concept} works:"


class TechnicalKnowledgeHierarchy:
    """Manage hierarchical technical knowledge structure"""
    pass


class SolutionValidator:
    """Validate technical solutions for applicability"""
    pass


class TechnicalEscalationManager:
    """Manage escalation to specialized technical staff"""
    pass


class TechnicalSessionAnalytics:
    """Analytics for technical support sessions"""
    
    async def record_interaction(self, session: TechnicalSession, strategy: Any, result: TechnicalSupportResult):
        """Record session interaction for analytics"""
        logger.info(f"Recording technical session interaction: {session.session_id}")


# Additional helper classes would be implemented here in a full production system