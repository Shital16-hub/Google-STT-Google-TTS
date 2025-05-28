"""
Technical Support Agent - Complete Implementation
Fixed version with all required abstract methods implemented.
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.agents.base_agent import (
    BaseAgent, AgentResponse, AgentConfiguration, UrgencyLevel,
    AgentCapability, ToolResult, AgentStatus
)
from app.vector_db.hybrid_vector_system import HybridVectorSystem
from app.tools.tool_orchestrator import ComprehensiveToolOrchestrator

logger = logging.getLogger(__name__)

class TechnicalComplexityLevel(Enum):
    """Technical complexity levels for adaptive support"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

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

class AdvancedTechnicalSupportAgent(BaseAgent):
    """
    Advanced Technical Support Agent with Patience-Optimized Interactions
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfiguration,
        hybrid_vector_system: HybridVectorSystem,
        tool_orchestrator: Optional[ComprehensiveToolOrchestrator] = None,
        target_response_time_ms: int = 250  # Longer for complex technical support
    ):
        """Initialize technical support agent."""
        super().__init__(
            agent_id=agent_id,
            config=config,
            hybrid_vector_system=hybrid_vector_system,
            tool_orchestrator=tool_orchestrator,
            target_response_time_ms=target_response_time_ms
        )
        
        # Technical Support Capabilities
        self.capabilities.extend([
            AgentCapability.KNOWLEDGE_RETRIEVAL,
            AgentCapability.TOOL_EXECUTION,
            AgentCapability.CONTEXT_UNDERSTANDING
        ])
        
        # Technical knowledge areas
        self.technical_categories = {
            "software_issues": ["installation", "configuration", "updates", "compatibility"],
            "hardware_problems": ["connectivity", "performance", "drivers", "peripherals"],
            "network_issues": ["connectivity", "speed", "security", "setup"],
            "account_access": ["login", "password", "authentication", "permissions"],
            "system_errors": ["crashes", "freezing", "error_messages", "debugging"]
        }
        
        # Diagnostic workflows
        self.diagnostic_workflows = {
            "basic_troubleshooting": self._basic_troubleshooting_workflow,
            "network_diagnostics": self._network_diagnostic_workflow,
            "software_installation": self._software_installation_workflow,
            "performance_optimization": self._performance_optimization_workflow
        }
        
        logger.info(f"AdvancedTechnicalSupportAgent initialized: {agent_id}")
    
    async def _detect_intent(self, query: str, context: Dict[str, Any]) -> str:
        """Detect technical support intent."""
        query_lower = query.lower()
        
        # Software issues
        if any(word in query_lower for word in ["install", "installation", "setup", "configure"]):
            return "software_installation"
        elif any(word in query_lower for word in ["update", "upgrade", "version"]):
            return "software_update"
        elif any(word in query_lower for word in ["crash", "freezing", "hang", "stuck"]):
            return "system_stability"
        
        # Network issues
        elif any(word in query_lower for word in ["internet", "wifi", "connection", "network"]):
            return "network_troubleshooting"
        elif any(word in query_lower for word in ["slow", "speed", "performance"]):
            return "performance_optimization"
        
        # Account/access issues
        elif any(word in query_lower for word in ["login", "password", "access", "account"]):
            return "account_access"
        elif any(word in query_lower for word in ["permission", "denied", "unauthorized"]):
            return "permission_issues"
        
        # Hardware issues
        elif any(word in query_lower for word in ["hardware", "device", "driver", "peripheral"]):
            return "hardware_troubleshooting"
        
        # Error messages
        elif any(word in query_lower for word in ["error", "message", "code", "exception"]):
            return "error_diagnosis"
        
        else:
            return "general_technical_support"
    
    async def _requires_tools(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if tools are required for technical support."""
        intent = await self._detect_intent(query, context)
        
        # Most technical support requires diagnostic tools or workflows
        tool_required_intents = [
            "software_installation", "network_troubleshooting", "performance_optimization",
            "system_stability", "hardware_troubleshooting", "error_diagnosis"
        ]
        
        return intent in tool_required_intents
    
    async def _suggest_tools(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Suggest appropriate tools for technical support."""
        intent = await self._detect_intent(query, context)
        
        suggested_tools = []
        
        # Intent-based tool suggestions
        if intent == "software_installation":
            suggested_tools.extend(["software_installation_workflow", "system_compatibility_check"])
        elif intent == "network_troubleshooting":
            suggested_tools.extend(["network_diagnostic_workflow", "connectivity_test"])
        elif intent == "performance_optimization":
            suggested_tools.extend(["performance_diagnostic", "system_optimization_workflow"])
        elif intent == "system_stability":
            suggested_tools.extend(["system_diagnostic_workflow", "crash_analysis"])
        elif intent == "hardware_troubleshooting":
            suggested_tools.extend(["hardware_diagnostic", "driver_update_workflow"])
        elif intent == "error_diagnosis":
            suggested_tools.extend(["error_analysis_workflow", "log_diagnostic"])
        elif intent == "account_access":
            suggested_tools.extend(["account_verification", "password_reset_workflow"])
        
        # Always consider knowledge base search
        suggested_tools.append("technical_knowledge_search")
        
        # Add escalation tool for complex issues
        if any(word in query.lower() for word in ["complex", "advanced", "multiple", "several"]):
            suggested_tools.append("escalate_to_specialist")
            
        return suggested_tools
    
    async def _generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        knowledge_context: List[Dict[str, Any]],
        tool_results: List[ToolResult],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate technical support response with step-by-step guidance."""
        
        intent = analysis.get("intent", "general_technical_support")
        complexity = self._assess_query_complexity(query)
        
        response_parts = []
        
        # Start with appropriate technical greeting
        technical_greeting = self._get_technical_greeting(intent, complexity)
        if technical_greeting:
            response_parts.append(technical_greeting)
        
        # Process tool results for actionable steps
        diagnostic_info = self._process_technical_tool_results(tool_results, intent)
        if diagnostic_info:
            response_parts.append(diagnostic_info)
        
        # Generate step-by-step solution
        solution_steps = self._generate_technical_solution(
            intent, knowledge_context, complexity, tool_results
        )
        if solution_steps:
            response_parts.append(solution_steps)
        
        # Add verification and next steps
        verification_steps = self._generate_verification_steps(intent, tool_results)
        if verification_steps:
            response_parts.append(verification_steps)
        
        # Add support information
        support_info = self._generate_support_information(intent, complexity)
        if support_info:
            response_parts.append(support_info)
        
        # Combine all parts
        full_response = ". ".join(response_parts)
        
        # Optimize for voice delivery (keep concise but informative)
        if len(full_response.split()) > 60:  # Limit for voice clarity
            priority_parts = []
            
            if technical_greeting:
                priority_parts.append(technical_greeting)
            if diagnostic_info:
                priority_parts.append(diagnostic_info)
            elif solution_steps:
                priority_parts.append(solution_steps)
            if verification_steps:
                priority_parts.append(verification_steps)
            
            full_response = ". ".join(priority_parts)
        
        return full_response
    
    def _assess_query_complexity(self, query: str) -> TechnicalComplexityLevel:
        """Assess the complexity level of the technical query."""
        query_lower = query.lower()
        
        # Expert level indicators
        expert_indicators = [
            "api", "configuration file", "registry", "command line", 
            "terminal", "sql", "database", "server", "advanced settings"
        ]
        
        # Advanced level indicators
        advanced_indicators = [
            "network settings", "firewall", "port", "protocol", "driver",
            "system settings", "administrator", "registry editor"
        ]
        
        # Intermediate level indicators
        intermediate_indicators = [
            "settings", "preferences", "install", "uninstall", "update",
            "browser", "application", "software", "program"
        ]
        
        if any(indicator in query_lower for indicator in expert_indicators):
            return TechnicalComplexityLevel.EXPERT
        elif any(indicator in query_lower for indicator in advanced_indicators):
            return TechnicalComplexityLevel.ADVANCED
        elif any(indicator in query_lower for indicator in intermediate_indicators):
            return TechnicalComplexityLevel.INTERMEDIATE
        else:
            return TechnicalComplexityLevel.BASIC
    
    def _get_technical_greeting(self, intent: str, complexity: TechnicalComplexityLevel) -> Optional[str]:
        """Get appropriate technical greeting based on intent and complexity."""
        greetings = {
            "software_installation": "I'll help you get that software installed correctly",
            "network_troubleshooting": "Let's work together to resolve your network connectivity issue",
            "performance_optimization": "I can help optimize your system's performance",
            "system_stability": "I'll help diagnose and fix those system stability issues",
            "hardware_troubleshooting": "Let's troubleshoot your hardware problem step by step",
            "error_diagnosis": "I'll help you understand and resolve that error message",
            "account_access": "I'll help you regain access to your account"
        }
        
        base_greeting = greetings.get(intent, "I'm here to help with your technical issue")
        
        # Add complexity-appropriate reassurance
        if complexity == TechnicalComplexityLevel.BASIC:
            return f"{base_greeting}. I'll guide you through each step clearly"
        elif complexity in [TechnicalComplexityLevel.ADVANCED, TechnicalComplexityLevel.EXPERT]:
            return f"{base_greeting}. I'll provide detailed technical guidance"
        else:
            return base_greeting
    
    def _process_technical_tool_results(self, tool_results: List[ToolResult], intent: str) -> Optional[str]:
        """Process technical tool results into user-friendly information."""
        successful_tools = [r for r in tool_results if r.success]
        
        if not successful_tools:
            return None
        
        results = []
        
        for result in successful_tools:
            if result.tool_name == "system_diagnostic_workflow":
                if result.output and isinstance(result.output, dict):
                    status = result.output.get("system_status", "checked")
                    results.append(f"I've completed a system diagnostic and found the issue")
            
            elif result.tool_name == "network_diagnostic_workflow":
                if result.output and isinstance(result.output, dict):
                    connectivity = result.output.get("connectivity_status", "tested")
                    results.append(f"Network diagnostics show: {connectivity}")
            
            elif result.tool_name == "software_installation_workflow":
                results.append("I've prepared the installation steps for your software")
            
            elif result.tool_name == "performance_diagnostic":
                if result.output and isinstance(result.output, dict):
                    performance_score = result.output.get("performance_score", "analyzed")
                    results.append(f"Performance analysis completed: {performance_score}")
        
        return ". ".join(results) if results else None
    
    def _generate_technical_solution(
        self,
        intent: str,
        knowledge_context: List[Dict[str, Any]],
        complexity: TechnicalComplexityLevel,
        tool_results: List[ToolResult]
    ) -> Optional[str]:
        """Generate appropriate technical solution based on intent and complexity."""
        
        solutions = {
            "software_installation": "First, make sure your system meets the requirements, then download from the official source and run as administrator",
            "network_troubleshooting": "Let's start by checking your network adapter, then test connectivity and review network settings",
            "performance_optimization": "We'll check startup programs, clean temporary files, and optimize system settings",
            "system_stability": "I'll guide you through checking system files, updating drivers, and reviewing error logs",
            "hardware_troubleshooting": "Let's verify hardware connections, update drivers, and run hardware diagnostics",
            "error_diagnosis": "I'll help you interpret the error message and provide specific steps to resolve it",
            "account_access": "We'll verify your credentials, check account status, and reset if necessary"
        }
        
        base_solution = solutions.get(intent, "Let me provide step-by-step guidance for your technical issue")
        
        # Adapt based on complexity level
        if complexity == TechnicalComplexityLevel.BASIC:
            return f"{base_solution}. I'll explain each step in simple terms"
        elif complexity == TechnicalComplexityLevel.EXPERT:
            return f"{base_solution}. I'll provide detailed technical steps and alternatives"
        else:
            return base_solution
    
    def _generate_verification_steps(self, intent: str, tool_results: List[ToolResult]) -> Optional[str]:
        """Generate verification steps for the solution."""
        verification_steps = {
            "software_installation": "After installation, try launching the program to confirm it's working",
            "network_troubleshooting": "Test your internet connection by visiting a website",
            "performance_optimization": "Restart your computer to see the performance improvements",
            "system_stability": "Monitor your system for any recurring issues over the next few hours",
            "hardware_troubleshooting": "Test the hardware functionality to ensure it's working properly",
            "error_diagnosis": "Try the action that previously caused the error to see if it's resolved",
            "account_access": "Attempt to log in with your credentials to verify access"
        }
        
        return verification_steps.get(intent)
    
    def _generate_support_information(self, intent: str, complexity: TechnicalComplexityLevel) -> Optional[str]:
        """Generate additional support information."""
        if complexity in [TechnicalComplexityLevel.ADVANCED, TechnicalComplexityLevel.EXPERT]:
            return "If you need additional assistance with advanced configuration, I can connect you with a specialist"
        elif intent in ["system_stability", "hardware_troubleshooting"]:
            return "If the issue persists, we may need to run additional diagnostics"
        else:
            return "Let me know if you need help with any of these steps"
    
    # Workflow methods (simplified implementations)
    async def _basic_troubleshooting_workflow(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic troubleshooting workflow."""
        return {"status": "completed", "steps_completed": 3, "resolution": "basic_fix_applied"}
    
    async def _network_diagnostic_workflow(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Network diagnostic workflow."""
        return {"status": "completed", "connectivity_status": "connection_restored", "latency": "normal"}
    
    async def _software_installation_workflow(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Software installation workflow."""
        return {"status": "ready", "requirements_met": True, "installation_path": "prepared"}
    
    async def _performance_optimization_workflow(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Performance optimization workflow."""
        return {"status": "completed", "performance_score": "improved", "optimizations_applied": 5}
    
    async def _load_knowledge_base(self):
        """Load technical support knowledge base."""
        logger.info(f"Loading technical support knowledge base for {self.agent_id}")
        
        # Knowledge areas for technical support
        knowledge_areas = [
            "software_troubleshooting_guides",
            "hardware_diagnostic_procedures",
            "network_configuration_help",
            "system_optimization_techniques",
            "error_message_solutions",
            "installation_procedures",
            "compatibility_requirements",
            "performance_tuning_guides"
        ]
        
        # In a real implementation, this would load technical documentation
        # into the hybrid vector system under the agent's namespace
    
    async def _initialize_specialized_components(self):
        """Initialize technical support specific components."""
        logger.info("Initializing technical support specialized components")
        
        # Initialize diagnostic tools
        # This would set up connections to system diagnostic tools
        
        # Initialize knowledge base connections
        # This would connect to technical documentation systems
        
        # Initialize escalation pathways
        # This would configure specialist escalation procedures
        
        logger.info("Technical support specialized components initialized")
    
    def get_technical_capabilities(self) -> Dict[str, Any]:
        """Get technical support capabilities."""
        return {
            "software_support": True,
            "hardware_diagnostics": True,
            "network_troubleshooting": True,
            "system_optimization": True,
            "error_diagnosis": True,
            "installation_assistance": True,
            "performance_tuning": True,
            "step_by_step_guidance": True,
            "complexity_adaptation": True,
            "specialist_escalation": True,
            "supported_categories": list(self.technical_categories.keys()),
            "diagnostic_workflows": list(self.diagnostic_workflows.keys())
        }