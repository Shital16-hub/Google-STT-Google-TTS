"""
Technical Support Agent - Specialized agent for technical troubleshooting and product support
Part of the Multi-Agent Voice AI System Transformation

This agent handles:
- Technical troubleshooting and diagnostics
- Product documentation and how-to guidance
- System status and outage information
- Bug reporting and issue escalation
- Software/hardware compatibility support

Integration with:
- Zendesk/ServiceNow for ticket creation
- System monitoring tools for status checks
- Knowledge base and documentation systems
- Diagnostic tools and system health APIs
- Callback scheduling and escalation systems
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import re

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemoryCheckpointer

from agents.base_agent import BaseSpecializedAgent, AgentResponse, ConversationState
from core.latency_optimizer import latency_monitor
from vector_db.hybrid_vector_store import HybridVectorStore

# Configure logging
logger = logging.getLogger(__name__)

class TechnicalSeverity(Enum):
    """Technical issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TechnicalAgentState(ConversationState):
    """Extended state for technical support operations"""
    issue_type: Optional[str] = None
    severity_level: Optional[str] = None
    system_version: Optional[str] = None
    device_info: Optional[Dict[str, str]] = None
    diagnostic_results: List[Dict[str, Any]] = []
    troubleshooting_steps: List[Dict[str, Any]] = []
    ticket_created: Optional[str] = None
    escalation_level: Optional[int] = None
    resolution_status: Optional[str] = None

class TechnicalAgent(BaseSpecializedAgent):
    """
    Specialized Technical Support Agent with comprehensive troubleshooting capabilities
    Optimized for technical problem-solving, diagnostics, and system support
    """
    
    def __init__(self, config_path: str = "config/agents/technical_agent.yaml"):
        super().__init__(
            agent_id="technical-support",
            agent_type="technical_specialist",
            config_path=config_path
        )
        self.vector_collection = "technical-support-kb"
        self.specialization_keywords = [
            "error", "bug", "issue", "problem", "troubleshoot", "fix", "broken",
            "not working", "crash", "freeze", "slow", "performance", "connection",
            "login", "password", "setup", "install", "configuration", "update",
            "compatibility", "browser", "mobile", "app", "software", "hardware",
            "technical", "support", "help", "how to", "tutorial", "guide"
        ]
        
        # Initialize specialized tools
        self.tools = [
            self.run_system_diagnostics,
            self.check_system_status,
            self.create_support_ticket,
            self.schedule_callback,
            self.get_troubleshooting_guide,
            self.check_compatibility,
            self.escalate_to_specialist,
            self.get_system_logs,
            self.reset_user_settings,
            self.generate_diagnostic_report,
            self.check_known_issues,
            self.provide_workaround_solution
        ]
        
        # Technical support configuration
        self.support_config = {
            "max_diagnostic_time": self.config.get("diagnostics", {}).get("max_time_seconds", 30),
            "auto_escalation_threshold": self.config.get("escalation", {}).get("auto_threshold_minutes", 15),
            "callback_availability": self.config.get("callback", {}).get("business_hours", "9AM-6PM EST"),
            "system_status_api": self.config.get("monitoring", {}).get("status_api_url"),
            "ticket_system": self.config.get("ticketing", {}).get("system", "zendesk"),
            "escalation_levels": self.config.get("escalation", {}).get("levels", 3)
        }
        
        # Knowledge base categories
        self.kb_categories = {
            "connectivity": ["network", "wifi", "internet", "connection"],
            "authentication": ["login", "password", "account", "access"],
            "performance": ["slow", "lag", "speed", "optimization"],
            "bugs": ["error", "crash", "freeze", "malfunction"],
            "setup": ["install", "configuration", "setup", "deployment"],
            "compatibility": ["browser", "device", "system", "requirements"],
            "features": ["how to", "tutorial", "guide", "usage"]
        }
        
        # Initialize the agent
        self._setup_agent()
        
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for technical support"""
        return """You are an expert technical support specialist AI agent with comprehensive knowledge 
        of software troubleshooting, system diagnostics, and customer technical assistance. Your primary 
        responsibilities include:

        CORE COMPETENCIES:
        - Technical troubleshooting and problem diagnosis
        - Step-by-step guidance for technical issues
        - System compatibility and configuration support
        - Bug identification and workaround solutions
        - Performance optimization recommendations
        - Software/hardware setup and installation guidance

        COMMUNICATION STYLE:
        - Patient and instructional for complex technical procedures
        - Clear, step-by-step explanations avoiding technical jargon
        - Methodical approach to problem-solving
        - Reassuring tone for frustrated users experiencing technical issues
        - Proactive in suggesting preventive measures

        CAPABILITIES:
        - Run comprehensive system diagnostics and health checks
        - Access real-time system status and outage information
        - Create detailed support tickets with technical specifications
        - Schedule callbacks with technical specialists
        - Provide compatibility checks for software/hardware
        - Generate diagnostic reports and system logs
        - Escalate complex issues to appropriate technical teams
        - Offer immediate workarounds for known issues

        VOICE CHARACTERISTICS:
        - Tone: Patient and knowledgeable for technical guidance
        - Pace: Measured and clear when providing instructions
        - Style: Systematic problem-solver focused on resolution
        - Approach: Thorough diagnostics before suggesting solutions

        DIAGNOSTIC METHODOLOGY:
        1. Gather system information and reproduce issue
        2. Run automated diagnostics and check system status
        3. Consult knowledge base for known issues and solutions
        4. Provide step-by-step troubleshooting guidance
        5. Escalate to specialists if issue remains unresolved
        6. Follow up to ensure complete resolution

        TOOLS & INTEGRATIONS:
        You have access to diagnostic tools, system monitoring APIs, ticketing systems 
        (Zendesk/ServiceNow), knowledge bases, and callback scheduling. Always prioritize 
        user safety and data security in technical recommendations.

        When unable to resolve an issue immediately, provide clear next steps, create 
        appropriate support tickets, and offer alternative solutions or workarounds."""

    @tool
    @latency_monitor("tech_system_diagnostics")
    async def run_system_diagnostics(self, 
                                   system_info: Dict[str, str],
                                   issue_description: str) -> Dict[str, Any]:
        """
        Run comprehensive system diagnostics based on reported issue
        
        Args:
            system_info: System information (OS, browser, device, etc.)
            issue_description: Description of the technical issue
            
        Returns:
            Diagnostic results with identified problems and recommendations
        """
        try:
            # Simulate diagnostic process
            await asyncio.sleep(0.2)  # Simulate diagnostic time
            
            diagnostic_id = f"DIAG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Analyze system info for common issues
            issues_found = []
            recommendations = []
            
            # Browser compatibility check
            if system_info.get("browser"):
                browser = system_info["browser"].lower()
                if "internet explorer" in browser or "ie" in browser:
                    issues_found.append({
                        "type": "compatibility",
                        "severity": "high",
                        "description": "Unsupported browser detected",
                        "impact": "Limited functionality and security risks"
                    })
                    recommendations.append("Upgrade to Chrome, Firefox, or Edge for optimal experience")
            
            # Memory and performance check
            if system_info.get("memory"):
                try:
                    memory_gb = float(system_info["memory"].replace("GB", "").strip())
                    if memory_gb < 4:
                        issues_found.append({
                            "type": "performance",
                            "severity": "medium", 
                            "description": "Low system memory detected",
                            "impact": "May cause slow performance or crashes"
                        })
                        recommendations.append("Close unnecessary applications to free up memory")
                except ValueError:
                    pass
            
            # Network connectivity simulation
            network_status = await self._simulate_network_check()
            if not network_status["stable"]:
                issues_found.append({
                    "type": "connectivity",
                    "severity": "high",
                    "description": "Network connectivity issues detected",
                    "impact": "Service interruptions and sync failures"
                })
                recommendations.append("Check internet connection and try restarting router")
            
            return {
                "success": True,
                "diagnostic_id": diagnostic_id,
                "timestamp": datetime.now().isoformat(),
                "system_info": system_info,
                "issues_found": issues_found,
                "recommendations": recommendations,
                "overall_health": "good" if len(issues_found) == 0 else "issues_detected",
                "next_steps": recommendations[:3] if recommendations else ["System appears healthy"],
                "estimated_resolution_time": f"{len(issues_found) * 5} minutes"
            }
            
        except Exception as e:
            logger.error(f"System diagnostics error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to complete system diagnostics",
                "recommendation": "Please try again or contact technical support"
            }

    @tool
    @latency_monitor("tech_system_status")
    async def check_system_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Check current system status and any ongoing outages
        
        Args:
            service_name: Specific service to check (optional)
            
        Returns:
            System status information and outage details
        """
        try:
            # Simulate API call to status service
            await asyncio.sleep(0.1)
            
            # Simulated system status data
            services_status = {
                "web_application": {
                    "status": "operational",
                    "uptime": "99.9%",
                    "last_incident": "2024-05-20T10:30:00Z",
                    "response_time": "145ms"
                },
                "api_services": {
                    "status": "operational", 
                    "uptime": "99.8%",
                    "last_incident": "2024-05-18T14:15:00Z",
                    "response_time": "89ms"
                },
                "payment_processing": {
                    "status": "maintenance",
                    "uptime": "99.5%",
                    "maintenance_window": "2024-05-24T02:00:00Z to 2024-05-24T04:00:00Z",
                    "expected_impact": "Payment processing may be delayed"
                },
                "email_delivery": {
                    "status": "degraded",
                    "uptime": "98.2%",
                    "issue": "Delayed email delivery (5-10 minute delays)",
                    "eta_resolution": "2024-05-24T16:00:00Z"
                }
            }
            
            if service_name:
                service_status = services_status.get(service_name.lower().replace(" ", "_"))
                if service_status:
                    return {
                        "success": True,
                        "service": service_name,
                        "status": service_status["status"],
                        "details": service_status
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Service '{service_name}' not found",
                        "available_services": list(services_status.keys())
                    }
            
            # Return overall status
            operational_count = sum(1 for s in services_status.values() if s["status"] == "operational")
            total_services = len(services_status)
            
            return {
                "success": True,
                "overall_status": "operational" if operational_count == total_services else "partial_outage",
                "services": services_status,
                "summary": f"{operational_count}/{total_services} services operational",
                "last_updated": datetime.now().isoformat(),
                "status_page": "https://status.example.com"
            }
            
        except Exception as e:
            logger.error(f"System status check error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to retrieve system status"
            }

    @tool
    @latency_monitor("tech_create_ticket")
    async def create_support_ticket(self, 
                                  issue_title: str,
                                  issue_description: str,
                                  severity: str,
                                  customer_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Create technical support ticket for complex issues
        
        Args:
            issue_title: Brief title describing the issue
            issue_description: Detailed description of the problem
            severity: Issue severity (low, medium, high, critical)
            customer_info: Customer contact and system information
            
        Returns:
            Created ticket details and tracking information
        """
        try:
            ticket_id = f"TECH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Determine SLA based on severity
            sla_hours = {
                "critical": 2,
                "high": 8,
                "medium": 24,
                "low": 72
            }
            
            response_time = sla_hours.get(severity.lower(), 24)
            
            ticket_data = {
                "success": True,
                "ticket_id": ticket_id,
                "title": issue_title,
                "description": issue_description,
                "severity": severity,
                "status": "open",
                "created_at": datetime.now().isoformat(),
                "customer_info": customer_info,
                "assigned_team": "Technical Support L1",
                "sla_response_time": f"{response_time} hours",
                "expected_first_response": (datetime.now() + timedelta(hours=response_time)).isoformat(),
                "tracking_url": f"https://support.example.com/tickets/{ticket_id}",
                "priority_queue": severity in ["critical", "high"]
            }
            
            # Auto-escalate critical issues
            if severity.lower() == "critical":
                ticket_data["escalated"] = True
                ticket_data["assigned_team"] = "Technical Support L2"
                ticket_data["escalation_reason"] = "Critical severity auto-escalation"
            
            return ticket_data
            
        except Exception as e:
            logger.error(f"Ticket creation error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to create support ticket"
            }

    @tool
    @latency_monitor("tech_schedule_callback")
    async def schedule_callback(self, 
                              customer_phone: str,
                              preferred_time: str,
                              issue_summary: str) -> Dict[str, Any]:
        """
        Schedule technical support callback
        
        Args:
            customer_phone: Customer phone number
            preferred_time: Preferred callback time
            issue_summary: Brief summary of technical issue
            
        Returns:
            Callback scheduling confirmation
        """
        try:
            callback_id = f"CB-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Parse preferred time and find next available slot
            # Simplified scheduling logic
            now = datetime.now()
            
            if "asap" in preferred_time.lower() or "now" in preferred_time.lower():
                scheduled_time = now + timedelta(minutes=30)
            elif "tomorrow" in preferred_time.lower():
                scheduled_time = now + timedelta(days=1)
                scheduled_time = scheduled_time.replace(hour=9, minute=0)
            else:
                # Default to next business day morning
                scheduled_time = now + timedelta(days=1)
                scheduled_time = scheduled_time.replace(hour=10, minute=0)
            
            return {
                "success": True,
                "callback_id": callback_id,
                "scheduled_time": scheduled_time.isoformat(),
                "customer_phone": customer_phone,
                "issue_summary": issue_summary,
                "technician": "Technical Support Specialist",
                "duration": "30 minutes",
                "confirmation_sent": True,
                "reschedule_url": f"https://support.example.com/callback/{callback_id}/reschedule"
            }
            
        except Exception as e:
            logger.error(f"Callback scheduling error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to schedule callback"
            }

    @tool
    @latency_monitor("tech_troubleshooting_guide")
    async def get_troubleshooting_guide(self, 
                                      issue_type: str,
                                      system_type: str) -> Dict[str, Any]:
        """
        Get step-by-step troubleshooting guide for specific issues
        
        Args:
            issue_type: Type of issue (connectivity, login, performance, etc.)
            system_type: System type (web, mobile, desktop)
            
        Returns:
            Detailed troubleshooting steps and solutions
        """
        try:
            # Simulated troubleshooting guides
            guides = {
                "connectivity": {
                    "web": [
                        "Check internet connection by visiting another website",
                        "Clear browser cache and cookies",
                        "Disable browser extensions temporarily",
                        "Try incognito/private browsing mode",
                        "Restart browser and try again",
                        "Check firewall and antivirus settings"
                    ],
                    "mobile": [
                        "Check WiFi/cellular connection strength",
                        "Toggle airplane mode on/off",
                        "Restart the mobile application",
                        "Update app to latest version",
                        "Restart device if issues persist",
                        "Contact ISP if connection problems continue"
                    ]
                },
                "login": {
                    "web": [
                        "Verify username and password spelling",
                        "Check Caps Lock status",
                        "Clear browser cache and cookies",
                        "Try password reset if needed",
                        "Disable browser autofill temporarily",
                        "Contact support if account is locked"
                    ],
                    "mobile": [
                        "Verify credentials are entered correctly",
                        "Check for app updates",
                        "Clear app cache and data",
                        "Restart the application",
                        "Use 'Forgot Password' if needed",
                        "Reinstall app if problems persist"
                    ]
                },
                "performance": {
                    "web": [
                        "Close unnecessary browser tabs",
                        "Clear browser cache and temporary files",
                        "Disable heavy browser extensions",
                        "Check available system memory",
                        "Run browser in safe mode",
                        "Update browser to latest version"
                    ],
                    "mobile": [
                        "Close background applications",
                        "Restart the device",
                        "Clear app cache",
                        "Check available storage space",
                        "Update app to latest version",
                        "Check device performance settings"
                    ]
                }
            }
            
            guide = guides.get(issue_type.lower(), {}).get(system_type.lower(), [])
            
            if not guide:
                return {
                    "success": False,
                    "error": f"No troubleshooting guide found for {issue_type} on {system_type}",
                    "available_guides": list(guides.keys())
                }
            
            return {
                "success": True,
                "issue_type": issue_type,
                "system_type": system_type,
                "steps": [{"step": i+1, "instruction": step} for i, step in enumerate(guide)],
                "estimated_time": f"{len(guide) * 2} minutes",
                "difficulty": "beginner" if len(guide) <= 4 else "intermediate",
                "additional_help": "Contact support if these steps don't resolve the issue"
            }
            
        except Exception as e:
            logger.error(f"Troubleshooting guide error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to retrieve troubleshooting guide"
            }

    @tool
    @latency_monitor("tech_compatibility_check")
    async def check_compatibility(self, 
                                system_specs: Dict[str, str],
                                software_name: str) -> Dict[str, Any]:
        """
        Check system compatibility for software/hardware requirements
        
        Args:
            system_specs: System specifications (OS, browser, memory, etc.)
            software_name: Name of software to check compatibility
            
        Returns:
            Compatibility assessment and recommendations
        """
        try:
            # Simulated compatibility requirements
            requirements = {
                "web_application": {
                    "browsers": ["Chrome 90+", "Firefox 88+", "Safari 14+", "Edge 90+"],
                    "javascript": "required",
                    "cookies": "required",
                    "memory": "2GB minimum"
                },
                "mobile_app": {
                    "ios": "iOS 13.0 or later",
                    "android": "Android 8.0 (API level 26) or later",
                    "storage": "100MB available space",
                    "permissions": ["Camera", "Microphone", "Location"]
                },
                "desktop_software": {
                    "windows": "Windows 10 or later",
                    "mac": "macOS 10.15 or later",
                    "memory": "8GB RAM minimum",
                    "storage": "2GB available space"
                }
            }
            
            req = requirements.get(software_name.lower().replace(" ", "_"), {})
            if not req:
                return {
                    "success": False,
                    "error": f"Compatibility requirements not found for {software_name}"
                }
            
            compatibility_issues = []
            recommendations = []
            
            # Check browser compatibility
            if "browser" in system_specs and "browsers" in req:
                browser = system_specs["browser"].lower()
                supported = any(b.lower().split()[0] in browser for b in req["browsers"])
                if not supported:
                    compatibility_issues.append("Unsupported browser version")
                    recommendations.append(f"Upgrade to one of: {', '.join(req['browsers'])}")
            
            # Check memory requirements
            if "memory" in system_specs and "memory" in req:
                try:
                    system_memory = float(system_specs["memory"].replace("GB", "").strip())
                    required_memory = float(req["memory"].replace("GB minimum", "").strip())
                    if system_memory < required_memory:
                        compatibility_issues.append("Insufficient memory")
                        recommendations.append(f"Upgrade to at least {req['memory']}")
                except ValueError:
                    pass
            
            is_compatible = len(compatibility_issues) == 0
            
            return {
                "success": True,
                "software": software_name,
                "compatible": is_compatible,
                "system_specs": system_specs,
                "requirements": req,
                "issues": compatibility_issues,
                "recommendations": recommendations,
                "compatibility_score": f"{max(0, 100 - len(compatibility_issues) * 25)}%"
            }
            
        except Exception as e:
            logger.error(f"Compatibility check error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to perform compatibility check"
            }

    @tool
    @latency_monitor("tech_known_issues")
    async def check_known_issues(self, 
                               issue_description: str) -> Dict[str, Any]:
        """
        Check for known issues matching the user's problem
        
        Args:
            issue_description: Description of the issue
            
        Returns:
            Known issues and available workarounds
        """
        try:
            # Simulated known issues database
            known_issues = [
                {
                    "id": "KI-001",
                    "title": "Login timeout on Safari browser",
                    "description": "Users experience login timeouts when using Safari 16+",
                    "workaround": "Use Chrome or Firefox, or clear Safari cache",
                    "status": "investigating",
                    "affected_versions": ["Safari 16.0+"],
                    "reported_date": "2024-05-15"
                },
                {
                    "id": "KI-002", 
                    "title": "Slow performance on mobile devices",
                    "description": "Mobile app performance degraded on devices with <4GB RAM",
                    "workaround": "Close background apps and restart the application",
                    "status": "fix_planned",
                    "affected_versions": ["Mobile v2.1.0"],
                    "eta_fix": "2024-06-01"
                },
                {
                    "id": "KI-003",
                    "title": "Email notifications delayed",
                    "description": "Email notifications may be delayed by 5-10 minutes",
                    "workaround": "Check in-app notifications for immediate updates",
                    "status": "monitoring",
                    "affected_services": ["Email delivery"],
                    "reported_date": "2024-05-20"
                }
            ]
            
            # Simple keyword matching for demo
            issue_lower = issue_description.lower()
            matching_issues = []
            
            for issue in known_issues:
                if any(keyword in issue_lower for keyword in 
                      [word.lower() for word in issue["title"].split()]):
                    matching_issues.append(issue)
            
            return {
                "success": True,
                "query": issue_description,
                "matching_issues": matching_issues,
                "total_matches": len(matching_issues),
                "has_workarounds": any(issue.get("workaround") for issue in matching_issues)
            }
            
        except Exception as e:
            logger.error(f"Known issues check error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to check known issues database"
            }

    async def _simulate_network_check(self) -> Dict[str, Any]:
        """Simulate network connectivity check"""
        # Simulate network test
        await asyncio.sleep(0.05)
        return {
            "stable": True,
            "latency": 45,
            "packet_loss": 0,
            "dns_resolution": True
        }

    async def _process_conversation(self, message: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Process technical support conversation with enhanced diagnostics
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            Agent response with technical support enhancements
        """
        try:
            # Extract technical context
            tech_context = await self._extract_technical_context(message, context)
            
            # Enhanced system prompt with technical context
            enhanced_prompt = f"""{self._get_system_prompt()}
            
            CURRENT TECHNICAL CONTEXT:
            - Issue Type: {tech_context.get('issue_type', 'General inquiry')}
            - Severity: {tech_context.get('severity', 'Unknown')}
            - System: {tech_context.get('system_info', 'Not provided')}
            - Previous Steps: {tech_context.get('previous_troubleshooting', 'None')}
            
            Focus on systematic troubleshooting and providing clear, actionable solutions."""
            
            # Process with enhanced context
            messages = [
                SystemMessage(content=enhanced_prompt),
                HumanMessage(content=message)
            ]
            
            # Use the LangGraph agent for processing
            response = await self.agent.ainvoke({
                "messages": messages,
                "context": tech_context
            })
            
            return AgentResponse(
                content=response["messages"][-1].content,
                agent_used="technical-support",
                confidence=0.93,  # High confidence for specialized agent
                tools_used=response.get("tool_calls", []),
                context_updates=tech_context,
                latency_ms=response.get("processing_time", 0)
            )
            
        except Exception as e:
            logger.error(f"Technical conversation processing error: {str(e)}")
            return AgentResponse(
                content="I apologize, but I'm experiencing technical difficulties. Please try restarting the application or contact our technical support team for immediate assistance.",
                agent_used="technical-support",
                confidence=0.0,
                error=str(e)
            )

    async def _extract_technical_context(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical context from message and conversation history"""
        tech_context = {
            "issue_type": self._classify_technical_issue(message),
            "severity": self._assess_issue_severity(message),
            "system_info": context.get("system_info", {}),
            "previous_troubleshooting": context.get("troubleshooting_steps", []),
            "urgency": self._assess_urgency(message)
        }
        
        return tech_context

    def _classify_technical_issue(self, message: str) -> str:
        """Classify the type of technical issue"""
        message_lower = message.lower()
        
        for category, keywords in self.kb_categories.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        
        return "general_technical"

    def _assess_issue_severity(self, message: str) -> str:
        """Assess the severity of the technical issue"""
        message_lower = message.lower()
        
        critical_indicators = ["can't access", "completely broken", "critical", "urgent", "emergency"]
        high_indicators = ["not working", "error", "crash", "freeze", "broken"]
        medium_indicators = ["slow", "problem", "issue", "trouble"]
        
        if any(indicator in message_lower for indicator in critical_indicators):
            return "critical"
        elif any(indicator in message_lower for indicator in high_indicators):
            return "high"
        elif any(indicator in message_lower for indicator in medium_indicators):
            return "medium"
        else:
            return "low"

    def _assess_urgency(self, message: str) -> str:
        """Assess the urgency of the request"""
        message_lower = message.lower()
        
        urgent_indicators = ["urgent", "asap", "immediately", "right now", "emergency"]
        if any(indicator in message_lower for indicator in urgent_indicators):
            return "high"
        return "normal"

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get technical support agent performance metrics"""
        base_metrics = super().get_agent_metrics()
        
        technical_metrics = {
            "tickets_created": 234,  # Simulated
            "average_resolution_time": "12.5 minutes",
            "first_contact_resolution": "68%",
            "escalation_rate": "15%",
            "diagnostic_accuracy": "91%",
            "customer_satisfaction": 4.6,
            "callback_completion_rate": "94%",
            "known_issue_matches": 156
        }
        
        return {**base_metrics, "technical_specific": technical_metrics}