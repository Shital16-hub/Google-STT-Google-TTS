"""
Empathy-Optimized Billing Support Agent
Handles billing inquiries with emotional intelligence and customer satisfaction focus.
Optimized for sensitive financial conversations with <180ms processing time.
"""
import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.agents.base_agent import (
    BaseAgent, AgentResponse, AgentConfiguration, UrgencyLevel,
    AgentCapability, ToolResult
)
from app.vector_db.hybrid_vector_system import HybridVectorSystem
from app.tools.tool_orchestrator import ComprehensiveToolOrchestrator

logger = logging.getLogger(__name__)

@dataclass
class BillingInquiry:
    """Structured billing inquiry with context."""
    inquiry_type: str
    amount_mentioned: Optional[Decimal] = None
    account_reference: Optional[str] = None
    date_mentioned: Optional[str] = None
    urgency_level: UrgencyLevel = UrgencyLevel.NORMAL
    emotional_state: str = "neutral"
    requires_refund: bool = False
    requires_investigation: bool = False

class EmotionalIntelligenceEngine:
    """Detects emotional state and adapts responses accordingly."""
    
    def __init__(self):
        self.emotion_patterns = {
            "frustrated": [
                "frustrated", "annoyed", "irritated", "angry", "mad",
                "this is ridiculous", "unacceptable", "terrible service",
                "keep happening", "again and again", "every time"
            ],
            "confused": [
                "don't understand", "confused", "what does this mean",
                "explain", "clarify", "what is this charge",
                "why am I", "how did this", "what's this for"
            ],
            "worried": [
                "worried", "concerned", "anxious", "stressed",
                "can't afford", "financial hardship", "tight budget",
                "need help", "struggling", "difficult time"
            ],
            "disappointed": [
                "disappointed", "expected better", "let down",
                "thought you were", "used to be better", "not what I expected"
            ],
            "urgent": [
                "urgent", "asap", "immediately", "right now",
                "can't wait", "need this fixed", "emergency",
                "today", "by tomorrow"
            ]
        }
        
        self.positive_indicators = [
            "thank you", "appreciate", "helpful", "good service",
            "satisfied", "happy", "pleased", "great job"
        ]
        
        self.escalation_triggers = [
            "speak to manager", "supervisor", "complaint", "legal action",
            "report you", "better business bureau", "cancel my account",
            "switch providers", "this is unacceptable"
        ]
    
    def analyze_emotion(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional state from query."""
        query_lower = query.lower()
        
        emotion_analysis = {
            "primary_emotion": "neutral",
            "intensity": 0.0,
            "positive_indicators": 0,
            "escalation_risk": 0.0,
            "empathy_required": False,
            "response_tone": "professional"
        }
        
        # Detect primary emotion
        emotion_scores = {}
        for emotion, patterns in self.emotion_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
            emotion_analysis["primary_emotion"] = primary_emotion
            emotion_analysis["intensity"] = min(1.0, emotion_scores[primary_emotion] / 3.0)
        
        # Check for positive indicators
        positive_count = sum(1 for indicator in self.positive_indicators if indicator in query_lower)
        emotion_analysis["positive_indicators"] = positive_count
        
        # Assess escalation risk
        escalation_count = sum(1 for trigger in self.escalation_triggers if trigger in query_lower)
        emotion_analysis["escalation_risk"] = min(1.0, escalation_count / 2.0)
        
        # Determine empathy requirement
        emotion_analysis["empathy_required"] = (
            emotion_analysis["primary_emotion"] in ["frustrated", "worried", "disappointed"] or
            emotion_analysis["intensity"] > 0.5 or
            emotion_analysis["escalation_risk"] > 0.3
        )
        
        # Set response tone
        if emotion_analysis["escalation_risk"] > 0.5:
            emotion_analysis["response_tone"] = "apologetic_solution_focused"
        elif emotion_analysis["empathy_required"]:
            emotion_analysis["response_tone"] = "empathetic_supportive"
        elif positive_count > 0:
            emotion_analysis["response_tone"] = "warm_appreciative"
        else:
            emotion_analysis["response_tone"] = "professional_helpful"
        
        return emotion_analysis

class BillingQueryProcessor:
    """Processes and categorizes billing-related queries."""
    
    def __init__(self):
        self.inquiry_types = {
            "refund_request": [
                "refund", "money back", "return", "credit", "reimburse",
                "charged wrong", "overcharged", "double charged", "cancel and refund"
            ],
            "payment_issue": [
                "payment failed", "can't pay", "payment not going through",
                "card declined", "payment method", "autopay", "billing cycle"
            ],
            "billing_explanation": [
                "what is this charge", "explain charge", "don't recognize",
                "what's this for", "why was I charged", "breakdown of charges"
            ],
            "account_inquiry": [
                "account balance", "statement", "invoice", "billing history",
                "payment history", "account status", "current charges"
            ],
            "subscription_management": [
                "cancel subscription", "change plan", "upgrade", "downgrade",
                "pause subscription", "subscription status", "plan details"
            ],
            "dispute_charge": [
                "dispute", "unauthorized charge", "fraud", "didn't authorize",
                "never ordered", "cancel this charge", "remove charge"
            ]
        }
        
        self.amount_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $1,234.56
            r'(\d+(?:,\d{3})*(?:\.\d{2})?) dollars?',  # 1234.56 dollars
            r'(\d+(?:,\d{3})*(?:\.\d{2})?) bucks?'  # 1234.56 bucks
        ]
        
        self.date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
            r'(\d{1,2}-\d{1,2}-\d{4})',  # MM-DD-YYYY
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}'
        ]
    
    def process_query(self, query: str, context: Dict[str, Any]) -> BillingInquiry:
        """Process billing query and extract structured information."""
        query_lower = query.lower()
        
        # Determine inquiry type
        inquiry_type = self._classify_inquiry_type(query_lower)
        
        # Extract monetary amounts
        amount = self._extract_amount(query)
        
        # Extract account references
        account_ref = self._extract_account_reference(query, context)
        
        # Extract dates
        date_mentioned = self._extract_date(query)
        
        # Determine urgency
        urgency = self._assess_urgency(query_lower, context)
        
        # Determine if refund/investigation needed
        requires_refund = inquiry_type in ["refund_request", "dispute_charge"]
        requires_investigation = inquiry_type in ["dispute_charge", "billing_explanation"]
        
        return BillingInquiry(
            inquiry_type=inquiry_type,
            amount_mentioned=amount,
            account_reference=account_ref,
            date_mentioned=date_mentioned,
            urgency_level=urgency,
            requires_refund=requires_refund,
            requires_investigation=requires_investigation
        )
    
    def _classify_inquiry_type(self, query_lower: str) -> str:
        """Classify the type of billing inquiry."""
        type_scores = {}
        
        for inquiry_type, patterns in self.inquiry_types.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                type_scores[inquiry_type] = score
        
        if type_scores:
            return max(type_scores.keys(), key=lambda k: type_scores[k])
        else:
            return "general_billing_inquiry"
    
    def _extract_amount(self, query: str) -> Optional[Decimal]:
        """Extract monetary amount from query."""
        for pattern in self.amount_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return Decimal(amount_str)
                except InvalidOperation:
                    continue
        return None
    
    def _extract_account_reference(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract account reference from query or context."""
        # Check context first
        if context.get("account_id"):
            return context["account_id"]
        
        # Look for account numbers in query
        account_patterns = [
            r'account\s+(?:number\s+)?(\d{6,12})',
            r'account\s+(?:#)?(\d{6,12})',
            r'customer\s+(?:#)?(\d{6,12})'
        ]
        
        for pattern in account_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_date(self, query: str) -> Optional[str]:
        """Extract date mentions from query."""
        for pattern in self.date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def _assess_urgency(self, query_lower: str, context: Dict[str, Any]) -> UrgencyLevel:
        """Assess urgency level of billing inquiry."""
        high_urgency_indicators = [
            "urgent", "asap", "immediately", "emergency",
            "account suspended", "service cut off", "overdue",
            "legal action", "collections", "fraud"
        ]
        
        medium_urgency_indicators = [
            "today", "this week", "soon", "quickly",
            "payment due", "deadline", "late fee"
        ]
        
        if any(indicator in query_lower for indicator in high_urgency_indicators):
            return UrgencyLevel.HIGH
        elif any(indicator in query_lower for indicator in medium_urgency_indicators):
            return UrgencyLevel.NORMAL
        else:
            return UrgencyLevel.LOW

class EmpathyResponseGenerator:
    """Generates empathetic responses based on emotional context."""
    
    def __init__(self):
        self.empathy_templates = {
            "frustrated": [
                "I completely understand your frustration, and I'm here to help resolve this for you.",
                "I can hear that this situation is really frustrating for you. Let me see what I can do to make this right.",
                "I'm sorry this has been such a frustrating experience. I want to help fix this immediately."
            ],
            "worried": [
                "I understand your concern about this, and I want to help put your mind at ease.",
                "I can see why this would be worrying. Let me help clarify this situation for you.",
                "I know billing issues can be stressful. I'm here to help you through this."
            ],
            "confused": [
                "I'd be happy to explain this clearly for you.",
                "Let me break this down so it makes perfect sense.",
                "I understand this can be confusing. I'll explain everything step by step."
            ],
            "disappointed": [
                "I'm sorry we've let you down. I want to make this right for you.",
                "I understand your disappointment, and I'm committed to resolving this.",
                "I'm sorry this hasn't met your expectations. Let me help improve your experience."
            ]
        }
        
        self.solution_connectors = [
            "Here's what I can do for you:",
            "Let me take care of this:",
            "I'm going to help you by:",
            "Here's how we'll resolve this:"
        ]
        
        self.reassurance_phrases = [
            "You're in good hands.",
            "I'll make sure this gets resolved.",
            "We'll get this sorted out for you.",
            "I'm here to help every step of the way."
        ]
    
    def generate_empathetic_opening(self, emotion: str, intensity: float) -> str:
        """Generate empathetic opening based on detected emotion."""
        if emotion in self.empathy_templates:
            templates = self.empathy_templates[emotion]
            # Choose template based on intensity
            if intensity > 0.7:
                return templates[0]  # Most empathetic
            elif intensity > 0.4:
                return templates[1]  # Moderate empathy
            else:
                return templates[2]  # Light empathy
        else:
            return "I'm here to help you with this billing question."
    
    def get_solution_connector(self) -> str:
        """Get a connecting phrase to introduce the solution."""
        import random
        return random.choice(self.solution_connectors)
    
    def get_reassurance_phrase(self) -> str:
        """Get a reassuring phrase for the end of response."""
        import random
        return random.choice(self.reassurance_phrases)

class BillingSupportAgent(BaseAgent):
    """
    Empathy-optimized billing support agent with emotional intelligence.
    Specialized in handling sensitive financial conversations with customer satisfaction focus.
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfiguration,
        hybrid_vector_system: HybridVectorSystem,
        tool_orchestrator: Optional[ComprehensiveToolOrchestrator] = None,
        target_response_time_ms: int = 180  # Slightly longer for thoughtful responses
    ):
        """Initialize billing support agent with empathy capabilities."""
        super().__init__(
            agent_id=agent_id,
            config=config,
            hybrid_vector_system=hybrid_vector_system,
            tool_orchestrator=tool_orchestrator,
            target_response_time_ms=target_response_time_ms
        )
        
        # Specialized components
        self.emotion_engine = EmotionalIntelligenceEngine()
        self.query_processor = BillingQueryProcessor()
        self.empathy_generator = EmpathyResponseGenerator()
        
        # Domain-specific capabilities
        self.capabilities.extend([
            AgentCapability.EMOTION_DETECTION,
            AgentCapability.EMPATHY_RESPONSE,  # Now properly defined
            AgentCapability.FINANCIAL_PROCESSING
        ])
        
        # Billing-specific knowledge
        self.billing_policies = {
            "refund_policy": "14-day refund policy for unused services",
            "dispute_timeline": "Disputes processed within 5-7 business days",
            "payment_methods": ["Credit Card", "Bank Transfer", "PayPal"],
            "billing_cycle": "Monthly on the anniversary of signup",
            "late_fee_policy": "$25 late fee after 15 days past due"
        }
        
        # Service level commitments
        self.service_commitments = {
            "response_time_hours": 2,
            "resolution_time_days": 3,
            "escalation_threshold": 0.7,
            "satisfaction_target": 0.9
        }
        
        logger.info(f"BillingSupportAgent initialized with empathy optimization")
    
    async def _detect_intent(self, query: str, context: Dict[str, Any]) -> str:
        """Detect billing-related intent."""
        billing_inquiry = self.query_processor.process_query(query, context)
        return billing_inquiry.inquiry_type
    
    async def _requires_tools(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if tools are required for billing inquiry."""
        billing_inquiry = self.query_processor.process_query(query, context)
        
        # Most billing inquiries require account access or payment processing
        tool_required_types = [
            "refund_request", "payment_issue", "account_inquiry",
            "subscription_management", "dispute_charge"
        ]
        
        return billing_inquiry.inquiry_type in tool_required_types
    
    async def _suggest_tools(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Suggest appropriate tools for billing inquiry."""
        billing_inquiry = self.query_processor.process_query(query, context)
        emotion_analysis = self.emotion_engine.analyze_emotion(query, context)
        
        suggested_tools = []
        
        # Payment processing tools
        if billing_inquiry.inquiry_type == "refund_request":
            suggested_tools.append("process_refund_workflow")
        elif billing_inquiry.inquiry_type == "payment_issue":
            suggested_tools.append("update_payment_method")
        elif billing_inquiry.inquiry_type == "subscription_management":
            suggested_tools.append("update_subscription_workflow")
        elif billing_inquiry.inquiry_type == "dispute_charge":
            suggested_tools.append("initiate_dispute_workflow")
        
        # Account management tools
        if billing_inquiry.account_reference or context.get("account_id"):
            suggested_tools.append("lookup_account_details")
        
        # Communication tools
        if emotion_analysis["escalation_risk"] > 0.5:
            suggested_tools.append("escalate_to_supervisor")
        
        # Always consider billing inquiry search
        suggested_tools.append("billing_inquiry_search")
        
        return suggested_tools
    
    async def _generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        knowledge_context: List[Dict[str, Any]],
        tool_results: List[ToolResult],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate empathetic billing support response."""
        
        # Analyze emotional context
        emotion_analysis = self.emotion_engine.analyze_emotion(query, context)
        billing_inquiry = self.query_processor.process_query(query, context)
        
        response_parts = []
        
        # Start with empathetic opening if needed
        if emotion_analysis["empathy_required"]:
            empathetic_opening = self.empathy_generator.generate_empathetic_opening(
                emotion_analysis["primary_emotion"],
                emotion_analysis["intensity"]
            )
            response_parts.append(empathetic_opening)
        
        # Process tool results for actionable information
        action_taken = self._process_billing_tool_results(tool_results, billing_inquiry)
        if action_taken:
            if emotion_analysis["empathy_required"]:
                connector = self.empathy_generator.get_solution_connector()
                response_parts.append(f"{connector} {action_taken}")
            else:
                response_parts.append(action_taken)
        
        # Add specific billing information
        billing_details = self._generate_billing_details(
            billing_inquiry, knowledge_context, emotion_analysis
        )
        if billing_details:
            response_parts.append(billing_details)
        
        # Add policy information if relevant
        policy_info = self._get_relevant_policy_info(billing_inquiry)
        if policy_info:
            response_parts.append(policy_info)
        
        # Add next steps or reassurance
        next_steps = self._generate_next_steps(billing_inquiry, emotion_analysis, bool(action_taken))
        if next_steps:
            response_parts.append(next_steps)
        
        # Add reassurance for emotional customers
        if emotion_analysis["empathy_required"] and emotion_analysis["intensity"] > 0.5:
            reassurance = self.empathy_generator.get_reassurance_phrase()
            response_parts.append(reassurance)
        
        # Combine response parts
        full_response = " ".join(response_parts)
        
        # Optimize for voice - keep concise but empathetic
        if len(full_response.split()) > 50:  # Max words for voice
            # Prioritize empathy and action
            priority_parts = []
            
            if emotion_analysis["empathy_required"]:
                priority_parts.append(response_parts[0])  # Empathetic opening
            
            if action_taken:
                priority_parts.append(action_taken)
            elif billing_details:
                priority_parts.append(billing_details)
            
            if next_steps:
                priority_parts.append(next_steps)
            
            full_response = " ".join(priority_parts)
        
        return full_response
    
    def _process_billing_tool_results(
        self,
        tool_results: List[ToolResult],
        billing_inquiry: BillingInquiry
    ) -> Optional[str]:
        """Process billing tool results into customer-friendly language."""
        successful_tools = [r for r in tool_results if r.success]
        
        if not successful_tools:
            return None
        
        actions = []
        
        for result in successful_tools:
            if result.tool_name == "process_refund_workflow":
                if result.output and isinstance(result.output, dict):
                    amount = result.output.get("refund_amount", "the amount")
                    timeline = result.output.get("processing_days", "3-5 business days")
                    actions.append(f"I've processed a refund of {amount} which will appear in {timeline}")
            
            elif result.tool_name == "update_subscription_workflow":
                if result.output and isinstance(result.output, dict):
                    action = result.output.get("action", "updated")
                    plan = result.output.get("new_plan", "your plan")
                    actions.append(f"I've {action} your subscription to {plan}")
            
            elif result.tool_name == "initiate_dispute_workflow":
                actions.append("I've initiated a billing dispute investigation on your behalf")
            
            elif result.tool_name == "lookup_account_details":
                if result.output and isinstance(result.output, dict):
                    balance = result.output.get("current_balance", "your current balance")
                    actions.append(f"I can see your account shows {balance}")
        
        return ". ".join(actions) if actions else None
    
    def _generate_billing_details(
        self,
        billing_inquiry: BillingInquiry,
        knowledge_context: List[Dict[str, Any]],
        emotion_analysis: Dict[str, Any]
    ) -> Optional[str]:
        """Generate specific billing details based on inquiry type."""
        
        if billing_inquiry.inquiry_type == "billing_explanation":
            if billing_inquiry.amount_mentioned:
                return f"The ${billing_inquiry.amount_mentioned} charge is for your monthly service fee"
            else:
                return "I can explain the charges on your account in detail"
        
        elif billing_inquiry.inquiry_type == "refund_request":
            if emotion_analysis["primary_emotion"] == "frustrated":
                return "I want to make this right for you with a full refund"
            else:
                return f"Based on our {self.billing_policies['refund_policy']}, you're eligible for a refund"
        
        elif billing_inquiry.inquiry_type == "payment_issue":
            return "I can help update your payment method to prevent future issues"
        
        elif billing_inquiry.inquiry_type == "subscription_management":
            return "I can help you modify your subscription to better meet your needs"
        
        elif billing_inquiry.inquiry_type == "dispute_charge":
            return f"I'll investigate this charge and we typically resolve disputes within {self.billing_policies['dispute_timeline']}"
        
        return None
    
    def _get_relevant_policy_info(self, billing_inquiry: BillingInquiry) -> Optional[str]:
        """Get relevant policy information for the inquiry."""
        
        if billing_inquiry.inquiry_type == "refund_request":
            return f"Our {self.billing_policies['refund_policy']} applies to your situation"
        
        elif billing_inquiry.inquiry_type == "dispute_charge":
            return f"Disputes are typically resolved within {self.billing_policies['dispute_timeline']}"
        
        elif billing_inquiry.inquiry_type == "payment_issue":
            if billing_inquiry.urgency_level == UrgencyLevel.HIGH:
                return f"We charge a {self.billing_policies['late_fee_policy']}, but I may be able to waive this"
        
        return None
    
    def _generate_next_steps(
        self,
        billing_inquiry: BillingInquiry,
        emotion_analysis: Dict[str, Any],
        action_taken: bool
    ) -> Optional[str]:
        """Generate appropriate next steps for the customer."""
        
        if action_taken:
            if billing_inquiry.inquiry_type == "refund_request":
                return "You'll receive an email confirmation shortly"
            elif billing_inquiry.inquiry_type == "dispute_charge":
                return "I'll email you updates as we investigate this charge"
            else:
                return "The changes will take effect immediately"
        
        elif emotion_analysis["escalation_risk"] > 0.5:
            return "Would you like me to have a supervisor review this case?"
        
        elif billing_inquiry.requires_investigation:
            return "I'll need to investigate this further and get back to you within 24 hours"
        
        elif emotion_analysis["primary_emotion"] == "confused":
            return "Would you like me to explain any other charges on your account?"
        
        return None
    
    async def _needs_escalation(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if billing inquiry needs escalation."""
        emotion_analysis = self.emotion_engine.analyze_emotion(query, context)
        billing_inquiry = self.query_processor.process_query(query, context)
        
        # Escalate for high emotional distress
        if emotion_analysis["escalation_risk"] > self.service_commitments["escalation_threshold"]:
            return True
        
        # Escalate for complex disputes
        if billing_inquiry.inquiry_type == "dispute_charge" and billing_inquiry.amount_mentioned:
            if billing_inquiry.amount_mentioned > Decimal("100.00"):
                return True
        
        # Escalate for multiple failed resolution attempts
        if context.get("previous_contact_count", 0) > 2:
            return True
        
        return False
    
    async def _load_knowledge_base(self):
        """Load billing-specific knowledge base."""
        logger.info(f"Loading billing support knowledge base for {self.agent_id}")
        
        # Knowledge areas for billing support
        knowledge_areas = [
            "billing_policies_procedures",
            "refund_processing_guidelines",
            "payment_troubleshooting",
            "subscription_management",
            "dispute_resolution_process",
            "common_billing_questions",
            "pricing_information",
            "account_management_procedures"
        ]
        
        # In a real implementation, this would load billing documentation
        # into the hybrid vector system under the agent's namespace
    
    async def _initialize_specialized_components(self):
        """Initialize billing-specific components."""
        logger.info("Initializing billing support specialized components")
        
        # Initialize payment processing connections
        # This would set up connections to payment processors, billing systems
        
        # Initialize customer satisfaction tracking
        # This would set up metrics collection for satisfaction scores
        
        # Initialize escalation pathways
        # This would configure supervisor notification and escalation procedures
        
        logger.info("Billing support specialized components initialized")
    
    def get_billing_capabilities(self) -> Dict[str, Any]:
        """Get billing support capabilities."""
        return {
            "emotional_intelligence": True,
            "empathy_optimization": True,
            "refund_processing": True,
            "payment_troubleshooting": True,
            "subscription_management": True,
            "dispute_resolution": True,
            "account_management": True,
            "multilingual_support": False,  # Would be enhanced
            "24_7_availability": True,
            "supervisor_escalation": True,
            "service_commitments": self.service_commitments,
            "billing_policies": self.billing_policies,
            "supported_payment_methods": self.billing_policies["payment_methods"]
        }
    
    def get_satisfaction_metrics(self) -> Dict[str, Any]:
        """Get customer satisfaction metrics."""
        # In a real implementation, this would return actual metrics
        return {
            "average_satisfaction_score": 4.2,
            "resolution_rate": 0.89,
            "first_contact_resolution": 0.76,
            "average_response_time_hours": 1.8,
            "escalation_rate": 0.12,
            "refund_approval_rate": 0.83,
            "dispute_resolution_success": 0.91
        }