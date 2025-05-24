"""
Billing Support Agent - Specialized agent for payment, refund, and billing inquiries
Part of the Multi-Agent Voice AI System Transformation

This agent handles:
- Payment processing and refund automation
- Billing inquiries and account updates  
- Subscription management and billing cycles
- Financial dispute resolution
- Account balance and transaction history

Integration with:
- Stripe/PayPal payment processing
- CRM systems for account management
- Email/SMS notification systems
- Financial reporting tools
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemoryCheckpointer

from app.agents.base_agent import BaseSpecializedAgent, AgentResponse, ConversationState
from app.core.latency_optimizer import latency_monitor
from app.vector_db.hybrid_vector_store import HybridVectorStore

# Configure logging
logger = logging.getLogger(__name__)

class BillingAgentState(ConversationState):
    """Extended state for billing-specific operations"""
    customer_id: Optional[str] = None
    account_balance: Optional[float] = None
    last_payment_date: Optional[datetime] = None
    subscription_status: Optional[str] = None
    pending_refunds: List[Dict[str, Any]] = []
    payment_methods: List[Dict[str, Any]] = []
    billing_cycle: Optional[str] = None
    outstanding_invoices: List[Dict[str, Any]] = []

class BillingAgent(BaseSpecializedAgent):
    """
    Specialized Billing Support Agent with payment processing capabilities
    Optimized for financial inquiries, refund processing, and account management
    """
    
    def __init__(self, config_path: str = "config/agents/billing_agent.yaml"):
        super().__init__(
            agent_id="billing-support",
            agent_type="billing_specialist",
            config_path=config_path
        )
        self.vector_collection = "billing-support-kb"
        self.specialization_keywords = [
            "payment", "refund", "billing", "invoice", "subscription", 
            "account", "balance", "charge", "credit", "debit", "transaction",
            "overdue", "late fee", "payment method", "cancel subscription",
            "upgrade", "downgrade", "proration", "dispute", "chargeback"
        ]
        
        # Initialize specialized tools
        self.tools = [
            self.process_refund,
            self.update_payment_method,
            self.check_account_balance,
            self.generate_invoice,
            self.apply_credit,
            self.schedule_payment,
            self.cancel_subscription,
            self.upgrade_subscription,
            self.calculate_proration,
            self.resolve_billing_dispute,
            self.send_payment_notification,
            self.validate_payment_history
        ]
        
        # Payment processing configuration
        self.payment_config = {
            "stripe_webhook_secret": self.config.get("stripe", {}).get("webhook_secret"),
            "max_refund_amount": self.config.get("limits", {}).get("max_refund", 5000.00),
            "auto_refund_threshold": self.config.get("limits", {}).get("auto_refund_threshold", 100.00),
            "payment_retry_attempts": self.config.get("retry", {}).get("max_attempts", 3),
            "late_fee_percentage": self.config.get("fees", {}).get("late_fee_percentage", 0.05)
        }
        
        # Initialize the agent
        self._setup_agent()
        
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for billing support"""
        return """You are a professional billing support specialist AI agent with expertise in payment processing, 
        refund management, and subscription billing. Your primary responsibilities include:

        CORE COMPETENCIES:
        - Payment processing and refund automation
        - Billing inquiry resolution and account management
        - Subscription lifecycle management (upgrades, downgrades, cancellations)
        - Financial dispute resolution and chargeback handling
        - Account balance reconciliation and transaction history

        COMMUNICATION STYLE:
        - Professional, empathetic, and solution-oriented
        - Clear explanations of billing processes and policies
        - Proactive in identifying and resolving financial concerns
        - Patient when handling complex billing situations
        - Focused on maintaining customer trust in financial matters

        CAPABILITIES:
        - Process refunds up to policy limits automatically
        - Update payment methods and billing information securely
        - Generate detailed invoices and account statements
        - Apply credits, discounts, and promotional adjustments
        - Schedule payment plans and manage payment retries
        - Handle subscription changes with accurate proration
        - Resolve billing disputes through systematic investigation
        - Send automated payment notifications and reminders

        VOICE CHARACTERISTICS:
        - Tone: Reassuring and professional for financial concerns
        - Pace: Measured and clear when explaining billing details
        - Style: Empathetic problem-solver focused on financial resolution
        - Approach: Systematic and thorough in addressing billing issues

        TOOLS & INTEGRATIONS:
        You have access to payment processing tools (Stripe, PayPal), CRM systems, 
        notification services, and financial reporting tools. Always prioritize customer 
        financial security and follow compliance requirements.

        When unable to fully resolve an issue, clearly explain next steps and provide 
        appropriate escalation paths to human billing specialists."""

    @tool
    @latency_monitor("billing_process_refund")
    async def process_refund(self, 
                           customer_id: str, 
                           transaction_id: str, 
                           refund_amount: float, 
                           reason: str) -> Dict[str, Any]:
        """
        Process customer refund with validation and automation
        
        Args:
            customer_id: Customer identifier
            transaction_id: Original transaction to refund
            refund_amount: Amount to refund (USD)
            reason: Reason for refund request
            
        Returns:
            Refund processing result with status and details
        """
        try:
            # Validate refund amount
            if refund_amount > self.payment_config["max_refund_amount"]:
                return {
                    "success": False,
                    "error": f"Refund amount exceeds maximum limit of ${self.payment_config['max_refund_amount']}",
                    "escalation_required": True
                }
            
            # Auto-process small refunds
            if refund_amount <= self.payment_config["auto_refund_threshold"]:
                refund_id = f"auto_refund_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Simulate payment processing
                await asyncio.sleep(0.1)  # Simulate API call latency
                
                result = {
                    "success": True,
                    "refund_id": refund_id,
                    "amount": refund_amount,
                    "status": "processed",
                    "estimated_arrival": "3-5 business days",
                    "reference_number": f"REF-{refund_id.upper()}",
                    "original_transaction": transaction_id,
                    "reason": reason,
                    "processed_at": datetime.now().isoformat()
                }
                
                # Send confirmation notification
                await self.send_payment_notification(
                    customer_id, 
                    "refund_processed", 
                    {"refund_details": result}
                )
                
                return result
            else:
                # Require manual approval for larger refunds
                return {
                    "success": False,
                    "message": "Refund requires manual approval",
                    "amount": refund_amount,
                    "approval_required": True,
                    "ticket_created": f"BILLING-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    "estimated_review_time": "24-48 hours"
                }
                
        except Exception as e:
            logger.error(f"Refund processing error: {str(e)}")
            return {
                "success": False,
                "error": "System error processing refund",
                "support_ticket": True
            }

    @tool
    @latency_monitor("billing_update_payment_method")
    async def update_payment_method(self, 
                                  customer_id: str, 
                                  payment_method_type: str,
                                  is_primary: bool = False) -> Dict[str, Any]:
        """
        Update customer payment method securely
        
        Args:
            customer_id: Customer identifier
            payment_method_type: Type of payment method (card, bank_account, paypal)
            is_primary: Whether to set as primary payment method
            
        Returns:
            Payment method update status and instructions
        """
        try:
            # Generate secure update link (simulated)
            update_token = f"pm_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                "success": True,
                "update_method": "secure_link",
                "secure_link": f"https://secure-billing.example.com/update/{update_token}",
                "link_expires": (datetime.now() + timedelta(hours=24)).isoformat(),
                "instructions": "Click the secure link to update your payment method. Link expires in 24 hours.",
                "payment_method_type": payment_method_type,
                "will_be_primary": is_primary,
                "current_methods_count": 2  # Simulated
            }
            
        except Exception as e:
            logger.error(f"Payment method update error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to generate secure update link",
                "alternative": "Please call customer service for manual update"
            }

    @tool  
    @latency_monitor("billing_check_balance")
    async def check_account_balance(self, customer_id: str) -> Dict[str, Any]:
        """
        Check customer account balance and payment status
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Account balance details and payment status
        """
        try:
            # Simulate account lookup
            await asyncio.sleep(0.05)
            
            return {
                "success": True,
                "customer_id": customer_id,
                "current_balance": -45.99,  # Negative indicates amount owed
                "account_status": "current",
                "next_payment_due": (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
                "last_payment": {
                    "amount": 29.99,
                    "date": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    "method": "Credit Card ****1234"
                },
                "subscription_details": {
                    "plan": "Professional Plan",
                    "monthly_amount": 29.99,
                    "next_billing_date": (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
                    "auto_renew": True
                },
                "outstanding_invoices": [
                    {
                        "invoice_id": "INV-2024-001",
                        "amount": 45.99,
                        "due_date": (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
                        "description": "Monthly Service Fee"
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Balance check error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to retrieve account information"
            }

    @tool
    @latency_monitor("billing_generate_invoice")
    async def generate_invoice(self, 
                             customer_id: str, 
                             amount: float, 
                             description: str,
                             due_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate new invoice for customer
        
        Args:
            customer_id: Customer identifier
            amount: Invoice amount
            description: Invoice description
            due_date: Payment due date (optional)
            
        Returns:
            Generated invoice details
        """
        try:
            invoice_id = f"INV-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            if not due_date:
                due_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            
            invoice_data = {
                "success": True,
                "invoice_id": invoice_id,
                "customer_id": customer_id,
                "amount": amount,
                "description": description,
                "issue_date": datetime.now().strftime('%Y-%m-%d'),
                "due_date": due_date,
                "status": "pending",
                "payment_url": f"https://billing.example.com/pay/{invoice_id}",
                "pdf_download": f"https://billing.example.com/invoice/{invoice_id}.pdf"
            }
            
            # Send invoice notification
            await self.send_payment_notification(
                customer_id, 
                "invoice_generated", 
                {"invoice": invoice_data}
            )
            
            return invoice_data
            
        except Exception as e:
            logger.error(f"Invoice generation error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to generate invoice"
            }

    @tool
    @latency_monitor("billing_apply_credit")
    async def apply_credit(self, 
                         customer_id: str, 
                         credit_amount: float, 
                         reason: str) -> Dict[str, Any]:
        """
        Apply account credit to customer account
        
        Args:
            customer_id: Customer identifier
            credit_amount: Credit amount to apply
            reason: Reason for credit application
            
        Returns:
            Credit application result
        """
        try:
            credit_id = f"CR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            return {
                "success": True,
                "credit_id": credit_id,
                "amount": credit_amount,
                "reason": reason,
                "applied_date": datetime.now().isoformat(),
                "new_account_balance": -45.99 + credit_amount,  # Simulated calculation
                "expires": (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d'),
                "reference": f"CREDIT-{credit_id}"
            }
            
        except Exception as e:
            logger.error(f"Credit application error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to apply account credit"
            }

    @tool
    @latency_monitor("billing_cancel_subscription")
    async def cancel_subscription(self, 
                                customer_id: str, 
                                cancellation_reason: str,
                                effective_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel customer subscription with proper handling
        
        Args:
            customer_id: Customer identifier
            cancellation_reason: Reason for cancellation
            effective_date: When cancellation should take effect
            
        Returns:
            Cancellation processing result
        """
        try:
            if not effective_date:
                # Cancel at end of current billing period
                effective_date = (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d')
            
            cancellation_id = f"CANCEL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            result = {
                "success": True,
                "cancellation_id": cancellation_id,
                "effective_date": effective_date,
                "reason": cancellation_reason,
                "service_continues_until": effective_date,
                "final_billing_date": effective_date,
                "refund_eligible": False,  # Simplified logic
                "confirmation_number": cancellation_id,
                "data_retention": "90 days after cancellation"
            }
            
            # Send cancellation confirmation
            await self.send_payment_notification(
                customer_id, 
                "subscription_cancelled", 
                {"cancellation": result}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Subscription cancellation error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to process cancellation"
            }

    @tool
    @latency_monitor("billing_send_notification")
    async def send_payment_notification(self, 
                                      customer_id: str, 
                                      notification_type: str,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send payment-related notifications to customer
        
        Args:
            customer_id: Customer identifier
            notification_type: Type of notification to send
            context: Additional context for notification
            
        Returns:
            Notification sending result
        """
        try:
            notification_templates = {
                "payment_reminder": "Your payment of ${amount} is due on {due_date}",
                "payment_received": "Payment of ${amount} received successfully",
                "refund_processed": "Refund of ${amount} has been processed",
                "invoice_generated": "New invoice #{invoice_id} for ${amount}",
                "subscription_cancelled": "Subscription cancelled effective {date}"
            }
            
            template = notification_templates.get(notification_type, "General billing notification")
            
            return {
                "success": True,
                "notification_id": f"NOTIF-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "type": notification_type,
                "customer_id": customer_id,
                "channels": ["email", "sms"],  # Configurable
                "sent_at": datetime.now().isoformat(),
                "template_used": template,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Notification sending error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to send notification"
            }

    @tool
    @latency_monitor("billing_resolve_dispute")
    async def resolve_billing_dispute(self, 
                                    customer_id: str, 
                                    dispute_type: str,
                                    dispute_details: str) -> Dict[str, Any]:
        """
        Handle billing disputes and chargebacks
        
        Args:
            customer_id: Customer identifier
            dispute_type: Type of dispute (chargeback, billing_error, etc.)
            dispute_details: Details of the dispute
            
        Returns:
            Dispute resolution status and next steps
        """
        try:
            dispute_id = f"DISPUTE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Auto-resolve simple billing errors
            if dispute_type == "billing_error" and "duplicate charge" in dispute_details.lower():
                return {
                    "success": True,
                    "dispute_id": dispute_id,
                    "resolution": "auto_resolved",
                    "action_taken": "Duplicate charge identified and refunded",
                    "refund_amount": 29.99,  # Simulated
                    "resolution_time": "immediate",
                    "case_closed": True
                }
            
            # For complex disputes, create case for manual review
            return {
                "success": True,
                "dispute_id": dispute_id,
                "status": "under_review",
                "estimated_resolution": "5-10 business days",
                "assigned_specialist": "Billing Dispute Team",
                "case_number": dispute_id,
                "next_update": (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
                "customer_actions_required": "Provide transaction receipts if available"
            }
            
        except Exception as e:
            logger.error(f"Dispute resolution error: {str(e)}")
            return {
                "success": False,
                "error": "Unable to process dispute"
            }

    async def _process_conversation(self, message: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Process billing-specific conversation with enhanced context
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            Agent response with billing-specific enhancements
        """
        try:
            # Extract billing context
            billing_context = await self._extract_billing_context(message, context)
            
            # Enhanced system prompt with billing context
            enhanced_prompt = f"""{self._get_system_prompt()}
            
            CURRENT BILLING CONTEXT:
            - Customer ID: {billing_context.get('customer_id', 'Not identified')}
            - Account Status: {billing_context.get('account_status', 'Unknown')}
            - Outstanding Balance: ${billing_context.get('balance', 0.00)}
            - Last Payment: {billing_context.get('last_payment_date', 'No recent payments')}
            
            Focus on resolving the customer's billing concern efficiently and empathetically."""
            
            # Process with enhanced context
            messages = [
                SystemMessage(content=enhanced_prompt),
                HumanMessage(content=message)
            ]
            
            # Use the LangGraph agent for processing
            response = await self.agent.ainvoke({
                "messages": messages,
                "context": billing_context
            })
            
            return AgentResponse(
                content=response["messages"][-1].content,
                agent_used="billing-support",
                confidence=0.95,  # High confidence for specialized agent
                tools_used=response.get("tool_calls", []),
                context_updates=billing_context,
                latency_ms=response.get("processing_time", 0)
            )
            
        except Exception as e:
            logger.error(f"Billing conversation processing error: {str(e)}")
            return AgentResponse(
                content="I apologize, but I'm experiencing technical difficulties with billing inquiries. Please contact our billing support team directly for immediate assistance.",
                agent_used="billing-support",
                confidence=0.0,
                error=str(e)
            )

    async def _extract_billing_context(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract billing-specific context from message and conversation history"""
        billing_context = {
            "customer_id": context.get("customer_id"),
            "account_status": "active",  # Simulated
            "balance": -45.99,  # Simulated
            "last_payment_date": "2024-04-15",  # Simulated
            "billing_inquiry_type": self._classify_billing_inquiry(message)
        }
        
        return billing_context

    def _classify_billing_inquiry(self, message: str) -> str:
        """Classify the type of billing inquiry"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["refund", "return", "money back"]):
            return "refund_request"
        elif any(word in message_lower for word in ["payment", "card", "method"]):
            return "payment_method"
        elif any(word in message_lower for word in ["balance", "owe", "bill"]):
            return "account_balance"
        elif any(word in message_lower for word in ["cancel", "subscription", "plan"]):
            return "subscription_management"
        elif any(word in message_lower for word in ["dispute", "charge", "error"]):
            return "billing_dispute"
        else:
            return "general_billing"

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get billing agent performance metrics"""
        base_metrics = super().get_agent_metrics()
        
        billing_metrics = {
            "refunds_processed": 145,  # Simulated
            "average_resolution_time": "2.3 minutes",
            "payment_method_updates": 89,
            "subscription_changes": 34,
            "billing_disputes_resolved": 12,
            "customer_satisfaction": 4.7,
            "auto_resolution_rate": 0.78
        }
        
        return {**base_metrics, "billing_specific": billing_metrics}