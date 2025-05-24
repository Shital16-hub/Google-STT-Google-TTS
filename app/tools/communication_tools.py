"""
Communication Tools - SMS and Email Integration with Mock Layer
Part of the Multi-Agent Voice AI System Transformation

This module provides comprehensive communication capabilities:
- SMS messaging via Twilio
- Email delivery via SendGrid, AWS SES
- Multi-channel notification orchestration
- Template management and personalization
- Delivery tracking and analytics
- Mock implementations for development

PRODUCTION SETUP:
1. Add real API credentials to environment variables
2. Set USE_MOCK_COMMUNICATIONS=False in configuration  
3. Install required SDKs: twilio, sendgrid, boto3
4. Configure notification templates and branding
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import uuid
import re
from dataclasses import dataclass
import html

from app.core.latency_optimizer import latency_monitor

# Configure logging
logger = logging.getLogger(__name__)

class CommunicationChannel(Enum):
    """Supported communication channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    VOICE = "voice"

class MessageStatus(Enum):
    """Message delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPENED = "opened"
    CLICKED = "clicked"

class NotificationType(Enum):
    """Types of notifications"""
    PAYMENT_CONFIRMATION = "payment_confirmation"
    PAYMENT_REMINDER = "payment_reminder"
    REFUND_PROCESSED = "refund_processed"
    TICKET_CREATED = "ticket_created"
    TICKET_UPDATED = "ticket_updated"
    WELCOME_EMAIL = "welcome_email"
    APPOINTMENT_REMINDER = "appointment_reminder"
    EMERGENCY_ALERT = "emergency_alert"
    BILLING_DISPUTE = "billing_dispute"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"

@dataclass
class CommunicationConfig:
    """Configuration for communication providers"""
    channel: CommunicationChannel
    provider: str
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    sender_id: Optional[str] = None
    enabled: bool = True
    use_mock: bool = True

@dataclass
class MessageTemplate:
    """Template for messages"""
    template_id: str
    name: str
    channel: CommunicationChannel
    subject: Optional[str] = None
    body: str = ""
    variables: List[str] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []

class CommunicationManager:
    """
    Unified communication manager for multi-channel messaging
    Handles SMS, email, and other notification channels
    """
    
    def __init__(self):
        # Configuration management
        self.use_mock_communications = os.getenv("USE_MOCK_COMMUNICATIONS", "true").lower() == "true"
        
        # Communication provider configurations
        self.comm_configs = self._load_communication_configurations()
        
        # Message templates
        self.templates = self._load_message_templates()
        
        # Mock data stores (for development)
        self.mock_messages = {}
        self.mock_delivery_logs = {}
        
        # Delivery tracking
        self.delivery_stats = {
            "total_sent": 0,
            "total_delivered": 0, 
            "total_failed": 0,
            "delivery_rate": 0.0
        }
        
        # Initialize mock data
        if self.use_mock_communications:
            self._initialize_mock_data()
            
        logger.info(f"Communication Manager initialized - Mock mode: {self.use_mock_communications}")

    def _load_communication_configurations(self) -> Dict[str, CommunicationConfig]:
        """Load communication provider configurations"""
        configs = {}
        
        # Twilio SMS
        configs["twilio"] = CommunicationConfig(
            channel=CommunicationChannel.SMS,
            provider="twilio",
            api_key=os.getenv("TWILIO_ACCOUNT_SID"),
            secret_key=os.getenv("TWILIO_AUTH_TOKEN"),
            sender_id=os.getenv("TWILIO_PHONE_NUMBER"),
            use_mock=self.use_mock_communications
        )
        
        # SendGrid Email
        configs["sendgrid"] = CommunicationConfig(
            channel=CommunicationChannel.EMAIL,
            provider="sendgrid",
            api_key=os.getenv("SENDGRID_API_KEY"),
            sender_id=os.getenv("SENDGRID_FROM_EMAIL", "noreply@example.com"),
            use_mock=self.use_mock_communications
        )
        
        # AWS SES Email
        configs["aws_ses"] = CommunicationConfig(
            channel=CommunicationChannel.EMAIL,
            provider="aws_ses",
            api_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            sender_id=os.getenv("AWS_SES_FROM_EMAIL", "noreply@example.com"),
            use_mock=self.use_mock_communications
        )
        
        return configs

    def _load_message_templates(self) -> Dict[str, MessageTemplate]:
        """Load message templates for different notification types"""
        templates = {}
        
        # Payment confirmation templates
        templates["payment_confirmation_email"] = MessageTemplate(
            template_id="payment_confirmation_email",
            name="Payment Confirmation Email",
            channel=CommunicationChannel.EMAIL,
            subject="Payment Confirmation - ${amount} ${currency}",
            body="""
            <html>
            <body>
                <h2>Payment Confirmation</h2>
                <p>Dear ${customer_name},</p>
                <p>Your payment has been successfully processed:</p>
                <ul>
                    <li><strong>Amount:</strong> ${amount} ${currency}</li>
                    <li><strong>Transaction ID:</strong> ${transaction_id}</li>
                    <li><strong>Date:</strong> ${payment_date}</li>
                    <li><strong>Payment Method:</strong> ${payment_method}</li>
                </ul>
                <p>Thank you for your business!</p>
                <p>Best regards,<br>Customer Support Team</p>
            </body>
            </html>
            """,
            variables=["customer_name", "amount", "currency", "transaction_id", "payment_date", "payment_method"]
        )
        
        templates["payment_confirmation_sms"] = MessageTemplate(
            template_id="payment_confirmation_sms",
            name="Payment Confirmation SMS",
            channel=CommunicationChannel.SMS,
            body="Payment confirmed: ${amount} ${currency}. Transaction ID: ${transaction_id}. Thank you!",
            variables=["amount", "currency", "transaction_id"]
        )
        
        # Ticket notification templates
        templates["ticket_created_email"] = MessageTemplate(
            template_id="ticket_created_email",
            name="Support Ticket Created",
            channel=CommunicationChannel.EMAIL,
            subject="Support Ticket Created - #${ticket_id}",
            body="""
            <html>
            <body>
                <h2>Support Ticket Created</h2>
                <p>Dear ${customer_name},</p>
                <p>Your support ticket has been created and assigned to our team:</p>
                <ul>
                    <li><strong>Ticket ID:</strong> #${ticket_id}</li>
                    <li><strong>Subject:</strong> ${ticket_subject}</li>
                    <li><strong>Priority:</strong> ${priority}</li>
                    <li><strong>Expected Response:</strong> ${sla_response_time}</li>
                </ul>
                <p>Track your ticket: <a href="${tracking_url}">View Ticket</a></p>
                <p>Our team will respond within ${sla_response_time}.</p>
                <p>Best regards,<br>Support Team</p>
            </body>
            </html>
            """,
            variables=["customer_name", "ticket_id", "ticket_subject", "priority", "sla_response_time", "tracking_url"]
        )
        
        # Welcome email template
        templates["welcome_email"] = MessageTemplate(
            template_id="welcome_email",
            name="Welcome Email",
            channel=CommunicationChannel.EMAIL,
            subject="Welcome to ${company_name}!",
            body="""
            <html>
            <body>
                <h1>Welcome to ${company_name}!</h1>
                <p>Dear ${customer_name},</p>
                <p>Thank you for joining us! We're excited to have you on board.</p>
                <p>Here's what you can expect:</p>
                <ul>
                    <li>24/7 customer support</li>
                    <li>Regular product updates</li>
                    <li>Access to our knowledge base</li>
                </ul>
                <p>Get started: <a href="${dashboard_url}">Access Your Dashboard</a></p>
                <p>If you have any questions, don't hesitate to reach out!</p>
                <p>Best regards,<br>The ${company_name} Team</p>
            </body>
            </html>
            """,
            variables=["customer_name", "company_name", "dashboard_url"]
        )
        
        # Emergency alert templates
        templates["emergency_alert_sms"] = MessageTemplate(
            template_id="emergency_alert_sms",
            name="Emergency Alert SMS",
            channel=CommunicationChannel.SMS,
            body="URGENT: ${alert_message}. Ticket #${ticket_id}. Call ${support_phone} for immediate assistance.",
            variables=["alert_message", "ticket_id", "support_phone"]
        )
        
        return templates

    def _initialize_mock_data(self):
        """Initialize mock communication data"""
        # Mock sent messages for testing
        self.mock_messages = {
            "msg_001": {
                "id": "msg_001",
                "channel": "email",
                "to": "test@example.com",
                "subject": "Test Email",
                "body": "This is a test email",
                "status": MessageStatus.DELIVERED.value,
                "sent_at": datetime.now().isoformat(),
                "delivered_at": (datetime.now() + timedelta(seconds=30)).isoformat()
            }
        }

    # =============================================================================
    # EMAIL COMMUNICATIONS
    # =============================================================================

    @latency_monitor("comm_send_email")
    async def send_email(self, 
                        email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send email message
        
        Args:
            email_data: Email details (to, subject, body, template_id, variables, etc.)
            
        Returns:
            Email sending result with message ID and delivery status
        """
        if self.use_mock_communications:
            return await self._mock_send_email(email_data)
        
        # Real email sending
        provider = email_data.get("provider", "sendgrid")
        
        if provider == "sendgrid":
            return await self._sendgrid_send_email(email_data)
        elif provider == "aws_ses":
            return await self._aws_ses_send_email(email_data)
        else:
            raise ValueError(f"Unsupported email provider: {provider}")

    async def _mock_send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for email sending"""
        await asyncio.sleep(0.1)  # Simulate sending time
        
        message_id = f"email_{uuid.uuid4().hex[:12]}"
        
        # Process template if provided
        subject = email_data.get("subject", "")
        body = email_data.get("body", "")
        
        if email_data.get("template_id"):
            template_result = await self._process_template(
                email_data["template_id"],
                email_data.get("variables", {})
            )
            if template_result["success"]:
                subject = template_result["subject"]
                body = template_result["body"]
        
        # Validate email
        to_email = email_data.get("to")
        if not to_email or not self._validate_email(to_email):
            return {
                "success": False,
                "error": "Invalid email address",
                "email": to_email
            }
        
        # Simulate delivery (95% success rate)
        import random
        success = random.random() < 0.95
        
        message_record = {
            "id": message_id,
            "channel": CommunicationChannel.EMAIL.value,
            "to": to_email,
            "from": email_data.get("from", "noreply@example.com"),
            "subject": subject,
            "body": body,
            "status": MessageStatus.SENT.value if success else MessageStatus.FAILED.value,
            "sent_at": datetime.now().isoformat(),
            "provider": "mock_sendgrid",
            "template_id": email_data.get("template_id")
        }
        
        if success:
            # Simulate delivery after a short delay
            message_record["delivered_at"] = (datetime.now() + timedelta(seconds=5)).isoformat()
            message_record["status"] = MessageStatus.DELIVERED.value
        
        self.mock_messages[message_id] = message_record
        self._update_delivery_stats(success)
        
        return {
            "success": success,
            "message_id": message_id,
            "status": message_record["status"],
            "to": to_email,
            "sent_at": message_record["sent_at"],
            "provider": "mock_sendgrid",
            "error": None if success else "Mock delivery failure"
        }

    @latency_monitor("comm_payment_confirmation")
    async def send_payment_confirmation(self, 
                                      payment_data: Dict[str, Any],
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send payment confirmation via email and SMS
        
        Args:
            payment_data: Payment information
            context: Additional context from workflow
            
        Returns:
            Multi-channel sending result
        """
        results = {}
        
        # Prepare template variables
        variables = {
            "customer_name": payment_data.get("customer_name", "Valued Customer"),
            "amount": f"{payment_data.get('amount', 0):.2f}",
            "currency": payment_data.get("currency", "USD").upper(),
            "transaction_id": payment_data.get("transaction_id", "N/A"),
            "payment_date": datetime.now().strftime("%B %d, %Y"),
            "payment_method": payment_data.get("payment_method", "Credit Card")
        }
        
        # Send email confirmation
        if payment_data.get("email"):
            email_result = await self.send_email({
                "to": payment_data["email"],
                "template_id": "payment_confirmation_email",
                "variables": variables
            })
            results["email"] = email_result
        
        # Send SMS confirmation if phone provided
        if payment_data.get("phone"):
            sms_result = await self.send_sms({
                "to": payment_data["phone"],
                "template_id": "payment_confirmation_sms",
                "variables": variables
            })
            results["sms"] = sms_result
        
        return {
            "success": any(r.get("success", False) for r in results.values()),
            "channels_sent": len([r for r in results.values() if r.get("success", False)]),
            "results": results
        }

    # =============================================================================
    # SMS COMMUNICATIONS
    # =============================================================================

    @latency_monitor("comm_send_sms")
    async def send_sms(self, 
                      sms_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send SMS message
        
        Args:
            sms_data: SMS details (to, body, template_id, variables, etc.)
            
        Returns:
            SMS sending result with message ID and delivery status
        """
        if self.use_mock_communications:
            return await self._mock_send_sms(sms_data)
        
        # Real SMS sending via Twilio
        return await self._twilio_send_sms(sms_data)

    async def _mock_send_sms(self, sms_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for SMS sending"""
        await asyncio.sleep(0.08)  # Simulate sending time
        
        message_id = f"sms_{uuid.uuid4().hex[:12]}"
        
        # Process template if provided
        body = sms_data.get("body", "")
        
        if sms_data.get("template_id"):
            template_result = await self._process_template(
                sms_data["template_id"],
                sms_data.get("variables", {})
            )
            if template_result["success"]:
                body = template_result["body"]
        
        # Validate phone number
        to_phone = sms_data.get("to")
        if not to_phone or not self._validate_phone(to_phone):
            return {
                "success": False,
                "error": "Invalid phone number",
                "phone": to_phone
            }
        
        # Check message length (SMS limit is typically 160 characters)
        if len(body) > 160:
            return {
                "success": False,
                "error": "Message exceeds SMS length limit (160 characters)",
                "length": len(body)
            }
        
        # Simulate delivery (90% success rate for SMS)
        import random
        success = random.random() < 0.90
        
        message_record = {
            "id": message_id,
            "channel": CommunicationChannel.SMS.value,
            "to": to_phone,
            "from": "+1-555-0123",  # Mock Twilio number
            "body": body,
            "status": MessageStatus.SENT.value if success else MessageStatus.FAILED.value,
            "sent_at": datetime.now().isoformat(),
            "provider": "mock_twilio",
            "template_id": sms_data.get("template_id"),
            "character_count": len(body)
        }
        
        if success:
            message_record["delivered_at"] = (datetime.now() + timedelta(seconds=2)).isoformat()
            message_record["status"] = MessageStatus.DELIVERED.value
        
        self.mock_messages[message_id] = message_record
        self._update_delivery_stats(success)
        
        return {
            "success": success,
            "message_id": message_id,
            "status": message_record["status"],
            "to": to_phone,
            "sent_at": message_record["sent_at"],
            "character_count": len(body),
            "provider": "mock_twilio",
            "error": None if success else "Mock delivery failure"
        }

    # =============================================================================
    # SPECIALIZED NOTIFICATION METHODS
    # =============================================================================

    @latency_monitor("comm_ticket_notification")
    async def send_ticket_notification(self, 
                                     ticket_data: Dict[str, Any],
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send support ticket creation notification
        
        Args:
            ticket_data: Ticket information
            context: Additional context from workflow
            
        Returns:
            Notification sending result
        """
        variables = {
            "customer_name": ticket_data.get("customer_name", "Valued Customer"),
            "ticket_id": ticket_data.get("ticket_id", "N/A"),
            "ticket_subject": ticket_data.get("title", "Support Request"),
            "priority": ticket_data.get("priority", "Medium").title(),
            "sla_response_time": ticket_data.get("sla_response_time", "24 hours"),
            "tracking_url": ticket_data.get("tracking_url", "https://support.example.com")
        }
        
        # Send email notification
        if ticket_data.get("customer_email"):
            return await self.send_email({
                "to": ticket_data["customer_email"],
                "template_id": "ticket_created_email",
                "variables": variables
            })
        
        return {"success": False, "error": "No customer email provided"}

    @latency_monitor("comm_welcome_email")
    async def send_welcome_email(self, 
                               customer_data: Dict[str, Any],
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send welcome email to new customers
        
        Args:
            customer_data: Customer information
            context: Additional context from workflow
            
        Returns:
            Welcome email sending result
        """
        variables = {
            "customer_name": customer_data.get("name", "Valued Customer"),
            "company_name": "Your Company",  # Configure this
            "dashboard_url": "https://app.example.com/dashboard"
        }
        
        if customer_data.get("email"):
            return await self.send_email({
                "to": customer_data["email"],
                "template_id": "welcome_email",
                "variables": variables
            })
        
        return {"success": False, "error": "No customer email provided"}

    @latency_monitor("comm_notify_on_call")
    async def notify_on_call_team(self, 
                                alert_data: Dict[str, Any],
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send emergency notification to on-call team
        
        Args:
            alert_data: Emergency alert information
            context: Additional context from workflow
            
        Returns:
            Emergency notification result
        """
        # On-call team contacts (configure these)
        on_call_contacts = [
            {"name": "Emergency Support", "phone": "+1-555-0911", "email": "emergency@example.com"}
        ]
        
        variables = {
            "alert_message": alert_data.get("message", "Critical system alert"),
            "ticket_id": alert_data.get("ticket_id", "N/A"),
            "support_phone": "+1-555-HELP"
        }
        
        results = {}
        
        # Send SMS to all on-call team members
        for contact in on_call_contacts:
            sms_result = await self.send_sms({
                "to": contact["phone"],
                "template_id": "emergency_alert_sms",
                "variables": variables
            })
            results[f"sms_{contact['name']}"] = sms_result
        
        return {
            "success": any(r.get("success", False) for r in results.values()),
            "notifications_sent": len([r for r in results.values() if r.get("success", False)]),
            "results": results
        }

    @latency_monitor("comm_dispute_update")
    async def send_dispute_update(self, 
                                dispute_data: Dict[str, Any],
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send billing dispute status update
        
        Args:
            dispute_data: Dispute information
            context: Additional context from workflow
            
        Returns:
            Dispute update notification result
        """
        # Create custom message for dispute update
        subject = f"Billing Dispute Update - Case #{dispute_data.get('dispute_id', 'N/A')}"
        
        body = f"""
        <html>
        <body>
            <h2>Billing Dispute Update</h2>
            <p>Dear {dispute_data.get('customer_name', 'Valued Customer')},</p>
            <p>We have an update regarding your billing dispute:</p>
            <ul>
                <li><strong>Case ID:</strong> #{dispute_data.get('dispute_id', 'N/A')}</li>
                <li><strong>Status:</strong> {dispute_data.get('status', 'Under Review').title()}</li>
                <li><strong>Amount:</strong> ${dispute_data.get('amount', 0):.2f}</li>
                <li><strong>Next Steps:</strong> {dispute_data.get('next_steps', 'We will contact you within 2 business days')}</li>
            </ul>
            <p>If you have any questions, please don't hesitate to contact us.</p>
            <p>Best regards,<br>Billing Support Team</p>
        </body>
        </html>
        """
        
        if dispute_data.get("customer_email"):
            return await self.send_email({
                "to": dispute_data["customer_email"],
                "subject": subject,
                "body": body
            })
        
        return {"success": False, "error": "No customer email provided"}

    # =============================================================================
    # TEMPLATE PROCESSING
    # =============================================================================

    async def _process_template(self, 
                              template_id: str, 
                              variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process message template with variable substitution
        
        Args:
            template_id: Template identifier
            variables: Variables to substitute in template
            
        Returns:
            Processed template with substituted content
        """
        if template_id not in self.templates:
            return {
                "success": False,
                "error": f"Template {template_id} not found"
            }
        
        template = self.templates[template_id]
        
        try:
            # Process subject (for email templates)
            processed_subject = template.subject
            if processed_subject:
                for var, value in variables.items():
                    processed_subject = processed_subject.replace(f"${{{var}}}", str(value))
            
            # Process body
            processed_body = template.body
            for var, value in variables.items():
                processed_body = processed_body.replace(f"${{{var}}}", str(value))
            
            return {
                "success": True,
                "template_id": template_id,
                "subject": processed_subject,
                "body": processed_body,
                "channel": template.channel.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Template processing error: {str(e)}"
            }

    # =============================================================================
    # REAL API IMPLEMENTATIONS (Templates)
    # =============================================================================

    async def _sendgrid_send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real SendGrid API implementation"""
        # TODO: Implement actual SendGrid API calls
        # import sendgrid
        # from sendgrid.helpers.mail import Mail
        
        logger.info("SendGrid API not implemented yet - using mock")
        return await self._mock_send_email(email_data)

    async def _aws_ses_send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real AWS SES API implementation"""
        # TODO: Implement actual AWS SES API calls
        # import boto3
        
        logger.info("AWS SES API not implemented yet - using mock")
        return await self._mock_send_email(email_data)

    async def _twilio_send_sms(self, sms_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real Twilio API implementation"""
        # TODO: Implement actual Twilio API calls
        # from twilio.rest import Client
        
        logger.info("Twilio API not implemented yet - using mock")
        return await self._mock_send_sms(sms_data)

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def _validate_email(self, email: str) -> bool:
        """Validate email address format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def _validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        import re
        # Simple validation for E.164 format
        pattern = r'^\+?[1-9]\d{1,14}$'
        cleaned_phone = re.sub(r'[^\d+]', '', phone)
        return re.match(pattern, cleaned_phone) is not None

    def _update_delivery_stats(self, success: bool):
        """Update delivery statistics"""
        self.delivery_stats["total_sent"] += 1
        if success:
            self.delivery_stats["total_delivered"] += 1
        else:
            self.delivery_stats["total_failed"] += 1
        
        # Calculate delivery rate
        total = self.delivery_stats["total_sent"]
        delivered = self.delivery_stats["total_delivered"]
        self.delivery_stats["delivery_rate"] = (delivered / total) * 100 if total > 0 else 0

    def get_communication_status(self) -> Dict[str, Any]:
        """Get communication system status"""
        return {
            "mock_mode": self.use_mock_communications,
            "configured_providers": {
                name: {
                    "channel": config.channel.value,
                    "provider": config.provider,
                    "enabled": config.enabled,
                    "has_credentials": bool(config.api_key)
                }
                for name, config in self.comm_configs.items()
            },
            "available_templates": {
                template_id: {
                    "name": template.name,
                    "channel": template.channel.value,
                    "variables": template.variables
                }
                for template_id, template in self.templates.items()
            },
            "delivery_stats": self.delivery_stats,
            "mock_data_counts": {
                "messages": len(self.mock_messages)
            }
        }

    def enable_production_mode(self):
        """Switch to production mode (real communications)"""
        self.use_mock_communications = False
        for config in self.comm_configs.values():
            config.use_mock = False
        logger.info("Switched to production communication mode - real APIs will be used")

    def enable_mock_mode(self):
        """Switch to mock mode (simulated communications)"""
        self.use_mock_communications = True
        for config in self.comm_configs.values():
            config.use_mock = True
        logger.info("Switched to mock communication mode - simulated sending will be used")

    def get_required_environment_variables(self) -> Dict[str, List[str]]:
        """Get required environment variables for production setup"""
        return {
            "twilio": [
                "TWILIO_ACCOUNT_SID",
                "TWILIO_AUTH_TOKEN",
                "TWILIO_PHONE_NUMBER"
            ],
            "sendgrid": [
                "SENDGRID_API_KEY",
                "SENDGRID_FROM_EMAIL"
            ],
            "aws_ses": [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SES_FROM_EMAIL"
            ]
        }

    def generate_setup_instructions(self) -> str:
        """Generate setup instructions for production communication setup"""
        return """
COMMUNICATION TOOLS SETUP INSTRUCTIONS:

1. ENVIRONMENT VARIABLES:
   Set the following environment variables for production:
   
   # Twilio SMS
   export TWILIO_ACCOUNT_SID="your-account-sid"
   export TWILIO_AUTH_TOKEN="your-auth-token"
   export TWILIO_PHONE_NUMBER="+1234567890"
   
   # SendGrid Email
   export SENDGRID_API_KEY="SG.your-api-key"
   export SENDGRID_FROM_EMAIL="noreply@yourdomain.com"
   
   # AWS SES Email (Alternative)
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_SES_FROM_EMAIL="noreply@yourdomain.com"

2. DISABLE MOCK MODE:
   export USE_MOCK_COMMUNICATIONS=false

3. INSTALL REQUIRED PACKAGES:
   pip install twilio sendgrid boto3

4. DOMAIN VERIFICATION:
   - For SendGrid: Verify your sending domain
   - For AWS SES: Verify domain and move out of sandbox mode
   - For Twilio: Verify your phone number

5. TEMPLATE CUSTOMIZATION:
   - Update email templates with your branding
   - Customize SMS message formats
   - Configure sender names and contact information

6. COMPLIANCE CONSIDERATIONS:
   - Implement unsubscribe mechanisms for marketing emails
   - Follow CAN-SPAM Act requirements
   - Respect SMS opt-out requests
   - Maintain communication preference records

7. UPDATE API IMPLEMENTATIONS:
   Uncomment and complete the real API implementation methods in this file.

8. TEST COMMUNICATIONS:
   Use small test campaigns to verify all integrations work correctly.
"""

    async def get_message_delivery_status(self, message_id: str) -> Dict[str, Any]:
        """Get delivery status for a specific message"""
        if self.use_mock_communications:
            if message_id in self.mock_messages:
                message = self.mock_messages[message_id]
                return {
                    "success": True,
                    "message_id": message_id,
                    "status": message["status"],
                    "sent_at": message["sent_at"],
                    "delivered_at": message.get("delivered_at"),
                    "channel": message["channel"]
                }
            else:
                return {
                    "success": False,
                    "error": f"Message {message_id} not found"
                }
        
        # TODO: Implement real delivery status checking
        return {"success": False, "error": "Real delivery status checking not implemented"}

    def get_delivery_analytics(self, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get communication delivery analytics"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # For mock mode, return simulated analytics
        if self.use_mock_communications:
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "totals": self.delivery_stats,
                "by_channel": {
                    "email": {
                        "sent": 150,
                        "delivered": 143,
                        "opened": 89,
                        "clicked": 23,
                        "bounced": 5,
                        "delivery_rate": 95.3,
                        "open_rate": 62.2,
                        "click_rate": 15.3
                    },
                    "sms": {
                        "sent": 89,
                        "delivered": 85,
                        "failed": 4,
                        "delivery_rate": 95.5
                    }
                },
                "templates_used": {
                    "payment_confirmation_email": 45,
                    "ticket_created_email": 32,
                    "welcome_email": 28,
                    "payment_confirmation_sms": 34
                }
            }
        
        # TODO: Implement real analytics from provider APIs
        return {"success": False, "error": "Real analytics not implemented"}

    async def send_bulk_communication(self, 
                                    recipients: List[Dict[str, Any]],
                                    template_id: str,
                                    batch_size: int = 100) -> Dict[str, Any]:
        """
        Send bulk communications with batching and rate limiting
        
        Args:
            recipients: List of recipient data with contact info and variables
            template_id: Template to use for all messages
            batch_size: Number of messages to send per batch
            
        Returns:
            Bulk sending results with success/failure counts
        """
        if template_id not in self.templates:
            return {
                "success": False,
                "error": f"Template {template_id} not found"
            }
        
        template = self.templates[template_id]
        channel = template.channel
        
        results = {
            "total_recipients": len(recipients),
            "successful_sends": 0,
            "failed_sends": 0,
            "batches_processed": 0,
            "errors": []
        }
        
        # Process recipients in batches
        for i in range(0, len(recipients), batch_size):
            batch = recipients[i:i + batch_size]
            batch_results = []
            
            # Process batch
            for recipient in batch:
                try:
                    if channel == CommunicationChannel.EMAIL:
                        if not recipient.get("email"):
                            results["failed_sends"] += 1
                            continue
                        
                        send_result = await self.send_email({
                            "to": recipient["email"],
                            "template_id": template_id,
                            "variables": recipient.get("variables", {})
                        })
                    elif channel == CommunicationChannel.SMS:
                        if not recipient.get("phone"):
                            results["failed_sends"] += 1
                            continue
                        
                        send_result = await self.send_sms({
                            "to": recipient["phone"],
                            "template_id": template_id,
                            "variables": recipient.get("variables", {})
                        })
                    else:
                        results["failed_sends"] += 1
                        continue
                    
                    if send_result.get("success"):
                        results["successful_sends"] += 1
                    else:
                        results["failed_sends"] += 1
                        results["errors"].append({
                            "recipient": recipient.get("email") or recipient.get("phone"),
                            "error": send_result.get("error", "Unknown error")
                        })
                
                except Exception as e:
                    results["failed_sends"] += 1
                    results["errors"].append({
                        "recipient": recipient.get("email") or recipient.get("phone"),
                        "error": str(e)
                    })
            
            results["batches_processed"] += 1
            
            # Rate limiting - small delay between batches
            if i + batch_size < len(recipients):
                await asyncio.sleep(1)
        
        results["success"] = results["successful_sends"] > 0
        results["success_rate"] = (results["successful_sends"] / results["total_recipients"]) * 100
        
        return results

    def create_custom_template(self, 
                             template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a custom message template
        
        Args:
            template_data: Template definition
            
        Returns:
            Template creation result
        """
        try:
            template_id = template_data.get("template_id")
            if not template_id:
                return {
                    "success": False,
                    "error": "Template ID is required"
                }
            
            if template_id in self.templates:
                return {
                    "success": False,
                    "error": f"Template {template_id} already exists"
                }
            
            # Validate channel
            channel_str = template_data.get("channel", "").lower()
            try:
                channel = CommunicationChannel(channel_str)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid channel: {channel_str}"
                }
            
            # Create template
            template = MessageTemplate(
                template_id=template_id,
                name=template_data.get("name", template_id),
                channel=channel,
                subject=template_data.get("subject"),
                body=template_data.get("body", ""),
                variables=template_data.get("variables", [])
            )
            
            self.templates[template_id] = template
            
            return {
                "success": True,
                "template_id": template_id,
                "message": "Template created successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Template creation error: {str(e)}"
            }

    def get_mock_data_summary(self) -> Dict[str, Any]:
        """Get summary of mock communication data"""
        return {
            "messages": {
                "total": len(self.mock_messages),
                "by_channel": {
                    channel.value: len([m for m in self.mock_messages.values() 
                                      if m["channel"] == channel.value])
                    for channel in CommunicationChannel
                },
                "by_status": {
                    status.value: len([m for m in self.mock_messages.values() 
                                     if m["status"] == status.value])
                    for status in MessageStatus
                }
            },
            "templates": {
                "total": len(self.templates),
                "by_channel": {
                    channel.value: len([t for t in self.templates.values() 
                                      if t.channel == channel])
                    for channel in CommunicationChannel
                }
            },
            "delivery_stats": self.delivery_stats
        }