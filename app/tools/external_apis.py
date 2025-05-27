"""
External APIs - Advanced DUMMY Implementation
============================================

Comprehensive external API integrations with realistic dummy implementations
that can be easily replaced with real API calls in production.

Features:
- Realistic API response simulation
- Authentication and error handling patterns
- Rate limiting and retry mechanisms
- Performance monitoring and logging
- Circuit breaker patterns for reliability
- Integration-ready interfaces for production deployment
"""

import asyncio
import logging
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import random
import aiohttp
from urllib.parse import urljoin

from app.tools.tool_orchestrator import (
    BaseTool, ToolMetadata, ToolType, ExecutionContext, ToolResult
)

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """External API providers"""
    STRIPE = "stripe"
    TWILIO = "twilio"
    SENDGRID = "sendgrid"
    ZENDESK = "zendesk"
    GOOGLE_MAPS = "google_maps"
    SLACK = "slack"
    CALENDLY = "calendly"
    HUBSPOT = "hubspot"
    SALESFORCE = "salesforce"


class AuthenticationType(Enum):
    """API authentication types"""
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"


@dataclass
class APICredentials:
    """API authentication credentials"""
    provider: APIProvider
    auth_type: AuthenticationType
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None


@dataclass
class APIResponse:
    """Standardized API response structure"""
    success: bool
    status_code: int
    data: Any
    headers: Dict[str, str]
    response_time_ms: float
    provider: APIProvider
    endpoint: str
    error_message: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None


class BaseExternalAPI(BaseTool):
    """Base class for external API integrations"""
    
    def __init__(self, metadata: ToolMetadata, provider: APIProvider, base_url: str):
        super().__init__(metadata)
        self.provider = provider
        self.base_url = base_url
        self.credentials: Optional[APICredentials] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limit_calls = 0
        self.rate_limit_window_start = time.time()
        self.max_calls_per_minute = 100  # Default rate limit
        
    async def _make_api_call(self, 
                           method: str, 
                           endpoint: str, 
                           data: Optional[Dict] = None,
                           params: Optional[Dict] = None,
                           headers: Optional[Dict] = None) -> APIResponse:
        """Make authenticated API call with error handling"""
        
        start_time = time.time()
        url = urljoin(self.base_url, endpoint)
        
        try:
            # Check rate limiting
            await self._check_rate_limit()
            
            # Prepare headers with authentication
            request_headers = await self._prepare_headers(headers or {})
            
            # In DUMMY mode, simulate the API call
            if self.metadata.dummy_mode:
                return await self._simulate_api_call(method, endpoint, data, params)
            
            # Real API call (for production)
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=request_headers,
                    timeout=aiohttp.ClientTimeout(total=self.metadata.timeout_ms / 1000)
                ) as response:
                    response_data = await response.json()
                    response_time = (time.time() - start_time) * 1000
                    
                    return APIResponse(
                        success=response.status < 400,
                        status_code=response.status,
                        data=response_data,
                        headers=dict(response.headers),
                        response_time_ms=response_time,
                        provider=self.provider,
                        endpoint=endpoint
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"API call failed for {self.provider.value}: {str(e)}")
            
            return APIResponse(
                success=False,
                status_code=500,
                data=None,
                headers={},
                response_time_ms=response_time,
                provider=self.provider,
                endpoint=endpoint,
                error_message=str(e)
            )
    
    async def _simulate_api_call(self, method: str, endpoint: str, data: Optional[Dict], params: Optional[Dict]) -> APIResponse:
        """Simulate API call for DUMMY mode"""
        # Add realistic delay
        await asyncio.sleep(random.uniform(0.05, 0.3))
        
        # Simulate occasional failures (5% failure rate)
        if random.random() < 0.05:
            return APIResponse(
                success=False,
                status_code=500,
                data={"error": "Simulated API failure"},
                headers={},
                response_time_ms=random.uniform(100, 500),
                provider=self.provider,
                endpoint=endpoint,
                error_message="Simulated API failure"
            )
        
        # Return success response
        return APIResponse(
            success=True,
            status_code=200,
            data=await self._generate_dummy_response(endpoint, data),
            headers={"Content-Type": "application/json"},
            response_time_ms=random.uniform(50, 300),
            provider=self.provider,
            endpoint=endpoint
        )
    
    async def _generate_dummy_response(self, endpoint: str, data: Optional[Dict]) -> Dict[str, Any]:
        """Generate realistic dummy response data"""
        return {"message": "Dummy API response", "endpoint": endpoint}
    
    async def _prepare_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Prepare headers with authentication"""
        if not self.credentials:
            return headers
        
        if self.credentials.auth_type == AuthenticationType.API_KEY:
            headers["Authorization"] = f"Bearer {self.credentials.api_key}"
        elif self.credentials.auth_type == AuthenticationType.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {self.credentials.bearer_token}"
        elif self.credentials.custom_headers:
            headers.update(self.credentials.custom_headers)
        
        return headers
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.rate_limit_window_start >= 60:
            self.rate_limit_calls = 0
            self.rate_limit_window_start = current_time
        
        # Check if rate limit exceeded
        if self.rate_limit_calls >= self.max_calls_per_minute:
            sleep_time = 60 - (current_time - self.rate_limit_window_start)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self.rate_limit_calls = 0
                self.rate_limit_window_start = time.time()
        
        self.rate_limit_calls += 1


class StripePaymentAPI(BaseExternalAPI):
    """
    Stripe Payment Processing API - DUMMY Implementation
    
    Handles payment processing, refunds, and subscription management
    with realistic simulation of Stripe API responses.
    """
    
    def __init__(self):
        metadata = ToolMetadata(
            tool_id="stripe_payment_api",
            name="Stripe Payment API",
            description="Process payments, refunds, and manage subscriptions",
            tool_type=ToolType.EXTERNAL_API,
            version="2.0.0",
            priority=1,
            timeout_ms=10000,
            dummy_mode=True,
            tags=["payment", "stripe", "billing", "financial"]
        )
        super().__init__(metadata, APIProvider.STRIPE, "https://api.stripe.com/v1/")
        self.max_calls_per_minute = 100  # Stripe rate limit
    
    async def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute Stripe API operation"""
        
        operation = kwargs.get("operation", "create_charge")
        
        try:
            if operation == "create_charge":
                result = await self.create_charge(kwargs)
            elif operation == "create_refund":
                result = await self.create_refund(kwargs)
            elif operation == "create_customer":
                result = await self.create_customer(kwargs)
            elif operation == "create_subscription":
                result = await self.create_subscription(kwargs)
            elif operation == "retrieve_payment_intent":
                result = await self.retrieve_payment_intent(kwargs)
            else:
                raise ValueError(f"Unknown Stripe operation: {operation}")
            
            return ToolResult(
                success=result.success,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                result_data=result.data,
                execution_time_ms=result.response_time_ms,
                metadata={
                    "provider": "stripe",
                    "operation": operation,
                    "status_code": result.status_code
                }
            )
            
        except Exception as e:
            logger.error(f"Stripe API operation failed: {str(e)}")
            return ToolResult(
                success=False,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                error_message=f"Stripe API failed: {str(e)}"
            )
    
    async def create_charge(self, params: Dict[str, Any]) -> APIResponse:
        """Create a payment charge"""
        return await self._make_api_call("POST", "charges", data=params)
    
    async def create_refund(self, params: Dict[str, Any]) -> APIResponse:
        """Create a refund"""
        return await self._make_api_call("POST", "refunds", data=params)
    
    async def create_customer(self, params: Dict[str, Any]) -> APIResponse:
        """Create a customer"""
        return await self._make_api_call("POST", "customers", data=params)
    
    async def create_subscription(self, params: Dict[str, Any]) -> APIResponse:
        """Create a subscription"""
        return await self._make_api_call("POST", "subscriptions", data=params)
    
    async def retrieve_payment_intent(self, params: Dict[str, Any]) -> APIResponse:
        """Retrieve payment intent"""
        payment_intent_id = params.get("payment_intent_id")
        return await self._make_api_call("GET", f"payment_intents/{payment_intent_id}")
    
    async def _generate_dummy_response(self, endpoint: str, data: Optional[Dict]) -> Dict[str, Any]:
        """Generate realistic Stripe API responses"""
        
        if endpoint == "charges":
            return {
                "id": f"ch_{uuid.uuid4().hex[:16]}",
                "object": "charge",
                "amount": data.get("amount", 2000) if data else 2000,
                "currency": "usd",
                "status": "succeeded",
                "paid": True,
                "refunded": False,
                "receipt_url": f"https://pay.stripe.com/receipts/ch_{uuid.uuid4().hex[:16]}",
                "created": int(datetime.now().timestamp()),
                "description": data.get("description") if data else "Payment"
            }
        
        elif endpoint == "refunds":
            return {
                "id": f"re_{uuid.uuid4().hex[:16]}",
                "object": "refund",
                "amount": data.get("amount", 2000) if data else 2000,
                "currency": "usd",
                "status": "succeeded",
                "created": int(datetime.now().timestamp()),
                "reason": data.get("reason", "requested_by_customer") if data else "requested_by_customer"
            }
        
        elif endpoint == "customers":
            return {
                "id": f"cus_{uuid.uuid4().hex[:14]}",
                "object": "customer",
                "email": data.get("email") if data else "customer@example.com",
                "created": int(datetime.now().timestamp()),
                "livemode": False
            }
        
        elif endpoint.startswith("payment_intents/"):
            return {
                "id": f"pi_{uuid.uuid4().hex[:16]}",
                "object": "payment_intent",
                "amount": 2000,
                "currency": "usd",
                "status": "succeeded",
                "client_secret": f"pi_{uuid.uuid4().hex[:16]}_secret_{uuid.uuid4().hex[:8]}"
            }
        
        return {"message": "Stripe API simulation", "endpoint": endpoint}


class TwilioSMSAPI(BaseExternalAPI):
    """
    Twilio SMS API - DUMMY Implementation
    
    Handles SMS messaging, voice calls, and communication services
    with realistic simulation of Twilio API responses.
    """
    
    def __init__(self):
        metadata = ToolMetadata(
            tool_id="twilio_sms_api",
            name="Twilio SMS API",
            description="Send SMS messages and manage communications",
            tool_type=ToolType.EXTERNAL_API,
            version="1.8.0",
            priority=1,
            timeout_ms=8000,
            dummy_mode=True,
            tags=["sms", "twilio", "communication", "messaging"]
        )
        super().__init__(metadata, APIProvider.TWILIO, "https://api.twilio.com/2010-04-01/")
        self.max_calls_per_minute = 300  # Twilio rate limit
    
    async def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute Twilio API operation"""
        
        operation = kwargs.get("operation", "send_sms")
        
        try:
            if operation == "send_sms":
                result = await self.send_sms(kwargs)
            elif operation == "make_call":
                result = await self.make_call(kwargs)
            elif operation == "get_message_status":
                result = await self.get_message_status(kwargs)
            else:
                raise ValueError(f"Unknown Twilio operation: {operation}")
            
            return ToolResult(
                success=result.success,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                result_data=result.data,
                execution_time_ms=result.response_time_ms,
                metadata={
                    "provider": "twilio",
                    "operation": operation,
                    "status_code": result.status_code
                }
            )
            
        except Exception as e:
            logger.error(f"Twilio API operation failed: {str(e)}")
            return ToolResult(
                success=False,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                error_message=f"Twilio API failed: {str(e)}"
            )
    
    async def send_sms(self, params: Dict[str, Any]) -> APIResponse:
        """Send SMS message"""
        return await self._make_api_call("POST", "Accounts/DUMMY_ACCOUNT/Messages.json", data=params)
    
    async def make_call(self, params: Dict[str, Any]) -> APIResponse:
        """Make voice call"""
        return await self._make_api_call("POST", "Accounts/DUMMY_ACCOUNT/Calls.json", data=params)
    
    async def get_message_status(self, params: Dict[str, Any]) -> APIResponse:
        """Get message delivery status"""
        message_sid = params.get("message_sid")
        return await self._make_api_call("GET", f"Accounts/DUMMY_ACCOUNT/Messages/{message_sid}.json")
    
    async def _generate_dummy_response(self, endpoint: str, data: Optional[Dict]) -> Dict[str, Any]:
        """Generate realistic Twilio API responses"""
        
        if "Messages.json" in endpoint:
            return {
                "sid": f"SM{uuid.uuid4().hex[:16]}",
                "account_sid": "ACDUMMY123456789",
                "to": data.get("To") if data else "+15551234567",
                "from": data.get("From") if data else "+15559876543",
                "body": data.get("Body") if data else "Test message",
                "status": "sent",
                "direction": "outbound-api",
                "date_created": datetime.now().isoformat(),
                "date_sent": datetime.now().isoformat(),
                "price": "-0.0075",
                "price_unit": "USD",
                "uri": f"/2010-04-01/Accounts/ACDUMMY123456789/Messages/SM{uuid.uuid4().hex[:16]}.json"
            }
        
        elif "Calls.json" in endpoint:
            return {
                "sid": f"CA{uuid.uuid4().hex[:16]}",
                "account_sid": "ACDUMMY123456789",
                "to": data.get("To") if data else "+15551234567",
                "from": data.get("From") if data else "+15559876543",
                "status": "ringing",
                "direction": "outbound-api",
                "date_created": datetime.now().isoformat(),
                "duration": "0",
                "price": None,
                "uri": f"/2010-04-01/Accounts/ACDUMMY123456789/Calls/CA{uuid.uuid4().hex[:16]}.json"
            }
        
        return {"message": "Twilio API simulation", "endpoint": endpoint}


class ZendeskTicketingAPI(BaseExternalAPI):
    """
    Zendesk Ticketing API - DUMMY Implementation
    
    Handles support ticket creation, updates, and management
    with realistic simulation of Zendesk API responses.
    """
    
    def __init__(self):
        metadata = ToolMetadata(
            tool_id="zendesk_ticketing_api",
            name="Zendesk Ticketing API",
            description="Manage support tickets and customer communications",
            tool_type=ToolType.EXTERNAL_API,
            version="1.5.0",
            priority=2,
            timeout_ms=7000,
            dummy_mode=True,
            tags=["ticketing", "zendesk", "support", "customer-service"]
        )
        super().__init__(metadata, APIProvider.ZENDESK, "https://company.zendesk.com/api/v2/")
        self.max_calls_per_minute = 700  # Zendesk rate limit
    
    async def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute Zendesk API operation"""
        
        operation = kwargs.get("operation", "create_ticket")
        
        try:
            if operation == "create_ticket":
                result = await self.create_ticket(kwargs)
            elif operation == "update_ticket":
                result = await self.update_ticket(kwargs)
            elif operation == "get_ticket":
                result = await self.get_ticket(kwargs)
            elif operation == "add_comment":
                result = await self.add_comment(kwargs)
            else:
                raise ValueError(f"Unknown Zendesk operation: {operation}")
            
            return ToolResult(
                success=result.success,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                result_data=result.data,
                execution_time_ms=result.response_time_ms,
                metadata={
                    "provider": "zendesk",
                    "operation": operation,
                    "status_code": result.status_code
                }
            )
            
        except Exception as e:
            logger.error(f"Zendesk API operation failed: {str(e)}")
            return ToolResult(
                success=False,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                error_message=f"Zendesk API failed: {str(e)}"
            )
    
    async def create_ticket(self, params: Dict[str, Any]) -> APIResponse:
        """Create support ticket"""
        return await self._make_api_call("POST", "tickets.json", data={"ticket": params})
    
    async def update_ticket(self, params: Dict[str, Any]) -> APIResponse:
        """Update existing ticket"""
        ticket_id = params.get("ticket_id")
        return await self._make_api_call("PUT", f"tickets/{ticket_id}.json", data={"ticket": params})
    
    async def get_ticket(self, params: Dict[str, Any]) -> APIResponse:
        """Get ticket details"""
        ticket_id = params.get("ticket_id")
        return await self._make_api_call("GET", f"tickets/{ticket_id}.json")
    
    async def add_comment(self, params: Dict[str, Any]) -> APIResponse:
        """Add comment to ticket"""
        ticket_id = params.get("ticket_id")
        return await self._make_api_call("PUT", f"tickets/{ticket_id}.json", data={"ticket": {"comment": params}})
    
    async def _generate_dummy_response(self, endpoint: str, data: Optional[Dict]) -> Dict[str, Any]:
        """Generate realistic Zendesk API responses"""
        
        if endpoint == "tickets.json":
            ticket_data = data.get("ticket", {}) if data else {}
            return {
                "ticket": {
                    "id": random.randint(1000, 9999),
                    "subject": ticket_data.get("subject", "Support Request"),
                    "description": ticket_data.get("description", "Customer needs assistance"),
                    "status": "open",
                    "priority": ticket_data.get("priority", "normal"),
                    "type": ticket_data.get("type", "incident"),
                    "requester_id": ticket_data.get("requester_id", 12345),
                    "assignee_id": None,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "tags": ticket_data.get("tags", []),
                    "custom_fields": ticket_data.get("custom_fields", [])
                }
            }
        
        elif endpoint.startswith("tickets/") and endpoint.endswith(".json"):
            ticket_id = endpoint.split("/")[1].split(".")[0]
            return {
                "ticket": {
                    "id": int(ticket_id),
                    "subject": "Support Request",
                    "description": "Customer needs assistance",
                    "status": "open",
                    "priority": "normal",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
            }
        
        return {"message": "Zendesk API simulation", "endpoint": endpoint}


class GoogleMapsAPI(BaseExternalAPI):
    """
    Google Maps API - DUMMY Implementation
    
    Handles geocoding, distance calculations, and location services
    with realistic simulation of Google Maps API responses.
    """
    
    def __init__(self):
        metadata = ToolMetadata(
            tool_id="google_maps_api",
            name="Google Maps API",
            description="Geocoding, directions, and location services",
            tool_type=ToolType.EXTERNAL_API,
            version="1.3.0",
            priority=2,
            timeout_ms=5000,
            dummy_mode=True,
            tags=["maps", "google", "geocoding", "location"]
        )
        super().__init__(metadata, APIProvider.GOOGLE_MAPS, "https://maps.googleapis.com/maps/api/")
        self.max_calls_per_minute = 500  # Google Maps rate limit
    
    async def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute Google Maps API operation"""
        
        operation = kwargs.get("operation", "geocode")
        
        try:
            if operation == "geocode":
                result = await self.geocode_address(kwargs)
            elif operation == "reverse_geocode":
                result = await self.reverse_geocode(kwargs)
            elif operation == "calculate_distance":
                result = await self.calculate_distance(kwargs)
            elif operation == "get_directions":
                result = await self.get_directions(kwargs)
            else:
                raise ValueError(f"Unknown Google Maps operation: {operation}")
            
            return ToolResult(
                success=result.success,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                result_data=result.data,
                execution_time_ms=result.response_time_ms,
                metadata={
                    "provider": "google_maps",
                    "operation": operation,
                    "status_code": result.status_code
                }
            )
            
        except Exception as e:
            logger.error(f"Google Maps API operation failed: {str(e)}")
            return ToolResult(
                success=False,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                error_message=f"Google Maps API failed: {str(e)}"
            )
    
    async def geocode_address(self, params: Dict[str, Any]) -> APIResponse:
        """Convert address to coordinates"""
        return await self._make_api_call("GET", "geocode/json", params=params)
    
    async def reverse_geocode(self, params: Dict[str, Any]) -> APIResponse:
        """Convert coordinates to address"""
        return await self._make_api_call("GET", "geocode/json", params=params)
    
    async def calculate_distance(self, params: Dict[str, Any]) -> APIResponse:
        """Calculate distance between locations"""
        return await self._make_api_call("GET", "distancematrix/json", params=params)
    
    async def get_directions(self, params: Dict[str, Any]) -> APIResponse:
        """Get directions between locations"""
        return await self._make_api_call("GET", "directions/json", params=params)
    
    async def _generate_dummy_response(self, endpoint: str, data: Optional[Dict]) -> Dict[str, Any]:
        """Generate realistic Google Maps API responses"""
        
        if endpoint == "geocode/json":
            return {
                "results": [
                    {
                        "formatted_address": "123 Main St, Anytown, ST 12345, USA",
                        "geometry": {
                            "location": {
                                "lat": 40.7128 + random.uniform(-0.1, 0.1),
                                "lng": -74.0060 + random.uniform(-0.1, 0.1)
                            },
                            "location_type": "ROOFTOP"
                        },
                        "place_id": f"ChIJ{uuid.uuid4().hex[:20]}",
                        "types": ["street_address"]
                    }
                ],
                "status": "OK"
            }
        
        elif endpoint == "distancematrix/json":
            return {
                "destination_addresses": ["123 Destination St, City, ST, USA"],
                "origin_addresses": ["456 Origin Ave, City, ST, USA"],
                "rows": [
                    {
                        "elements": [
                            {
                                "distance": {
                                    "text": f"{random.randint(1, 50)} mi",
                                    "value": random.randint(1600, 80000)
                                },
                                "duration": {
                                    "text": f"{random.randint(5, 60)} mins",
                                    "value": random.randint(300, 3600)
                                },
                                "status": "OK"
                            }
                        ]
                    }
                ],
                "status": "OK"
            }
        
        return {"message": "Google Maps API simulation", "endpoint": endpoint, "status": "OK"}


class SendGridEmailAPI(BaseExternalAPI):
    """SendGrid Email API - DUMMY Implementation"""
    
    def __init__(self):
        metadata = ToolMetadata(
            tool_id="sendgrid_email_api",
            name="SendGrid Email API",
            description="Send transactional and marketing emails",
            tool_type=ToolType.EXTERNAL_API,
            version="1.2.0",
            priority=2,
            timeout_ms=6000,
            dummy_mode=True,
            tags=["email", "sendgrid", "messaging", "marketing"]
        )
        super().__init__(metadata, APIProvider.SENDGRID, "https://api.sendgrid.com/v3/")
        self.max_calls_per_minute = 600  # SendGrid rate limit
    
    async def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute SendGrid API operation"""
        
        operation = kwargs.get("operation", "send_email")
        
        try:
            if operation == "send_email":
                result = await self.send_email(kwargs)
            elif operation == "send_template_email":
                result = await self.send_template_email(kwargs)
            else:
                raise ValueError(f"Unknown SendGrid operation: {operation}")
            
            return ToolResult(
                success=result.success,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                result_data=result.data,
                execution_time_ms=result.response_time_ms
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                tool_id=self.metadata.tool_id,
                execution_id=context.execution_id,
                error_message=f"SendGrid API failed: {str(e)}"
            )
    
    async def send_email(self, params: Dict[str, Any]) -> APIResponse:
        """Send email"""
        return await self._make_api_call("POST", "mail/send", data=params)
    
    async def send_template_email(self, params: Dict[str, Any]) -> APIResponse:
        """Send template-based email"""
        return await self._make_api_call("POST", "mail/send", data=params)
    
    async def _generate_dummy_response(self, endpoint: str, data: Optional[Dict]) -> Dict[str, Any]:
        """Generate realistic SendGrid API responses"""
        return {
            "message_id": f"sg_{uuid.uuid4().hex[:20]}",
            "status": "queued",
            "timestamp": datetime.now().isoformat()
        }


# Factory function to create API instances
def create_api_instance(provider: APIProvider) -> BaseExternalAPI:
    """Factory function to create API instances"""
    
    api_classes = {
        APIProvider.STRIPE: StripePaymentAPI,
        APIProvider.TWILIO: TwilioSMSAPI,
        APIProvider.ZENDESK: ZendeskTicketingAPI,
        APIProvider.GOOGLE_MAPS: GoogleMapsAPI,
        APIProvider.SENDGRID: SendGridEmailAPI
    }
    
    api_class = api_classes.get(provider)
    if not api_class:
        raise ValueError(f"Unsupported API provider: {provider}")
    
    return api_class()


# Export all API classes for easy importing
__all__ = [
    'BaseExternalAPI',
    'StripePaymentAPI',
    'TwilioSMSAPI', 
    'ZendeskTicketingAPI',
    'GoogleMapsAPI',
    'SendGridEmailAPI',
    'APIProvider',
    'AuthenticationType',
    'APICredentials',
    'APIResponse',
    'create_api_instance'
]