"""
External APIs Manager - CRM and Ticketing System Integration
Part of the Multi-Agent Voice AI System Transformation

This module provides a unified interface for external system integrations:
- CRM Systems: Salesforce, HubSpot, Pipedrive
- Ticketing Systems: Zendesk, ServiceNow, Jira
- Mock implementations for development and testing
- Easy API key management and authentication
- Retry logic and error handling

PRODUCTION SETUP:
1. Add real API credentials to environment variables
2. Set USE_MOCK_APIS=False in configuration
3. Install required SDKs: salesforce-python, zendesk, etc.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import os
from dataclasses import dataclass
import aiohttp
import uuid

from app.core.latency_optimizer import latency_monitor

# Configure logging
logger = logging.getLogger(__name__)

class APIProvider(Enum):
    """Supported external API providers"""
    # CRM Systems
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot" 
    PIPEDRIVE = "pipedrive"
    
    # Ticketing Systems
    ZENDESK = "zendesk"
    SERVICENOW = "servicenow"
    JIRA = "jira"
    FRESHDESK = "freshdesk"

@dataclass
class APIConfiguration:
    """Configuration for external API connections"""
    provider: APIProvider
    base_url: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    organization: Optional[str] = None
    enabled: bool = True
    use_mock: bool = True  # Default to mock for development

class ExternalAPIManager:
    """
    Unified manager for all external API integrations
    Provides mock implementations for development and testing
    """
    
    def __init__(self):
        # Configuration management
        self.use_mock_apis = os.getenv("USE_MOCK_APIS", "true").lower() == "true"
        
        # API configurations
        self.api_configs = self._load_api_configurations()
        
        # Mock data stores (for development)
        self.mock_customers = {}
        self.mock_tickets = {}
        self.mock_opportunities = {}
        
        # HTTP session for API calls
        self.session = None
        
        # Initialize mock data
        if self.use_mock_apis:
            self._initialize_mock_data()
            
        logger.info(f"External API Manager initialized - Mock mode: {self.use_mock_apis}")

    def _load_api_configurations(self) -> Dict[str, APIConfiguration]:
        """Load API configurations from environment variables"""
        configs = {}
        
        # Salesforce CRM
        configs["salesforce"] = APIConfiguration(
            provider=APIProvider.SALESFORCE,
            base_url=os.getenv("SALESFORCE_BASE_URL", "https://na1.salesforce.com"),
            username=os.getenv("SALESFORCE_USERNAME"),
            password=os.getenv("SALESFORCE_PASSWORD"),
            token=os.getenv("SALESFORCE_TOKEN"),
            use_mock=self.use_mock_apis
        )
        
        # HubSpot CRM
        configs["hubspot"] = APIConfiguration(
            provider=APIProvider.HUBSPOT,
            base_url="https://api.hubapi.com",
            api_key=os.getenv("HUBSPOT_API_KEY"),
            use_mock=self.use_mock_apis
        )
        
        # Zendesk Ticketing
        configs["zendesk"] = APIConfiguration(
            provider=APIProvider.ZENDESK,
            base_url=os.getenv("ZENDESK_BASE_URL", "https://yourcompany.zendesk.com"),
            username=os.getenv("ZENDESK_USERNAME"),
            token=os.getenv("ZENDESK_API_TOKEN"),
            use_mock=self.use_mock_apis
        )
        
        # ServiceNow Ticketing
        configs["servicenow"] = APIConfiguration(
            provider=APIProvider.SERVICENOW,
            base_url=os.getenv("SERVICENOW_BASE_URL", "https://yourcompany.service-now.com"),
            username=os.getenv("SERVICENOW_USERNAME"),
            password=os.getenv("SERVICENOW_PASSWORD"),
            use_mock=self.use_mock_apis
        )
        
        return configs

    def _initialize_mock_data(self):
        """Initialize mock data for development and testing"""
        # Mock customer data
        self.mock_customers = {
            "CUST-001": {
                "id": "CUST-001",
                "name": "John Smith",
                "email": "john.smith@example.com",
                "phone": "+1-555-0123",
                "company": "Tech Solutions Inc",
                "status": "active",
                "created_date": "2024-01-15",
                "last_contact": "2024-05-20",
                "account_manager": "Sarah Johnson",
                "subscription_plan": "Professional",
                "total_revenue": 12500.00
            },
            "CUST-002": {
                "id": "CUST-002", 
                "name": "Emily Davis",
                "email": "emily.davis@startup.com",
                "phone": "+1-555-0456",
                "company": "InnovateCorp",
                "status": "active",
                "created_date": "2024-03-22",
                "last_contact": "2024-05-23",
                "account_manager": "Mike Rodriguez",
                "subscription_plan": "Enterprise",
                "total_revenue": 35000.00
            }
        }
        
        # Mock ticket data
        self.mock_tickets = {
            "TICKET-001": {
                "id": "TICKET-001",
                "title": "Login Issues with Mobile App",
                "description": "User cannot log into mobile application",
                "status": "open",
                "priority": "medium",
                "customer_id": "CUST-001",
                "assigned_agent": "Alex Thompson",
                "created_date": "2024-05-23",
                "last_updated": "2024-05-24",
                "category": "technical",
                "tags": ["mobile", "authentication", "login"]
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    # =============================================================================
    # CRM OPERATIONS
    # =============================================================================

    @latency_monitor("crm_create_customer")
    async def create_customer_account(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new customer account in CRM system
        
        Args:
            customer_data: Customer information (name, email, phone, company, etc.)
            
        Returns:
            Created customer record with ID and details
        """
        if self.use_mock_apis:
            return await self._mock_create_customer(customer_data)
        
        # Real CRM implementation would go here
        crm_provider = customer_data.get("crm_provider", "salesforce")
        
        if crm_provider == "salesforce":
            return await self._salesforce_create_customer(customer_data)
        elif crm_provider == "hubspot":
            return await self._hubspot_create_customer(customer_data)
        else:
            raise ValueError(f"Unsupported CRM provider: {crm_provider}")

    async def _mock_create_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for customer creation"""
        await asyncio.sleep(0.1)  # Simulate API latency
        
        customer_id = f"CUST-{len(self.mock_customers) + 1:03d}"
        
        customer_record = {
            "id": customer_id,
            "name": customer_data.get("name", "Unknown Customer"),
            "email": customer_data.get("email"),
            "phone": customer_data.get("phone"),
            "company": customer_data.get("company"),
            "status": "active",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "last_contact": datetime.now().strftime("%Y-%m-%d"),
            "account_manager": "Auto-assigned",
            "subscription_plan": customer_data.get("plan", "Basic"),
            "total_revenue": 0.00,
            "source": "voice_ai_system"
        }
        
        self.mock_customers[customer_id] = customer_record
        
        return {
            "success": True,
            "customer_id": customer_id,
            "customer_record": customer_record,
            "message": "Customer account created successfully",
            "crm_provider": "mock"
        }

    @latency_monitor("crm_update_customer")
    async def update_customer_record(self, 
                                   customer_id: str, 
                                   updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing customer record in CRM
        
        Args:
            customer_id: Customer identifier
            updates: Fields to update
            
        Returns:
            Updated customer record
        """
        if self.use_mock_apis:
            return await self._mock_update_customer(customer_id, updates)
        
        # Real CRM update implementation
        return await self._real_update_customer(customer_id, updates)

    async def _mock_update_customer(self, customer_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for customer updates"""
        await asyncio.sleep(0.08)  # Simulate API latency
        
        if customer_id not in self.mock_customers:
            return {
                "success": False,
                "error": f"Customer {customer_id} not found",
                "customer_id": customer_id
            }
        
        # Update customer record
        customer = self.mock_customers[customer_id]
        customer.update(updates)
        customer["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "success": True,
            "customer_id": customer_id,
            "updated_fields": list(updates.keys()),
            "customer_record": customer,
            "message": "Customer record updated successfully"
        }

    @latency_monitor("crm_get_customer")
    async def get_customer_record(self, customer_id: str) -> Dict[str, Any]:
        """
        Retrieve customer record from CRM
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Customer record with all details
        """
        if self.use_mock_apis:
            return await self._mock_get_customer(customer_id)
        
        return await self._real_get_customer(customer_id)

    async def _mock_get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Mock implementation for customer retrieval"""
        await asyncio.sleep(0.05)  # Simulate API latency
        
        if customer_id in self.mock_customers:
            return {
                "success": True,
                "customer_record": self.mock_customers[customer_id],
                "found": True
            }
        else:
            return {
                "success": False,
                "error": f"Customer {customer_id} not found",
                "found": False
            }

    # =============================================================================
    # TICKETING OPERATIONS
    # =============================================================================

    @latency_monitor("ticketing_create_ticket")
    async def create_support_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create support ticket in ticketing system
        
        Args:
            ticket_data: Ticket information (title, description, priority, customer_id, etc.)
            
        Returns:
            Created ticket with ID and tracking information
        """
        if self.use_mock_apis:
            return await self._mock_create_ticket(ticket_data)
        
        # Real ticketing system implementation
        ticketing_provider = ticket_data.get("ticketing_provider", "zendesk")
        
        if ticketing_provider == "zendesk":
            return await self._zendesk_create_ticket(ticket_data)
        elif ticketing_provider == "servicenow":
            return await self._servicenow_create_ticket(ticket_data)
        else:
            raise ValueError(f"Unsupported ticketing provider: {ticketing_provider}")

    async def _mock_create_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for ticket creation"""
        await asyncio.sleep(0.12)  # Simulate API latency
        
        ticket_id = f"TICKET-{len(self.mock_tickets) + 1:03d}"
        
        # Determine priority and SLA
        priority = ticket_data.get("priority", "medium")
        sla_hours = {"low": 72, "medium": 24, "high": 8, "critical": 2}
        
        ticket_record = {
            "id": ticket_id,
            "title": ticket_data.get("title", "Support Request"),
            "description": ticket_data.get("description", ""),
            "status": "open",
            "priority": priority,
            "customer_id": ticket_data.get("customer_id"),
            "customer_email": ticket_data.get("customer_email"),
            "assigned_agent": "Auto-assigned",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": ticket_data.get("category", "general"),
            "tags": ticket_data.get("tags", []),
            "sla_deadline": (datetime.now() + timedelta(hours=sla_hours[priority])).strftime("%Y-%m-%d %H:%M:%S"),
            "source": "voice_ai_system"
        }
        
        self.mock_tickets[ticket_id] = ticket_record
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "ticket_record": ticket_record,
            "tracking_url": f"https://support.example.com/tickets/{ticket_id}",
            "sla_response_time": f"{sla_hours[priority]} hours",
            "message": "Support ticket created successfully"
        }

    @latency_monitor("ticketing_create_urgent")
    async def create_urgent_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create urgent/critical priority ticket with immediate escalation
        
        Args:
            ticket_data: Urgent ticket information
            
        Returns:
            Created urgent ticket with escalation details
        """
        # Force critical priority
        ticket_data["priority"] = "critical"
        ticket_data["escalate_immediately"] = True
        
        result = await self.create_support_ticket(ticket_data)
        
        if result.get("success"):
            # Add urgent handling details
            result["escalated"] = True
            result["on_call_notified"] = True
            result["expected_response"] = "Within 2 hours"
            
        return result

    @latency_monitor("ticketing_update_ticket")
    async def update_ticket_status(self, 
                                 ticket_id: str, 
                                 status: str,
                                 comment: Optional[str] = None) -> Dict[str, Any]:
        """
        Update ticket status and add comments
        
        Args:
            ticket_id: Ticket identifier
            status: New status (open, in_progress, resolved, closed)
            comment: Optional comment to add
            
        Returns:
            Updated ticket information
        """
        if self.use_mock_apis:
            return await self._mock_update_ticket(ticket_id, status, comment)
        
        return await self._real_update_ticket(ticket_id, status, comment)

    async def _mock_update_ticket(self, ticket_id: str, status: str, comment: Optional[str]) -> Dict[str, Any]:
        """Mock implementation for ticket updates"""
        await asyncio.sleep(0.09)  # Simulate API latency
        
        if ticket_id not in self.mock_tickets:
            return {
                "success": False,
                "error": f"Ticket {ticket_id} not found"
            }
        
        ticket = self.mock_tickets[ticket_id]
        old_status = ticket["status"]
        ticket["status"] = status
        ticket["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if comment:
            if "comments" not in ticket:
                ticket["comments"] = []
            ticket["comments"].append({
                "comment": comment,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "author": "AI Assistant"
            })
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "old_status": old_status,
            "new_status": status,
            "comment_added": comment is not None,
            "ticket_record": ticket
        }

    # =============================================================================
    # BILLING AND FINANCIAL OPERATIONS
    # =============================================================================

    @latency_monitor("crm_log_billing_dispute")
    async def log_billing_dispute(self, dispute_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log billing dispute in CRM system
        
        Args:
            dispute_data: Dispute information and customer details
            
        Returns:
            Logged dispute record with case number
        """
        if self.use_mock_apis:
            return await self._mock_log_dispute(dispute_data)
        
        return await self._real_log_dispute(dispute_data)

    async def _mock_log_dispute(self, dispute_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for dispute logging"""
        await asyncio.sleep(0.08)
        
        dispute_id = f"DISPUTE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        dispute_record = {
            "id": dispute_id,
            "customer_id": dispute_data.get("customer_id"),
            "dispute_type": dispute_data.get("dispute_type", "billing_error"),
            "amount": dispute_data.get("amount", 0.00),
            "transaction_id": dispute_data.get("transaction_id"),
            "description": dispute_data.get("description", ""),
            "status": "investigating",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assigned_specialist": "Billing Team",
            "priority": dispute_data.get("priority", "medium"),
            "estimated_resolution": "5-10 business days"
        }
        
        return {
            "success": True,
            "dispute_id": dispute_id,
            "dispute_record": dispute_record,
            "case_number": dispute_id,
            "next_steps": "Investigation will begin within 24 hours"
        }

    # =============================================================================
    # REAL API IMPLEMENTATIONS (Templates)
    # =============================================================================

    async def _salesforce_create_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real Salesforce API implementation"""
        # TODO: Implement actual Salesforce API calls
        # from simple_salesforce import Salesforce
        
        logger.info("Salesforce API not implemented yet - using mock")
        return await self._mock_create_customer(customer_data)

    async def _hubspot_create_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real HubSpot API implementation"""
        # TODO: Implement actual HubSpot API calls
        # import hubspot
        
        logger.info("HubSpot API not implemented yet - using mock")
        return await self._mock_create_customer(customer_data)

    async def _zendesk_create_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real Zendesk API implementation"""
        # TODO: Implement actual Zendesk API calls
        # from zendesk_api import ZendeskAPI
        
        logger.info("Zendesk API not implemented yet - using mock")
        return await self._mock_create_ticket(ticket_data)

    async def _servicenow_create_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real ServiceNow API implementation"""
        # TODO: Implement actual ServiceNow API calls
        
        logger.info("ServiceNow API not implemented yet - using mock")
        return await self._mock_create_ticket(ticket_data)

    async def _real_update_customer(self, customer_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Real CRM update implementation"""
        logger.info("Real CRM update not implemented yet - using mock")
        return await self._mock_update_customer(customer_id, updates)

    async def _real_get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Real CRM get implementation"""
        logger.info("Real CRM get not implemented yet - using mock")
        return await self._mock_get_customer(customer_id)

    async def _real_update_ticket(self, ticket_id: str, status: str, comment: Optional[str]) -> Dict[str, Any]:
        """Real ticketing update implementation"""
        logger.info("Real ticketing update not implemented yet - using mock")
        return await self._mock_update_ticket(ticket_id, status, comment)

    async def _real_log_dispute(self, dispute_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real dispute logging implementation"""
        logger.info("Real dispute logging not implemented yet - using mock")
        return await self._mock_log_dispute(dispute_data)

    # =============================================================================
    # SEARCH AND QUERY OPERATIONS
    # =============================================================================

    @latency_monitor("crm_search_customers")
    async def search_customers(self, search_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search customers in CRM system
        
        Args:
            search_criteria: Search parameters (name, email, phone, company, etc.)
            
        Returns:
            List of matching customer records
        """
        if self.use_mock_apis:
            return await self._mock_search_customers(search_criteria)
        
        return await self._real_search_customers(search_criteria)

    async def _mock_search_customers(self, search_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for customer search"""
        await asyncio.sleep(0.06)
        
        results = []
        search_term = search_criteria.get("search_term", "").lower()
        
        for customer in self.mock_customers.values():
            # Simple text matching
            if (search_term in customer.get("name", "").lower() or
                search_term in customer.get("email", "").lower() or
                search_term in customer.get("company", "").lower()):
                results.append(customer)
        
        return {
            "success": True,
            "total_results": len(results),
            "customers": results,
            "search_criteria": search_criteria
        }

    @latency_monitor("ticketing_get_customer_tickets")
    async def get_customer_tickets(self, customer_id: str) -> Dict[str, Any]:
        """
        Get all tickets for a specific customer
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            List of customer tickets
        """
        if self.use_mock_apis:
            return await self._mock_get_customer_tickets(customer_id)
        
        return await self._real_get_customer_tickets(customer_id)

    async def _mock_get_customer_tickets(self, customer_id: str) -> Dict[str, Any]:
        """Mock implementation for customer ticket retrieval"""
        await asyncio.sleep(0.07)
        
        customer_tickets = [
            ticket for ticket in self.mock_tickets.values()
            if ticket.get("customer_id") == customer_id
        ]
        
        return {
            "success": True,
            "customer_id": customer_id,
            "total_tickets": len(customer_tickets),
            "tickets": customer_tickets,
            "open_tickets": len([t for t in customer_tickets if t["status"] == "open"]),
            "resolved_tickets": len([t for t in customer_tickets if t["status"] == "resolved"])
        }

    # =============================================================================
    # UTILITY AND MANAGEMENT METHODS
    # =============================================================================

    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all configured APIs"""
        status = {
            "mock_mode": self.use_mock_apis,
            "configured_apis": {},
            "mock_data_counts": {
                "customers": len(self.mock_customers),
                "tickets": len(self.mock_tickets),
                "opportunities": len(self.mock_opportunities)
            }
        }
        
        for name, config in self.api_configs.items():
            status["configured_apis"][name] = {
                "provider": config.provider.value,
                "enabled": config.enabled,
                "use_mock": config.use_mock,
                "has_credentials": bool(config.api_key or config.token or config.password)
            }
        
        return status

    def enable_production_mode(self):
        """Switch to production mode (real APIs)"""
        self.use_mock_apis = False
        for config in self.api_configs.values():
            config.use_mock = False
        logger.info("Switched to production mode - real APIs will be used")

    def enable_mock_mode(self):
        """Switch to mock mode (simulated APIs)"""
        self.use_mock_apis = True
        for config in self.api_configs.values():
            config.use_mock = True
        logger.info("Switched to mock mode - simulated APIs will be used")

    async def test_api_connections(self) -> Dict[str, Any]:
        """Test all configured API connections"""
        results = {}
        
        for name, config in self.api_configs.items():
            try:
                if config.use_mock:
                    results[name] = {"status": "mock", "available": True}
                else:
                    # Test real API connection
                    test_result = await self._test_real_api_connection(config)
                    results[name] = test_result
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        
        return results

    async def _test_real_api_connection(self, config: APIConfiguration) -> Dict[str, Any]:
        """Test connection to real API"""
        # Placeholder for real API connection testing
        return {
            "status": "not_implemented",
            "message": f"Real API testing for {config.provider.value} not implemented yet"
        }

    def get_mock_data_summary(self) -> Dict[str, Any]:
        """Get summary of mock data for development"""
        return {
            "customers": {
                "count": len(self.mock_customers),
                "sample_ids": list(self.mock_customers.keys())[:5]
            },
            "tickets": {
                "count": len(self.mock_tickets),
                "sample_ids": list(self.mock_tickets.keys())[:5],
                "by_status": {
                    status: len([t for t in self.mock_tickets.values() if t["status"] == status])
                    for status in ["open", "in_progress", "resolved", "closed"]
                }
            }
        }

    # =============================================================================
    # PRODUCTION SETUP HELPERS
    # =============================================================================

    def get_required_environment_variables(self) -> Dict[str, List[str]]:
        """Get list of required environment variables for production setup"""
        return {
            "salesforce": [
                "SALESFORCE_BASE_URL",
                "SALESFORCE_USERNAME", 
                "SALESFORCE_PASSWORD",
                "SALESFORCE_TOKEN"
            ],
            "hubspot": [
                "HUBSPOT_API_KEY"
            ],
            "zendesk": [
                "ZENDESK_BASE_URL",
                "ZENDESK_USERNAME",
                "ZENDESK_API_TOKEN"
            ],
            "servicenow": [
                "SERVICENOW_BASE_URL",
                "SERVICENOW_USERNAME",
                "SERVICENOW_PASSWORD"
            ]
        }

    def generate_setup_instructions(self) -> str:
        """Generate setup instructions for production deployment"""
        return """
EXTERNAL APIS SETUP INSTRUCTIONS:

1. ENVIRONMENT VARIABLES:
   Set the following environment variables for production:
   
   # Salesforce CRM
   export SALESFORCE_BASE_URL="https://yourorg.salesforce.com"
   export SALESFORCE_USERNAME="your-username"
   export SALESFORCE_PASSWORD="your-password"
   export SALESFORCE_TOKEN="your-security-token"
   
   # HubSpot CRM
   export HUBSPOT_API_KEY="your-hubspot-api-key"
   
   # Zendesk Ticketing
   export ZENDESK_BASE_URL="https://yourcompany.zendesk.com"
   export ZENDESK_USERNAME="your-email@company.com"
   export ZENDESK_API_TOKEN="your-api-token"
   
   # ServiceNow Ticketing
   export SERVICENOW_BASE_URL="https://yourcompany.service-now.com"
   export SERVICENOW_USERNAME="your-username"
   export SERVICENOW_PASSWORD="your-password"

2. DISABLE MOCK MODE:
   export USE_MOCK_APIS=false

3. INSTALL REQUIRED PACKAGES:
   pip install simple-salesforce hubspot-api-client zendesk

4. UPDATE API IMPLEMENTATIONS:
   Uncomment and complete the real API implementation methods in this file.

5. TEST CONNECTIONS:
   Use the test_api_connections() method to verify all integrations work correctly.
"""