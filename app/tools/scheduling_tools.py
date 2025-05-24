"""
Scheduling Tools - Calendar Integration with Mock Layer
Part of the Multi-Agent Voice AI System Transformation

This module provides comprehensive scheduling capabilities:
- Google Calendar integration
- Outlook/Microsoft Graph integration  
- Calendly-style booking links
- Appointment scheduling and management
- Availability checking and conflict resolution
- Mock implementations for development

PRODUCTION SETUP:
1. Add real API credentials to environment variables
2. Set USE_MOCK_SCHEDULING=False in configuration
3. Install required SDKs: google-auth, microsoft-graph-api
4. Configure OAuth2 flows for calendar access
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import os
import uuid
from dataclasses import dataclass
import pytz

from app.core.latency_optimizer import latency_monitor

# Configure logging
logger = logging.getLogger(__name__)

class CalendarProvider(Enum):
    """Supported calendar providers"""
    GOOGLE_CALENDAR = "google_calendar"
    OUTLOOK = "outlook"
    OFFICE365 = "office365"
    CALENDLY = "calendly"
    ZOOM = "zoom"

class AppointmentStatus(Enum):
    """Appointment status"""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"

class AppointmentType(Enum):
    """Types of appointments"""
    ONBOARDING_CALL = "onboarding_call"
    TECHNICAL_SUPPORT = "technical_support"
    SALES_CONSULTATION = "sales_consultation"
    FOLLOW_UP = "follow_up"
    EMERGENCY_CALLBACK = "emergency_callback"
    BILLING_DISCUSSION = "billing_discussion"

@dataclass
class CalendarConfig:
    """Configuration for calendar providers"""
    provider: CalendarProvider
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    api_key: Optional[str] = None
    calendar_id: Optional[str] = None
    timezone: str = "UTC"
    enabled: bool = True
    use_mock: bool = True

@dataclass
class TimeSlot:
    """Available time slot"""
    start_time: datetime
    end_time: datetime
    available: bool = True
    buffer_minutes: int = 15

@dataclass
class Appointment:
    """Appointment details"""
    appointment_id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    attendees: List[Dict[str, str]]
    appointment_type: AppointmentType
    status: AppointmentStatus
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    calendar_event_id: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class SchedulingManager:
    """
    Comprehensive scheduling manager with multi-provider calendar integration
    Handles appointment booking, availability checking, and calendar management
    """
    
    def __init__(self):
        # Configuration management
        self.use_mock_scheduling = os.getenv("USE_MOCK_SCHEDULING", "true").lower() == "true"
        
        # Calendar provider configurations
        self.calendar_configs = self._load_calendar_configurations()
        
        # Mock data stores (for development)
        self.mock_appointments = {}
        self.mock_availability = {}
        self.mock_calendars = {}
        
        # Business hours and scheduling rules
        self.business_hours = {
            "monday": {"start": "09:00", "end": "17:00"},
            "tuesday": {"start": "09:00", "end": "17:00"},
            "wednesday": {"start": "09:00", "end": "17:00"},
            "thursday": {"start": "09:00", "end": "17:00"},
            "friday": {"start": "09:00", "end": "17:00"},
            "saturday": {"start": "10:00", "end": "14:00"},
            "sunday": {"start": None, "end": None}  # Closed
        }
        
        # Appointment duration defaults (in minutes)
        self.appointment_durations = {
            AppointmentType.ONBOARDING_CALL: 60,
            AppointmentType.TECHNICAL_SUPPORT: 30,
            AppointmentType.SALES_CONSULTATION: 45,
            AppointmentType.FOLLOW_UP: 30,
            AppointmentType.EMERGENCY_CALLBACK: 15,
            AppointmentType.BILLING_DISCUSSION: 30
        }
        
        # Initialize mock data
        if self.use_mock_scheduling:
            self._initialize_mock_data()
            
        logger.info(f"Scheduling Manager initialized - Mock mode: {self.use_mock_scheduling}")

    def _load_calendar_configurations(self) -> Dict[str, CalendarConfig]:
        """Load calendar provider configurations"""
        configs = {}
        
        # Google Calendar
        configs["google"] = CalendarConfig(
            provider=CalendarProvider.GOOGLE_CALENDAR,
            client_id=os.getenv("GOOGLE_CALENDAR_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET"),
            calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary"),
            timezone=os.getenv("CALENDAR_TIMEZONE", "America/New_York"),
            use_mock=self.use_mock_scheduling
        )
        
        # Microsoft Outlook
        configs["outlook"] = CalendarConfig(
            provider=CalendarProvider.OUTLOOK,
            client_id=os.getenv("OUTLOOK_CLIENT_ID"),
            client_secret=os.getenv("OUTLOOK_CLIENT_SECRET"),
            timezone=os.getenv("CALENDAR_TIMEZONE", "America/New_York"),
            use_mock=self.use_mock_scheduling
        )
        
        # Calendly
        configs["calendly"] = CalendarConfig(
            provider=CalendarProvider.CALENDLY,
            api_key=os.getenv("CALENDLY_API_KEY"),
            use_mock=self.use_mock_scheduling
        )
        
        return configs

    def _initialize_mock_data(self):
        """Initialize mock scheduling data"""
        # Mock existing appointments
        now = datetime.now()
        
        self.mock_appointments = {
            "apt_001": Appointment(
                appointment_id="apt_001",
                title="Technical Support Call",
                description="Help with login issues",
                start_time=now + timedelta(days=1, hours=2),
                end_time=now + timedelta(days=1, hours=2, minutes=30),
                attendees=[
                    {"name": "John Smith", "email": "john.smith@example.com"},
                    {"name": "Support Agent", "email": "support@company.com"}
                ],
                appointment_type=AppointmentType.TECHNICAL_SUPPORT,
                status=AppointmentStatus.SCHEDULED,
                meeting_link="https://zoom.us/j/123456789"
            ),
            "apt_002": Appointment(
                appointment_id="apt_002", 
                title="Onboarding Call",
                description="Welcome and product overview",
                start_time=now + timedelta(days=2, hours=3),
                end_time=now + timedelta(days=2, hours=4),
                attendees=[
                    {"name": "Emily Davis", "email": "emily.davis@startup.com"},
                    {"name": "Account Manager", "email": "sales@company.com"}
                ],
                appointment_type=AppointmentType.ONBOARDING_CALL,
                status=AppointmentStatus.CONFIRMED,
                meeting_link="https://zoom.us/j/987654321"
            )
        }

    # =============================================================================
    # APPOINTMENT SCHEDULING
    # =============================================================================

    @latency_monitor("scheduling_create_appointment")
    async def schedule_appointment(self, 
                                 appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule a new appointment
        
        Args:
            appointment_data: Appointment details (type, attendees, preferred_time, etc.)
            
        Returns:
            Scheduled appointment details with calendar integration
        """
        if self.use_mock_scheduling:
            return await self._mock_schedule_appointment(appointment_data)
        
        # Real calendar integration
        provider = appointment_data.get("calendar_provider", "google")
        
        if provider == "google":
            return await self._google_schedule_appointment(appointment_data)
        elif provider == "outlook":
            return await self._outlook_schedule_appointment(appointment_data)
        else:
            raise ValueError(f"Unsupported calendar provider: {provider}")

    async def _mock_schedule_appointment(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for appointment scheduling"""
        await asyncio.sleep(0.12)  # Simulate scheduling time
        
        appointment_id = f"apt_{uuid.uuid4().hex[:8]}"
        
        # Parse appointment type
        try:
            apt_type = AppointmentType(appointment_data.get("appointment_type", "technical_support"))
        except ValueError:
            apt_type = AppointmentType.TECHNICAL_SUPPORT
        
        # Determine duration
        duration_minutes = appointment_data.get("duration", self.appointment_durations[apt_type])
        
        # Parse preferred time or find next available slot
        preferred_time = appointment_data.get("preferred_time")
        if preferred_time:
            if isinstance(preferred_time, str):
                start_time = datetime.fromisoformat(preferred_time.replace('Z', '+00:00'))
            else:
                start_time = preferred_time
        else:
            # Find next available slot
            availability = await self.check_availability(
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=14),
                duration_minutes=duration_minutes
            )
            
            if availability["success"] and availability["available_slots"]:
                slot = availability["available_slots"][0]
                start_time = slot["start_time"]
            else:
                return {
                    "success": False,
                    "error": "No available time slots found",
                    "suggested_times": []
                }
        
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Check for conflicts
        conflict_check = await self._check_appointment_conflicts(start_time, end_time)
        if not conflict_check["available"]:
            return {
                "success": False,
                "error": "Time slot conflicts with existing appointment",
                "conflicting_appointment": conflict_check["conflict"]
            }
        
        # Create appointment
        appointment = Appointment(
            appointment_id=appointment_id,
            title=appointment_data.get("title", f"{apt_type.value.replace('_', ' ').title()} Appointment"),
            description=appointment_data.get("description", "Scheduled via AI Assistant"),
            start_time=start_time,
            end_time=end_time,
            attendees=appointment_data.get("attendees", []),
            appointment_type=apt_type,
            status=AppointmentStatus.SCHEDULED,
            location=appointment_data.get("location"),
            meeting_link=self._generate_meeting_link(appointment_id)
        )
        
        self.mock_appointments[appointment_id] = appointment
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "title": appointment.title,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_minutes,
            "meeting_link": appointment.meeting_link,
            "status": appointment.status.value,
            "calendar_event_created": True,
            "confirmation_sent": True
        }

    @latency_monitor("scheduling_check_availability")
    async def check_availability(self, 
                               start_date: datetime,
                               end_date: datetime,
                               duration_minutes: int = 30) -> Dict[str, Any]:
        """
        Check availability for appointments within date range
        
        Args:
            start_date: Start of availability check period
            end_date: End of availability check period
            duration_minutes: Required appointment duration
            
        Returns:
            Available time slots and booking options
        """
        if self.use_mock_scheduling:
            return await self._mock_check_availability(start_date, end_date, duration_minutes)
        
        return await self._real_check_availability(start_date, end_date, duration_minutes)

    async def _mock_check_availability(self, start_date: datetime, end_date: datetime, duration_minutes: int) -> Dict[str, Any]:
        """Mock implementation for availability checking"""
        await asyncio.sleep(0.08)  # Simulate availability check time
        
        available_slots = []
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            day_name = current_date.strftime("%A").lower()
            
            # Check if day has business hours
            if day_name in self.business_hours and self.business_hours[day_name]["start"]:
                start_hour, start_minute = map(int, self.business_hours[day_name]["start"].split(":"))
                end_hour, end_minute = map(int, self.business_hours[day_name]["end"].split(":"))
                
                # Create time slots for the day
                day_start = datetime.combine(current_date, datetime.min.time().replace(hour=start_hour, minute=start_minute))
                day_end = datetime.combine(current_date, datetime.min.time().replace(hour=end_hour, minute=end_minute))
                
                # Generate 30-minute slots
                current_slot = day_start
                while current_slot + timedelta(minutes=duration_minutes) <= day_end:
                    slot_end = current_slot + timedelta(minutes=duration_minutes)
                    
                    # Check if slot conflicts with existing appointments
                    conflict_check = await self._check_appointment_conflicts(current_slot, slot_end)
                    
                    if conflict_check["available"]:
                        available_slots.append({
                            "start_time": current_slot,
                            "end_time": slot_end,
                            "formatted_time": current_slot.strftime("%Y-%m-%d %I:%M %p"),
                            "day_of_week": current_slot.strftime("%A"),
                            "duration_minutes": duration_minutes
                        })
                    
                    current_slot += timedelta(minutes=30)  # 30-minute intervals
            
            current_date += timedelta(days=1)
        
        return {
            "success": True,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "duration_minutes": duration_minutes,
            "total_slots_found": len(available_slots),
            "available_slots": available_slots[:20],  # Return first 20 slots
            "business_hours": self.business_hours
        }

    async def _check_appointment_conflicts(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Check if proposed time conflicts with existing appointments"""
        for apt_id, appointment in self.mock_appointments.items():
            if appointment.status in [AppointmentStatus.CANCELLED]:
                continue
            
            # Check for time overlap
            if (start_time < appointment.end_time and end_time > appointment.start_time):
                return {
                    "available": False,
                    "conflict": {
                        "appointment_id": apt_id,
                        "title": appointment.title,
                        "start_time": appointment.start_time.isoformat(),
                        "end_time": appointment.end_time.isoformat()
                    }
                }
        
        return {"available": True, "conflict": None}

    # =============================================================================
    # SPECIALIZED SCHEDULING METHODS
    # =============================================================================

    @latency_monitor("scheduling_followup")
    async def schedule_followup(self, 
                              context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule follow-up appointment based on context
        
        Args:
            context_data: Context from previous interaction (ticket, customer info, etc.)
            
        Returns:
            Scheduled follow-up appointment details
        """
        # Determine follow-up timing based on context
        ticket_priority = context_data.get("priority", "medium").lower()
        
        if ticket_priority == "critical":
            followup_hours = 4  # 4 hours for critical issues
        elif ticket_priority == "high":
            followup_hours = 24  # 1 day for high priority
        elif ticket_priority == "medium":
            followup_hours = 72  # 3 days for medium priority
        else:
            followup_hours = 168  # 1 week for low priority
        
        preferred_time = datetime.now() + timedelta(hours=followup_hours)
        
        appointment_data = {
            "appointment_type": "follow_up",
            "title": f"Follow-up: {context_data.get('ticket_subject', 'Support Issue')}",
            "description": f"Follow-up call for ticket #{context_data.get('ticket_id', 'N/A')}",
            "preferred_time": preferred_time,
            "duration": 30,
            "attendees": [
                {
                    "name": context_data.get("customer_name", "Customer"),
                    "email": context_data.get("customer_email", "")
                }
            ]
        }
        
        return await self.schedule_appointment(appointment_data)

    @latency_monitor("scheduling_onboarding")
    async def schedule_onboarding_call(self, 
                                     customer_data: Dict[str, Any],
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Schedule onboarding call for new customers
        
        Args:
            customer_data: New customer information
            context: Additional context from workflow
            
        Returns:
            Scheduled onboarding appointment details
        """
        # Schedule onboarding within 2-3 business days
        preferred_time = datetime.now() + timedelta(days=2)
        
        appointment_data = {
            "appointment_type": "onboarding_call",
            "title": f"Welcome Call - {customer_data.get('name', 'New Customer')}",
            "description": "Product overview and account setup assistance",
            "preferred_time": preferred_time,
            "duration": 60,
            "attendees": [
                {
                    "name": customer_data.get("name", "New Customer"),
                    "email": customer_data.get("email", "")
                },
                {
                    "name": "Account Manager",
                    "email": "sales@company.com"
                }
            ]
        }
        
        return await self.schedule_appointment(appointment_data)

    @latency_monitor("scheduling_emergency")
    async def schedule_emergency_callback(self, 
                                        emergency_data: Dict[str, Any],
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Schedule emergency callback with immediate availability
        
        Args:
            emergency_data: Emergency situation details
            context: Additional context from workflow
            
        Returns:
            Emergency callback appointment details
        """
        # Schedule emergency callback within next 2 hours
        now = datetime.now()
        
        # Find the earliest available emergency slot
        for minutes_ahead in [15, 30, 45, 60, 90, 120]:
            preferred_time = now + timedelta(minutes=minutes_ahead)
            
            # Skip non-business hours for regular callbacks
            if preferred_time.hour < 8 or preferred_time.hour > 18:
                continue
            
            appointment_data = {
                "appointment_type": "emergency_callback",
                "title": f"URGENT: {emergency_data.get('issue_type', 'Emergency Support')}",
                "description": f"Emergency callback for: {emergency_data.get('description', 'Critical issue')}",
                "preferred_time": preferred_time,
                "duration": 15,
                "attendees": [
                    {
                        "name": emergency_data.get("customer_name", "Customer"),
                        "email": emergency_data.get("customer_email", ""),
                        "phone": emergency_data.get("customer_phone", "")
                    }
                ]
            }
            
            result = await self.schedule_appointment(appointment_data)
            if result.get("success"):
                # Mark as high priority
                result["priority"] = "emergency"
                result["escalated"] = True
                return result
        
        # If no slots available, create urgent request
        return {
            "success": False,
            "error": "No emergency slots available",
            "alternative": "Manual emergency escalation required",
            "contact_phone": "+1-555-EMERGENCY"
        }

    # =============================================================================
    # APPOINTMENT MANAGEMENT
    # =============================================================================

    @latency_monitor("scheduling_reschedule")
    async def reschedule_appointment(self, 
                                   appointment_id: str,
                                   new_time: datetime,
                                   reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Reschedule an existing appointment
        
        Args:
            appointment_id: Appointment to reschedule
            new_time: New appointment time
            reason: Reason for rescheduling
            
        Returns:
            Rescheduling result with updated details
        """
        if self.use_mock_scheduling:
            return await self._mock_reschedule_appointment(appointment_id, new_time, reason)
        
        return await self._real_reschedule_appointment(appointment_id, new_time, reason)

    async def _mock_reschedule_appointment(self, appointment_id: str, new_time: datetime, reason: Optional[str]) -> Dict[str, Any]:
        """Mock implementation for appointment rescheduling"""
        await asyncio.sleep(0.1)
        
        if appointment_id not in self.mock_appointments:
            return {
                "success": False,
                "error": f"Appointment {appointment_id} not found"
            }
        
        appointment = self.mock_appointments[appointment_id]
        
        # Calculate new end time
        duration = appointment.end_time - appointment.start_time
        new_end_time = new_time + duration
        
        # Check for conflicts at new time
        conflict_check = await self._check_appointment_conflicts(new_time, new_end_time)
        if not conflict_check["available"]:
            return {
                "success": False,
                "error": "New time slot conflicts with existing appointment",
                "conflicting_appointment": conflict_check["conflict"]
            }
        
        # Update appointment
        old_start_time = appointment.start_time
        appointment.start_time = new_time
        appointment.end_time = new_end_time
        appointment.status = AppointmentStatus.RESCHEDULED
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "old_time": old_start_time.isoformat(),
            "new_time": new_time.isoformat(),
            "new_end_time": new_end_time.isoformat(),
            "reason": reason,
            "status": appointment.status.value,
            "calendar_updated": True,
            "notifications_sent": True
        }

    @latency_monitor("scheduling_cancel")
    async def cancel_appointment(self, 
                               appointment_id: str,
                               reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel an existing appointment
        
        Args:
            appointment_id: Appointment to cancel
            reason: Reason for cancellation
            
        Returns:
            Cancellation result
        """
        if self.use_mock_scheduling:
            return await self._mock_cancel_appointment(appointment_id, reason)
        
        return await self._real_cancel_appointment(appointment_id, reason)

    async def _mock_cancel_appointment(self, appointment_id: str, reason: Optional[str]) -> Dict[str, Any]:
        """Mock implementation for appointment cancellation"""
        await asyncio.sleep(0.08)
        
        if appointment_id not in self.mock_appointments:
            return {
                "success": False,
                "error": f"Appointment {appointment_id} not found"
            }
        
        appointment = self.mock_appointments[appointment_id]
        appointment.status = AppointmentStatus.CANCELLED
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "title": appointment.title,
            "cancelled_time": appointment.start_time.isoformat(),
            "reason": reason,
            "status": appointment.status.value,
            "calendar_updated": True,
            "notifications_sent": True,
            "refund_eligible": (appointment.start_time - datetime.now()).total_seconds() > 86400  # 24 hours notice
        }

    @latency_monitor("scheduling_get_appointment")
    async def get_appointment_details(self, appointment_id: str) -> Dict[str, Any]:
        """
        Get details for a specific appointment
        
        Args:
            appointment_id: Appointment identifier
            
        Returns:
            Appointment details and status
        """
        if self.use_mock_scheduling:
            return await self._mock_get_appointment(appointment_id)
        
        return await self._real_get_appointment(appointment_id)

    async def _mock_get_appointment(self, appointment_id: str) -> Dict[str, Any]:
        """Mock implementation for appointment retrieval"""
        await asyncio.sleep(0.05)
        
        if appointment_id not in self.mock_appointments:
            return {
                "success": False,
                "error": f"Appointment {appointment_id} not found"
            }
        
        appointment = self.mock_appointments[appointment_id]
        
        return {
            "success": True,
            "appointment": {
                "appointment_id": appointment.appointment_id,
                "title": appointment.title,
                "description": appointment.description,
                "start_time": appointment.start_time.isoformat(),
                "end_time": appointment.end_time.isoformat(),
                "duration_minutes": int((appointment.end_time - appointment.start_time).total_seconds() / 60),
                "attendees": appointment.attendees,
                "appointment_type": appointment.appointment_type.value,
                "status": appointment.status.value,
                "location": appointment.location,
                "meeting_link": appointment.meeting_link,
                "created_at": appointment.created_at.isoformat() if appointment.created_at else None
            }
        }

    # =============================================================================
    # CALENDAR INTEGRATION HELPERS
    # =============================================================================

    def _generate_meeting_link(self, appointment_id: str) -> str:
        """Generate meeting link for appointment"""
        # In production, integrate with Zoom, Teams, etc.
        meeting_id = f"meet_{appointment_id}"
        return f"https://zoom.us/j/{meeting_id}"

    async def _send_calendar_invite(self, appointment: Appointment) -> Dict[str, Any]:
        """Send calendar invitation to attendees"""
        # Mock calendar invite sending
        await asyncio.sleep(0.05)
        
        return {
            "success": True,
            "invites_sent": len(appointment.attendees),
            "method": "calendar_invite"
        }

    # =============================================================================
    # REAL API IMPLEMENTATIONS (Templates)
    # =============================================================================

    async def _google_schedule_appointment(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real Google Calendar API implementation"""
        # TODO: Implement actual Google Calendar API calls
        # from google.oauth2.credentials import Credentials
        # from googleapiclient.discovery import build
        
        logger.info("Google Calendar API not implemented yet - using mock")
        return await self._mock_schedule_appointment(appointment_data)

    async def _outlook_schedule_appointment(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real Microsoft Outlook API implementation"""
        # TODO: Implement actual Microsoft Graph API calls
        # import requests
        
        logger.info("Outlook API not implemented yet - using mock")
        return await self._mock_schedule_appointment(appointment_data)

    async def _real_check_availability(self, start_date: datetime, end_date: datetime, duration_minutes: int) -> Dict[str, Any]:
        """Real availability checking implementation"""
        logger.info("Real availability checking not implemented yet - using mock")
        return await self._mock_check_availability(start_date, end_date, duration_minutes)

    async def _real_reschedule_appointment(self, appointment_id: str, new_time: datetime, reason: Optional[str]) -> Dict[str, Any]:
        """Real appointment rescheduling implementation"""
        logger.info("Real rescheduling not implemented yet - using mock")
        return await self._mock_reschedule_appointment(appointment_id, new_time, reason)

    async def _real_cancel_appointment(self, appointment_id: str, reason: Optional[str]) -> Dict[str, Any]:
        """Real appointment cancellation implementation"""
        logger.info("Real cancellation not implemented yet - using mock")
        return await self._mock_cancel_appointment(appointment_id, reason)

    async def _real_get_appointment(self, appointment_id: str) -> Dict[str, Any]:
        """Real appointment retrieval implementation"""
        logger.info("Real appointment retrieval not implemented yet - using mock")
        return await self._mock_get_appointment(appointment_id)

    # =============================================================================
    # BOOKING LINK GENERATION
    # =============================================================================

    @latency_monitor("scheduling_generate_booking_link")
    async def generate_booking_link(self, 
                                  link_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate shareable booking link (Calendly-style)
        
        Args:
            link_config: Configuration for booking link (duration, type, availability, etc.)
            
        Returns:
            Generated booking link and configuration
        """
        link_id = f"book_{uuid.uuid4().hex[:12]}"
        
        booking_link = {
            "link_id": link_id,
            "url": f"https://booking.example.com/{link_id}",
            "appointment_type": link_config.get("appointment_type", "consultation"),
            "duration_minutes": link_config.get("duration", 30),
            "availability_window_days": link_config.get("window_days", 14),
            "buffer_minutes": link_config.get("buffer", 15),
            "max_bookings_per_day": link_config.get("max_per_day", 8),
            "business_hours": self.business_hours,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=90)).isoformat(),
            "active": True
        }
        
        return {
            "success": True,
            "booking_link": booking_link,
            "instructions": "Share this link with customers to allow self-scheduling"
        }

    # =============================================================================
    # ANALYTICS AND REPORTING
    # =============================================================================

    def get_scheduling_analytics(self, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get scheduling analytics and metrics"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter appointments in date range
        appointments_in_range = [
            apt for apt in self.mock_appointments.values()
            if start_date <= apt.start_time <= end_date
        ]
        
        # Calculate metrics
        total_appointments = len(appointments_in_range)
        by_status = {}
        by_type = {}
        
        for apt in appointments_in_range:
            # Count by status
            status = apt.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # Count by type
            apt_type = apt.appointment_type.value
            by_type[apt_type] = by_type.get(apt_type, 0) + 1
        
        # Calculate rates
        completed = by_status.get("completed", 0)
        no_shows = by_status.get("no_show", 0)
        cancelled = by_status.get("cancelled", 0)
        
        completion_rate = (completed / total_appointments * 100) if total_appointments > 0 else 0
        no_show_rate = (no_shows / total_appointments * 100) if total_appointments > 0 else 0
        cancellation_rate = (cancelled / total_appointments * 100) if total_appointments > 0 else 0
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "totals": {
                "total_appointments": total_appointments,
                "completed": completed,
                "no_shows": no_shows,
                "cancelled": cancelled,
                "scheduled": by_status.get("scheduled", 0),
                "confirmed": by_status.get("confirmed", 0)
            },
            "rates": {
                "completion_rate": round(completion_rate, 1),
                "no_show_rate": round(no_show_rate, 1),
                "cancellation_rate": round(cancellation_rate, 1)
            },
            "by_appointment_type": by_type,
            "average_duration": 35,  # Simulated
            "busiest_days": ["Tuesday", "Wednesday", "Thursday"],
            "peak_hours": ["10:00-11:00", "14:00-15:00", "16:00-17:00"]
        }

    # =============================================================================
    # UTILITY AND MANAGEMENT METHODS
    # =============================================================================

    def get_scheduling_status(self) -> Dict[str, Any]:
        """Get scheduling system status and configuration"""
        return {
            "mock_mode": self.use_mock_scheduling,
            "configured_providers": {
                name: {
                    "provider": config.provider.value,
                    "enabled": config.enabled,
                    "has_credentials": bool(config.client_id and config.client_secret)
                }
                for name, config in self.calendar_configs.items()
            },
            "business_hours": self.business_hours,
            "appointment_durations": {
                apt_type.value: duration 
                for apt_type, duration in self.appointment_durations.items()
            },
            "mock_data_counts": {
                "appointments": len(self.mock_appointments),
                "by_status": {
                    status.value: len([a for a in self.mock_appointments.values() if a.status == status])
                    for status in AppointmentStatus
                }
            }
        }

    def update_business_hours(self, new_hours: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Update business hours configuration"""
        try:
            # Validate format
            for day, hours in new_hours.items():
                if day.lower() not in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                    return {
                        "success": False,
                        "error": f"Invalid day: {day}"
                    }
                
                if hours.get("start") and hours.get("end"):
                    # Validate time format
                    try:
                        datetime.strptime(hours["start"], "%H:%M")
                        datetime.strptime(hours["end"], "%H:%M")
                    except ValueError:
                        return {
                            "success": False,
                            "error": f"Invalid time format for {day}"
                        }
            
            # Update business hours
            self.business_hours.update(new_hours)
            
            return {
                "success": True,
                "message": "Business hours updated successfully",
                "new_hours": self.business_hours
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to update business hours: {str(e)}"
            }

    def enable_production_mode(self):
        """Switch to production mode (real calendar APIs)"""
        self.use_mock_scheduling = False
        for config in self.calendar_configs.values():
            config.use_mock = False
        logger.info("Switched to production scheduling mode - real APIs will be used")

    def enable_mock_mode(self):
        """Switch to mock mode (simulated scheduling)"""
        self.use_mock_scheduling = True
        for config in self.calendar_configs.values():
            config.use_mock = True
        logger.info("Switched to mock scheduling mode - simulated scheduling will be used")

    def get_required_environment_variables(self) -> Dict[str, List[str]]:
        """Get required environment variables for production setup"""
        return {
            "google_calendar": [
                "GOOGLE_CALENDAR_CLIENT_ID",
                "GOOGLE_CALENDAR_CLIENT_SECRET",
                "GOOGLE_CALENDAR_ID"
            ],
            "outlook": [
                "OUTLOOK_CLIENT_ID",
                "OUTLOOK_CLIENT_SECRET"
            ],
            "calendly": [
                "CALENDLY_API_KEY"
            ],
            "general": [
                "CALENDAR_TIMEZONE"
            ]
        }

    def generate_setup_instructions(self) -> str:
        """Generate setup instructions for production calendar integration"""
        return """
SCHEDULING TOOLS SETUP INSTRUCTIONS:

1. ENVIRONMENT VARIABLES:
   Set the following environment variables for production:
   
   # Google Calendar
   export GOOGLE_CALENDAR_CLIENT_ID="your-client-id"
   export GOOGLE_CALENDAR_CLIENT_SECRET="your-client-secret"
   export GOOGLE_CALENDAR_ID="primary"  # or specific calendar ID
   
   # Microsoft Outlook
   export OUTLOOK_CLIENT_ID="your-app-id"
   export OUTLOOK_CLIENT_SECRET="your-client-secret"
   
   # Calendly (if using)
   export CALENDLY_API_KEY="your-api-key"
   
   # General
   export CALENDAR_TIMEZONE="America/New_York"

2. DISABLE MOCK MODE:
   export USE_MOCK_SCHEDULING=false

3. INSTALL REQUIRED PACKAGES:
   pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
   pip install msal requests

4. OAUTH2 SETUP:
   - Google: Create project in Google Cloud Console, enable Calendar API
   - Microsoft: Register app in Azure AD, configure Graph API permissions
   - Set up OAuth2 consent screens and redirect URIs

5. CALENDAR PERMISSIONS:
   - Grant read/write access to calendars
   - Configure service account for server-to-server access (recommended)

6. BUSINESS CONFIGURATION:
   - Update business hours in the configuration
   - Set appointment types and durations
   - Configure meeting link providers (Zoom, Teams, etc.)

7. WEBHOOK SETUP (Optional):
   - Configure calendar webhooks for real-time updates
   - Handle calendar event changes and notifications

8. UPDATE API IMPLEMENTATIONS:
   Uncomment and complete the real API implementation methods in this file.

9. TEST INTEGRATIONS:
   Test calendar access and appointment creation before going live.
"""

    def get_mock_data_summary(self) -> Dict[str, Any]:
        """Get summary of mock scheduling data"""
        return {
            "appointments": {
                "total": len(self.mock_appointments),
                "by_status": {
                    status.value: len([a for a in self.mock_appointments.values() if a.status == status])
                    for status in AppointmentStatus
                },
                "by_type": {
                    apt_type.value: len([a for a in self.mock_appointments.values() if a.appointment_type == apt_type])
                    for apt_type in AppointmentType
                }
            },
            "upcoming_appointments": len([
                a for a in self.mock_appointments.values() 
                if a.start_time > datetime.now() and a.status not in [AppointmentStatus.CANCELLED]
            ]),
            "business_hours_configured": len([
                day for day, hours in self.business_hours.items() 
                if hours["start"] is not None
            ]),
            "appointment_types_available": len(self.appointment_durations)
        }