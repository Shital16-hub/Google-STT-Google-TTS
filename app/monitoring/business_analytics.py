"""
Business Analytics System for Multi-Agent Voice AI
Comprehensive business intelligence, KPI tracking, and revenue analytics
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum
import json
import statistics
from decimal import Decimal

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import redis
import structlog

# Configure structured logging
logger = structlog.get_logger(__name__)

class BusinessMetricType(Enum):
    """Types of business metrics"""
    REVENUE = "revenue"
    COST = "cost"
    SATISFACTION = "satisfaction"
    EFFICIENCY = "efficiency"
    UTILIZATION = "utilization"
    CONVERSION = "conversion"
    RETENTION = "retention"

class RevenueCategory(Enum):
    """Revenue categories"""
    ROADSIDE_SERVICES = "roadside_services"
    SUBSCRIPTION_FEES = "subscription_fees"
    PREMIUM_SUPPORT = "premium_support"
    TOOL_USAGE = "tool_usage"
    EMERGENCY_FEES = "emergency_fees"

class CostCategory(Enum):
    """Cost categories"""
    API_CALLS = "api_calls"
    INFRASTRUCTURE = "infrastructure"
    HUMAN_ESCALATION = "human_escalation"
    TOOL_EXECUTION = "tool_execution"
    STORAGE = "storage"

@dataclass
class BusinessMetric:
    """Individual business metric data point"""
    timestamp: datetime
    metric_type: BusinessMetricType
    category: str
    value: float
    agent_id: Optional[str] = None
    customer_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "category": self.category,
            "value": self.value,
            "agent_id": self.agent_id,
            "customer_id": self.customer_id,
            "metadata": self.metadata
        }

@dataclass
class CustomerInteractionSummary:
    """Summary of customer interaction for business analysis"""
    session_id: str
    customer_id: str
    agent_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    interaction_type: str  # call, chat, callback
    resolution_status: str  # resolved, escalated, abandoned
    satisfaction_score: Optional[float] = None
    revenue_generated: float = 0.0
    cost_incurred: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    escalation_reason: Optional[str] = None
    
    @property
    def profit_margin(self) -> float:
        """Calculate profit margin for this interaction"""
        if self.revenue_generated == 0:
            return -self.cost_incurred
        return self.revenue_generated - self.cost_incurred
    
    @property
    def roi_percentage(self) -> float:
        """Calculate ROI percentage"""
        if self.cost_incurred == 0:
            return 100.0 if self.revenue_generated > 0 else 0.0
        return ((self.revenue_generated - self.cost_incurred) / self.cost_incurred) * 100

@dataclass
class AgentPerformanceReport:
    """Comprehensive agent performance report"""
    agent_id: str
    date_range: Tuple[datetime, datetime]
    
    # Volume metrics
    total_interactions: int = 0
    successful_resolutions: int = 0
    escalations: int = 0
    abandoned_sessions: int = 0
    
    # Time metrics
    average_handling_time: float = 0.0
    average_response_time: float = 0.0
    
    # Quality metrics
    average_satisfaction: float = 0.0
    first_call_resolution_rate: float = 0.0
    
    # Financial metrics
    total_revenue: float = 0.0
    total_cost: float = 0.0
    average_revenue_per_interaction: float = 0.0
    cost_per_interaction: float = 0.0
    
    # Efficiency metrics
    utilization_rate: float = 0.0
    tools_usage_frequency: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_interactions == 0:
            return 0.0
        return (self.successful_resolutions / self.total_interactions) * 100
    
    @property
    def escalation_rate(self) -> float:
        """Calculate escalation rate percentage"""
        if self.total_interactions == 0:
            return 0.0
        return (self.escalations / self.total_interactions) * 100
    
    @property
    def profit_margin(self) -> float:
        """Calculate total profit margin"""
        return self.total_revenue - self.total_cost
    
    @property
    def roi_percentage(self) -> float:
        """Calculate ROI percentage"""
        if self.total_cost == 0:
            return 100.0 if self.total_revenue > 0 else 0.0
        return ((self.total_revenue - self.total_cost) / self.total_cost) * 100

class RevenueTracker:
    """Track revenue generation across agents and services"""
    
    def __init__(self):
        self.revenue_rates = {
            # Roadside services pricing
            "tow_truck_dispatch": 150.00,
            "jump_start": 75.00,
            "flat_tire_change": 85.00,
            "lockout_service": 90.00,
            "emergency_fuel": 65.00,
            
            # Subscription and support fees
            "premium_support_hourly": 45.00,
            "priority_routing": 15.00,
            "callback_service": 25.00,
            
            # Tool usage fees
            "payment_processing": 2.50,
            "sms_notification": 0.05,
            "email_service": 0.02,
            "maps_api_call": 0.01,
        }
        
        self.emergency_multipliers = {
            "after_hours": 1.5,
            "holiday": 2.0,
            "severe_weather": 1.75,
            "high_traffic": 1.25
        }
    
    def calculate_service_revenue(self, 
                                service_type: str, 
                                quantity: int = 1,
                                emergency_factors: List[str] = None) -> float:
        """Calculate revenue for a specific service"""
        base_rate = self.revenue_rates.get(service_type, 0.0)
        total_revenue = base_rate * quantity
        
        # Apply emergency multipliers
        if emergency_factors:
            for factor in emergency_factors:
                multiplier = self.emergency_multipliers.get(factor, 1.0)
                total_revenue *= multiplier
        
        return round(total_revenue, 2)
    
    def calculate_subscription_revenue(self, 
                                    plan_type: str, 
                                    months: int = 1) -> float:
        """Calculate subscription revenue"""
        monthly_rates = {
            "basic": 29.99,
            "premium": 59.99,
            "enterprise": 149.99
        }
        
        monthly_rate = monthly_rates.get(plan_type, 0.0)
        return round(monthly_rate * months, 2)

class CostTracker:
    """Track operational costs across agents and services"""
    
    def __init__(self):
        self.api_costs = {
            # OpenAI API costs (per 1K tokens)
            "gpt4o_mini_input": 0.000150,
            "gpt4o_mini_output": 0.000600,
            
            # Google Cloud costs
            "stt_per_15_seconds": 0.024,
            "tts_per_1m_chars": 4.00,
            
            # Vector database costs (per query)
            "qdrant_query": 0.0001,
            "faiss_query": 0.00001,
            "redis_operation": 0.000001,
            
            # External service costs
            "twilio_sms": 0.0075,
            "sendgrid_email": 0.0000695,
            "stripe_transaction": 0.029,  # 2.9% + $0.30
            "maps_api_call": 0.005
        }
        
        self.infrastructure_costs = {
            # Per hour costs
            "compute_instance": 0.096,
            "database_instance": 0.045,
            "load_balancer": 0.025,
            "storage_gb_month": 0.10
        }
        
        self.human_escalation_costs = {
            "tier_1_support": 25.00,  # per hour
            "tier_2_support": 45.00,
            "specialist": 75.00,
            "manager": 95.00
        }
    
    def calculate_api_cost(self, 
                          service: str, 
                          usage_amount: float) -> float:
        """Calculate API usage costs"""
        unit_cost = self.api_costs.get(service, 0.0)
        return round(unit_cost * usage_amount, 6)
    
    def calculate_llm_cost(self, 
                          model: str, 
                          input_tokens: int, 
                          output_tokens: int) -> float:
        """Calculate LLM usage costs"""
        input_cost = self.calculate_api_cost(f"{model}_input", input_tokens / 1000)
        output_cost = self.calculate_api_cost(f"{model}_output", output_tokens / 1000)
        return input_cost + output_cost
    
    def calculate_escalation_cost(self, 
                                tier: str, 
                                duration_minutes: float) -> float:
        """Calculate human escalation costs"""
        hourly_rate = self.human_escalation_costs.get(tier, 0.0)
        hours = duration_minutes / 60.0
        return round(hourly_rate * hours, 2)

class BusinessAnalytics:
    """Main business analytics system"""
    
    def __init__(self, 
                 db_engine, 
                 redis_client: Optional[redis.Redis] = None):
        self.db_engine = db_engine
        self.redis_client = redis_client
        self.revenue_tracker = RevenueTracker()
        self.cost_tracker = CostTracker()
        self.metrics_cache: Dict[str, Any] = {}
        
    async def record_interaction(self, interaction: CustomerInteractionSummary):
        """Record a customer interaction for analytics"""
        try:
            # Store in database
            await self._store_interaction_db(interaction)
            
            # Update real-time metrics in Redis
            if self.redis_client:
                await self._update_realtime_metrics(interaction)
            
            # Update cached metrics
            await self._update_cached_metrics(interaction)
            
            logger.info(
                "Interaction recorded",
                session_id=interaction.session_id,
                agent_id=interaction.agent_id,
                revenue=interaction.revenue_generated,
                cost=interaction.cost_incurred
            )
            
        except Exception as e:
            logger.error(
                "Failed to record interaction",
                session_id=interaction.session_id,
                error=str(e)
            )
    
    async def record_business_metric(self, metric: BusinessMetric):
        """Record a business metric"""
        try:
            # Store in database
            await self._store_metric_db(metric)
            
            # Update Redis for real-time dashboards
            if self.redis_client:
                await self._store_metric_redis(metric)
            
            logger.debug(
                "Business metric recorded",
                metric_type=metric.metric_type.value,
                category=metric.category,
                value=metric.value
            )
            
        except Exception as e:
            logger.error("Failed to record business metric", error=str(e))
    
    async def calculate_agent_revenue(self, 
                                   agent_id: str, 
                                   service_type: str, 
                                   quantity: int = 1,
                                   emergency_factors: List[str] = None) -> float:
        """Calculate and record revenue for agent service"""
        revenue = self.revenue_tracker.calculate_service_revenue(
            service_type, quantity, emergency_factors
        )
        
        # Record revenue metric
        metric = BusinessMetric(
            timestamp=datetime.utcnow(),
            metric_type=BusinessMetricType.REVENUE,
            category=service_type,
            value=revenue,
            agent_id=agent_id,
            metadata={
                "quantity": quantity,
                "emergency_factors": emergency_factors or [],
                "base_rate": self.revenue_tracker.revenue_rates.get(service_type, 0.0)
            }
        )
        
        await self.record_business_metric(metric)
        return revenue
    
    async def calculate_agent_cost(self, 
                                 agent_id: str, 
                                 cost_category: str, 
                                 usage_data: Dict[str, Any]) -> float:
        """Calculate and record costs for agent operations"""
        total_cost = 0.0
        
        if cost_category == "llm_usage":
            cost = self.cost_tracker.calculate_llm_cost(
                usage_data["model"],
                usage_data["input_tokens"],
                usage_data["output_tokens"]
            )
            total_cost += cost
        
        elif cost_category == "api_calls":
            for service, amount in usage_data.items():
                cost = self.cost_tracker.calculate_api_cost(service, amount)
                total_cost += cost
        
        elif cost_category == "escalation":
            cost = self.cost_tracker.calculate_escalation_cost(
                usage_data["tier"],
                usage_data["duration_minutes"]
            )
            total_cost += cost
        
        # Record cost metric
        metric = BusinessMetric(
            timestamp=datetime.utcnow(),
            metric_type=BusinessMetricType.COST,
            category=cost_category,
            value=total_cost,
            agent_id=agent_id,
            metadata=usage_data
        )
        
        await self.record_business_metric(metric)
        return total_cost
    
    async def generate_agent_report(self, 
                                  agent_id: str, 
                                  start_date: datetime, 
                                  end_date: datetime) -> AgentPerformanceReport:
        """Generate comprehensive agent performance report"""
        try:
            # Query interaction data
            interactions = await self._get_agent_interactions(agent_id, start_date, end_date)
            
            # Calculate metrics
            report = AgentPerformanceReport(
                agent_id=agent_id,
                date_range=(start_date, end_date)
            )
            
            if interactions:
                # Volume metrics
                report.total_interactions = len(interactions)
                report.successful_resolutions = sum(
                    1 for i in interactions if i.resolution_status == "resolved"
                )
                report.escalations = sum(
                    1 for i in interactions if i.resolution_status == "escalated"
                )
                report.abandoned_sessions = sum(
                    1 for i in interactions if i.resolution_status == "abandoned"
                )
                
                # Time metrics
                durations = [i.duration_minutes for i in interactions]
                report.average_handling_time = statistics.mean(durations)
                
                # Quality metrics
                satisfaction_scores = [
                    i.satisfaction_score for i in interactions 
                    if i.satisfaction_score is not None
                ]
                if satisfaction_scores:
                    report.average_satisfaction = statistics.mean(satisfaction_scores)
                
                # Financial metrics
                report.total_revenue = sum(i.revenue_generated for i in interactions)
                report.total_cost = sum(i.cost_incurred for i in interactions)
                report.average_revenue_per_interaction = report.total_revenue / len(interactions)
                report.cost_per_interaction = report.total_cost / len(interactions)
                
                # Tool usage
                all_tools = [tool for i in interactions for tool in i.tools_used]
                report.tools_usage_frequency = dict(Counter(all_tools))
                
                # First call resolution
                first_call_resolutions = sum(
                    1 for i in interactions 
                    if i.resolution_status == "resolved" and len(i.tools_used) <= 2
                )
                report.first_call_resolution_rate = (
                    first_call_resolutions / len(interactions) * 100
                )
            
            return report
            
        except Exception as e:
            logger.error(
                "Failed to generate agent report",
                agent_id=agent_id,
                error=str(e)
            )
            raise
    
    async def get_business_dashboard_data(self, 
                                       date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get comprehensive business dashboard data"""
        if not date_range:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            date_range = (start_date, end_date)
        
        try:
            dashboard_data = {
                "summary": await self._get_business_summary(date_range),
                "revenue_breakdown": await self._get_revenue_breakdown(date_range),
                "cost_analysis": await self._get_cost_analysis(date_range),
                "agent_performance": await self._get_agents_comparison(date_range),
                "customer_satisfaction": await self._get_satisfaction_trends(date_range),
                "operational_efficiency": await self._get_efficiency_metrics(date_range),
                "trends": await self._get_business_trends(date_range)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Failed to generate dashboard data", error=str(e))
            raise
    
    async def calculate_roi_by_agent(self, 
                                   start_date: datetime, 
                                   end_date: datetime) -> Dict[str, float]:
        """Calculate ROI for each agent"""
        try:
            query = """
            SELECT 
                agent_id,
                SUM(revenue_generated) as total_revenue,
                SUM(cost_incurred) as total_cost
            FROM customer_interactions 
            WHERE start_time >= :start_date AND start_time <= :end_date
            GROUP BY agent_id
            """
            
            async with AsyncSession(self.db_engine) as session:
                result = await session.execute(
                    text(query), 
                    {"start_date": start_date, "end_date": end_date}
                )
                rows = result.fetchall()
            
            roi_data = {}
            for row in rows:
                agent_id, revenue, cost = row
                if cost > 0:
                    roi = ((revenue - cost) / cost) * 100
                else:
                    roi = 100.0 if revenue > 0 else 0.0
                roi_data[agent_id] = round(roi, 2)
            
            return roi_data
            
        except Exception as e:
            logger.error("Failed to calculate ROI by agent", error=str(e))
            return {}
    
    async def get_peak_usage_analysis(self) -> Dict[str, Any]:
        """Analyze peak usage patterns for capacity planning"""
        try:
            # Get hourly usage patterns
            query = """
            SELECT 
                EXTRACT(hour FROM start_time) as hour,
                EXTRACT(dow FROM start_time) as day_of_week,
                agent_id,
                COUNT(*) as interaction_count,
                AVG(duration_minutes) as avg_duration
            FROM customer_interactions 
            WHERE start_time >= :start_date
            GROUP BY EXTRACT(hour FROM start_time), EXTRACT(dow FROM start_time), agent_id
            ORDER BY hour, day_of_week, agent_id
            """
            
            start_date = datetime.utcnow() - timedelta(days=30)
            
            async with AsyncSession(self.db_engine) as session:
                result = await session.execute(text(query), {"start_date": start_date})
                rows = result.fetchall()
            
            # Process data for analysis
            hourly_patterns = defaultdict(list)
            daily_patterns = defaultdict(list)
            agent_patterns = defaultdict(lambda: defaultdict(list))
            
            for row in rows:
                hour, day_of_week, agent_id, count, avg_duration = row
                
                hourly_patterns[hour].append(count)
                daily_patterns[day_of_week].append(count)
                agent_patterns[agent_id][hour].append(count)
            
            # Calculate peak times
            peak_hours = sorted(
                [(hour, statistics.mean(counts)) for hour, counts in hourly_patterns.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            peak_days = sorted(
                [(day, statistics.mean(counts)) for day, counts in daily_patterns.items()],
                key=lambda x: x[1], reverse=True
            )
            
            return {
                "peak_hours": [{"hour": h, "avg_interactions": round(c, 1)} for h, c in peak_hours],
                "peak_days": [{"day": d, "avg_interactions": round(c, 1)} for d, c in peak_days],
                "agent_peak_patterns": {
                    agent_id: sorted(
                        [(hour, statistics.mean(counts)) for hour, counts in hours.items()],
                        key=lambda x: x[1], reverse=True
                    )[:3]
                    for agent_id, hours in agent_patterns.items()
                },
                "capacity_recommendations": await self._generate_capacity_recommendations(
                    hourly_patterns, agent_patterns
                )
            }
            
        except Exception as e:
            logger.error("Failed to analyze peak usage", error=str(e))
            return {}
    
    async def _store_interaction_db(self, interaction: CustomerInteractionSummary):
        """Store interaction in database"""
        try:
            query = """
            INSERT INTO customer_interactions (
                session_id, customer_id, agent_id, start_time, end_time,
                duration_minutes, interaction_type, resolution_status,
                satisfaction_score, revenue_generated, cost_incurred,
                tools_used, escalation_reason
            ) VALUES (
                :session_id, :customer_id, :agent_id, :start_time, :end_time,
                :duration_minutes, :interaction_type, :resolution_status,
                :satisfaction_score, :revenue_generated, :cost_incurred,
                :tools_used, :escalation_reason
            )
            """
            
            async with AsyncSession(self.db_engine) as session:
                await session.execute(text(query), {
                    "session_id": interaction.session_id,
                    "customer_id": interaction.customer_id,
                    "agent_id": interaction.agent_id,
                    "start_time": interaction.start_time,
                    "end_time": interaction.end_time,
                    "duration_minutes": interaction.duration_minutes,
                    "interaction_type": interaction.interaction_type,
                    "resolution_status": interaction.resolution_status,
                    "satisfaction_score": interaction.satisfaction_score,
                    "revenue_generated": interaction.revenue_generated,
                    "cost_incurred": interaction.cost_incurred,
                    "tools_used": json.dumps(interaction.tools_used),
                    "escalation_reason": interaction.escalation_reason
                })
                await session.commit()
                
        except Exception as e:
            logger.error("Failed to store interaction in DB", error=str(e))
            raise
    
    async def _store_metric_db(self, metric: BusinessMetric):
        """Store business metric in database"""
        try:
            query = """
            INSERT INTO business_metrics (
                timestamp, metric_type, category, value, agent_id,
                customer_id, metadata
            ) VALUES (
                :timestamp, :metric_type, :category, :value, :agent_id,
                :customer_id, :metadata
            )
            """
            
            async with AsyncSession(self.db_engine) as session:
                await session.execute(text(query), {
                    "timestamp": metric.timestamp,
                    "metric_type": metric.metric_type.value,
                    "category": metric.category,
                    "value": metric.value,
                    "agent_id": metric.agent_id,
                    "customer_id": metric.customer_id,
                    "metadata": json.dumps(metric.metadata)
                })
                await session.commit()
                
        except Exception as e:
            logger.error("Failed to store metric in DB", error=str(e))
            raise
    
    async def _store_metric_redis(self, metric: BusinessMetric):
        """Store metric in Redis for real-time access"""
        if not self.redis_client:
            return
        
        try:
            # Store latest value
            key = f"business_metrics:{metric.metric_type.value}:{metric.category}"
            if metric.agent_id:
                key += f":{metric.agent_id}"
            
            await self.redis_client.setex(
                key, 3600, json.dumps(metric.to_dict())
            )
            
            # Add to time series
            ts_key = f"business_metrics:ts:{metric.metric_type.value}:{metric.category}"
            timestamp = int(metric.timestamp.timestamp() * 1000)
            await self.redis_client.zadd(ts_key, {json.dumps(metric.to_dict()): timestamp})
            
            # Keep only last 24 hours
            cutoff = timestamp - (24 * 60 * 60 * 1000)
            await self.redis_client.zremrangebyscore(ts_key, 0, cutoff)
            
        except Exception as e:
            logger.error("Failed to store metric in Redis", error=str(e))
    
    async def _get_business_summary(self, date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Get high-level business summary"""
        start_date, end_date = date_range
        
        try:
            query = """
            SELECT 
                COUNT(*) as total_interactions,
                SUM(revenue_generated) as total_revenue,
                SUM(cost_incurred) as total_cost,
                AVG(satisfaction_score) as avg_satisfaction,
                COUNT(CASE WHEN resolution_status = 'resolved' THEN 1 END) as resolved_count,
                COUNT(CASE WHEN resolution_status = 'escalated' THEN 1 END) as escalated_count
            FROM customer_interactions 
            WHERE start_time >= :start_date AND start_time <= :end_date
            """
            
            async with AsyncSession(self.db_engine) as session:
                result = await session.execute(
                    text(query), 
                    {"start_date": start_date, "end_date": end_date}
                )
                row = result.fetchone()
            
            if row:
                total_interactions, total_revenue, total_cost, avg_satisfaction, resolved_count, escalated_count = row
                
                return {
                    "total_interactions": total_interactions or 0,
                    "total_revenue": float(total_revenue or 0),
                    "total_cost": float(total_cost or 0),
                    "profit_margin": float((total_revenue or 0) - (total_cost or 0)),
                    "average_satisfaction": float(avg_satisfaction or 0),
                    "resolution_rate": (resolved_count / total_interactions * 100) if total_interactions > 0 else 0,
                    "escalation_rate": (escalated_count / total_interactions * 100) if total_interactions > 0 else 0,
                    "roi_percentage": (((total_revenue or 0) - (total_cost or 0)) / (total_cost or 1)) * 100
                }
            
            return {}
            
        except Exception as e:
            logger.error("Failed to get business summary", error=str(e))
            return {}
    
    async def _generate_capacity_recommendations(self, 
                                               hourly_patterns: Dict[int, List[int]],
                                               agent_patterns: Dict[str, Dict[int, List[int]]]) -> List[str]:
        """Generate capacity planning recommendations"""
        recommendations = []
        
        try:
            # Find peak hours across all agents
            peak_load = max(
                statistics.mean(counts) for counts in hourly_patterns.values()
            )
            avg_load = statistics.mean([
                statistics.mean(counts) for counts in hourly_patterns.values()
            ])
            
            if peak_load > avg_load * 2:
                recommendations.append(
                    f"Consider auto-scaling during peak hours (peak load {peak_load:.1f} vs avg {avg_load:.1f})"
                )
            
            # Analyze agent-specific patterns
            for agent_id, hours in agent_patterns.items():
                agent_peak = max(statistics.mean(counts) for counts in hours.values())
                agent_avg = statistics.mean([statistics.mean(counts) for counts in hours.values()])
                
                if agent_peak > agent_avg * 3:
                    recommendations.append(
                        f"Agent {agent_id} shows high peak variability - consider load balancing"
                    )
            
            if len(recommendations) == 0:
                recommendations.append("Current capacity appears well-balanced")
            
        except Exception as e:
            logger.error("Failed to generate capacity recommendations", error=str(e))
            recommendations.append("Unable to generate recommendations due to insufficient data")
        
        return recommendations

# Global business analytics instance
business_analytics: Optional[BusinessAnalytics] = None

def initialize_business_analytics(db_engine, redis_client: Optional[redis.Redis] = None):
    """Initialize the global business analytics system"""
    global business_analytics
    
    if business_analytics is None:
        business_analytics = BusinessAnalytics(db_engine, redis_client)
        logger.info("Business analytics system initialized")
    
    return business_analytics

def get_business_analytics() -> BusinessAnalytics:
    """Get the global business analytics instance"""
    if business_analytics is None:
        raise RuntimeError("Business analytics not initialized. Call initialize_business_analytics() first.")
    return business_analytics