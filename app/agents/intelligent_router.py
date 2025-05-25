"""
Intelligent Multi-Layer Router for Agent Selection.
Optimized for <15ms routing decisions with confidence scoring and fallback mechanisms.
"""
import asyncio
import logging
import time
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.latency_config import LatencyConfig

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Routing strategy types."""
    KEYWORD_BASED = "keyword_based"
    ML_SIMILARITY = "ml_similarity"
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"

@dataclass
class RoutingDecision:
    """Result of routing decision with confidence and reasoning."""
    agent_id: str
    confidence: float
    reasoning: str
    strategy_used: RoutingStrategy
    processing_time: float
    alternatives: List[Tuple[str, float]] = None  # Alternative agents with scores
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "strategy_used": self.strategy_used.value,
            "processing_time": self.processing_time,
            "alternatives": self.alternatives or []
        }

class IntelligentRouter:
    """
    Multi-layer intelligent router for agent selection with sub-15ms response times.
    
    Features:
    - Keyword-based routing for immediate decisions
    - ML similarity scoring for nuanced routing
    - Hybrid approach combining multiple strategies
    - Learning from routing decisions and user feedback
    - Real-time performance optimization
    """
    
    def __init__(self, performance_tracker=None):
        """Initialize the intelligent router."""
        self.performance_tracker = performance_tracker
        
        # Agent registry reference (set during initialization)
        self.agent_registry = None
        
        # Routing strategies and models
        self.tfidf_vectorizer = None
        self.agent_vectors = {}
        self.keyword_patterns = {}
        
        # Performance tracking
        self.routing_stats = {
            'total_requests': 0,
            'strategy_usage': defaultdict(int),
            'agent_selection_count': defaultdict(int),
            'avg_processing_time': 0.0,
            'confidence_scores': deque(maxlen=1000)
        }
        
        # Learning and optimization
        self.routing_feedback = defaultdict(list)
        self.agent_performance_scores = defaultdict(float)
        
        # Cache for frequent queries
        self.routing_cache = {}
        self.cache_max_size = 1000
        
        logger.info("IntelligentRouter initialized")
    
    async def init(self, agent_registry):
        """Initialize router with agent registry."""
        self.agent_registry = agent_registry
        
        # Initialize routing models
        await self._init_keyword_patterns()
        await self._init_ml_models()
        
        # Start background optimization
        asyncio.create_task(self._optimization_loop())
        
        logger.info("✅ Intelligent router initialized")
    
    async def _init_keyword_patterns(self):
        """Initialize keyword patterns for each agent type."""
        self.keyword_patterns = {
            "roadside-assistance": {
                "primary": {
                    "tow", "stuck", "breakdown", "accident", "flat tire", 
                    "battery", "jump start", "emergency", "stranded", "wreck"
                },
                "secondary": {
                    "car", "vehicle", "auto", "truck", "motorcycle", "help",
                    "need assistance", "problem", "issue", "broken down"
                },
                "negative": {
                    "bill", "payment", "account", "technical", "app", "website"
                }
            },
            "billing-support": {
                "primary": {
                    "bill", "payment", "charge", "refund", "account", "balance",
                    "invoice", "subscription", "pricing", "cost", "fee"
                },
                "secondary": {
                    "money", "card", "credit", "debit", "bank", "transaction",
                    "receipt", "statement", "overcharge", "discount"
                },
                "negative": {
                    "tow", "breakdown", "technical", "app problem", "not working"
                }
            },
            "technical-support": {
                "primary": {
                    "app", "website", "login", "password", "not working", "broken",
                    "error", "bug", "glitch", "technical", "system", "crash"
                },
                "secondary": {
                    "phone", "mobile", "computer", "browser", "internet",
                    "connection", "sync", "update", "install", "download"
                },
                "negative": {
                    "tow", "payment", "bill", "emergency", "accident"
                }
            }
        }
        
        logger.debug("Keyword patterns initialized")
    
    async def _init_ml_models(self):
        """Initialize ML models for similarity-based routing."""
        try:
            # Get agent descriptions for TF-IDF training
            agent_descriptions = {}
            
            if self.agent_registry:
                active_agents = await self.agent_registry.get_active_agents()
                
                for agent_id in active_agents:
                    config = await self.agent_registry.get_agent_config(agent_id)
                    if config:
                        description = self._build_agent_description(agent_id, config)
                        agent_descriptions[agent_id] = description
            
            if agent_descriptions:
                # Train TF-IDF vectorizer
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True
                )
                
                descriptions = list(agent_descriptions.values())
                agent_ids = list(agent_descriptions.keys())
                
                # Fit vectorizer and create agent vectors
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
                
                for i, agent_id in enumerate(agent_ids):
                    self.agent_vectors[agent_id] = tfidf_matrix[i].toarray()[0]
                
                logger.info(f"ML models initialized with {len(agent_descriptions)} agents")
            
        except Exception as e:
            logger.warning(f"Could not initialize ML models: {e}")
    
    def _build_agent_description(self, agent_id: str, config: Dict[str, Any]) -> str:
        """Build description text for agent from configuration."""
        parts = []
        
        # Add system prompt
        if "specialization" in config and "system_prompt" in config["specialization"]:
            parts.append(config["specialization"]["system_prompt"])
        
        # Add routing keywords
        if "routing" in config:
            primary_keywords = config["routing"].get("primary_keywords", [])
            secondary_keywords = config["routing"].get("secondary_keywords", [])
            parts.extend(primary_keywords)
            parts.extend(secondary_keywords)
        
        # Add tool descriptions
        if "tools" in config:
            for tool in config["tools"]:
                if "description" in tool:
                    parts.append(tool["description"])
        
        return " ".join(parts)
    
    async def route_request(
        self,
        user_input: str,
        conversation_history: List[Any] = None,
        user_context: Dict[str, Any] = None,
        session_metadata: Dict[str, Any] = None
    ) -> RoutingDecision:
        """
        Route user request to the most appropriate agent.
        
        Args:
            user_input: User's text input
            conversation_history: Previous conversation messages
            user_context: User profile and preferences
            session_metadata: Session-specific metadata
            
        Returns:
            Routing decision with confidence and reasoning
        """
        start_time = time.time()
        
        try:
            # Check cache first for frequent queries
            cache_key = self._generate_cache_key(user_input)
            if cache_key in self.routing_cache:
                cached_decision = self.routing_cache[cache_key]
                cached_decision.processing_time = time.time() - start_time
                logger.debug(f"🚀 Cache hit for routing: {user_input[:30]}...")
                return cached_decision
            
            # Preprocess input
            processed_input = self._preprocess_input(user_input)
            
            # Try multiple routing strategies in order of speed
            routing_decision = None
            
            # Strategy 1: Rule-based routing (fastest)
            routing_decision = await self._rule_based_routing(
                processed_input, user_context, session_metadata
            )
            
            if routing_decision and routing_decision.confidence >= 0.9:
                # High confidence rule-based decision
                pass
            else:
                # Strategy 2: Keyword-based routing
                keyword_decision = await self._keyword_based_routing(processed_input)
                
                if not routing_decision or keyword_decision.confidence > routing_decision.confidence:
                    routing_decision = keyword_decision
                
                # Strategy 3: ML similarity (if time permits and confidence is low)
                if routing_decision.confidence < 0.8 and self.tfidf_vectorizer:
                    ml_decision = await self._ml_similarity_routing(processed_input)
                    
                    if ml_decision.confidence > routing_decision.confidence:
                        routing_decision = ml_decision
            
            # Apply user preferences and context
            routing_decision = await self._apply_user_preferences(
                routing_decision, user_context, session_metadata
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            routing_decision.processing_time = processing_time
            
            await self._update_routing_stats(routing_decision)
            
            # Cache decision if processing time is acceptable
            if processing_time < LatencyConfig.TARGET_ROUTING_TIME:
                self._cache_routing_decision(cache_key, routing_decision)
            
            # Track performance
            if self.performance_tracker:
                await self.performance_tracker.track_routing_decision(
                    processing_time=processing_time,
                    strategy=routing_decision.strategy_used.value,
                    confidence=routing_decision.confidence
                )
            
            logger.debug(f"🎯 Routed to {routing_decision.agent_id} "
                        f"(confidence: {routing_decision.confidence:.2f}, "
                        f"time: {processing_time*1000:.1f}ms)")
            
            return routing_decision
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Routing error: {e}")
            
            # Return fallback decision
            return RoutingDecision(
                agent_id="technical-support",  # Safe fallback
                confidence=0.5,
                reasoning=f"Fallback due to routing error: {str(e)}",
                strategy_used=RoutingStrategy.RULE_BASED,
                processing_time=processing_time
            )
    
    def _preprocess_input(self, user_input: str) -> str:
        """Preprocess user input for routing."""
        # Convert to lowercase
        processed = user_input.lower().strip()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove common filler words that don't help with routing
        filler_words = {'um', 'uh', 'like', 'you know', 'i mean'}
        words = processed.split()
        processed = ' '.join(word for word in words if word not in filler_words)
        
        return processed
    
    async def _rule_based_routing(
        self,
        user_input: str,
        user_context: Dict[str, Any] = None,
        session_metadata: Dict[str, Any] = None
    ) -> Optional[RoutingDecision]:
        """Apply rule-based routing for immediate decisions."""
        
        # Emergency detection - highest priority
        emergency_patterns = [
            r'\b(emergency|urgent|accident|crash|collision)\b',
            r'\b(stuck|stranded|breakdown|broken down)\b',
            r'\b(help|need help|assistance)\b.*\b(road|highway|freeway)\b'
        ]
        
        for pattern in emergency_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return RoutingDecision(
                    agent_id="roadside-assistance",
                    confidence=0.95,
                    reasoning="Emergency situation detected",
                    strategy_used=RoutingStrategy.RULE_BASED,
                    processing_time=0.0
                )
        
        # Specific service requests
        if re.search(r'\b(tow|towing|tow truck)\b', user_input, re.IGNORECASE):
            return RoutingDecision(
                agent_id="roadside-assistance",
                confidence=0.9,
                reasoning="Towing service request",
                strategy_used=RoutingStrategy.RULE_BASED,
                processing_time=0.0
            )
        
        # Payment/billing specific
        if re.search(r'\b(refund|charged|billing error|overcharged)\b', user_input, re.IGNORECASE):
            return RoutingDecision(
                agent_id="billing-support",
                confidence=0.9,
                reasoning="Billing issue detected",
                strategy_used=RoutingStrategy.RULE_BASED,
                processing_time=0.0
            )
        
        # Technical issues
        if re.search(r'\b(app|website|login|password)\b.*\b(not working|broken|error)\b', user_input, re.IGNORECASE):
            return RoutingDecision(
                agent_id="technical-support",
                confidence=0.9,
                reasoning="Technical issue detected",
                strategy_used=RoutingStrategy.RULE_BASED,
                processing_time=0.0
            )
        
        return None
    
    async def _keyword_based_routing(self, user_input: str) -> RoutingDecision:
        """Perform keyword-based routing with scoring."""
        agent_scores = {}
        
        for agent_id, patterns in self.keyword_patterns.items():
            score = 0
            matched_keywords = []
            
            # Check primary keywords (high weight)
            for keyword in patterns["primary"]:
                if keyword in user_input:
                    score += 3
                    matched_keywords.append(keyword)
            
            # Check secondary keywords (medium weight)
            for keyword in patterns["secondary"]:
                if keyword in user_input:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Subtract for negative keywords
            for keyword in patterns["negative"]:
                if keyword in user_input:
                    score -= 2
            
            agent_scores[agent_id] = {
                'score': max(0, score),
                'keywords': matched_keywords
            }
        
        # Find best agent
        best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x]['score'])
        best_score = agent_scores[best_agent]['score']
        
        if best_score == 0:
            # No keywords matched, default to technical support
            return RoutingDecision(
                agent_id="technical-support",
                confidence=0.3,
                reasoning="No specific keywords matched, defaulting to technical support",
                strategy_used=RoutingStrategy.KEYWORD_BASED,
                processing_time=0.0
            )
        
        # Calculate confidence based on score and competition
        total_score = sum(data['score'] for data in agent_scores.values())
        confidence = min(0.9, best_score / max(1, total_score) * 0.8 + 0.1)
        
        # Get alternatives
        alternatives = [
            (agent_id, data['score'] / max(1, total_score))
            for agent_id, data in agent_scores.items()
            if agent_id != best_agent and data['score'] > 0
        ]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        keywords_matched = agent_scores[best_agent]['keywords']
        reasoning = f"Matched keywords: {', '.join(keywords_matched[:3])}"
        
        return RoutingDecision(
            agent_id=best_agent,
            confidence=confidence,
            reasoning=reasoning,
            strategy_used=RoutingStrategy.KEYWORD_BASED,
            processing_time=0.0,
            alternatives=alternatives[:2]
        )
    
    async def _ml_similarity_routing(self, user_input: str) -> RoutingDecision:
        """Perform ML-based similarity routing."""
        if not self.tfidf_vectorizer or not self.agent_vectors:
            return RoutingDecision(
                agent_id="technical-support",
                confidence=0.4,
                reasoning="ML models not available",
                strategy_used=RoutingStrategy.ML_SIMILARITY,
                processing_time=0.0
            )
        
        try:
            # Vectorize user input
            user_vector = self.tfidf_vectorizer.transform([user_input]).toarray()[0]
            
            # Calculate similarities
            similarities = {}
            for agent_id, agent_vector in self.agent_vectors.items():
                similarity = cosine_similarity([user_vector], [agent_vector])[0][0]
                similarities[agent_id] = similarity
            
            # Find best match
            best_agent = max(similarities.keys(), key=lambda x: similarities[x])
            best_similarity = similarities[best_agent]
            
            # Calculate confidence
            confidence = min(0.85, best_similarity * 0.8 + 0.2)
            
            # Get alternatives
            alternatives = [
                (agent_id, sim) for agent_id, sim in similarities.items()
                if agent_id != best_agent
            ]
            alternatives.sort(key=lambda x: x[1], reverse=True)
            
            return RoutingDecision(
                agent_id=best_agent,
                confidence=confidence,
                reasoning=f"ML similarity: {best_similarity:.3f}",
                strategy_used=RoutingStrategy.ML_SIMILARITY,
                processing_time=0.0,
                alternatives=alternatives[:2]
            )
            
        except Exception as e:
            logger.error(f"ML routing error: {e}")
            return RoutingDecision(
                agent_id="technical-support",
                confidence=0.4,
                reasoning=f"ML routing failed: {str(e)}",
                strategy_used=RoutingStrategy.ML_SIMILARITY,
                processing_time=0.0
            )
    
    async def _apply_user_preferences(
        self,
        decision: RoutingDecision,
        user_context: Dict[str, Any] = None,
        session_metadata: Dict[str, Any] = None
    ) -> RoutingDecision:
        """Apply user preferences and context to routing decision."""
        if not user_context:
            return decision
        
        # Check user preferences
        user_profile = user_context.get("user_profile", {})
        preferred_agents = user_profile.get("preferred_agents", [])
        
        # If user has a preference for the selected agent, boost confidence
        if decision.agent_id in preferred_agents:
            decision.confidence = min(0.95, decision.confidence + 0.1)
            decision.reasoning += " (user preference)"
        
        # Check recent interaction history
        interaction_history = user_profile.get("interaction_history", {})
        
        # If user had a bad experience with the selected agent recently, consider alternatives
        for session_id, history in interaction_history.items():
            if (history.get("primary_agent") == decision.agent_id and 
                history.get("satisfaction", 1.0) < 0.5 and
                decision.alternatives):
                
                # Switch to best alternative if confidence is not very high
                if decision.confidence < 0.8:
                    best_alt = decision.alternatives[0]
                    decision.agent_id = best_alt[0]
                    decision.confidence = min(0.8, best_alt[1] + 0.2)
                    decision.reasoning += " (avoiding recent poor experience)"
                break
        
        return decision
    
    def _generate_cache_key(self, user_input: str) -> str:
        """Generate cache key for user input."""
        # Normalize input and create hash
        normalized = re.sub(r'\W+', ' ', user_input.lower()).strip()
        words = normalized.split()
        
        # Use first 10 words to create cache key
        key_words = words[:10]
        return ' '.join(key_words)
    
    def _cache_routing_decision(self, cache_key: str, decision: RoutingDecision):
        """Cache routing decision for future use."""
        # Manage cache size
        if len(self.routing_cache) >= self.cache_max_size:
            # Remove oldest entries
            oldest_keys = list(self.routing_cache.keys())[:100]
            for key in oldest_keys:
                del self.routing_cache[key]
        
        # Create cacheable decision (without processing time)
        cached_decision = RoutingDecision(
            agent_id=decision.agent_id,
            confidence=decision.confidence,
            reasoning=decision.reasoning + " (cached)",
            strategy_used=decision.strategy_used,
            processing_time=0.0,  # Will be updated when retrieved
            alternatives=decision.alternatives
        )
        
        self.routing_cache[cache_key] = cached_decision
    
    async def _update_routing_stats(self, decision: RoutingDecision):
        """Update routing statistics."""
        self.routing_stats['total_requests'] += 1
        self.routing_stats['strategy_usage'][decision.strategy_used] += 1
        self.routing_stats['agent_selection_count'][decision.agent_id] += 1
        self.routing_stats['confidence_scores'].append(decision.confidence)
        
        # Update average processing time
        total = self.routing_stats['total_requests']
        current_avg = self.routing_stats['avg_processing_time']
        self.routing_stats['avg_processing_time'] = (
            (current_avg * (total - 1) + decision.processing_time) / total
        )
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._optimize_routing()
            except Exception as e:
                logger.error(f"Error in routing optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_routing(self):
        """Optimize routing based on performance feedback."""
        try:
            # Analyze routing performance
            if len(self.routing_stats['confidence_scores']) > 100:
                avg_confidence = np.mean(list(self.routing_stats['confidence_scores']))
                
                if avg_confidence < 0.7:
                    logger.warning(f"Low average routing confidence: {avg_confidence:.2f}")
                    # Consider retraining ML models or updating keyword patterns
            
            # Update agent performance scores based on feedback
            for agent_id, feedback_list in self.routing_feedback.items():
                if feedback_list:
                    avg_satisfaction = np.mean([f['satisfaction'] for f in feedback_list])
                    self.agent_performance_scores[agent_id] = avg_satisfaction
            
            # Clean up old cache entries
            await self._cleanup_cache()
            
        except Exception as e:
            logger.error(f"Error during routing optimization: {e}")
    
    async def _cleanup_cache(self):
        """Clean up old cache entries."""
        if len(self.routing_cache) > self.cache_max_size * 0.8:
            # Remove 20% of oldest entries
            remove_count = int(len(self.routing_cache) * 0.2)
            oldest_keys = list(self.routing_cache.keys())[:remove_count]
            
            for key in oldest_keys:
                del self.routing_cache[key]
            
            logger.debug(f"Cleaned up {remove_count} old routing cache entries")
    
    async def add_routing_feedback(
        self,
        agent_id: str,
        user_satisfaction: float,
        routing_correct: bool,
        user_input: str = ""
    ):
        """Add feedback about routing decision."""
        feedback = {
            'timestamp': time.time(),
            'satisfaction': user_satisfaction,
            'routing_correct': routing_correct,
            'user_input': user_input[:100] if user_input else ""
        }
        
        self.routing_feedback[agent_id].append(feedback)
        
        # Keep only recent feedback (last 100 entries per agent)
        if len(self.routing_feedback[agent_id]) > 100:
            self.routing_feedback[agent_id] = self.routing_feedback[agent_id][-100:]
    
    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        stats = dict(self.routing_stats)
        
        # Add confidence statistics
        if self.routing_stats['confidence_scores']:
            confidence_scores = list(self.routing_stats['confidence_scores'])
            stats['confidence_stats'] = {
                'avg': np.mean(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores),
                'std': np.std(confidence_scores)
            }
        
        # Add agent performance scores
        stats['agent_performance'] = dict(self.agent_performance_scores)
        
        # Add cache statistics
        stats['cache_stats'] = {
            'size': len(self.routing_cache),
            'max_size': self.cache_max_size,
            'hit_rate': 0.0  # Would need to track cache hits to calculate this
        }
        
        return stats
    
    async def health_check(self) -> bool:
        """Perform health check on routing system."""
        try:
            # Check if agent registry is available
            if not self.agent_registry:
                return False
            
            # Test basic routing functionality
            test_input = "I need help with my account"
            decision = await self.route_request(test_input)
            
            if not decision or not decision.agent_id:
                return False
            
            # Check processing time
            if decision.processing_time > LatencyConfig.TARGET_ROUTING_TIME * 2:
                logger.warning(f"Routing performance degraded: {decision.processing_time*1000:.1f}ms")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Router health check failed: {e}")
            return False
    
    async def get_agent_suitability(
        self, 
        user_input: str, 
        agent_id: str
    ) -> float:
        """Get suitability score for a specific agent."""
        try:
            # Use keyword-based scoring for the specific agent
            if agent_id in self.keyword_patterns:
                patterns = self.keyword_patterns[agent_id]
                score = 0
                
                processed_input = self._preprocess_input(user_input)
                
                # Check primary keywords
                for keyword in patterns["primary"]:
                    if keyword in processed_input:
                        score += 3
                
                # Check secondary keywords
                for keyword in patterns["secondary"]:
                    if keyword in processed_input:
                        score += 1
                
                # Subtract for negative keywords
                for keyword in patterns["negative"]:
                    if keyword in processed_input:
                        score -= 2
                
                # Normalize score to 0-1 range
                return min(1.0, max(0.0, score / 10.0))
            
            return 0.5  # Default moderate suitability
            
        except Exception as e:
            logger.error(f"Error calculating agent suitability: {e}")
            return 0.0