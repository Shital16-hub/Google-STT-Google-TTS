"""
ML-Based Intelligent Agent Router with Confidence Scoring
Implements advanced routing algorithms with context-aware decision making.
Target: <15ms routing decision with >90% accuracy.
"""
import asyncio
import logging
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import re
from collections import defaultdict, Counter

from app.agents.registry import AgentRegistry
from app.agents.base_agent import BaseAgent, UrgencyLevel
from app.vector_db.hybrid_vector_system import HybridVectorSystem

logger = logging.getLogger(__name__)

class RoutingStrategy(str, Enum):
    """Routing strategies for agent selection."""
    KEYWORD_MATCHING = "keyword_matching"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ML_CLASSIFICATION = "ml_classification"
    HYBRID_INTELLIGENT = "hybrid_intelligent"
    CONTEXT_AWARE = "context_aware"

class RoutingDecision(str, Enum):
    """Types of routing decisions."""
    DIRECT_MATCH = "direct_match"
    CONFIDENCE_BASED = "confidence_based"
    FALLBACK = "fallback"
    ESCALATION = "escalation"
    LOAD_BALANCED = "load_balanced"

@dataclass
class RoutingResult:
    """Result of agent routing decision."""
    selected_agent_id: str
    confidence: float
    routing_time_ms: float
    strategy_used: RoutingStrategy
    decision_type: RoutingDecision
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    routing_factors: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentPerformanceProfile:
    """Performance profile for routing decisions."""
    agent_id: str
    success_rate: float = 0.0
    average_response_time_ms: float = 0.0
    average_confidence: float = 0.0
    specialization_score: float = 0.0
    current_load: int = 0
    max_concurrent: int = 10
    last_updated: float = field(default_factory=time.time)

class KeywordMatcher:
    """Advanced keyword matching with weights and synonyms."""
    
    def __init__(self):
        self.keyword_patterns = {}
        self.synonym_map = {}
        self.weight_boosters = {}
        
        # Initialize with domain-specific patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize keyword patterns for different domains."""
        
        # Roadside assistance patterns
        self.keyword_patterns["roadside-assistance"] = {
            "primary": [
                "tow", "towing", "stuck", "breakdown", "accident", "emergency",
                "stranded", "crash", "collision", "vehicle", "car trouble"
            ],
            "secondary": [
                "jump start", "flat tire", "dead battery", "won't start",
                "help", "assistance", "roadside", "aaa", "service"
            ],
            "urgency_indicators": [
                "urgent", "emergency", "critical", "asap", "immediate",
                "dangerous", "traffic", "highway", "unsafe"
            ]
        }
        
        # Billing support patterns
        self.keyword_patterns["billing-support"] = {
            "primary": [
                "bill", "billing", "payment", "charge", "invoice", "refund",
                "subscription", "plan", "pricing", "cost", "fee"
            ],
            "secondary": [
                "account", "money", "card", "credit", "debit", "bank",
                "transaction", "receipt", "statement", "balance"
            ],
            "urgency_indicators": [
                "overcharge", "unauthorized", "dispute", "fraud", "error",
                "wrong amount", "cancel", "immediate refund"
            ]
        }
        
        # Technical support patterns
        self.keyword_patterns["technical-support"] = {
            "primary": [
                "not working", "error", "bug", "issue", "problem", "broken",
                "setup", "install", "configure", "technical", "support"
            ],
            "secondary": [
                "login", "password", "access", "connection", "network",
                "app", "software", "system", "device", "troubleshoot"
            ],
            "urgency_indicators": [
                "critical system", "production down", "server error",
                "data loss", "security breach", "urgent fix"
            ]
        }
        
        # Synonym mapping for better matching
        self.synonym_map = {
            "tow": ["towing", "towed", "haul", "pull"],
            "breakdown": ["broken down", "broke down", "not running"],
            "stuck": ["stranded", "trapped", "immobilized"],
            "payment": ["bill", "charge", "fee", "cost"],
            "refund": ["return", "credit", "reimbursement"],
            "error": ["bug", "issue", "problem", "fault"],
            "install": ["setup", "configure", "set up"]
        }
        
        # Weight boosters for context
        self.weight_boosters = {
            "location_mentioned": 1.3,
            "phone_number_mentioned": 1.2,
            "urgency_high": 1.5,
            "previous_context": 1.4
        }
    
    def calculate_match_score(
        self,
        query: str,
        agent_id: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate match score for agent based on keywords."""
        if agent_id not in self.keyword_patterns:
            return 0.0
        
        patterns = self.keyword_patterns[agent_id]
        query_lower = query.lower()
        score = 0.0
        
        # Primary keywords (highest weight)
        for keyword in patterns["primary"]:
            if keyword in query_lower:
                score += 2.0
                # Check for synonyms
                synonyms = self.synonym_map.get(keyword, [])
                for synonym in synonyms:
                    if synonym in query_lower:
                        score += 1.0
        
        # Secondary keywords (medium weight)
        for keyword in patterns["secondary"]:
            if keyword in query_lower:
                score += 1.0
        
        # Urgency indicators (context boost)
        urgency_boost = 0.0
        for indicator in patterns["urgency_indicators"]:
            if indicator in query_lower:
                urgency_boost += 0.5
        
        # Apply context boosters
        if context.get("location"):
            score *= self.weight_boosters.get("location_mentioned", 1.0)
        
        if context.get("phone_number"):
            score *= self.weight_boosters.get("phone_number_mentioned", 1.0)
        
        if urgency_boost > 0:
            score *= self.weight_boosters.get("urgency_high", 1.0)
        
        # Previous context boost
        if context.get("previous_agent_id") == agent_id:
            score *= self.weight_boosters.get("previous_context", 1.0)
        
        # Normalize score
        max_possible_score = len(patterns["primary"]) * 2.0 + len(patterns["secondary"]) * 1.0
        normalized_score = min(1.0, score / max_possible_score) if max_possible_score > 0 else 0.0
        
        return normalized_score

class SemanticSimilarityMatcher:
    """Semantic similarity matching using vector embeddings."""
    
    def __init__(self, hybrid_vector_system: HybridVectorSystem):
        self.hybrid_vector_system = hybrid_vector_system
        self.agent_embeddings = {}
        self.similarity_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize semantic matcher with agent embeddings."""
        # Create embeddings for each agent's specialization
        agent_descriptions = {
            "roadside-assistance": (
                "Emergency roadside assistance including towing, jump starts, "
                "flat tire repair, accident response, and vehicle recovery services"
            ),
            "billing-support": (
                "Billing and payment support including invoices, refunds, "
                "subscription management, payment processing, and account billing"
            ),
            "technical-support": (
                "Technical support for software issues, system errors, "
                "installation problems, configuration help, and troubleshooting"
            )
        }
        
        for agent_id, description in agent_descriptions.items():
            # This would use your actual embedding model
            embedding = await self._create_embedding(description)
            self.agent_embeddings[agent_id] = embedding
    
    async def calculate_similarity_score(
        self,
        query: str,
        agent_id: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity score."""
        if agent_id not in self.agent_embeddings:
            return 0.0
        
        # Check cache first
        cache_key = self._get_cache_key(query, agent_id)
        cached_score = self._get_cached_score(cache_key)
        if cached_score is not None:
            return cached_score
        
        try:
            # Create query embedding
            query_embedding = await self._create_embedding(query)
            agent_embedding = self.agent_embeddings[agent_id]
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, agent_embedding)
            
            # Cache the result
            self._cache_score(cache_key, similarity)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text."""
        # This would use your actual embedding model
        # For now, returning a placeholder vector
        return np.random.rand(1536).astype(np.float32)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_cache_key(self, query: str, agent_id: str) -> str:
        """Generate cache key for query-agent pair."""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        return f"{agent_id}:{query_hash}"
    
    def _get_cached_score(self, cache_key: str) -> Optional[float]:
        """Get cached similarity score."""
        if cache_key in self.similarity_cache:
            entry = self.similarity_cache[cache_key]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                return entry["score"]
            else:
                del self.similarity_cache[cache_key]
        return None
    
    def _cache_score(self, cache_key: str, score: float):
        """Cache similarity score."""
        self.similarity_cache[cache_key] = {
            "score": score,
            "timestamp": time.time()
        }

class MLClassifier:
    """Machine learning classifier for agent routing."""
    
    def __init__(self):
        self.feature_extractors = {
            "query_length": self._extract_query_length,
            "question_words": self._extract_question_words,
            "urgency_indicators": self._extract_urgency_indicators,
            "entity_types": self._extract_entity_types,
            "sentiment_score": self._extract_sentiment,
            "complexity_score": self._extract_complexity
        }
        
        # Simple rule-based classifier weights
        self.agent_weights = {
            "roadside-assistance": {
                "urgency_indicators": 2.0,
                "entity_types": 1.5,
                "location_words": 2.0,
                "vehicle_words": 2.5
            },
            "billing-support": {
                "money_words": 2.5,
                "account_words": 2.0,
                "sentiment_score": 1.5,
                "question_words": 1.0
            },
            "technical-support": {
                "technical_words": 2.5,
                "complexity_score": 2.0,
                "error_words": 2.0,
                "question_words": 1.5
            }
        }
    
    def classify_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Classify query and return confidence scores for each agent."""
        features = self._extract_features(query, context)
        scores = {}
        
        for agent_id, weights in self.agent_weights.items():
            score = 0.0
            
            # Calculate weighted feature score
            for feature_name, feature_value in features.items():
                weight = weights.get(feature_name, 0.0)
                score += feature_value * weight
            
            # Normalize score
            max_possible = sum(weights.values())
            normalized_score = min(1.0, score / max_possible) if max_possible > 0 else 0.0
            scores[agent_id] = normalized_score
        
        return scores
    
    def _extract_features(self, query: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from query and context."""
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = extractor(query, context)
            except Exception as e:
                logger.error(f"Error extracting feature {feature_name}: {e}")
                features[feature_name] = 0.0
        
        # Domain-specific features
        features.update(self._extract_domain_features(query, context))
        
        return features
    
    def _extract_query_length(self, query: str, context: Dict[str, Any]) -> float:
        """Extract query length feature."""
        length = len(query.split())
        return min(1.0, length / 20.0)  # Normalize to 0-1
    
    def _extract_question_words(self, query: str, context: Dict[str, Any]) -> float:
        """Extract question words feature."""
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        query_lower = query.lower()
        count = sum(1 for word in question_words if word in query_lower)
        return min(1.0, count / 3.0)  # Normalize
    
    def _extract_urgency_indicators(self, query: str, context: Dict[str, Any]) -> float:
        """Extract urgency indicators."""
        urgency_words = [
            "urgent", "emergency", "asap", "immediate", "critical",
            "help", "now", "quickly", "fast"
        ]
        query_lower = query.lower()
        count = sum(1 for word in urgency_words if word in query_lower)
        return min(1.0, count / 2.0)
    
    def _extract_entity_types(self, query: str, context: Dict[str, Any]) -> float:
        """Extract entity types (simplified)."""
        # Simple entity detection
        entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        return min(1.0, len(entities) / 5.0)
    
    def _extract_sentiment(self, query: str, context: Dict[str, Any]) -> float:
        """Extract sentiment score."""
        positive_words = ["good", "great", "excellent", "love", "happy"]
        negative_words = ["bad", "terrible", "hate", "frustrated", "angry", "problem"]
        
        query_lower = query.lower()
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        
        # Return negative sentiment intensity (problems need support)
        return min(1.0, negative_count / 3.0)
    
    def _extract_complexity(self, query: str, context: Dict[str, Any]) -> float:
        """Extract complexity score."""
        # Simple complexity based on length and structure
        word_count = len(query.split())
        sentence_count = len([s for s in query.split('.') if s.strip()])
        
        complexity = (word_count / 50.0) + (sentence_count / 10.0)
        return min(1.0, complexity)
    
    def _extract_domain_features(self, query: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract domain-specific features."""
        query_lower = query.lower()
        features = {}
        
        # Vehicle/location words for roadside
        vehicle_words = ["car", "truck", "vehicle", "auto", "tow", "highway", "road"]
        location_words = ["street", "highway", "parking", "lot", "address", "location"]
        
        features["vehicle_words"] = min(1.0, sum(1 for word in vehicle_words if word in query_lower) / 3.0)
        features["location_words"] = min(1.0, sum(1 for word in location_words if word in query_lower) / 3.0)
        
        # Money/account words for billing
        money_words = ["payment", "bill", "charge", "refund", "money", "cost", "price"]
        account_words = ["account", "subscription", "plan", "invoice", "statement"]
        
        features["money_words"] = min(1.0, sum(1 for word in money_words if word in query_lower) / 3.0)
        features["account_words"] = min(1.0, sum(1 for word in account_words if word in query_lower) / 3.0)
        
        # Technical words for support
        technical_words = ["error", "bug", "install", "setup", "login", "password", "system"]
        error_words = ["not working", "broken", "error", "issue", "problem", "fault"]
        
        features["technical_words"] = min(1.0, sum(1 for word in technical_words if word in query_lower) / 3.0)
        features["error_words"] = min(1.0, sum(1 for phrase in error_words if phrase in query_lower) / 3.0)
        
        return features

class ContextAnalyzer:
    """Analyzes conversation context for better routing decisions."""
    
    def __init__(self):
        self.context_weights = {
            "previous_agent_continuity": 0.3,
            "conversation_phase": 0.2,
            "user_satisfaction": 0.2,
            "urgency_escalation": 0.3
        }
    
    def analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for routing insights."""
        analysis = {
            "continuity_preference": None,
            "phase_requirements": None,
            "satisfaction_indicators": "neutral",
            "urgency_level": UrgencyLevel.NORMAL,
            "context_score": 0.0
        }
        
        # Previous agent continuity
        previous_agent = context.get("previous_agent_id")
        if previous_agent and context.get("conversation_ongoing", False):
            analysis["continuity_preference"] = previous_agent
        
        # Conversation phase
        phase = context.get("conversation_phase", "initial")
        if phase in ["follow_up", "clarification"]:
            analysis["phase_requirements"] = "continuation"
        
        # Satisfaction indicators
        if context.get("user_frustrated", False):
            analysis["satisfaction_indicators"] = "negative"
            analysis["urgency_level"] = UrgencyLevel.HIGH
        elif context.get("user_satisfied", False):
            analysis["satisfaction_indicators"] = "positive"
        
        # Urgency escalation
        if context.get("escalation_requested", False):
            analysis["urgency_level"] = UrgencyLevel.CRITICAL
        
        # Calculate context score
        score = 0.5  # Base score
        if analysis["continuity_preference"]:
            score += 0.2
        if analysis["satisfaction_indicators"] == "negative":
            score += 0.3  # Negative satisfaction increases routing importance
        
        analysis["context_score"] = min(1.0, score)
        
        return analysis

class IntelligentAgentRouter:
    """
    Advanced ML-based agent router with context awareness and confidence scoring.
    Implements hybrid routing strategies for optimal agent selection.
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        hybrid_vector_system: HybridVectorSystem,
        confidence_threshold: float = 0.85,
        fallback_threshold: float = 0.6,
        enable_ml_routing: bool = True,
        routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID_INTELLIGENT
    ):
        """Initialize the intelligent agent router."""
        self.agent_registry = agent_registry
        self.hybrid_vector_system = hybrid_vector_system
        self.confidence_threshold = confidence_threshold
        self.fallback_threshold = fallback_threshold
        self.enable_ml_routing = enable_ml_routing
        self.routing_strategy = routing_strategy
        
        # Routing components
        self.keyword_matcher = KeywordMatcher()
        self.semantic_matcher = SemanticSimilarityMatcher(hybrid_vector_system)
        self.ml_classifier = MLClassifier()
        self.context_analyzer = ContextAnalyzer()
        
        # Performance tracking
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.routing_stats = {
            "total_routes": 0,
            "successful_routes": 0,
            "fallback_routes": 0,
            "average_routing_time_ms": 0.0,
            "strategy_usage": defaultdict(int),
            "agent_selection_counts": defaultdict(int)
        }
        
        # Caching
        self.routing_cache = {}
        self.cache_ttl = 60  # 1 minute
        
        self.initialized = False
        logger.info("Intelligent Agent Router initialized with ML capabilities")
    
    async def initialize(self):
        """Initialize the router and all components."""
        logger.info("🚀 Initializing Intelligent Agent Router...")
        
        try:
            # Initialize semantic matcher
            await self.semantic_matcher.initialize()
            
            # Initialize agent performance profiles
            await self._initialize_agent_profiles()
            
            # Start background tasks
            asyncio.create_task(self._background_profile_updates())
            
            self.initialized = True
            logger.info("✅ Intelligent Agent Router initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Router initialization failed: {e}")
            raise
    
    async def route_query(
        self,
        query: str,
        context: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        preferred_agent_id: Optional[str] = None
    ) -> RoutingResult:
        """
        Route query to the most appropriate agent using intelligent routing.
        Target: <15ms routing decision.
        """
        if not self.initialized:
            await self.initialize()
        
        routing_start = time.time()
        
        logger.debug(f"Routing query: {query[:100]}...")
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, context, preferred_agent_id)
            cached_result = self._get_cached_routing(cache_key)
            
            if cached_result:
                cached_result.routing_time_ms = (time.time() - routing_start) * 1000
                logger.debug(f"Cache hit for routing: {cached_result.routing_time_ms:.2f}ms")
                return cached_result
            
            # Get available agents
            available_agents = await self.agent_registry.list_active_agents()
            if not available_agents:
                raise Exception("No active agents available")
            
            # Handle preferred agent
            if preferred_agent_id:
                preferred_agent = await self.agent_registry.get_agent(preferred_agent_id)
                if preferred_agent and preferred_agent.status.value == "active":
                    routing_time = (time.time() - routing_start) * 1000
                    result = RoutingResult(
                        selected_agent_id=preferred_agent_id,
                        confidence=1.0,
                        routing_time_ms=routing_time,
                        strategy_used=RoutingStrategy.KEYWORD_MATCHING,
                        decision_type=RoutingDecision.DIRECT_MATCH,
                        routing_factors={"preferred_agent": True}
                    )
                    self._update_routing_stats(result)
                    return result
            
            # Analyze context
            context_analysis = self.context_analyzer.analyze_context(context)
            
            # Execute routing strategy
            if self.routing_strategy == RoutingStrategy.HYBRID_INTELLIGENT:
                result = await self._hybrid_intelligent_routing(
                    query, context, context_analysis, available_agents
                )
            elif self.routing_strategy == RoutingStrategy.KEYWORD_MATCHING:
                result = await self._keyword_based_routing(
                    query, context, available_agents
                )
            elif self.routing_strategy == RoutingStrategy.SEMANTIC_SIMILARITY:
                result = await self._semantic_based_routing(
                    query, context, available_agents
                )
            elif self.routing_strategy == RoutingStrategy.ML_CLASSIFICATION:
                result = await self._ml_based_routing(
                    query, context, available_agents
                )
            else:
                result = await self._context_aware_routing(
                    query, context, context_analysis, available_agents
                )
            
            # Calculate final routing time
            result.routing_time_ms = (time.time() - routing_start) * 1000
            
            # Update agent load
            await self._update_agent_load(result.selected_agent_id, increment=True)
            
            # Cache result
            self._cache_routing_result(cache_key, result)
            
            # Update statistics
            self._update_routing_stats(result)
            
            # Log performance
            if result.routing_time_ms > 15.0:
                logger.warning(f"⚠️ Routing exceeded target: {result.routing_time_ms:.2f}ms > 15ms")
            else:
                logger.debug(f"✅ Routed query in {result.routing_time_ms:.2f}ms to {result.selected_agent_id}")
            
            return result
            
        except Exception as e:
            routing_time = (time.time() - routing_start) * 1000
            logger.error(f"❌ Routing error: {e}")
            
            # Fallback to first available agent
            available_agents = await self.agent_registry.list_active_agents()
            fallback_agent_id = available_agents[0].agent_id if available_agents else "unknown"
            
            result = RoutingResult(
                selected_agent_id=fallback_agent_id,
                confidence=0.3,
                routing_time_ms=routing_time,
                strategy_used=RoutingStrategy.KEYWORD_MATCHING,
                decision_type=RoutingDecision.FALLBACK,
                metadata={"error": str(e)}
            )
            
            self._update_routing_stats(result)
            return result
    
    async def _hybrid_intelligent_routing(
        self,
        query: str,
        context: Dict[str, Any],
        context_analysis: Dict[str, Any],
        available_agents: List[BaseAgent]
    ) -> RoutingResult:
        """Execute hybrid intelligent routing combining multiple strategies."""
        
        agent_scores = {}
        routing_factors = {}
        
        # 1. Keyword matching (40% weight)
        keyword_scores = {}
        for agent in available_agents:
            score = self.keyword_matcher.calculate_match_score(query, agent.agent_id, context)
            keyword_scores[agent.agent_id] = score
        
        routing_factors["keyword_scores"] = keyword_scores
        
        # 2. Semantic similarity (30% weight)
        semantic_scores = {}
        for agent in available_agents:
            score = await self.semantic_matcher.calculate_similarity_score(
                query, agent.agent_id, context
            )
            semantic_scores[agent.agent_id] = score
        
        routing_factors["semantic_scores"] = semantic_scores
        
        # 3. ML classification (20% weight)
        ml_scores = {}
        if self.enable_ml_routing:
            ml_scores = self.ml_classifier.classify_query(query, context)
        
        routing_factors["ml_scores"] = ml_scores
        
        # 4. Performance and load balancing (10% weight)
        performance_scores = {}
        for agent in available_agents:
            profile = self.agent_profiles.get(agent.agent_id)
            if profile:
                # Consider success rate, response time, and current load
                load_factor = 1.0 - (profile.current_load / max(profile.max_concurrent, 1))
                perf_score = (profile.success_rate * 0.6 + 
                             (1.0 - min(profile.average_response_time_ms / 1000.0, 1.0)) * 0.4)
                performance_scores[agent.agent_id] = perf_score * load_factor
            else:
                performance_scores[agent.agent_id] = 0.5  # Default
        
        routing_factors["performance_scores"] = performance_scores
        
        # 5. Context continuity boost
        continuity_boost = {}
        preferred_agent = context_analysis.get("continuity_preference")
        for agent in available_agents:
            if agent.agent_id == preferred_agent:
                continuity_boost[agent.agent_id] = 0.3
            else:
                continuity_boost[agent.agent_id] = 0.0
        
        routing_factors["continuity_boost"] = continuity_boost
        
        # Combine scores with weights
        weights = {
            "keyword": 0.4,
            "semantic": 0.3,
            "ml": 0.2,
            "performance": 0.1
        }
        
        for agent in available_agents:
            agent_id = agent.agent_id
            
            combined_score = (
                weights["keyword"] * keyword_scores.get(agent_id, 0.0) +
                weights["semantic"] * semantic_scores.get(agent_id, 0.0) +
                weights["ml"] * ml_scores.get(agent_id, 0.0) +
                weights["performance"] * performance_scores.get(agent_id, 0.0) +
                continuity_boost.get(agent_id, 0.0)
            )
            
            agent_scores[agent_id] = combined_score
        
        # Select best agent
        best_agent_id = max(agent_scores.keys(), key=lambda k: agent_scores[k])
        confidence = agent_scores[best_agent_id]
        
        # Create alternatives list
        alternatives = [
            (agent_id, score) for agent_id, score in sorted(
                agent_scores.items(), key=lambda x: x[1], reverse=True
            ) if agent_id != best_agent_id
        ][:3]  # Top 3 alternatives
        
        # Determine decision type
        if confidence >= self.confidence_threshold:
            decision_type = RoutingDecision.CONFIDENCE_BASED
        elif confidence >= self.fallback_threshold:
            decision_type = RoutingDecision.CONFIDENCE_BASED
        else:
            decision_type = RoutingDecision.FALLBACK
        
        return RoutingResult(
            selected_agent_id=best_agent_id,
            confidence=confidence,
            routing_time_ms=0.0,  # Will be set by caller
            strategy_used=RoutingStrategy.HYBRID_INTELLIGENT,
            decision_type=decision_type,
            alternatives=alternatives,
            routing_factors=routing_factors
        )
    
    async def _keyword_based_routing(
        self,
        query: str,
        context: Dict[str, Any],
        available_agents: List[BaseAgent]
    ) -> RoutingResult:
        """Execute keyword-based routing."""
        agent_scores = {}
        
        for agent in available_agents:
            score = self.keyword_matcher.calculate_match_score(query, agent.agent_id, context)
            agent_scores[agent.agent_id] = score
        
        best_agent_id = max(agent_scores.keys(), key=lambda k: agent_scores[k])
        confidence = agent_scores[best_agent_id]
        
        alternatives = [
            (agent_id, score) for agent_id, score in sorted(
                agent_scores.items(), key=lambda x: x[1], reverse=True
            ) if agent_id != best_agent_id
        ][:2]
        
        return RoutingResult(
            selected_agent_id=best_agent_id,
            confidence=confidence,
            routing_time_ms=0.0,
            strategy_used=RoutingStrategy.KEYWORD_MATCHING,
            decision_type=RoutingDecision.DIRECT_MATCH if confidence > 0.8 else RoutingDecision.CONFIDENCE_BASED,
            alternatives=alternatives,
            routing_factors={"keyword_scores": agent_scores}
        )
    
    async def _semantic_based_routing(
        self,
        query: str,
        context: Dict[str, Any],
        available_agents: List[BaseAgent]
    ) -> RoutingResult:
        """Execute semantic similarity-based routing."""
        agent_scores = {}
        
        for agent in available_agents:
            score = await self.semantic_matcher.calculate_similarity_score(
                query, agent.agent_id, context
            )
            agent_scores[agent.agent_id] = score
        
        best_agent_id = max(agent_scores.keys(), key=lambda k: agent_scores[k])
        confidence = agent_scores[best_agent_id]
        
        alternatives = [
            (agent_id, score) for agent_id, score in sorted(
                agent_scores.items(), key=lambda x: x[1], reverse=True
            ) if agent_id != best_agent_id
        ][:2]
        
        return RoutingResult(
            selected_agent_id=best_agent_id,
            confidence=confidence,
            routing_time_ms=0.0,
            strategy_used=RoutingStrategy.SEMANTIC_SIMILARITY,
            decision_type=RoutingDecision.CONFIDENCE_BASED,
            alternatives=alternatives,
            routing_factors={"semantic_scores": agent_scores}
        )
    
    async def _ml_based_routing(
        self,
        query: str,
        context: Dict[str, Any],
        available_agents: List[BaseAgent]
    ) -> RoutingResult:
        """Execute ML classification-based routing."""
        ml_scores = self.ml_classifier.classify_query(query, context)
        
        # Filter scores for available agents
        agent_scores = {
            agent.agent_id: ml_scores.get(agent.agent_id, 0.0)
            for agent in available_agents
        }
        
        best_agent_id = max(agent_scores.keys(), key=lambda k: agent_scores[k])
        confidence = agent_scores[best_agent_id]
        
        alternatives = [
            (agent_id, score) for agent_id, score in sorted(
                agent_scores.items(), key=lambda x: x[1], reverse=True
            ) if agent_id != best_agent_id
        ][:2]
        
        return RoutingResult(
            selected_agent_id=best_agent_id,
            confidence=confidence,
            routing_time_ms=0.0,
            strategy_used=RoutingStrategy.ML_CLASSIFICATION,
            decision_type=RoutingDecision.CONFIDENCE_BASED,
            alternatives=alternatives,
            routing_factors={"ml_scores": agent_scores}
        )
    
    async def _context_aware_routing(
        self,
        query: str,
        context: Dict[str, Any],
        context_analysis: Dict[str, Any],
        available_agents: List[BaseAgent]
    ) -> RoutingResult:
        """Execute context-aware routing."""
        # Prefer continuity if applicable
        preferred_agent = context_analysis.get("continuity_preference")
        
        if preferred_agent:
            for agent in available_agents:
                if agent.agent_id == preferred_agent:
                    return RoutingResult(
                        selected_agent_id=preferred_agent,
                        confidence=0.9,
                        routing_time_ms=0.0,
                        strategy_used=RoutingStrategy.CONTEXT_AWARE,
                        decision_type=RoutingDecision.DIRECT_MATCH,
                        routing_factors={"context_continuity": True}
                    )
        
        # Fall back to keyword matching
        return await self._keyword_based_routing(query, context, available_agents)
    
    async def _initialize_agent_profiles(self):
        """Initialize performance profiles for all agents."""
        agents = await self.agent_registry.list_active_agents()
        
        for agent in agents:
            self.agent_profiles[agent.agent_id] = AgentPerformanceProfile(
                agent_id=agent.agent_id,
                success_rate=0.95,  # Default values
                average_response_time_ms=200.0,
                average_confidence=0.8,
                specialization_score=0.9,
                current_load=0,
                max_concurrent=10
            )
    
    async def _background_profile_updates(self):
        """Background task to update agent performance profiles."""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                agents = await self.agent_registry.list_active_agents()
                
                for agent in agents:
                    stats = agent.get_stats()
                    
                    if agent.agent_id in self.agent_profiles:
                        profile = self.agent_profiles[agent.agent_id]
                        
                        # Update performance metrics
                        total_queries = stats.total_queries
                        if total_queries > 0:
                            profile.success_rate = stats.successful_responses / total_queries
                        
                        profile.average_response_time_ms = stats.average_response_time_ms
                        profile.average_confidence = stats.average_confidence
                        profile.last_updated = time.time()
                
            except Exception as e:
                logger.error(f"Error updating agent profiles: {e}")
                await asyncio.sleep(60)  # Wait longer before retrying
    
    async def _update_agent_load(self, agent_id: str, increment: bool = True):
        """Update agent load tracking."""
        if agent_id in self.agent_profiles:
            profile = self.agent_profiles[agent_id]
            if increment:
                profile.current_load = min(profile.current_load + 1, profile.max_concurrent)
            else:
                profile.current_load = max(profile.current_load - 1, 0)
    
    def _generate_cache_key(
        self,
        query: str,
        context: Dict[str, Any],
        preferred_agent_id: Optional[str]
    ) -> str:
        """Generate cache key for routing decision."""
        # Simplified cache key based on query hash and key context elements
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        context_hash = hashlib.md5(str(sorted(context.items())).encode()).hexdigest()[:8]
        preferred_hash = hashlib.md5((preferred_agent_id or "").encode()).hexdigest()[:4]
        
        return f"{query_hash}:{context_hash}:{preferred_hash}"
    
    def _get_cached_routing(self, cache_key: str) -> Optional[RoutingResult]:
        """Get cached routing result."""
        if cache_key in self.routing_cache:
            entry = self.routing_cache[cache_key]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                return entry["result"]
            else:
                del self.routing_cache[cache_key]
        return None
    
    def _cache_routing_result(self, cache_key: str, result: RoutingResult):
        """Cache routing result."""
        self.routing_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # Manage cache size
        if len(self.routing_cache) > 1000:
            # Remove oldest entry
            oldest_key = min(
                self.routing_cache.keys(),
                key=lambda k: self.routing_cache[k]["timestamp"]
            )
            del self.routing_cache[oldest_key]
    
    def _update_routing_stats(self, result: RoutingResult):
        """Update routing statistics."""
        self.routing_stats["total_routes"] += 1
        
        if result.confidence >= self.confidence_threshold:
            self.routing_stats["successful_routes"] += 1
        
        if result.decision_type == RoutingDecision.FALLBACK:
            self.routing_stats["fallback_routes"] += 1
        
        # Update average routing time
        total_routes = self.routing_stats["total_routes"]
        current_avg = self.routing_stats["average_routing_time_ms"]
        
        self.routing_stats["average_routing_time_ms"] = (
            (current_avg * (total_routes - 1) + result.routing_time_ms) / total_routes
        )
        
        # Update strategy usage
        self.routing_stats["strategy_usage"][result.strategy_used.value] += 1
        
        # Update agent selection counts
        self.routing_stats["agent_selection_counts"][result.selected_agent_id] += 1
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total_routes = self.routing_stats["total_routes"]
        
        return {
            **self.routing_stats,
            "success_rate": (
                self.routing_stats["successful_routes"] / max(total_routes, 1)
            ) * 100,
            "fallback_rate": (
                self.routing_stats["fallback_routes"] / max(total_routes, 1)
            ) * 100,
            "agent_distribution": dict(self.routing_stats["agent_selection_counts"]),
            "strategy_distribution": dict(self.routing_stats["strategy_usage"]),
            "performance_target_met": self.routing_stats["average_routing_time_ms"] <= 15.0
        }
    
    async def shutdown(self):
        """Shutdown the router."""
        logger.info("Shutting down Intelligent Agent Router...")
        
        # Clear caches
        self.routing_cache.clear()
        self.agent_profiles.clear()
        
        self.initialized = False
        logger.info("✅ Intelligent Agent Router shutdown complete")