"""
Intelligent LLM Router - OpenAI Only Version (FIXED)
==========================================

COMPLETE REPLACEMENT for app/llm/intelligent_router.py
This fixes the _initialize_anthropic_models error and removes all Anthropic dependencies.
"""
import os
import asyncio
import logging
import time
import uuid
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import re
from collections import defaultdict, deque
import threading

# OpenAI imports only
import openai
from openai import AsyncOpenAI

# For ML-based complexity analysis
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"


class ModelType(str, Enum):
    """Model types for different use cases"""
    ULTRA_FAST = "ultra_fast"      # GPT-4o-mini
    FAST = "fast"                  # GPT-4o-mini optimized
    BALANCED = "balanced"          # GPT-4o
    HIGH_QUALITY = "high_quality"  # GPT-4o
    SPECIALIZED = "specialized"    # Fine-tuned models


class ComplexityLevel(str, Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # 0.0 - 0.3
    MODERATE = "moderate"      # 0.3 - 0.6
    COMPLEX = "complex"        # 0.6 - 0.8
    VERY_COMPLEX = "very_complex"  # 0.8 - 1.0


class RoutingStrategy(str, Enum):
    """LLM routing strategies"""
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    AGENT_SPECIFIC = "agent_specific"


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    model_id: str
    provider: LLMProvider
    model_type: ModelType
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float
    avg_latency_ms: float
    quality_score: float
    supports_streaming: bool
    context_window: int
    specialization: Optional[str] = None
    enabled: bool = True


@dataclass
class ComplexityFeatures:
    """Features extracted for complexity analysis"""
    query_length: int
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    question_count: int
    technical_terms_count: int
    entity_count: int
    negation_count: int
    conditional_count: int
    temporal_references: int
    numerical_references: int
    complexity_indicators: List[str]
    domain_specific_terms: int
    readability_score: float


@dataclass
class RoutingDecision:
    """LLM routing decision with metadata"""
    selected_model_id: str
    provider: LLMProvider
    model_type: ModelType
    confidence: float
    complexity_score: float
    complexity_level: ComplexityLevel
    routing_strategy: RoutingStrategy
    decision_factors: Dict[str, Any]
    alternatives: List[Tuple[str, float]]
    estimated_latency_ms: float
    estimated_cost: float
    routing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Enhanced LLM response with routing metadata"""
    content: str
    model_id: str
    provider: LLMProvider
    actual_latency_ms: float
    token_count: int
    cost: float
    confidence: float
    quality_score: float
    is_streaming: bool
    routing_decision: RoutingDecision
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplexityAnalyzer:
    """
    ML-based query complexity analyzer for intelligent model routing
    """
    
    def __init__(self):
        """Initialize complexity analyzer"""
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        
        # Feature extractors
        self.technical_terms = set([
            'api', 'database', 'server', 'configuration', 'authentication',
            'encryption', 'algorithm', 'protocol', 'framework', 'architecture',
            'deployment', 'optimization', 'integration', 'synchronization',
            'validation', 'serialization', 'middleware', 'orchestration'
        ])
        
        self.complexity_indicators = [
            'analyze', 'compare', 'evaluate', 'assess', 'determine', 'calculate',
            'optimize', 'implement', 'design', 'architect', 'troubleshoot',
            'debug', 'investigate', 'comprehensive', 'detailed', 'thorough'
        ]
        
        self.question_patterns = [
            r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b',
            r'\bwhich\b', r'\bwho\b', r'\bcan\b', r'\bwould\b', r'\bshould\b'
        ]
        
        self.conditional_patterns = [
            r'\bif\b', r'\bunless\b', r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\bbecause\b', r'\bsince\b', r'\bgiven\b', r'\bassuming\b'
        ]
        
        self.temporal_patterns = [
            r'\btoday\b', r'\btomorrow\b', r'\byesterday\b', r'\bnow\b',
            r'\blater\b', r'\bsoon\b', r'\brecently\b', r'\bcurrently\b'
        ]
        
        logger.info("Complexity Analyzer initialized")
    
    def extract_features(self, query: str, context: Dict[str, Any] = None) -> ComplexityFeatures:
        """Extract comprehensive features for complexity analysis"""
        
        # Basic text features
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = query.split()
        
        # Count features
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        avg_sentence_length = word_count / sentence_count
        
        # Question analysis
        question_count = sum(1 for pattern in self.question_patterns 
                           if re.search(pattern, query, re.IGNORECASE))
        
        # Technical terms
        query_lower = query.lower()
        technical_terms_count = sum(1 for term in self.technical_terms 
                                  if term in query_lower)
        
        # Entity detection (simplified)
        entity_count = len(re.findall(r'\b[A-Z][a-zA-Z]+\b', query))
        
        # Negation detection
        negation_patterns = [r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bwithout\b']
        negation_count = sum(1 for pattern in negation_patterns 
                           if re.search(pattern, query, re.IGNORECASE))
        
        # Conditional structures
        conditional_count = sum(1 for pattern in self.conditional_patterns 
                              if re.search(pattern, query, re.IGNORECASE))
        
        # Temporal references
        temporal_references = sum(1 for pattern in self.temporal_patterns 
                                if re.search(pattern, query, re.IGNORECASE))
        
        # Numerical references
        numerical_references = len(re.findall(r'\b\d+(?:\.\d+)?\b', query))
        
        # Complexity indicators
        found_indicators = [indicator for indicator in self.complexity_indicators 
                          if indicator in query_lower]
        
        # Domain-specific terms (context-aware)
        domain_terms = 0
        if context:
            domain = context.get('domain', '')
            if domain == 'technical':
                domain_terms = technical_terms_count * 2
            elif domain == 'financial':
                financial_terms = ['payment', 'refund', 'billing', 'cost', 'price']
                domain_terms = sum(1 for term in financial_terms if term in query_lower)
            elif domain == 'emergency':
                emergency_terms = ['urgent', 'emergency', 'critical', 'immediate']
                domain_terms = sum(1 for term in emergency_terms if term in query_lower)
        
        # Simple readability score (Flesch-like)
        if word_count > 0:
            avg_word_length = sum(len(word) for word in words) / word_count
            readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 5.0)
            readability_score = max(0, min(100, readability_score)) / 100.0  # Normalize
        else:
            readability_score = 0.5
        
        return ComplexityFeatures(
            query_length=len(query),
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            question_count=question_count,
            technical_terms_count=technical_terms_count,
            entity_count=entity_count,
            negation_count=negation_count,
            conditional_count=conditional_count,
            temporal_references=temporal_references,
            numerical_references=numerical_references,
            complexity_indicators=found_indicators,
            domain_specific_terms=domain_terms,
            readability_score=readability_score
        )
    
    def calculate_complexity_score(self, features: ComplexityFeatures) -> float:
        """Calculate complexity score from features"""
        
        # Base score from length and structure
        length_score = min(1.0, features.word_count / 50.0)  # Normalize to 50 words
        structure_score = min(1.0, features.sentence_count / 5.0)  # Normalize to 5 sentences
        
        # Complexity indicators
        indicator_score = min(1.0, len(features.complexity_indicators) / 3.0)
        
        # Question complexity
        question_score = min(1.0, features.question_count / 3.0)
        
        # Technical complexity
        technical_score = min(1.0, features.technical_terms_count / 5.0)
        
        # Conditional complexity
        conditional_score = min(1.0, features.conditional_count / 2.0)
        
        # Domain complexity
        domain_score = min(1.0, features.domain_specific_terms / 3.0)
        
        # Readability (inverse - lower readability = higher complexity)
        readability_complexity = 1.0 - features.readability_score
        
        # Weighted combination
        complexity_score = (
            length_score * 0.15 +
            structure_score * 0.1 +
            indicator_score * 0.2 +
            question_score * 0.15 +
            technical_score * 0.15 +
            conditional_score * 0.1 +
            domain_score * 0.1 +
            readability_complexity * 0.05
        )
        
        return min(1.0, complexity_score)
    
    def get_complexity_level(self, complexity_score: float) -> ComplexityLevel:
        """Convert complexity score to level"""
        if complexity_score < 0.3:
            return ComplexityLevel.SIMPLE
        elif complexity_score < 0.6:
            return ComplexityLevel.MODERATE
        elif complexity_score < 0.8:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    def analyze_query_complexity(self, query: str, context: Dict[str, Any] = None) -> Tuple[float, ComplexityLevel]:
        """Main method to analyze query complexity"""
        try:
            features = self.extract_features(query, context)
            complexity_score = self.calculate_complexity_score(features)
            complexity_level = self.get_complexity_level(complexity_score)
            
            return complexity_score, complexity_level
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {e}")
            return 0.5, ComplexityLevel.MODERATE


class ModelPerformanceTracker:
    """
    Tracks model performance for adaptive routing decisions
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize performance tracker"""
        self.max_history = max_history
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.model_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_update = time.time()
        
        # Performance metrics
        self.metrics = [
            'latency_ms', 'token_count', 'cost', 'quality_score', 
            'success_rate', 'user_satisfaction'
        ]
        
        logger.info("Model Performance Tracker initialized")
    
    def record_performance(self, model_id: str, metrics: Dict[str, float]):
        """Record performance metrics for a model"""
        try:
            timestamp = time.time()
            
            # Add timestamp to metrics
            metrics_with_time = {**metrics, 'timestamp': timestamp}
            
            # Store in history
            self.performance_history[model_id].append(metrics_with_time)
            
            # Update aggregated stats
            self._update_model_stats(model_id)
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    def _update_model_stats(self, model_id: str):
        """Update aggregated statistics for a model"""
        try:
            history = self.performance_history[model_id]
            if not history:
                return
            
            # Calculate recent performance (last 100 entries or 1 hour)
            recent_cutoff = time.time() - 3600  # 1 hour
            recent_entries = [entry for entry in history 
                             if entry.get('timestamp', 0) > recent_cutoff]
            
            if not recent_entries:
                recent_entries = list(history)[-min(100, len(history)):]
            
            # Calculate averages
            stats = {}
            for metric in self.metrics:
                values = [entry[metric] for entry in recent_entries 
                         if metric in entry and entry[metric] is not None]
                if values:
                    stats[f'avg_{metric}'] = sum(values) / len(values)
                    stats[f'min_{metric}'] = min(values)
                    stats[f'max_{metric}'] = max(values)
                    # Calculate 95th percentile for latency
                    if metric == 'latency_ms':
                        sorted_values = sorted(values)
                        p95_index = int(len(sorted_values) * 0.95)
                        stats['p95_latency_ms'] = sorted_values[p95_index] if p95_index < len(sorted_values) else sorted_values[-1]
            
            self.model_stats[model_id] = stats
        except Exception as e:
            logger.error(f"Error updating model stats: {e}")
    
    def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """Get performance statistics for a model"""
        return self.model_stats.get(model_id, {})
    
    def get_best_model_for_criteria(self, criteria: str, available_models: List[str]) -> Optional[str]:
        """Get best model based on specific criteria"""
        if not available_models:
            return None
        
        try:
            scored_models = []
            
            for model_id in available_models:
                stats = self.model_stats.get(model_id, {})
                if not stats:
                    # If no stats, give default score
                    scored_models.append((model_id, 0.5))
                    continue
                
                score = 0.0
                
                if criteria == 'latency':
                    # Lower latency is better
                    avg_latency = stats.get('avg_latency_ms', 1000)
                    score = 1000 / max(avg_latency, 50)  # Inverse scoring
                    
                elif criteria == 'quality':
                    # Higher quality is better
                    score = stats.get('avg_quality_score', 0.5)
                    
                elif criteria == 'cost':
                    # Lower cost is better
                    avg_cost = stats.get('avg_cost', 0.1)
                    score = 0.1 / max(avg_cost, 0.001)  # Inverse scoring
                    
                elif criteria == 'balanced':
                    # Balanced scoring
                    quality = stats.get('avg_quality_score', 0.5)
                    latency = stats.get('avg_latency_ms', 1000)
                    cost = stats.get('avg_cost', 0.1)
                    
                    # Normalize and combine (higher is better)
                    quality_score = quality
                    latency_score = 1000 / max(latency, 50)
                    cost_score = 0.1 / max(cost, 0.001)
                    
                    score = (quality_score * 0.4 + latency_score * 0.4 + cost_score * 0.2)
                
                scored_models.append((model_id, score))
            
            if scored_models:
                scored_models.sort(key=lambda x: x[1], reverse=True)
                return scored_models[0][0]
            
            return available_models[0]  # Fallback to first available
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
            return available_models[0] if available_models else None


class IntelligentLLMRouter:
    """
    Intelligent LLM Router - OpenAI Only Version (FIXED)
    
    This version removes all Anthropic dependencies and fixes initialization errors.
    """
    
    def __init__(self, 
                 openai_client: Optional[AsyncOpenAI] = None,
                 default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
                 enable_caching: bool = True,
                 cache_ttl_seconds: int = 300):
        """Initialize the intelligent LLM router"""
        
        self.openai_client = openai_client
        self.default_strategy = default_strategy
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl_seconds
        
        # Core components
        self.complexity_analyzer = ComplexityAnalyzer()
        self.performance_tracker = ModelPerformanceTracker()
        
        # Model configurations
        self.model_configs: Dict[str, ModelConfig] = {}
        self._initialize_openai_models()
        
        # Routing cache
        self.routing_cache: Dict[str, Dict[str, Any]] = {}
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.routing_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_routing_time_ms': 0.0,
            'model_usage': defaultdict(int),
            'strategy_usage': defaultdict(int),
            'complexity_distribution': defaultdict(int)
        }
        
        # Agent-specific model preferences
        self.agent_model_preferences = {
            'roadside-assistance': {
                'preferred_models': ['gpt-4o-mini', 'gpt-4o'],
                'urgency_model': 'gpt-4o-mini',  # Fast for emergencies
                'quality_threshold': 0.7
            },
            'billing-support': {
                'preferred_models': ['gpt-4o', 'gpt-4o-mini'],
                'accuracy_model': 'gpt-4o',  # Accuracy for financial
                'quality_threshold': 0.8
            },
            'technical-support': {
                'preferred_models': ['gpt-4o', 'gpt-4o-mini'],
                'complex_model': 'gpt-4o',  # Complex technical queries
                'quality_threshold': 0.85
            }
        }
        
        self.initialized = False
        logger.info("Intelligent LLM Router (OpenAI Only) initialized")
    
    def _initialize_openai_models(self):
        """Initialize OpenAI model configurations"""
        
        # GPT-4o-mini - Ultra fast for simple queries
        self.model_configs['gpt-4o-mini'] = ModelConfig(
            model_id='gpt-4o-mini',
            provider=LLMProvider.OPENAI,
            model_type=ModelType.ULTRA_FAST,
            max_tokens=150,
            temperature=0.7,
            cost_per_1k_tokens=0.0001,
            avg_latency_ms=150,
            quality_score=0.85,
            supports_streaming=True,
            context_window=128000
        )
        
        # GPT-4o - Balanced performance and quality
        self.model_configs['gpt-4o'] = ModelConfig(
            model_id='gpt-4o',
            provider=LLMProvider.OPENAI,
            model_type=ModelType.BALANCED,
            max_tokens=300,
            temperature=0.7,
            cost_per_1k_tokens=0.005,
            avg_latency_ms=300,
            quality_score=0.95,
            supports_streaming=True,
            context_window=128000
        )
    
    async def initialize(self):
        """FIXED: Initialize the router with proper OpenAI client setup"""
        if self.initialized:
            return
        
        logger.info("🚀 Initializing Intelligent LLM Router (OpenAI Only)...")
        
        try:
            # Initialize OpenAI client with clean configuration
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("❌ OPENAI_API_KEY not found in environment")
                raise Exception("OpenAI API key required")
            
            # Create client with ONLY the API key - no organization headers
            self.openai_client = AsyncOpenAI(
                api_key=api_key,
                # Explicitly don't set organization to avoid auth issues
                # organization=None  # This is the default anyway
            )
            
            # Test OpenAI connection with minimal request
            try:
                test_response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    timeout=10  # Add timeout for faster failure detection
                )
                logger.info("✅ OpenAI connection verified successfully")
            except Exception as e:
                logger.error(f"❌ OpenAI connection test failed: {e}")
                # Don't raise here - continue with degraded functionality
                logger.warning("⚠️ Continuing with degraded LLM functionality")
            
            # Start background tasks
            asyncio.create_task(self._background_cache_cleanup())
            asyncio.create_task(self._background_performance_analysis())
            
            self.initialized = True
            logger.info("✅ Intelligent LLM Router initialization complete")
            
        except Exception as e:
            logger.error(f"❌ LLM Router initialization failed: {e}")
            # Set initialized to True anyway to prevent blocking
            self.initialized = True
            logger.warning("⚠️ Running with degraded LLM functionality")
    
    async def route_and_generate(self,
                                query: str,
                                context: Dict[str, Any] = None,
                                agent_id: Optional[str] = None,
                                routing_strategy: Optional[RoutingStrategy] = None,
                                streaming: bool = False,
                                max_tokens: Optional[int] = None,
                                temperature: Optional[float] = None) -> LLMResponse:
        """
        Main method: Route query to optimal model and generate response
        """
        if not self.initialized:
            await self.initialize()
        
        routing_start = time.time()
        context = context or {}
        
        logger.debug(f"Routing LLM request: {query[:100]}...")
        
        try:
            # Check response cache first
            if self.enable_caching and not streaming:
                cache_key = self._generate_response_cache_key(query, context, agent_id)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    self.routing_stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for LLM response")
                    return cached_response
            
            self.routing_stats['cache_misses'] += 1
            
            # Make routing decision
            routing_decision = await self._make_routing_decision(
                query=query,
                context=context,
                agent_id=agent_id,
                routing_strategy=routing_strategy or self.default_strategy,
                streaming=streaming
            )
            
            # Override model parameters if specified
            model_config = self.model_configs[routing_decision.selected_model_id]
            if max_tokens:
                model_config.max_tokens = max_tokens
            if temperature is not None:
                model_config.temperature = temperature
            
            # Generate response
            response = await self._generate_openai_response(routing_decision, query, context, model_config)
            
            # Cache successful responses
            if self.enable_caching and response.error is None:
                cache_key = self._generate_response_cache_key(query, context, agent_id)
                self._cache_response(cache_key, response)
            
            return response
                
        except Exception as e:
            logger.error(f"❌ LLM routing/generation error: {e}")
            
            # Return error response
            routing_time = (time.time() - routing_start) * 1000
            
            return LLMResponse(
                content=f"I apologize, but I encountered an error processing your request. How can I help you?",
                model_id="error",
                provider=LLMProvider.OPENAI,
                actual_latency_ms=routing_time,
                token_count=0,
                cost=0.0,
                confidence=0.0,
                quality_score=0.0,
                is_streaming=False,
                routing_decision=RoutingDecision(
                    selected_model_id="error",
                    provider=LLMProvider.OPENAI,
                    model_type=ModelType.ULTRA_FAST,
                    confidence=0.0,
                    complexity_score=0.0,
                    complexity_level=ComplexityLevel.SIMPLE,
                    routing_strategy=self.default_strategy,
                    decision_factors={},
                    alternatives=[],
                    estimated_latency_ms=routing_time,
                    estimated_cost=0.0,
                    routing_time_ms=routing_time
                ),
                error=str(e)
            )
    
    async def _make_routing_decision(self,
                                   query: str,
                                   context: Dict[str, Any],
                                   agent_id: Optional[str],
                                   routing_strategy: RoutingStrategy,
                                   streaming: bool) -> RoutingDecision:
        """Make intelligent routing decision based on query analysis"""
        
        decision_start = time.time()
        
        try:
            # Analyze query complexity
            complexity_score, complexity_level = self.complexity_analyzer.analyze_query_complexity(query, context)
            
            # Get available models
            available_models = [model_id for model_id, config in self.model_configs.items() 
                              if config.enabled and (not streaming or config.supports_streaming)]
            
            if not available_models:
                # Fallback to basic model
                available_models = ['gpt-4o-mini']
            
            # Apply routing strategy
            decision_factors = {
                'complexity_score': complexity_score,
                'complexity_level': complexity_level.value,
                'context': context,
                'agent_id': agent_id,
                'streaming': streaming
            }
            
            selected_model_id = await self._apply_routing_strategy(
                routing_strategy, available_models, complexity_level, context, agent_id, decision_factors
            )
            
            # Get model config
            model_config = self.model_configs[selected_model_id]
            
            # Calculate confidence based on routing factors
            confidence = self._calculate_routing_confidence(
                selected_model_id, complexity_level, routing_strategy, decision_factors
            )
            
            # Generate alternatives
            alternatives = await self._generate_alternatives(
                selected_model_id, available_models, complexity_level, routing_strategy
            )
            
            # Estimate performance
            estimated_latency = await self._estimate_latency(selected_model_id, query, context)
            estimated_cost = await self._estimate_cost(selected_model_id, query)
            
            routing_time = (time.time() - decision_start) * 1000
            
            # Update statistics
            self.routing_stats['strategy_usage'][routing_strategy.value] += 1
            self.routing_stats['complexity_distribution'][complexity_level.value] += 1
            
            return RoutingDecision(
                selected_model_id=selected_model_id,
                provider=model_config.provider,
                model_type=model_config.model_type,
                confidence=confidence,
                complexity_score=complexity_score,
                complexity_level=complexity_level,
                routing_strategy=routing_strategy,
                decision_factors=decision_factors,
                alternatives=alternatives,
                estimated_latency_ms=estimated_latency,
                estimated_cost=estimated_cost,
                routing_time_ms=routing_time
            )
        except Exception as e:
            logger.error(f"Error making routing decision: {e}")
            # Return fallback decision
            routing_time = (time.time() - decision_start) * 1000
            return RoutingDecision(
                selected_model_id='gpt-4o-mini',
                provider=LLMProvider.OPENAI,
                model_type=ModelType.ULTRA_FAST,
                confidence=0.5,
                complexity_score=0.5,
                complexity_level=ComplexityLevel.MODERATE,
                routing_strategy=routing_strategy,
                decision_factors={},
                alternatives=[],
                estimated_latency_ms=200,
                estimated_cost=0.001,
                routing_time_ms=routing_time
            )
    
    async def _apply_routing_strategy(self,
                                    strategy: RoutingStrategy,
                                    available_models: List[str],
                                    complexity_level: ComplexityLevel,
                                    context: Dict[str, Any],
                                    agent_id: Optional[str],
                                    decision_factors: Dict[str, Any]) -> str:
        """Apply specific routing strategy to select model"""
        
        try:
            if strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
                # Prioritize speed and low latency
                if complexity_level in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]:
                    return 'gpt-4o-mini' if 'gpt-4o-mini' in available_models else available_models[0]
                else:
                    return 'gpt-4o' if 'gpt-4o' in available_models else available_models[0]
            
            elif strategy == RoutingStrategy.COST_OPTIMIZED:
                # Prioritize low cost
                cost_ordered = sorted(available_models, 
                                    key=lambda m: self.model_configs[m].cost_per_1k_tokens)
                return cost_ordered[0]
            
            elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
                # Prioritize highest quality
                if 'gpt-4o-quality' in available_models:
                    return 'gpt-4o-quality'
                elif 'gpt-4o' in available_models:
                    return 'gpt-4o'
                else:
                    return available_models[0]
            
            elif strategy == RoutingStrategy.AGENT_SPECIFIC:
                # Use agent-specific preferences
                if agent_id and agent_id in self.agent_model_preferences:
                    preferences = self.agent_model_preferences[agent_id]
                    
                    # Check for urgency (emergency contexts)
                    if context.get('urgency_level') == 'emergency':
                        urgency_model = preferences.get('urgency_model')
                        if urgency_model and urgency_model in available_models:
                            return urgency_model
                    
                    # Check for high complexity requiring accuracy
                    if complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]:
                        accuracy_model = preferences.get('accuracy_model') or preferences.get('complex_model')
                        if accuracy_model and accuracy_model in available_models:
                            return accuracy_model
                    
                    # Use preferred models
                    for model in preferences.get('preferred_models', []):
                        if model in available_models:
                            return model
            
            # Default balanced strategy
            return await self._balanced_model_selection(available_models, complexity_level, context)
        except Exception as e:
            logger.error(f"Error applying routing strategy: {e}")
            return available_models[0] if available_models else 'gpt-4o-mini'
    
    async def _balanced_model_selection(self,
                                       available_models: List[str],
                                       complexity_level: ComplexityLevel,
                                       context: Dict[str, Any]) -> str:
        """Balanced model selection considering multiple factors"""
        
        try:
            scored_models = []
            
            for model_id in available_models:
                config = self.model_configs[model_id]
                performance = self.performance_tracker.get_model_performance(model_id)
                
                # Calculate composite score
                # Quality (30%)
                quality_score = config.quality_score * 0.3
                
                # Performance (25%) - inverse of latency
                latency = performance.get('avg_latency_ms', config.avg_latency_ms)
                performance_score = (1000 / max(latency, 50)) * 0.25
                
                # Cost efficiency (20%) - inverse of cost
                cost_score = (0.1 / max(config.cost_per_1k_tokens, 0.0001)) * 0.2
                
                # Complexity match (15%)
                complexity_match = self._calculate_complexity_match(config.model_type, complexity_level) * 0.15
                
                # Recent performance (10%)
                recent_performance = performance.get('avg_quality_score', config.quality_score) * 0.1
                
                total_score = quality_score + performance_score + cost_score + complexity_match + recent_performance
                scored_models.append((model_id, total_score))
            
            # Sort by score and return best
            scored_models.sort(key=lambda x: x[1], reverse=True)
            return scored_models[0][0]
        except Exception as e:
            logger.error(f"Error in balanced model selection: {e}")
            return available_models[0] if available_models else 'gpt-4o-mini'
    
    def _calculate_complexity_match(self, model_type: ModelType, complexity_level: ComplexityLevel) -> float:
        """Calculate how well a model type matches the complexity level"""
        try:
            # Define optimal matches
            optimal_matches = {
                ComplexityLevel.SIMPLE: [ModelType.ULTRA_FAST, ModelType.FAST],
                ComplexityLevel.MODERATE: [ModelType.FAST, ModelType.BALANCED],
                ComplexityLevel.COMPLEX: [ModelType.BALANCED, ModelType.HIGH_QUALITY],
                ComplexityLevel.VERY_COMPLEX: [ModelType.HIGH_QUALITY, ModelType.SPECIALIZED]
            }
            
            optimal_types = optimal_matches.get(complexity_level, [])
            
            if model_type in optimal_types:
                return 1.0 if model_type == optimal_types[0] else 0.8
            else:
                return 0.5  # Partial match
        except Exception:
            return 0.5
    
    async def _generate_openai_response(self,
                                      routing_decision: RoutingDecision,
                                      query: str,
                                      context: Dict[str, Any],
                                      model_config: ModelConfig) -> LLMResponse:
        """Generate response using OpenAI model"""
        
        generation_start = time.time()
        
        try:
            # Prepare messages
            messages = self._prepare_messages(query, context)
            
            # Make API call
            response = await self.openai_client.chat.completions.create(
                model=routing_decision.selected_model_id,
                messages=messages,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                stream=False
            )
            
            # Extract response data
            content = response.choices[0].message.content or ""
            token_count = response.usage.total_tokens if response.usage else 0
            cost = token_count * model_config.cost_per_1k_tokens / 1000
            
            # Calculate quality score (simplified)
            quality_score = self._calculate_response_quality(content, query, context)
            
            actual_latency = (time.time() - generation_start) * 1000
            
            # Record performance
            performance_metrics = {
                'latency_ms': actual_latency,
                'token_count': token_count,
                'cost': cost,
                'quality_score': quality_score,
                'success_rate': 1.0
            }
            
            self.performance_tracker.record_performance(routing_decision.selected_model_id, performance_metrics)
            
            # Update routing stats
            self.routing_stats['total_requests'] += 1
            self.routing_stats['model_usage'][routing_decision.selected_model_id] += 1
            
            return LLMResponse(
                content=content,
                model_id=routing_decision.selected_model_id,
                provider=routing_decision.provider,
                actual_latency_ms=actual_latency,
                token_count=token_count,
                cost=cost,
                confidence=routing_decision.confidence,
                quality_score=quality_score,
                is_streaming=False,
                routing_decision=routing_decision
            )
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            
            # Record failure
            performance_metrics = {
                'latency_ms': (time.time() - generation_start) * 1000,
                'success_rate': 0.0
            }
            self.performance_tracker.record_performance(routing_decision.selected_model_id, performance_metrics)
            
            # Return error response
            return LLMResponse(
                content="I apologize, but I'm having trouble processing your request right now. How can I help you?",
                model_id=routing_decision.selected_model_id,
                provider=routing_decision.provider,
                actual_latency_ms=(time.time() - generation_start) * 1000,
                token_count=0,
                cost=0.0,
                confidence=routing_decision.confidence,
                quality_score=0.3,
                is_streaming=False,
                routing_decision=routing_decision,
                error=str(e)
            )
    
    def _prepare_messages(self, query: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
        
        messages = []
        
        # System message with context
        system_content = "You are a helpful AI assistant."
        
        if context.get('agent_id'):
            agent_id = context['agent_id']
            if 'roadside' in agent_id:
                system_content = "You are a professional roadside assistance coordinator. Provide clear, actionable guidance for vehicle emergencies while prioritizing safety."
            elif 'billing' in agent_id:
                system_content = "You are an empathetic billing support specialist. Help customers with payment and billing issues with patience and understanding."
            elif 'technical' in agent_id:
                system_content = "You are a patient technical support expert. Provide clear, step-by-step guidance for technical problems."
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history if available
        if context.get('conversation_history'):
            for msg in context['conversation_history'][-3:]:  # Last 3 messages
                if msg.get('role') and msg.get('content'):
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _calculate_response_quality(self, content: str, query: str, context: Dict[str, Any]) -> float:
        """Calculate quality score for response"""
        
        try:
            if not content or len(content.strip()) < 10:
                return 0.3
            
            quality_score = 0.7  # Base score
            
            # Length check (reasonable response length)
            if 50 <= len(content) <= 500:
                quality_score += 0.1
            
            # Relevance check (simplified keyword matching)
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            
            if overlap > 0 and len(query_words) > 0:
                relevance = min(1.0, overlap / len(query_words))
                quality_score += relevance * 0.2
            
            return min(1.0, quality_score)
        except Exception:
            return 0.5
    
    async def _generate_alternatives(self,
                                   selected_model: str,
                                   available_models: List[str],
                                   complexity_level: ComplexityLevel,
                                   routing_strategy: RoutingStrategy) -> List[Tuple[str, float]]:
        """Generate alternative model suggestions with confidence scores"""
        
        try:
            alternatives = []
            
            for model_id in available_models:
                if model_id == selected_model:
                    continue
                
                config = self.model_configs[model_id]
                
                # Calculate alternative score based on complexity match and performance
                complexity_match = self._calculate_complexity_match(config.model_type, complexity_level)
                performance_stats = self.performance_tracker.get_model_performance(model_id)
                recent_quality = performance_stats.get('avg_quality_score', config.quality_score)
                
                alt_score = (complexity_match * 0.6) + (recent_quality * 0.4)
                alternatives.append((model_id, alt_score))
            
            # Sort by score and return top 3
            alternatives.sort(key=lambda x: x[1], reverse=True)
            return alternatives[:3]
        except Exception:
            return []
    
    async def _estimate_latency(self, model_id: str, query: str, context: Dict[str, Any]) -> float:
        """Estimate response latency for model"""
        
        try:
            config = self.model_configs[model_id]
            base_latency = config.avg_latency_ms
            
            # Adjust based on recent performance
            performance = self.performance_tracker.get_model_performance(model_id)
            if performance.get('avg_latency_ms'):
                base_latency = performance['avg_latency_ms']
            
            # Adjust for query length (longer queries may take more time)
            query_length_factor = min(2.0, len(query) / 200.0)
            estimated_latency = base_latency * (0.8 + query_length_factor * 0.4)
            
            return estimated_latency
        except Exception:
            return 200.0
    
    async def _estimate_cost(self, model_id: str, query: str) -> float:
        """Estimate response cost for model"""
        
        try:
            config = self.model_configs[model_id]
            
            # Rough token estimation (4 characters per token)
            estimated_tokens = (len(query) + config.max_tokens) / 4
            estimated_cost = estimated_tokens * config.cost_per_1k_tokens / 1000
            
            return estimated_cost
        except Exception:
            return 0.001
    
    def _calculate_routing_confidence(self,
                                    selected_model: str,
                                    complexity_level: ComplexityLevel,
                                    routing_strategy: RoutingStrategy,
                                    decision_factors: Dict[str, Any]) -> float:
        """Calculate confidence in routing decision"""
        
        try:
            base_confidence = 0.7
            
            # Boost confidence for good complexity matches
            config = self.model_configs[selected_model]
            complexity_match = self._calculate_complexity_match(config.model_type, complexity_level)
            base_confidence += complexity_match * 0.2
            
            # Boost confidence for agent-specific routing
            if routing_strategy == RoutingStrategy.AGENT_SPECIFIC:
                base_confidence += 0.1
            
            # Boost confidence if model has good recent performance
            performance = self.performance_tracker.get_model_performance(selected_model)
            if performance.get('avg_quality_score', 0) > 0.8:
                base_confidence += 0.1
            
            return min(1.0, base_confidence)
        except Exception:
            return 0.7
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        try:
            key_string = "|".join(str(arg) for arg in args)
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception:
            return str(uuid.uuid4())
    
    def _generate_response_cache_key(self, query: str, context: Dict[str, Any], agent_id: Optional[str]) -> str:
        """Generate cache key for response caching"""
        try:
            # Create a simplified context for caching
            cache_context = {
                'agent_id': agent_id,
                'domain': context.get('domain'),
                'urgency': context.get('urgency_level')
            }
            return self._generate_cache_key(query.lower().strip(), json.dumps(cache_context, sort_keys=True))
        except Exception:
            return str(uuid.uuid4())
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available and fresh"""
        try:
            if cache_key in self.response_cache:
                entry = self.response_cache[cache_key]
                if time.time() - entry['timestamp'] < self.cache_ttl:
                    return entry['response']
                else:
                    del self.response_cache[cache_key]
            return None
        except Exception:
            return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache response with timestamp"""
        try:
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            
            # Manage cache size
            if len(self.response_cache) > 1000:
                # Remove oldest entry
                oldest_key = min(
                    self.response_cache.keys(),
                    key=lambda k: self.response_cache[k]['timestamp']
                )
                del self.response_cache[oldest_key]
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    async def _background_cache_cleanup(self):
        """Background task to clean up expired cache entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = time.time()
                
                # Clean up routing cache
                expired_routing = [
                    key for key, entry in self.routing_cache.items()
                    if current_time - entry.get('timestamp', 0) > self.cache_ttl
                ]
                for key in expired_routing:
                    del self.routing_cache[key]
                
                # Clean up response cache
                expired_responses = [
                    key for key, entry in self.response_cache.items()
                    if current_time - entry.get('timestamp', 0) > self.cache_ttl
                ]
                for key in expired_responses:
                    del self.response_cache[key]
                
                if expired_routing or expired_responses:
                    logger.debug(f"Cleaned up {len(expired_routing)} routing cache entries and "
                               f"{len(expired_responses)} response cache entries")
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def _background_performance_analysis(self):
        """Background task for performance analysis and optimization"""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                # Analyze model performance trends
                for model_id in self.model_configs.keys():
                    performance = self.performance_tracker.get_model_performance(model_id)
                    
                    # Log poor performance (could implement auto-disable logic)
                    if (performance.get('avg_quality_score', 1.0) < 0.5 or 
                        performance.get('avg_latency_ms', 0) > 2000):
                        
                        logger.warning(f"Model {model_id} showing poor performance, "
                                     f"quality: {performance.get('avg_quality_score', 0):.2f}, "
                                     f"latency: {performance.get('avg_latency_ms', 0):.0f}ms")
                
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        try:
            return {
                **self.routing_stats,
                'cache_hit_rate': (
                    self.routing_stats['cache_hits'] / 
                    max(self.routing_stats['cache_hits'] + self.routing_stats['cache_misses'], 1)
                ) * 100,
                'model_performance': {
                    model_id: self.performance_tracker.get_model_performance(model_id)
                    for model_id in self.model_configs.keys()
                },
                'active_models': [
                    model_id for model_id, config in self.model_configs.items()
                    if config.enabled
                ]
            }
        except Exception as e:
            logger.error(f"Error getting routing stats: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown the router and cleanup resources"""
        logger.info("Shutting down Intelligent LLM Router...")
        
        try:
            # Clear caches
            self.routing_cache.clear()
            self.response_cache.clear()
            
            self.initialized = False
            logger.info("✅ Intelligent LLM Router shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Utility functions for easy integration
def create_llm_router_for_agent(agent_type: str, **kwargs) -> IntelligentLLMRouter:
    """Create optimized LLM router for specific agent types"""
    
    agent_strategies = {
        "roadside-assistance": RoutingStrategy.PERFORMANCE_OPTIMIZED,  # Speed for emergencies
        "billing-support": RoutingStrategy.QUALITY_OPTIMIZED,         # Accuracy for financial
        "technical-support": RoutingStrategy.BALANCED                 # Balance for complex issues
    }
    
    default_strategy = agent_strategies.get(agent_type, RoutingStrategy.BALANCED)
    
    return IntelligentLLMRouter(
        default_strategy=default_strategy,
        **kwargs
    )


# Export main classes and functions
__all__ = [
    'IntelligentLLMRouter',
    'LLMResponse',
    'RoutingDecision',
    'ComplexityAnalyzer',
    'ModelPerformanceTracker',
    'LLMProvider',
    'ModelType',
    'ComplexityLevel',
    'RoutingStrategy',
    'create_llm_router_for_agent'
]