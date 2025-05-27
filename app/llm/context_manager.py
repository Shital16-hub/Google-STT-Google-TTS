"""
LLM Context Manager - Intelligent Conversation Memory Management
===============================================================

Advanced context management system for multi-agent LLM interactions with intelligent
compression, relevance scoring, and cross-agent context sharing capabilities.
Optimizes context window usage while maintaining conversation coherence and quality.

Features:
- Intelligent conversation memory with semantic compression
- Context window optimization for different LLM models
- Multi-agent context sharing and isolation
- Context relevance scoring and automatic pruning
- Memory persistence across sessions with Redis integration
- Adaptive context length based on conversation complexity
- Cross-conversation learning and pattern recognition
- Agent-specific context personalization
"""

import asyncio
import logging
import time
import uuid
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
import re
from collections import defaultdict, deque
import threading

# For semantic similarity and embeddings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# For text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Redis for persistence
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class ContextType(str, Enum):
    """Types of context information"""
    CONVERSATION = "conversation"
    USER_PROFILE = "user_profile"
    AGENT_SPECIFIC = "agent_specific"
    SYSTEM = "system"
    KNOWLEDGE = "knowledge"
    EMOTIONAL = "emotional"
    PROCEDURAL = "procedural"


class CompressionStrategy(str, Enum):
    """Context compression strategies"""
    NONE = "none"
    SIMPLE_TRUNCATION = "simple_truncation"
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC_COMPRESSION = "semantic_compression"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class RelevanceLevel(str, Enum):
    """Relevance levels for context scoring"""
    CRITICAL = "critical"      # Must retain (>0.9)
    HIGH = "high"             # Important to retain (0.7-0.9)
    MEDIUM = "medium"         # Moderate importance (0.5-0.7)
    LOW = "low"               # Can be compressed (0.3-0.5)
    MINIMAL = "minimal"       # Can be discarded (<0.3)


@dataclass
class ContextItem:
    """Individual context item with metadata"""
    item_id: str
    content: str
    context_type: ContextType
    relevance_score: float
    timestamp: float
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Semantic features
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    
    # Usage tracking
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    importance_boost: float = 0.0
    
    # Relationships
    related_items: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert numpy array to list for JSON serialization
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextItem':
        """Create from dictionary"""
        # Convert embedding back to numpy array
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


@dataclass
class ContextWindow:
    """Context window for a specific conversation/agent"""
    window_id: str
    max_tokens: int
    current_tokens: int
    items: List[ContextItem] = field(default_factory=list)
    compression_strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE
    
    # Optimization settings
    target_utilization: float = 0.8  # Use 80% of available tokens
    critical_reserve: float = 0.1    # Reserve 10% for critical context
    
    # Performance tracking
    compressions_performed: int = 0
    avg_relevance_score: float = 0.0
    last_optimization: float = field(default_factory=time.time)


@dataclass
class ConversationContext:
    """Complete conversation context with multi-agent support"""
    conversation_id: str
    user_id: Optional[str]
    start_time: float
    last_activity: float
    
    # Context windows per agent
    agent_contexts: Dict[str, ContextWindow] = field(default_factory=dict)
    
    # Shared context across agents
    shared_context: List[ContextItem] = field(default_factory=list)
    
    # Conversation metadata
    primary_language: str = "en"
    conversation_phase: str = "active"
    complexity_score: float = 0.5
    
    # User profile context
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_compressions: int = 0
    avg_context_utilization: float = 0.0


class SemanticAnalyzer:
    """
    Semantic analysis for context relevance and compression
    """
    
    def __init__(self):
        """Initialize semantic analyzer"""
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Domain-specific keywords for relevance scoring
        self.domain_keywords = {
            'roadside-assistance': [
                'tow', 'breakdown', 'accident', 'stuck', 'emergency', 'location',
                'vehicle', 'car', 'truck', 'highway', 'road', 'service'
            ],
            'billing-support': [
                'payment', 'bill', 'charge', 'refund', 'account', 'subscription',
                'invoice', 'cost', 'price', 'card', 'bank', 'transaction'
            ],
            'technical-support': [
                'error', 'bug', 'install', 'setup', 'login', 'password',
                'system', 'software', 'connection', 'network', 'device'
            ]
        }
        
        # Importance indicators
        self.importance_patterns = [
            r'\b(important|critical|urgent|essential|vital)\b',
            r'\b(remember|note|warning|caution)\b',
            r'\b(always|never|must|should)\b',
            r'\b(first|second|third|final|last)\b'
        ]
        
        logger.info("Semantic Analyzer initialized")
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        
        # Simple keyword extraction using TF-IDF
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words 
                         if word.isalpha() and word not in self.stop_words and len(word) > 2]
        
        # Count word frequencies
        word_freq = defaultdict(int)
        for word in filtered_words:
            word_freq[word] += 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified implementation)"""
        
        # Simple entity extraction using patterns
        entities = []
        
        # Proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(proper_nouns[:5])  # Limit to top 5
        
        # Phone numbers
        phone_numbers = re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s?\d{3}-\d{4}\b', text)
        entities.extend(phone_numbers)
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities.extend(emails)
        
        # Addresses (simplified)
        addresses = re.findall(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b', text)
        entities.extend(addresses)
        
        return list(set(entities))  # Remove duplicates
    
    def calculate_relevance_score(self, 
                                context_item: ContextItem,
                                current_query: str,
                                agent_id: Optional[str] = None,
                                conversation_history: List[str] = None) -> float:
        """Calculate relevance score for context item"""
        
        base_score = 0.5
        content = context_item.content.lower()
        query = current_query.lower()
        
        # 1. Direct keyword overlap (30% weight)
        content_words = set(word_tokenize(content))
        query_words = set(word_tokenize(query))
        content_words = {w for w in content_words if w not in self.stop_words}
        query_words = {w for w in query_words if w not in self.stop_words}
        
        if query_words:
            overlap_ratio = len(content_words & query_words) / len(query_words)
            base_score += overlap_ratio * 0.3
        
        # 2. Domain relevance (25% weight)
        if agent_id and agent_id in self.domain_keywords:
            domain_words = set(self.domain_keywords[agent_id])
            domain_overlap = len(content_words & domain_words) / max(len(domain_words), 1)
            base_score += domain_overlap * 0.25
        
        # 3. Temporal relevance (20% weight)
        age_hours = (time.time() - context_item.timestamp) / 3600
        temporal_score = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
        base_score += temporal_score * 0.2
        
        # 4. Usage frequency (15% weight)
        usage_score = min(1.0, context_item.access_count / 10)  # Max at 10 accesses
        base_score += usage_score * 0.15
        
        # 5. Importance indicators (10% weight)
        importance_score = 0.0
        for pattern in self.importance_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                importance_score += 0.2
        
        base_score += min(1.0, importance_score) * 0.1
        
        # Apply importance boost
        base_score += context_item.importance_boost
        
        return min(1.0, base_score)
    
    def find_similar_contexts(self, 
                            target_item: ContextItem,
                            context_pool: List[ContextItem],
                            threshold: float = 0.7) -> List[Tuple[ContextItem, float]]:
        """Find semantically similar context items"""
        
        similar_items = []
        target_words = set(word_tokenize(target_item.content.lower()))
        target_words = {w for w in target_words if w not in self.stop_words}
        
        for item in context_pool:
            if item.item_id == target_item.item_id:
                continue
            
            item_words = set(word_tokenize(item.content.lower()))
            item_words = {w for w in item_words if w not in self.stop_words}
            
            # Calculate Jaccard similarity
            if target_words and item_words:
                intersection = len(target_words & item_words)
                union = len(target_words | item_words)
                similarity = intersection / union if union > 0 else 0.0
                
                if similarity >= threshold:
                    similar_items.append((item, similarity))
        
        # Sort by similarity
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return similar_items
    
    def summarize_context_cluster(self, context_items: List[ContextItem]) -> str:
        """Create a summary from a cluster of related context items"""
        
        if not context_items:
            return ""
        
        if len(context_items) == 1:
            return context_items[0].content
        
        # Extract key information from all items
        all_content = " ".join(item.content for item in context_items)
        
        # Extract most frequent keywords
        keywords = self.extract_keywords(all_content, max_keywords=8)
        
        # Create a concise summary
        summary = f"Context summary ({len(context_items)} items): "
        
        # Include most recent item content (usually most relevant)
        latest_item = max(context_items, key=lambda x: x.timestamp)
        summary += latest_item.content[:100]  # First 100 chars
        
        # Add keyword context
        if keywords:
            summary += f" Key topics: {', '.join(keywords[:5])}"
        
        return summary


class ContextCompressor:
    """
    Intelligent context compression with multiple strategies
    """
    
    def __init__(self, semantic_analyzer: SemanticAnalyzer):
        """Initialize context compressor"""
        self.semantic_analyzer = semantic_analyzer
        
        # Compression strategies
        self.strategies = {
            CompressionStrategy.SIMPLE_TRUNCATION: self._simple_truncation,
            CompressionStrategy.SLIDING_WINDOW: self._sliding_window,
            CompressionStrategy.SEMANTIC_COMPRESSION: self._semantic_compression,
            CompressionStrategy.HIERARCHICAL: self._hierarchical_compression,
            CompressionStrategy.ADAPTIVE: self._adaptive_compression
        }
        
        logger.info("Context Compressor initialized")
    
    def compress_context(self,
                        context_window: ContextWindow,
                        target_tokens: int,
                        current_query: str,
                        agent_id: Optional[str] = None) -> List[ContextItem]:
        """Compress context using specified strategy"""
        
        strategy = context_window.compression_strategy
        compression_func = self.strategies.get(strategy, self._adaptive_compression)
        
        logger.debug(f"Compressing context using {strategy.value} strategy")
        
        compressed_items = compression_func(
            context_window.items,
            target_tokens,
            current_query,
            agent_id
        )
        
        # Update window metadata
        context_window.compressions_performed += 1
        context_window.last_optimization = time.time()
        
        return compressed_items
    
    def _simple_truncation(self,
                          items: List[ContextItem],
                          target_tokens: int,
                          current_query: str,
                          agent_id: Optional[str]) -> List[ContextItem]:
        """Simple truncation - keep most recent items"""
        
        # Sort by timestamp (most recent first)
        sorted_items = sorted(items, key=lambda x: x.timestamp, reverse=True)
        
        selected_items = []
        current_tokens = 0
        
        for item in sorted_items:
            item_tokens = len(item.content.split()) * 1.3  # Rough token estimation
            
            if current_tokens + item_tokens <= target_tokens:
                selected_items.append(item)
                current_tokens += item_tokens
            else:
                break
        
        return selected_items
    
    def _sliding_window(self,
                       items: List[ContextItem],
                       target_tokens: int,
                       current_query: str,
                       agent_id: Optional[str]) -> List[ContextItem]:
        """Sliding window - keep recent conversation flow"""
        
        # Sort by timestamp
        sorted_items = sorted(items, key=lambda x: x.timestamp)
        
        # Keep last N items that fit in token limit
        selected_items = []
        current_tokens = 0
        
        for item in reversed(sorted_items):
            item_tokens = len(item.content.split()) * 1.3
            
            if current_tokens + item_tokens <= target_tokens:
                selected_items.insert(0, item)  # Maintain chronological order
                current_tokens += item_tokens
            else:
                break
        
        return selected_items
    
    def _semantic_compression(self,
                            items: List[ContextItem],
                            target_tokens: int,
                            current_query: str,
                            agent_id: Optional[str]) -> List[ContextItem]:
        """Semantic compression - prioritize relevant content"""
        
        # Calculate relevance scores
        scored_items = []
        for item in items:
            relevance = self.semantic_analyzer.calculate_relevance_score(
                item, current_query, agent_id
            )
            scored_items.append((item, relevance))
        
        # Sort by relevance (highest first)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Select items by relevance until token limit
        selected_items = []
        current_tokens = 0
        
        for item, relevance in scored_items:
            item_tokens = len(item.content.split()) * 1.3
            
            # Always include critical items, even if over limit
            if relevance > 0.9 or current_tokens + item_tokens <= target_tokens:
                selected_items.append(item)
                current_tokens += item_tokens
                
                # Stop if we're significantly over limit (except critical items)
                if current_tokens > target_tokens * 1.2 and relevance <= 0.9:
                    break
        
        return selected_items
    
    def _hierarchical_compression(self,
                                items: List[ContextItem],
                                target_tokens: int,
                                current_query: str,
                                agent_id: Optional[str]) -> List[ContextItem]:
        """Hierarchical compression - cluster and summarize similar items"""
        
        if len(items) <= 5:
            # Too few items for clustering
            return self._semantic_compression(items, target_tokens, current_query, agent_id)
        
        # Group items by similarity
        clusters = self._cluster_similar_items(items)
        
        compressed_items = []
        current_tokens = 0
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single item - add directly
                item = cluster[0]
                item_tokens = len(item.content.split()) * 1.3
                
                if current_tokens + item_tokens <= target_tokens:
                    compressed_items.append(item)
                    current_tokens += item_tokens
            else:
                # Multiple items - create summary
                summary_content = self.semantic_analyzer.summarize_context_cluster(cluster)
                
                # Create summarized context item
                latest_item = max(cluster, key=lambda x: x.timestamp)
                summary_item = ContextItem(
                    item_id=f"summary_{uuid.uuid4().hex[:8]}",
                    content=summary_content,
                    context_type=latest_item.context_type,
                    relevance_score=max(item.relevance_score for item in cluster),
                    timestamp=latest_item.timestamp,
                    agent_id=latest_item.agent_id,
                    metadata={'type': 'cluster_summary', 'original_count': len(cluster)}
                )
                
                summary_tokens = len(summary_content.split()) * 1.3
                
                if current_tokens + summary_tokens <= target_tokens:
                    compressed_items.append(summary_item)
                    current_tokens += summary_tokens
        
        return compressed_items
    
    def _adaptive_compression(self,
                            items: List[ContextItem],
                            target_tokens: int,
                            current_query: str,
                            agent_id: Optional[str]) -> List[ContextItem]:
        """Adaptive compression - choose best strategy based on context"""
        
        # Analyze context characteristics
        total_items = len(items)
        avg_age_hours = sum((time.time() - item.timestamp) / 3600 for item in items) / max(total_items, 1)
        
        # Choose strategy based on characteristics
        if total_items <= 5:
            # Few items - use semantic compression
            return self._semantic_compression(items, target_tokens, current_query, agent_id)
        
        elif avg_age_hours > 2:
            # Older context - use hierarchical compression
            return self._hierarchical_compression(items, target_tokens, current_query, agent_id)
        
        else:
            # Recent active conversation - use sliding window with semantic boost
            window_items = self._sliding_window(items, int(target_tokens * 0.7), current_query, agent_id)
            
            # Add most relevant older items to fill remaining space
            remaining_tokens = target_tokens - sum(len(item.content.split()) * 1.3 for item in window_items)
            
            if remaining_tokens > 0:
                older_items = [item for item in items if item not in window_items]
                semantic_items = self._semantic_compression(older_items, remaining_tokens, current_query, agent_id)
                window_items.extend(semantic_items)
            
            return window_items
    
    def _cluster_similar_items(self, items: List[ContextItem]) -> List[List[ContextItem]]:
        """Cluster similar context items for hierarchical compression"""
        
        if len(items) <= 3:
            return [[item] for item in items]  # Each item in its own cluster
        
        # Simple clustering based on content similarity
        clusters = []
        processed = set()
        
        for item in items:
            if item.item_id in processed:
                continue
            
            cluster = [item]
            processed.add(item.item_id)
            
            # Find similar items
            similar_items = self.semantic_analyzer.find_similar_contexts(
                item, items, threshold=0.6
            )
            
            for similar_item, similarity in similar_items:
                if similar_item.item_id not in processed and len(cluster) < 5:
                    cluster.append(similar_item)
                    processed.add(similar_item.item_id)
            
            clusters.append(cluster)
        
        return clusters


class LLMContextManager:
    """
    Advanced LLM Context Manager for intelligent conversation memory management.
    
    Provides sophisticated context management with semantic compression, multi-agent
    support, and cross-conversation learning capabilities for voice AI applications.
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 1,
                 enable_persistence: bool = True,
                 default_max_tokens: int = 4000,
                 compression_threshold: float = 0.9):
        """Initialize LLM context manager"""
        
        self.redis_client = redis_client
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.enable_persistence = enable_persistence
        self.default_max_tokens = default_max_tokens
        self.compression_threshold = compression_threshold
        
        # Core components
        self.semantic_analyzer = SemanticAnalyzer()
        self.compressor = ContextCompressor(self.semantic_analyzer)
        
        # In-memory storage for active contexts
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.context_locks: Dict[str, asyncio.Lock] = {}
        
        # Model-specific token limits
        self.model_token_limits = {
            'gpt-4o': 128000,
            'gpt-4o-mini': 128000,
            'gpt-4': 8192,
            'gpt-3.5-turbo': 16385,
            'claude-3-opus': 200000,
            'claude-3-sonnet': 200000,
            'claude-3-haiku': 200000,
        }
        
        # Agent-specific context configurations
        self.agent_context_configs = {
            'roadside-assistance': {
                'max_tokens': 3000,
                'compression_strategy': CompressionStrategy.SEMANTIC_COMPRESSION,
                'priority_types': [ContextType.PROCEDURAL, ContextType.SYSTEM],
                'retention_hours': 24
            },
            'billing-support': {
                'max_tokens': 4000,
                'compression_strategy': CompressionStrategy.HIERARCHICAL,
                'priority_types': [ContextType.USER_PROFILE, ContextType.CONVERSATION],
                'retention_hours': 72
            },
            'technical-support': {
                'max_tokens': 5000,
                'compression_strategy': CompressionStrategy.ADAPTIVE,
                'priority_types': [ContextType.PROCEDURAL, ContextType.KNOWLEDGE],
                'retention_hours': 48
            }
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_contexts': 0,
            'compressions_performed': 0,
            'avg_compression_ratio': 0.0,
            'avg_context_utilization': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        self.initialized = False
        logger.info("LLM Context Manager initialized")
    
    async def initialize(self):
        """Initialize context manager and Redis connection"""
        
        if self.initialized:
            return
        
        logger.info("🚀 Initializing LLM Context Manager...")
        
        try:
            # Initialize Redis client if not provided
            if not self.redis_client and self.enable_persistence:
                self.redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info(f"✅ Redis connection established: {self.redis_host}:{self.redis_port}")
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._background_cleanup())
            self.optimization_task = asyncio.create_task(self._background_optimization())
            
            self.initialized = True
            logger.info("✅ LLM Context Manager initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Context manager initialization failed: {e}")
            if self.enable_persistence:
                logger.warning("⚠️ Running without persistence - contexts will be memory-only")
                self.enable_persistence = False
            self.initialized = True
    
    async def get_context_for_llm(self,
                                 conversation_id: str,
                                 agent_id: str,
                                 current_query: str,
                                 model_id: str = "gpt-4o",
                                 max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get optimized context for LLM request
        
        Args:
            conversation_id: Unique conversation identifier
            agent_id: Agent requesting context
            current_query: Current user query
            model_id: LLM model being used
            max_tokens: Override token limit
            
        Returns:
            List of messages formatted for LLM
        """
        
        if not self.initialized:
            await self.initialize()
        
        context_start = time.time()
        
        # Get or create conversation context
        conversation_context = await self._get_or_create_context(conversation_id)
        
        # Get agent-specific context window
        context_window = await self._get_agent_context_window(
            conversation_context, agent_id, model_id, max_tokens
        )
        
        # Check if compression is needed
        available_tokens = context_window.max_tokens * context_window.target_utilization
        current_usage = sum(len(item.content.split()) * 1.3 for item in context_window.items)
        
        if current_usage > available_tokens:
            logger.debug(f"Context compression needed: {current_usage:.0f} > {available_tokens:.0f} tokens")
            
            compressed_items = self.compressor.compress_context(
                context_window=context_window,
                target_tokens=int(available_tokens),
                current_query=current_query,
                agent_id=agent_id
            )
            
            context_window.items = compressed_items
            context_window.current_tokens = sum(len(item.content.split()) * 1.3 for item in compressed_items)
            
            self.performance_metrics['compressions_performed'] += 1
            
            # Update compression ratio
            if current_usage > 0:
                compression_ratio = context_window.current_tokens / current_usage
                current_avg = self.performance_metrics['avg_compression_ratio']
                total_compressions = self.performance_metrics['compressions_performed']
                self.performance_metrics['avg_compression_ratio'] = (
                    (current_avg * (total_compressions - 1) + compression_ratio) / total_compressions
                )
        
        # Convert to LLM message format
        messages = self._format_context_for_llm(context_window, agent_id, current_query)
        
        # Update access tracking
        for item in context_window.items:
            item.access_count += 1
            item.last_accessed = time.time()
        
        # Update performance metrics
        processing_time = (time.time() - context_start) * 1000
        utilization = context_window.current_tokens / context_window.max_tokens
        
        current_avg = self.performance_metrics['avg_context_utilization']
        total_contexts = self.performance_metrics['total_contexts'] + 1
        self.performance_metrics['avg_context_utilization'] = (
            (current_avg * (total_contexts - 1) + utilization) / total_contexts
        )
        self.performance_metrics['total_contexts'] = total_contexts
        
        logger.debug(f"Context prepared in {processing_time:.1f}ms: "
                    f"{len(messages)} messages, {context_window.current_tokens:.0f} tokens "
                    f"({utilization:.1%} utilization)")
        
        return messages
    
    async def add_context(self,
                         conversation_id: str,
                         content: str,
                         context_type: ContextType,
                         agent_id: Optional[str] = None,
                         user_id: Optional[str] = None,
                         importance_boost: float = 0.0,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Add new context item to conversation
        
        Args:
            conversation_id: Conversation identifier
            content: Context content
            context_type: Type of context
            agent_id: Agent that generated/owns this context
            user_id: User identifier
            importance_boost: Additional importance score
            metadata: Additional metadata
        """
        
        if not self.initialized:
            await self.initialize()
        
        # Get conversation context
        conversation_context = await self._get_or_create_context(conversation_id, user_id)
        
        # Create context item
        context_item = ContextItem(
            item_id=str(uuid.uuid4()),
            content=content,
            context_type=context_type,
            relevance_score=0.5,  # Will be calculated later
            timestamp=time.time(),
            agent_id=agent_id,
            user_id=user_id,
            session_id=conversation_id,
            importance_boost=importance_boost,
            keywords=self.semantic_analyzer.extract_keywords(content),
            entities=self.semantic_analyzer.extract_entities(content),
            metadata=metadata or {}
        )
        
        # Add to appropriate context window(s)
        if agent_id:
            # Add to agent-specific context
            if agent_id not in conversation_context.agent_contexts:
                conversation_context.agent_contexts[agent_id] = await self._create_agent_context_window(
                    agent_id, conversation_id
                )
            
            conversation_context.agent_contexts[agent_id].items.append(context_item)
        
        # Add to shared context if it's a system or user message
        if context_type in [ContextType.SYSTEM, ContextType.USER_PROFILE, ContextType.CONVERSATION]:
            conversation_context.shared_context.append(context_item)
        
        # Persist if enabled
        if self.enable_persistence:
            await self._persist_context(conversation_context)
        
        logger.debug(f"Added context item: {context_type.value} for agent {agent_id}")
    
    async def update_user_profile(self,
                                 conversation_id: str,
                                 user_id: str,
                                 profile_updates: Dict[str, Any]):
        """Update user profile context"""
        
        conversation_context = await self._get_or_create_context(conversation_id, user_id)
        conversation_context.user_preferences.update(profile_updates)
        
        # Add as context item
        profile_content = f"User preferences updated: {json.dumps(profile_updates, indent=2)}"
        
        await self.add_context(
            conversation_id=conversation_id,
            content=profile_content,
            context_type=ContextType.USER_PROFILE,
            user_id=user_id,
            importance_boost=0.2,
            metadata={'type': 'profile_update', 'updates': profile_updates}
        )
    
    async def get_cross_agent_context(self,
                                     conversation_id: str,
                                     requesting_agent_id: str,
                                     context_types: Optional[List[ContextType]] = None) -> List[ContextItem]:
        """Get context shared across agents"""
        
        conversation_context = await self._get_or_create_context(conversation_id)
        
        # Get shared context
        shared_items = conversation_context.shared_context.copy()
        
        # Add relevant context from other agents
        for agent_id, agent_context in conversation_context.agent_contexts.items():
            if agent_id == requesting_agent_id:
                continue
            
            # Add high-relevance items from other agents
            for item in agent_context.items:
                if (item.relevance_score > 0.8 and
                    (not context_types or item.context_type in context_types)):
                    shared_items.append(item)
        
        # Sort by relevance and timestamp
        shared_items.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
        
        return shared_items[:10]  # Limit to top 10 items
    
    async def optimize_context_window(self,
                                     conversation_id: str,
                                     agent_id: str,
                                     current_query: str):
        """Manually trigger context window optimization"""
        
        conversation_context = await self._get_or_create_context(conversation_id)
        
        if agent_id in conversation_context.agent_contexts:
            context_window = conversation_context.agent_contexts[agent_id]
            
            # Recalculate relevance scores
            for item in context_window.items:
                item.relevance_score = self.semantic_analyzer.calculate_relevance_score(
                    item, current_query, agent_id
                )
            
            # Remove low-relevance items
            context_window.items = [
                item for item in context_window.items
                if item.relevance_score > 0.3 or item.context_type == ContextType.SYSTEM
            ]
            
            # Update metadata
            context_window.last_optimization = time.time()
            context_window.avg_relevance_score = (
                sum(item.relevance_score for item in context_window.items) /
                max(len(context_window.items), 1)
            )
            
            logger.debug(f"Optimized context window for {agent_id}: "
                        f"{len(context_window.items)} items retained")
    
    async def _get_or_create_context(self,
                                    conversation_id: str,
                                    user_id: Optional[str] = None) -> ConversationContext:
        """Get existing context or create new one"""
        
        # Check memory cache first
        if conversation_id in self.active_contexts:
            self.performance_metrics['cache_hits'] += 1
            context = self.active_contexts[conversation_id]
            context.last_activity = time.time()
            return context
        
        self.performance_metrics['cache_misses'] += 1
        
        # Try to load from persistence
        if self.enable_persistence and self.redis_client:
            try:
                context_data = await self.redis_client.get(f"context:{conversation_id}")
                if context_data:
                    context_dict = json.loads(context_data)
                    context = self._conversation_context_from_dict(context_dict)
                    
                    # Cache in memory
                    self.active_contexts[conversation_id] = context
                    self.context_locks[conversation_id] = asyncio.Lock()
                    
                    logger.debug(f"Loaded context from Redis: {conversation_id}")
                    return context
                    
            except Exception as e:
                logger.error(f"Error loading context from Redis: {e}")
        
        # Create new context
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            start_time=time.time(),
            last_activity=time.time()
        )
        
        # Cache in memory
        self.active_contexts[conversation_id] = context
        self.context_locks[conversation_id] = asyncio.Lock()
        
        logger.debug(f"Created new context: {conversation_id}")
        return context
    
    async def _get_agent_context_window(self,
                                       conversation_context: ConversationContext,
                                       agent_id: str,
                                       model_id: str,
                                       max_tokens: Optional[int]) -> ContextWindow:
        """Get or create agent-specific context window"""
        
        if agent_id not in conversation_context.agent_contexts:
            conversation_context.agent_contexts[agent_id] = await self._create_agent_context_window(
                agent_id, conversation_context.conversation_id, model_id, max_tokens
            )
        
        context_window = conversation_context.agent_contexts[agent_id]
        
        # Update token limit if specified
        if max_tokens:
            context_window.max_tokens = max_tokens
        elif model_id in self.model_token_limits:
            # Reserve space for response generation
            available_tokens = int(self.model_token_limits[model_id] * 0.7)
            context_window.max_tokens = min(context_window.max_tokens, available_tokens)
        
        # Add shared context items
        for shared_item in conversation_context.shared_context:
            # Check if item is already in agent context
            if not any(item.item_id == shared_item.item_id for item in context_window.items):
                # Calculate relevance for this agent
                shared_item.relevance_score = self.semantic_analyzer.calculate_relevance_score(
                    shared_item, "", agent_id
                )
                
                # Add if relevant
                if shared_item.relevance_score > 0.5:
                    context_window.items.append(shared_item)
        
        return context_window
    
    async def _create_agent_context_window(self,
                                          agent_id: str,
                                          conversation_id: str,
                                          model_id: str = "gpt-4o",
                                          max_tokens: Optional[int] = None) -> ContextWindow:
        """Create new agent-specific context window"""
        
        # Get agent configuration
        agent_config = self.agent_context_configs.get(agent_id, {})
        
        # Determine token limit
        if max_tokens:
            token_limit = max_tokens
        elif agent_id in self.agent_context_configs:
            token_limit = agent_config['max_tokens']
        elif model_id in self.model_token_limits:
            token_limit = int(self.model_token_limits[model_id] * 0.6)  # 60% for context
        else:
            token_limit = self.default_max_tokens
        
        # Determine compression strategy
        compression_strategy = agent_config.get(
            'compression_strategy',
            CompressionStrategy.ADAPTIVE
        )
        
        context_window = ContextWindow(
            window_id=f"{conversation_id}:{agent_id}",
            max_tokens=token_limit,
            current_tokens=0,
            compression_strategy=compression_strategy
        )
        
        logger.debug(f"Created context window for {agent_id}: {token_limit} tokens, {compression_strategy.value}")
        
        return context_window
    
    def _format_context_for_llm(self,
                               context_window: ContextWindow,
                               agent_id: str,
                               current_query: str) -> List[Dict[str, str]]:
        """Format context items as LLM messages"""
        
        messages = []
        
        # Add system message with agent context
        system_content = self._get_agent_system_message(agent_id)
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        # Sort context items by timestamp
        sorted_items = sorted(context_window.items, key=lambda x: x.timestamp)
        
        # Group consecutive items by role to avoid role switching
        current_role = None
        current_content = []
        
        for item in sorted_items:
            # Determine message role based on context type
            if item.context_type == ContextType.SYSTEM:
                role = "system"
            elif item.context_type in [ContextType.CONVERSATION, ContextType.USER_PROFILE]:
                # Determine if this was user or assistant message
                role = "user" if "user:" in item.content.lower() else "assistant"
            else:
                role = "assistant"  # Default for agent-generated content
            
            # Add to current group or start new group
            if role == current_role:
                current_content.append(item.content)
            else:
                # Finish previous group
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content)
                    })
                
                # Start new group
                current_role = role
                current_content = [item.content]
        
        # Add final group
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": "\n".join(current_content)
            })
        
        # Add current user query
        messages.append({"role": "user", "content": current_query})
        
        return messages
    
    def _get_agent_system_message(self, agent_id: str) -> str:
        """Get system message for specific agent"""
        
        system_messages = {
            'roadside-assistance': (
                "You are a professional roadside assistance coordinator. "
                "Prioritize safety and provide clear, actionable guidance for vehicle emergencies."
            ),
            'billing-support': (
                "You are an empathetic billing support specialist. "
                "Help customers with payment and billing issues with patience and understanding."
            ),
            'technical-support': (
                "You are a patient technical support expert. "
                "Provide clear, step-by-step guidance for technical problems."
            )
        }
        
        return system_messages.get(agent_id, "You are a helpful AI assistant.")
    
    def _conversation_context_from_dict(self, data: Dict[str, Any]) -> ConversationContext:
        """Convert dictionary to ConversationContext"""
        
        # Convert agent contexts
        agent_contexts = {}
        for agent_id, window_data in data.get('agent_contexts', {}).items():
            # Convert context items
            items = []
            for item_data in window_data.get('items', []):
                items.append(ContextItem.from_dict(item_data))
            
            # Create context window
            context_window = ContextWindow(
                window_id=window_data['window_id'],
                max_tokens=window_data['max_tokens'],
                current_tokens=window_data['current_tokens'],
                items=items,
                compression_strategy=CompressionStrategy(window_data.get('compression_strategy', 'adaptive'))
            )
            
            agent_contexts[agent_id] = context_window
        
        # Convert shared context
        shared_context = []
        for item_data in data.get('shared_context', []):
            shared_context.append(ContextItem.from_dict(item_data))
        
        # Create conversation context
        return ConversationContext(
            conversation_id=data['conversation_id'],
            user_id=data.get('user_id'),
            start_time=data['start_time'],
            last_activity=data['last_activity'],
            agent_contexts=agent_contexts,
            shared_context=shared_context,
            primary_language=data.get('primary_language', 'en'),
            conversation_phase=data.get('conversation_phase', 'active'),
            complexity_score=data.get('complexity_score', 0.5),
            user_preferences=data.get('user_preferences', {}),
            interaction_history=data.get('interaction_history', [])
        )
    
    def _conversation_context_to_dict(self, context: ConversationContext) -> Dict[str, Any]:
        """Convert ConversationContext to dictionary"""
        
        # Convert agent contexts
        agent_contexts = {}
        for agent_id, window in context.agent_contexts.items():
            agent_contexts[agent_id] = {
                'window_id': window.window_id,
                'max_tokens': window.max_tokens,
                'current_tokens': window.current_tokens,
                'compression_strategy': window.compression_strategy.value,
                'items': [item.to_dict() for item in window.items]
            }
        
        return {
            'conversation_id': context.conversation_id,
            'user_id': context.user_id,
            'start_time': context.start_time,
            'last_activity': context.last_activity,
            'agent_contexts': agent_contexts,
            'shared_context': [item.to_dict() for item in context.shared_context],
            'primary_language': context.primary_language,
            'conversation_phase': context.conversation_phase,
            'complexity_score': context.complexity_score,
            'user_preferences': context.user_preferences,
            'interaction_history': context.interaction_history
        }
    
    async def _persist_context(self, context: ConversationContext):
        """Persist context to Redis"""
        
        if not self.enable_persistence or not self.redis_client:
            return
        
        try:
            context_dict = self._conversation_context_to_dict(context)
            context_json = json.dumps(context_dict, default=str)
            
            # Set with expiration (24 hours default)
            expiration = 86400  # 24 hours
            
            await self.redis_client.setex(
                f"context:{context.conversation_id}",
                expiration,
                context_json
            )
            
        except Exception as e:
            logger.error(f"Error persisting context: {e}")
    
    async def _background_cleanup(self):
        """Background task to clean up expired contexts"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = time.time()
                expired_contexts = []
                
                # Find expired contexts (inactive for > 1 hour)
                for conversation_id, context in self.active_contexts.items():
                    if current_time - context.last_activity > 3600:  # 1 hour
                        expired_contexts.append(conversation_id)
                
                # Clean up expired contexts
                for conversation_id in expired_contexts:
                    if conversation_id in self.active_contexts:
                        del self.active_contexts[conversation_id]
                    if conversation_id in self.context_locks:
                        del self.context_locks[conversation_id]
                
                if expired_contexts:
                    logger.info(f"Cleaned up {len(expired_contexts)} expired contexts")
                
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
    
    async def _background_optimization(self):
        """Background task for context optimization"""
        
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Optimize context windows that haven't been optimized recently
                current_time = time.time()
                
                for context in self.active_contexts.values():
                    for agent_id, window in context.agent_contexts.items():
                        # Skip if optimized recently
                        if current_time - window.last_optimization < 1800:  # 30 minutes
                            continue
                        
                        # Remove very old, low-relevance items
                        cutoff_time = current_time - 7200  # 2 hours
                        
                        window.items = [
                            item for item in window.items
                            if (item.timestamp > cutoff_time or 
                                item.relevance_score > 0.7 or
                                item.context_type == ContextType.SYSTEM)
                        ]
                        
                        window.last_optimization = current_time
                        
                        logger.debug(f"Background optimization for {agent_id}: "
                                   f"{len(window.items)} items retained")
                
            except Exception as e:
                logger.error(f"Error in background optimization: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        return {
            **self.performance_metrics,
            'active_contexts': len(self.active_contexts),
            'avg_items_per_context': (
                sum(len(ctx.shared_context) + sum(len(w.items) for w in ctx.agent_contexts.values())
                    for ctx in self.active_contexts.values()) / 
                max(len(self.active_contexts), 1)
            ),
            'cache_hit_rate': (
                self.performance_metrics['cache_hits'] / 
                max(self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1)
            ) * 100,
            'compression_effectiveness': self.performance_metrics['avg_compression_ratio'],
            'context_utilization': self.performance_metrics['avg_context_utilization'] * 100
        }
    
    async def clear_context(self, conversation_id: str, agent_id: Optional[str] = None):
        """Clear context for conversation or specific agent"""
        
        if conversation_id in self.active_contexts:
            context = self.active_contexts[conversation_id]
            
            if agent_id:
                # Clear specific agent context
                if agent_id in context.agent_contexts:
                    del context.agent_contexts[agent_id]
                    logger.info(f"Cleared context for agent {agent_id} in conversation {conversation_id}")
            else:
                # Clear entire conversation context
                del self.active_contexts[conversation_id]
                if conversation_id in self.context_locks:
                    del self.context_locks[conversation_id]
                
                # Remove from persistence
                if self.enable_persistence and self.redis_client:
                    try:
                        await self.redis_client.delete(f"context:{conversation_id}")
                    except Exception as e:
                        logger.error(f"Error removing context from Redis: {e}")
                
                logger.info(f"Cleared entire context for conversation {conversation_id}")
    
    async def shutdown(self):
        """Shutdown context manager and cleanup resources"""
        
        logger.info("Shutting down LLM Context Manager...")
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        # Persist all active contexts
        if self.enable_persistence:
            for context in self.active_contexts.values():
                await self._persist_context(context)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Clear memory
        self.active_contexts.clear()
        self.context_locks.clear()
        
        self.initialized = False
        logger.info("✅ LLM Context Manager shutdown complete")


# Utility functions for easy integration

def create_context_manager_for_agent(agent_type: str, **kwargs) -> LLMContextManager:
    """Create optimized context manager for specific agent types"""
    
    agent_configs = {
        "roadside-assistance": {
            'default_max_tokens': 3000,
            'compression_threshold': 0.8  # More aggressive compression for speed
        },
        "billing-support": {
            'default_max_tokens': 4000,
            'compression_threshold': 0.9  # Preserve more context for accuracy
        },
        "technical-support": {
            'default_max_tokens': 5000,
            'compression_threshold': 0.85  # Balanced for complex explanations
        }
    }
    
    config = agent_configs.get(agent_type, {})
    config.update(kwargs)
    
    return LLMContextManager(**config)


# Export main classes and functions
__all__ = [
    'LLMContextManager',
    'ContextItem',
    'ContextWindow',
    'ConversationContext',
    'ContextType',
    'CompressionStrategy',
    'RelevanceLevel',
    'SemanticAnalyzer',
    'ContextCompressor',
    'create_context_manager_for_agent'
]