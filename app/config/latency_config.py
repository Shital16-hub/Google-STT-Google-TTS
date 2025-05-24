"""
Latency Configuration for Multi-Agent Voice AI System
Comprehensive latency optimization settings targeting sub-650ms end-to-end response
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LatencyTier(Enum):
    """Latency performance tiers"""
    ULTRA_LOW = "ultra_low"      # <200ms total
    LOW = "low"                  # <500ms total  
    STANDARD = "standard"        # <1000ms total
    RELAXED = "relaxed"          # <2000ms total

class OptimizationLevel(Enum):
    """Optimization aggressiveness levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class STTLatencyConfig:
    """Speech-to-Text latency optimization configuration"""
    # Target latencies (milliseconds)
    target_latency: int = 120
    max_acceptable_latency: int = 200
    
    # Google Cloud STT optimization
    google_stt_config: Dict[str, Any] = field(default_factory=lambda: {
        "model": "telephony_short",  # Optimized for phone calls
        "language_code": "en-US",
        "sample_rate_hertz": 8000,   # Phone quality
        "encoding": "MULAW",         # Efficient encoding
        "enable_automatic_punctuation": False,  # Reduce processing
        "enable_word_time_offsets": False,     # Skip if not needed
        "max_alternatives": 1,       # Single best result
        "profanity_filter": False,   # Skip if not needed
        "enable_word_confidence": False,       # Skip confidence scores
        "use_enhanced": False,       # Basic model for speed
        "single_utterance": True,    # Single phrase mode
        "interim_results": True      # Stream partial results
    })
    
    # Voice Activity Detection
    vad_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "sensitivity": 0.7,          # Balance false positives/negatives
        "min_speech_duration": 100,  # ms
        "min_silence_duration": 200, # ms
        "pre_speech_pad": 50,        # ms before speech
        "post_speech_pad": 100       # ms after speech
    })
    
    # Streaming optimization
    streaming_config: Dict[str, Any] = field(default_factory=lambda: {
        "chunk_size": 1024,          # Bytes per chunk
        "chunk_duration_ms": 60,     # Milliseconds per chunk
        "buffer_size": 4096,         # Total buffer size
        "enable_early_termination": True,
        "silence_threshold": -40     # dB silence level
    })
    
    # Fallback STT service (AssemblyAI Nano)
    fallback_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "api_key_env": "ASSEMBLYAI_API_KEY",
        "model": "nano",             # Ultra-fast model
        "target_latency": 270,       # ms
        "auto_fallback_threshold": 300  # Switch if primary > 300ms
    })

@dataclass 
class LLMLatencyConfig:
    """Large Language Model latency optimization configuration"""
    # Target latencies
    target_latency: int = 280
    max_acceptable_latency: int = 400
    
    # Primary LLM (GPT-4o-mini)
    primary_llm_config: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4o-mini",
        "temperature": 0.3,          # Lower for consistency and speed
        "max_tokens": 150,           # Shorter responses for voice
        "top_p": 0.8,               # Focus on likely tokens
        "frequency_penalty": 0.1,    # Reduce repetition
        "presence_penalty": 0.1,     # Encourage conciseness
        "stream": True,              # Enable streaming
        "timeout": 10,               # 10 second timeout
        "max_retries": 2            # Quick failure recovery
    })
    
    # Fallback LLM (Claude 3.5 Haiku)
    fallback_llm_config: Dict[str, Any] = field(default_factory=lambda: {
        "model": "claude-3-5-haiku-20241022",
        "temperature": 0.2,
        "max_tokens": 120,           # Even shorter for fallback
        "timeout": 8,
        "max_retries": 1,
        "auto_fallback_threshold": 350  # Switch if primary > 350ms
    })
    
    # Context optimization
    context_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_context_length": 2048,  # Tokens
        "context_compression": True,  # Compress old messages
        "relevance_threshold": 0.7,   # Keep relevant context only
        "sliding_window": 10,         # Keep last N exchanges
        "summarization_threshold": 1500  # Summarize if context > N tokens
    })
    
    # Voice-specific optimizations
    voice_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "prefer_short_responses": True,
        "conversational_style": True,
        "avoid_complex_formatting": True,
        "optimize_for_speech": True,
        "max_sentence_length": 20,    # Words per sentence
        "use_contractions": True      # "don't" vs "do not"
    })

@dataclass
class TTSLatencyConfig:
    """Text-to-Speech latency optimization configuration"""
    # Target latencies
    target_latency: int = 150      # Time to first audio chunk
    streaming_chunk_size: int = 1024  # Bytes
    
    # Dual streaming architecture
    dual_streaming_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "primary_engine": "orca_streaming",  # PADRI engine
        "fallback_engine": "google_cloud",
        "chunk_processing": "word_by_word",
        "parallel_processing": True,
        "early_audio_return": True,
        "max_concurrent_chunks": 4
    })
    
    # Google Cloud TTS optimization
    google_tts_config: Dict[str, Any] = field(default_factory=lambda: {
        "voice": {
            "language_code": "en-US",
            "name": "en-US-Neural2-C",   # Fast, natural voice
            "ssml_gender": "NEUTRAL"
        },
        "audio_config": {
            "audio_encoding": "LINEAR16", # Uncompressed for speed
            "sample_rate_hertz": 8000,   # Phone quality 
            "speaking_rate": 1.1,        # Slightly faster
            "pitch": 0.0,
            "volume_gain_db": 0.0,
            "effects_profile_id": []     # No effects for speed
        },
        "streaming": True,
        "enable_time_pointing": False    # Skip timing info
    })
    
    # OrcaTTS Streaming configuration
    orca_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_path": "./models/orca_streaming.onnx",
        "chunk_length_ms": 50,       # Very small chunks
        "overlap_ms": 10,           # Small overlap for smoothness
        "sample_rate": 8000,
        "enable_gpu": False,         # CPU for consistency
        "num_threads": 4,
        "streaming_factor": 0.8     # Start streaming at 80% completion
    })
    
    # Voice characteristics optimization
    voice_config: Dict[str, Any] = field(default_factory=lambda: {
        "optimize_for_phone": True,
        "enhance_clarity": True,
        "reduce_background_noise": True,
        "normalize_volume": True,
        "apply_compression": True,   # Audio compression for size
        "bit_rate": 32000           # 32kbps for voice
    })

@dataclass
class VectorLatencyConfig:
    """Vector database latency optimization configuration"""
    # Target latencies by tier
    target_latencies: Dict[str, int] = field(default_factory=lambda: {
        "redis": 1,      # <1ms
        "faiss": 5,      # <5ms
        "qdrant": 50,    # <50ms
        "total": 10      # <10ms overall
    })
    
    # Redis optimization
    redis_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "connection_pooling": True,
        "persistent_connections": True,
        "pipeline_requests": True,
        "compression": False,        # Skip compression for speed
        "serialization": "pickle",   # Fast serialization
        "key_expiry_optimization": True,
        "memory_optimization": "speed_over_space"
    })
    
    # FAISS optimization  
    faiss_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "index_type": "IVF",        # Inverted file for speed
        "nprobe": 8,                # Reduced for speed
        "parallel_search": True,
        "use_precomputed_tables": True,
        "quantization": "SQ8",      # 8-bit quantization
        "memory_mapping": False,    # Keep in RAM
        "search_threads": 4
    })
    
    # Qdrant optimization
    qdrant_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "prefer_grpc": True,        # Faster than HTTP
        "connection_pooling": True,
        "hnsw_ef": 32,             # Reduced for speed
        "quantization": "scalar",   # Use quantization
        "payload_exclusion": True,  # Exclude heavy payloads from search
        "exact_search_threshold": 100  # Use exact search for small collections
    })

@dataclass
class NetworkLatencyConfig:
    """Network and infrastructure latency optimization"""
    # Target network latency
    target_latency: int = 50
    max_acceptable_latency: int = 100
    
    # WebSocket optimization
    websocket_config: Dict[str, Any] = field(default_factory=lambda: {
        "compression": True,
        "keep_alive": True,
        "ping_interval": 20,        # seconds
        "ping_timeout": 10,         # seconds
        "close_timeout": 10,        # seconds
        "max_size": 1048576,       # 1MB max message size
        "max_queue": 32,           # Max queued messages
        "write_limit": 1048576,    # Write buffer limit
        "read_limit": 1048576      # Read buffer limit
    })
    
    # HTTP optimization
    http_config: Dict[str, Any] = field(default_factory=lambda: {
        "keep_alive": True,
        "connection_pooling": True,
        "max_connections": 100,
        "max_keepalive_connections": 20,
        "keepalive_expiry": 30,    # seconds
        "timeout": {
            "connect": 5,           # seconds
            "read": 10,            # seconds  
            "write": 10,           # seconds
            "pool": 10             # seconds
        }
    })
    
    # CDN and caching
    caching_config: Dict[str, Any] = field(default_factory=lambda: {
        "edge_caching": True,
        "static_asset_caching": True,
        "api_response_caching": True,
        "cache_headers": True,
        "compression": "gzip",
        "cache_ttl": {
            "static": 86400,        # 24 hours
            "api": 300,            # 5 minutes
            "dynamic": 60          # 1 minute
        }
    })

@dataclass
class AgentRoutingLatencyConfig:
    """Agent routing and orchestration latency optimization"""
    # Target routing latency
    target_latency: int = 15
    max_acceptable_latency: int = 30
    
    # Intelligent routing optimization
    routing_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "enable_caching": True,
        "cache_ttl": 300,          # 5 minutes
        "confidence_threshold": 0.85,
        "early_termination": True,
        "parallel_evaluation": True,
        "max_evaluation_time": 20,  # ms
        "fallback_agent": "general"
    })
    
    # Agent warm-up configuration
    warmup_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "warmup_on_startup": True,
        "keep_warm_agents": ["roadside-assistance"],  # Critical agents
        "warmup_interval": 300,     # seconds
        "health_check_interval": 60 # seconds
    })

class LatencyConfig:
    """Main latency configuration class"""
    
    def __init__(self, tier: LatencyTier = LatencyTier.LOW):
        self.tier = tier
        self.optimization_level = OptimizationLevel.BALANCED
        
        # Component configurations
        self.stt = STTLatencyConfig()
        self.llm = LLMLatencyConfig()
        self.tts = TTSLatencyConfig()
        self.vector = VectorLatencyConfig()
        self.network = NetworkLatencyConfig()
        self.routing = AgentRoutingLatencyConfig()
        
        # Overall targets based on tier
        self.total_latency_targets = self._get_tier_targets()
        
        # Apply tier-specific optimizations
        self._apply_tier_optimizations()

    def _get_tier_targets(self) -> Dict[str, int]:
        """Get latency targets based on performance tier"""
        targets = {
            LatencyTier.ULTRA_LOW: {
                "stt": 100, "routing": 10, "vector": 5, 
                "llm": 200, "tts": 100, "network": 30, "total": 450
            },
            LatencyTier.LOW: {
                "stt": 120, "routing": 15, "vector": 10,
                "llm": 280, "tts": 150, "network": 50, "total": 625
            },
            LatencyTier.STANDARD: {
                "stt": 200, "routing": 30, "vector": 25,
                "llm": 400, "tts": 250, "network": 100, "total": 1000
            },
            LatencyTier.RELAXED: {
                "stt": 400, "routing": 50, "vector": 50,
                "llm": 800, "tts": 500, "network": 200, "total": 2000
            }
        }
        return targets[self.tier]

    def _apply_tier_optimizations(self):
        """Apply tier-specific optimizations"""
        if self.tier == LatencyTier.ULTRA_LOW:
            # Ultra-aggressive optimizations
            self.stt.google_stt_config.update({
                "enable_automatic_punctuation": False,
                "max_alternatives": 1,
                "single_utterance": True
            })
            
            self.llm.primary_llm_config.update({
                "max_tokens": 100,
                "temperature": 0.1,
                "timeout": 8
            })
            
            self.tts.target_latency = 100
            self.vector.target_latencies["total"] = 5
            
        elif self.tier == LatencyTier.LOW:
            # Balanced optimizations (current settings)
            pass
            
        elif self.tier == LatencyTier.STANDARD:
            # Relaxed optimizations for reliability
            self.llm.primary_llm_config.update({
                "max_tokens": 200,
                "temperature": 0.5,
                "timeout": 15
            })
            
            self.tts.target_latency = 250
            self.vector.target_latencies["total"] = 25

    def get_component_budget(self) -> Dict[str, Dict[str, int]]:
        """Get detailed latency budget breakdown"""
        return {
            "stt": {
                "audio_processing": int(self.stt.target_latency * 0.3),
                "vad": int(self.stt.target_latency * 0.1), 
                "transcription": int(self.stt.target_latency * 0.6)
            },
            "routing": {
                "intent_classification": int(self.routing.target_latency * 0.4),
                "agent_selection": int(self.routing.target_latency * 0.3),
                "context_preparation": int(self.routing.target_latency * 0.3)
            },
            "vector": {
                "query_embedding": int(self.vector.target_latencies["total"] * 0.2),
                "similarity_search": int(self.vector.target_latencies["total"] * 0.6),
                "result_retrieval": int(self.vector.target_latencies["total"] * 0.2)
            },
            "llm": {
                "context_building": int(self.llm.target_latency * 0.1),
                "inference": int(self.llm.target_latency * 0.8),
                "response_formatting": int(self.llm.target_latency * 0.1)
            },
            "tts": {
                "text_processing": int(self.tts.target_latency * 0.2),
                "synthesis": int(self.tts.target_latency * 0.6),
                "audio_encoding": int(self.tts.target_latency * 0.2)
            },
            "network": {
                "request_overhead": int(self.network.target_latency * 0.4),
                "response_overhead": int(self.network.target_latency * 0.4),
                "connection_management": int(self.network.target_latency * 0.2)
            }
        }

    def get_monitoring_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get monitoring and alerting thresholds"""
        return {
            "warning_thresholds": {
                "stt": self.stt.target_latency * 1.2,
                "routing": self.routing.target_latency * 1.2,
                "vector": self.vector.target_latencies["total"] * 1.2,
                "llm": self.llm.target_latency * 1.2,
                "tts": self.tts.target_latency * 1.2,
                "network": self.network.target_latency * 1.2,
                "total": self.total_latency_targets["total"] * 1.1
            },
            "critical_thresholds": {
                "stt": self.stt.max_acceptable_latency,
                "routing": self.routing.max_acceptable_latency,
                "vector": self.vector.target_latencies["total"] * 2,
                "llm": self.llm.max_acceptable_latency,
                "tts": self.tts.target_latency * 2,
                "network": self.network.max_acceptable_latency,
                "total": self.total_latency_targets["total"] * 1.5
            },
            "percentile_targets": {
                "p50": self.total_latency_targets["total"],
                "p90": self.total_latency_targets["total"] * 1.3,
                "p95": self.total_latency_targets["total"] * 1.5,
                "p99": self.total_latency_targets["total"] * 2.0
            }
        }

    def get_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Get optimization recommendations based on current configuration"""
        return {
            "immediate_wins": [
                "Enable Redis vector caching for frequent queries",
                "Use FAISS hot tier for active agents",
                "Implement connection pooling for all services",
                "Enable gRPC for Qdrant connections",
                "Use streaming for TTS and LLM responses"
            ],
            "infrastructure": [
                "Deploy Redis cluster for high availability",
                "Use SSD storage for FAISS indices",
                "Implement CDN for static assets",
                "Enable HTTP/2 for better multiplexing",
                "Use load balancing for agent instances"
            ],
            "agent_specific": [
                "Pre-warm critical agents (roadside assistance)",
                "Cache common routing decisions",
                "Optimize context length per agent type",
                "Use agent-specific TTS voice caching",
                "Implement smart context compression"
            ],
            "monitoring": [
                "Set up real-time latency dashboards",
                "Implement automated alerting",
                "Track latency percentiles over time",
                "Monitor resource utilization",
                "Set up synthetic transaction monitoring"
            ]
        }

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate latency configuration settings"""
        errors = []
        
        # Check target consistency
        total_component_time = (
            self.stt.target_latency +
            self.routing.target_latency +
            self.vector.target_latencies["total"] +
            self.llm.target_latency +
            self.tts.target_latency +
            self.network.target_latency
        )
        
        if total_component_time > self.total_latency_targets["total"]:
            errors.append(
                f"Component latencies sum ({total_component_time}ms) exceeds "
                f"total target ({self.total_latency_targets['total']}ms)"
            )
        
        # Validate individual components
        if self.stt.target_latency <= 0:
            errors.append("STT target latency must be positive")
            
        if self.llm.primary_llm_config["max_tokens"] < 50:
            errors.append("LLM max_tokens too low for meaningful responses")
            
        if self.tts.target_latency < 50:
            errors.append("TTS target latency too aggressive")
            
        # Check resource constraints
        if self.vector.faiss_optimization["search_threads"] > 8:
            errors.append("FAISS search threads may exceed available cores")
            
        return len(errors) == 0, errors

    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        
        if env == "production":
            return {
                "optimization_level": OptimizationLevel.AGGRESSIVE,
                "monitoring_interval": 5,  # seconds
                "enable_detailed_tracing": True,
                "fallback_mechanisms": True,
                "circuit_breaker_enabled": True
            }
        elif env == "staging":
            return {
                "optimization_level": OptimizationLevel.BALANCED,
                "monitoring_interval": 10,
                "enable_detailed_tracing": True,
                "fallback_mechanisms": True,
                "circuit_breaker_enabled": False
            }
        else:  # development
            return {
                "optimization_level": OptimizationLevel.CONSERVATIVE,
                "monitoring_interval": 30,
                "enable_detailed_tracing": False,
                "fallback_mechanisms": False,
                "circuit_breaker_enabled": False
            }

# Global configuration instances
latency_config = LatencyConfig(LatencyTier.LOW)

# Environment-specific validation
is_valid, validation_errors = latency_config.validate_configuration()
if not is_valid:
    logger.warning(f"Latency configuration issues: {validation_errors}")

# Apply environment overrides
env_overrides = latency_config.get_environment_overrides()
for key, value in env_overrides.items():
    if hasattr(latency_config, key):
        setattr(latency_config, key, value)