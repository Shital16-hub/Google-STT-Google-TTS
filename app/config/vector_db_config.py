"""
Vector Database Configuration for Multi-Agent Voice AI System
Supports hybrid 3-tier architecture: Redis Cache + FAISS Hot Tier + Qdrant Cold Storage
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VectorDBType(Enum):
    """Supported vector database types"""
    QDRANT = "qdrant"
    FAISS = "faiss"
    REDIS = "redis"

class DistanceMetric(Enum):
    """Vector similarity distance metrics"""
    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

@dataclass
class RedisVectorConfig:
    """Redis vector cache configuration for sub-1ms responses"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    db: int = int(os.getenv("REDIS_DB", "0"))
    
    # Performance settings
    max_connections: int = 100
    connection_pool_max_connections: int = 50
    socket_timeout: float = 1.0
    socket_connect_timeout: float = 1.0
    
    # Cache settings
    default_ttl: int = 300  # 5 minutes
    max_memory_policy: str = "allkeys-lru"
    max_memory: str = "2gb"
    
    # Vector-specific settings
    vector_key_prefix: str = "vector:"
    hot_queries_key: str = "hot_queries"
    cache_hit_threshold: int = 5  # Promote to cache after 5 hits
    
    def get_redis_url(self) -> str:
        """Generate Redis connection URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

@dataclass
class FAISSConfig:
    """FAISS in-memory hot tier configuration for <5ms responses"""
    # Index types
    index_type: str = "IVF"  # IVF, HNSW, Flat
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    
    # HNSW parameters (if using HNSW)
    hnsw_m: int = 16  # Number of bi-directional links for HNSW
    hnsw_ef_construction: int = 200  # Size of dynamic candidate list
    hnsw_ef_search: int = 50  # Size of search list
    
    # Memory management
    max_memory_usage_gb: float = 4.0
    auto_promotion_threshold: int = 100  # queries/hour to promote to FAISS
    auto_demotion_threshold: int = 10   # queries/hour to demote from FAISS
    
    # Performance settings
    use_gpu: bool = False  # Enable if GPU available
    omp_threads: int = 4   # OpenMP threads for parallel processing
    
    # Storage paths
    index_storage_path: str = "./data/faiss_indices"
    metadata_storage_path: str = "./data/faiss_metadata"

@dataclass
class QdrantConfig:
    """Qdrant cold storage configuration for <50ms responses"""
    # Connection settings
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    grpc_port: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    https: bool = os.getenv("QDRANT_HTTPS", "false").lower() == "true"
    
    # Performance settings
    prefer_grpc: bool = True  # Use gRPC for better performance
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Connection pooling
    max_connections: int = 100
    connection_pool_size: int = 20
    
    # Collection settings
    shard_number: int = 1
    replication_factor: int = 1
    write_consistency_factor: int = 1
    on_disk_payload: bool = True  # Store payload on disk to save RAM
    
    # Indexing parameters
    hnsw_config: Dict[str, Any] = field(default_factory=lambda: {
        "m": 16,              # Number of edges per node
        "ef_construct": 100,  # Size of dynamic candidate list
        "full_scan_threshold": 10000,  # Switch to full scan below this
        "max_indexing_threads": 0,     # Use all available threads
        "on_disk": False      # Keep index in memory for speed
    })
    
    # Quantization settings for memory optimization
    scalar_quantization: Dict[str, Any] = field(default_factory=lambda: {
        "type": "int8",       # int8 quantization
        "quantile": 0.99,     # Quantile for clipping
        "always_ram": True    # Keep quantized vectors in RAM
    })
    
    # Storage paths
    storage_path: str = "./data/qdrant_storage"
    snapshots_path: str = "./data/qdrant_snapshots"
    
    def get_qdrant_url(self) -> str:
        """Generate Qdrant connection URL"""
        protocol = "https" if self.https else "http"
        return f"{protocol}://{self.host}:{self.port}"

@dataclass
class CollectionConfig:
    """Configuration for vector collections/indices"""
    name: str
    vector_size: int = 1536  # OpenAI ada-002 embedding size
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # Tier assignments
    redis_enabled: bool = True
    faiss_enabled: bool = True
    qdrant_enabled: bool = True
    
    # Agent-specific settings
    agent_id: Optional[str] = None
    specialization_keywords: List[str] = field(default_factory=list)
    priority_level: int = 1  # 1=highest, 5=lowest
    
    # Performance thresholds
    faiss_promotion_threshold: int = 100  # queries/hour
    redis_cache_ttl: int = 300  # seconds
    auto_optimize_schedule: str = "0 2 * * *"  # Daily at 2 AM
    
    # Metadata schema
    payload_schema: Dict[str, str] = field(default_factory=dict)

class HybridVectorConfig:
    """Main configuration class for hybrid vector database system"""
    
    def __init__(self):
        self.redis = RedisVectorConfig()
        self.faiss = FAISSConfig()
        self.qdrant = QdrantConfig()
        
        # System-wide settings
        self.enable_performance_monitoring: bool = True
        self.enable_auto_optimization: bool = True
        self.enable_health_checks: bool = True
        
        # Latency targets (milliseconds)
        self.latency_targets = {
            "redis": 1,      # <1ms
            "faiss": 5,      # <5ms  
            "qdrant": 50,    # <50ms
            "total": 10      # <10ms overall
        }
        
        # Agent collections configuration
        self.collections = self._init_agent_collections()

    def _init_agent_collections(self) -> Dict[str, CollectionConfig]:
        """Initialize collections for each specialized agent"""
        return {
            "roadside-assistance": CollectionConfig(
                name="roadside_assistance_v1",
                agent_id="roadside-assistance",
                specialization_keywords=["tow", "breakdown", "emergency", "roadside"],
                priority_level=1,  # Highest priority for emergencies
                payload_schema={
                    "procedure_type": "str",
                    "urgency_level": "int",
                    "service_area": "str",
                    "estimated_cost": "float"
                }
            ),
            "billing-support": CollectionConfig(
                name="billing_support_v1", 
                agent_id="billing-support",
                specialization_keywords=["payment", "bill", "refund", "charge"],
                priority_level=2,
                payload_schema={
                    "policy_type": "str",
                    "account_type": "str",
                    "resolution_steps": "list",
                    "escalation_required": "bool"
                }
            ),
            "technical-support": CollectionConfig(
                name="technical_support_v1",
                agent_id="technical-support", 
                specialization_keywords=["technical", "troubleshoot", "manual", "setup"],
                priority_level=3,
                payload_schema={
                    "product_category": "str",
                    "difficulty_level": "int", 
                    "prerequisites": "list",
                    "tools_required": "list"
                }
            )
        }

    def get_collection_config(self, agent_id: str) -> Optional[CollectionConfig]:
        """Get collection configuration for specific agent"""
        return self.collections.get(agent_id)

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        
        if env == "production":
            return {
                "redis": {
                    "max_memory": "8gb",
                    "max_connections": 200
                },
                "faiss": {
                    "max_memory_usage_gb": 16.0,
                    "omp_threads": 8
                },
                "qdrant": {
                    "shard_number": 2,
                    "replication_factor": 2,
                    "max_connections": 200
                }
            }
        elif env == "staging":
            return {
                "redis": {
                    "max_memory": "4gb",
                    "max_connections": 100
                },
                "faiss": {
                    "max_memory_usage_gb": 8.0,
                    "omp_threads": 4
                },
                "qdrant": {
                    "shard_number": 1,
                    "replication_factor": 1,
                    "max_connections": 100
                }
            }
        else:  # development
            return {
                "redis": {
                    "max_memory": "1gb",
                    "max_connections": 50
                },
                "faiss": {
                    "max_memory_usage_gb": 2.0,
                    "omp_threads": 2
                },
                "qdrant": {
                    "shard_number": 1,
                    "replication_factor": 1,
                    "max_connections": 50
                }
            }

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check required environment variables
            required_vars = []
            if not all(os.getenv(var) for var in required_vars):
                logger.warning("Some optional environment variables not set")
            
            # Validate latency targets
            if any(target <= 0 for target in self.latency_targets.values()):
                raise ValueError("All latency targets must be positive")
            
            # Validate collection configurations
            for agent_id, config in self.collections.items():
                if config.vector_size <= 0:
                    raise ValueError(f"Invalid vector size for agent {agent_id}")
                if config.priority_level < 1 or config.priority_level > 5:
                    raise ValueError(f"Priority level must be 1-5 for agent {agent_id}")
            
            logger.info("Vector database configuration validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring and alerting configuration"""
        return {
            "metrics": {
                "collection_interval": 10,  # seconds
                "retention_period": "7d",
                "exporters": ["prometheus", "statsd"]
            },
            "alerts": {
                "latency_threshold_ms": 100,
                "error_rate_threshold": 0.05,
                "memory_usage_threshold": 0.85,
                "disk_usage_threshold": 0.80
            },
            "health_checks": {
                "interval": 30,  # seconds
                "timeout": 5,    # seconds
                "retries": 3
            }
        }

# Global configuration instance
vector_config = HybridVectorConfig()

# Environment-specific overrides
env_config = vector_config.get_environment_config()
for service, settings in env_config.items():
    service_config = getattr(vector_config, service)
    for key, value in settings.items():
        if hasattr(service_config, key):
            setattr(service_config, key, value)

# Validate on import
if not vector_config.validate_config():
    raise RuntimeError("Vector database configuration validation failed")