"""
Production Settings for Multi-Agent Voice AI System
Comprehensive production-ready configuration with security, monitoring, and scalability
"""

import os
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class SecurityConfig:
    """Security configuration for production deployment"""
    # API Security
    api_key_required: bool = True
    api_key_header: str = "X-API-Key"
    api_rate_limiting: bool = True
    max_requests_per_minute: int = 1000
    max_requests_per_hour: int = 10000
    
    # JWT Configuration
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # CORS Settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: [
        "https://yourdomain.com",
        "https://api.yourdomain.com"
    ])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # SSL/TLS
    ssl_enabled: bool = True
    ssl_cert_path: str = "/etc/ssl/certs/app.crt"
    ssl_key_path: str = "/etc/ssl/private/app.key"
    ssl_ca_path: Optional[str] = "/etc/ssl/certs/ca.crt"
    min_tls_version: str = "1.2"
    
    # Data Encryption
    encryption_key: str = os.getenv("ENCRYPTION_KEY", secrets.token_urlsafe(32))
    encrypt_sensitive_data: bool = True
    hash_algorithm: str = "SHA-256"
    
    # Input Validation
    max_input_length: int = 10000
    sanitize_inputs: bool = True
    validate_audio_format: bool = True
    allowed_audio_formats: List[str] = field(default_factory=lambda: ["wav", "mp3", "m4a"])
    
    # Session Security
    session_timeout_minutes: int = 30
    secure_cookies: bool = True
    httponly_cookies: bool = True
    same_site_cookies: str = "strict"

@dataclass
class MonitoringConfig:
    """Comprehensive monitoring and observability configuration"""
    # Metrics Collection
    metrics_enabled: bool = True
    metrics_endpoint: str = "/metrics"
    metrics_port: int = 8000
    collection_interval: int = 10  # seconds
    
    # Prometheus Configuration
    prometheus_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "namespace": "voice_ai",
        "subsystem": "multi_agent",
        "labels": {
            "service": "voice-ai-system",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "production")
        },
        "buckets": [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    })
    
    # Health Checks
    health_check_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "endpoint": "/health",
        "detailed_endpoint": "/health/detailed",
        "interval": 30,  # seconds
        "timeout": 5,    # seconds
        "checks": [
            "database_connection",
            "vector_db_connection", 
            "redis_connection",
            "external_apis",
            "disk_space",
            "memory_usage"
        ]
    })
    
    # Distributed Tracing
    tracing_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "service_name": "voice-ai-multi-agent",
        "jaeger_endpoint": os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
        "sampling_rate": 0.1,  # 10% sampling in production
        "trace_all_requests": False,
        "trace_database_queries": True,
        "trace_external_calls": True
    })
    
    # Logging Configuration
    logging_config: Dict[str, Any] = field(default_factory=lambda: {
        "level": LogLevel.INFO.value,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": ["console", "file", "elasticsearch"],
        "file_path": "/var/log/voice-ai/app.log",
        "max_file_size": "100MB",
        "backup_count": 10,
        "elasticsearch_host": os.getenv("ELASTICSEARCH_HOST", "localhost:9200"),
        "elasticsearch_index": "voice-ai-logs"
    })
    
    # Alerting Configuration
    alerting_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "pagerduty_key": os.getenv("PAGERDUTY_INTEGRATION_KEY"),
        "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
        "email_smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "email_smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "alert_rules": {
            "high_latency": {"threshold": 1000, "duration": "5m"},
            "error_rate": {"threshold": 0.05, "duration": "2m"},
            "memory_usage": {"threshold": 0.85, "duration": "10m"},
            "disk_usage": {"threshold": 0.80, "duration": "5m"}
        }
    })

@dataclass
class ScalabilityConfig:
    """Scalability and performance configuration"""
    # Application Scaling
    min_workers: int = 4
    max_workers: int = 32
    worker_scaling_threshold: float = 0.8  # CPU utilization
    worker_timeout: int = 300  # seconds
    
    # Database Connection Pooling
    db_pool_config: Dict[str, Any] = field(default_factory=lambda: {
        "min_connections": 5,
        "max_connections": 100,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600,  # 1 hour
        "pool_pre_ping": True
    })
    
    # Redis Connection Pooling
    redis_pool_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_connections": 100,
        "retry_on_timeout": True,
        "socket_keepalive": True,
        "socket_keepalive_options": {
            "TCP_KEEPINTVL": 1,
            "TCP_KEEPCNT": 3,
            "TCP_USER_TIMEOUT": 5000
        }
    })
    
    # Auto-scaling Configuration
    autoscaling_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "cpu_target": 70,      # Target CPU utilization %
        "memory_target": 80,   # Target memory utilization %
        "scale_up_threshold": 80,
        "scale_down_threshold": 30,
        "cooldown_period": 300,  # seconds
        "min_replicas": 2,
        "max_replicas": 20
    })
    
    # Load Balancing
    load_balancer_config: Dict[str, Any] = field(default_factory=lambda: {
        "algorithm": "round_robin",  # round_robin, least_connections, ip_hash
        "health_check_interval": 30,
        "unhealthy_threshold": 3,
        "healthy_threshold": 2,
        "timeout": 5,
        "sticky_sessions": False
    })

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    # Caching Strategy
    caching_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "default_ttl": 300,  # 5 minutes
        "max_memory": "2GB",
        "eviction_policy": "allkeys-lru",
        "cache_layers": {
            "l1": {"type": "memory", "size": "512MB", "ttl": 60},
            "l2": {"type": "redis", "size": "2GB", "ttl": 300},
            "l3": {"type": "disk", "size": "10GB", "ttl": 3600}
        }
    })
    
    # Request Processing
    request_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_request_size": "50MB",
        "request_timeout": 30,  # seconds
        "keep_alive_timeout": 65,
        "max_keepalive_requests": 1000,
        "enable_compression": True,
        "compression_level": 6,
        "compression_threshold": 1024  # bytes
    })
    
    # Async Processing
    async_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_concurrent_requests": 1000,
        "task_queue_size": 10000,
        "worker_concurrency": 4,
        "background_task_timeout": 300,
        "enable_request_batching": True,
        "batch_size": 10,
        "batch_timeout": 100  # milliseconds
    })

@dataclass
class BackupConfig:
    """Backup and disaster recovery configuration"""
    # Database Backups
    database_backup: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "schedule": "0 2 * * *",  # Daily at 2 AM
        "retention_days": 30,
        "backup_location": "/backups/database",
        "compression": True,
        "encryption": True
    })
    
    # Vector Database Backups
    vector_backup: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "schedule": "0 3 * * *",  # Daily at 3 AM
        "retention_days": 14,
        "backup_location": "/backups/vectors",
        "incremental": True,
        "compression": True
    })
    
    # Configuration Backups
    config_backup: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "schedule": "0 4 * * 0",  # Weekly on Sunday at 4 AM
        "retention_weeks": 12,
        "backup_location": "/backups/config",
        "version_control": True
    })
    
    # Disaster Recovery
    disaster_recovery: Dict[str, Any] = field(default_factory=lambda: {
        "rto": 4,  # Recovery Time Objective (hours)
        "rpo": 1,  # Recovery Point Objective (hours)
        "backup_regions": ["us-west-2", "us-east-1"],
        "failover_automatic": True,
        "health_check_interval": 60,  # seconds
        "failover_threshold": 3  # failed checks before failover
    })

class ProductionSettings:
    """Main production configuration class"""
    
    def __init__(self, environment: Environment = Environment.PRODUCTION):
        self.environment = environment
        
        # ✅ Basic Application Configuration
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8000"))
        self.WORKERS = int(os.getenv("WORKERS", "1"))
        
        # ✅ CORS settings
        allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
        if allowed_origins_env:
            self.ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_env.split(",")]
        else:
            self.ALLOWED_ORIGINS = ["*"]  # Default for development
        
        # ✅ Base URL for webhook callbacks
        self.BASE_URL = os.getenv("BASE_URL", f"http://localhost:{self.PORT}")
        
        # Core configurations
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.scalability = ScalabilityConfig()
        self.performance = PerformanceConfig()
        self.backup = BackupConfig()
        
        # Environment-specific overrides
        self._apply_environment_settings()
        
        # Feature flags
        self.feature_flags = self._init_feature_flags()
        
        # Resource limits
        self.resource_limits = self._init_resource_limits()

    def _apply_environment_settings(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.DEVELOPMENT:
            # Development overrides
            self.DEBUG = True  # ✅ Override for development
            self.security.api_rate_limiting = False
            self.security.ssl_enabled = False
            self.monitoring.tracing_config["sampling_rate"] = 1.0
            self.monitoring.logging_config["level"] = LogLevel.DEBUG.value
            self.scalability.min_workers = 1
            self.scalability.max_workers = 4
            
        elif self.environment == Environment.STAGING:
            # Staging overrides
            self.DEBUG = False  # ✅ Override for staging
            self.security.max_requests_per_minute = 500
            self.monitoring.tracing_config["sampling_rate"] = 0.5
            self.scalability.min_workers = 2
            self.scalability.max_workers = 8
            
        else:  # Production
            # Production settings (defaults are already production-ready)
            self.DEBUG = False  # ✅ Override for production
            pass

    def _init_feature_flags(self) -> Dict[str, bool]:
        """Initialize feature flags for gradual rollouts"""
        return {
            "enable_new_tts_engine": os.getenv("FF_NEW_TTS_ENGINE", "false").lower() == "true",
            "enable_advanced_routing": os.getenv("FF_ADVANCED_ROUTING", "true").lower() == "true",
            "enable_conversation_memory": os.getenv("FF_CONVERSATION_MEMORY", "true").lower() == "true",
            "enable_tool_orchestration": os.getenv("FF_TOOL_ORCHESTRATION", "false").lower() == "true",
            "enable_multi_language": os.getenv("FF_MULTI_LANGUAGE", "false").lower() == "true",
            "enable_voice_cloning": os.getenv("FF_VOICE_CLONING", "false").lower() == "true",
            "enable_sentiment_analysis": os.getenv("FF_SENTIMENT_ANALYSIS", "true").lower() == "true",
            "enable_analytics": os.getenv("FF_ANALYTICS", "true").lower() == "true"
        }

    def _init_resource_limits(self) -> Dict[str, Dict[str, Any]]:
        """Initialize resource limits and quotas"""
        return {
            "compute": {
                "max_cpu_cores": int(os.getenv("MAX_CPU_CORES", "16")),
                "max_memory_gb": int(os.getenv("MAX_MEMORY_GB", "32")),
                "max_concurrent_sessions": int(os.getenv("MAX_CONCURRENT_SESSIONS", "1000"))
            },
            "storage": {
                "max_vector_storage_gb": int(os.getenv("MAX_VECTOR_STORAGE_GB", "100")),
                "max_audio_storage_gb": int(os.getenv("MAX_AUDIO_STORAGE_GB", "50")),
                "max_log_storage_gb": int(os.getenv("MAX_LOG_STORAGE_GB", "20"))
            },
            "network": {
                "max_bandwidth_mbps": int(os.getenv("MAX_BANDWIDTH_MBPS", "1000")),
                "max_requests_per_second": int(os.getenv("MAX_RPS", "5000")),
                "max_websocket_connections": int(os.getenv("MAX_WS_CONNECTIONS", "10000"))
            },
            "api_quotas": {
                "openai_requests_per_minute": int(os.getenv("OPENAI_RPM", "3500")),
                "google_stt_requests_per_minute": int(os.getenv("GOOGLE_STT_RPM", "1000")),
                "google_tts_requests_per_minute": int(os.getenv("GOOGLE_TTS_RPM", "1000"))
            }
        }

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration with production optimizations"""
        return {
            "postgresql": {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DB", "voice_ai"),
                "username": os.getenv("POSTGRES_USER", "voice_ai"),
                "password": os.getenv("POSTGRES_PASSWORD"),
                "pool_size": self.scalability.db_pool_config["max_connections"],
                "max_overflow": self.scalability.db_pool_config["max_overflow"],
                "ssl_mode": "require" if self.environment == Environment.PRODUCTION else "prefer",
                "statement_timeout": 30000,  # 30 seconds
                "lock_timeout": 5000,        # 5 seconds
                "idle_in_transaction_session_timeout": 300000  # 5 minutes
            }
        }

    def get_external_api_config(self) -> Dict[str, Dict[str, Any]]:
        """Get external API configurations with production settings"""
        return {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "max_retries": 3,
                "timeout": 30,
                "rate_limit_rpm": self.resource_limits["api_quotas"]["openai_requests_per_minute"]
            },
            "google_cloud": {
                "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
                "stt_config": {
                    "timeout": 30,
                    "max_retries": 2,
                    "rate_limit_rpm": self.resource_limits["api_quotas"]["google_stt_requests_per_minute"]
                },
                "tts_config": {
                    "timeout": 30,
                    "max_retries": 2,
                    "rate_limit_rpm": self.resource_limits["api_quotas"]["google_tts_requests_per_minute"]
                }
            },
            "twilio": {
                "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
                "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
                "webhook_url": os.getenv("TWILIO_WEBHOOK_URL"),
                "timeout": 30
            }
        }

    def validate_configuration(self) -> tuple[bool, List[str]]:
        """Validate production configuration"""
        errors = []
        
        # Check required environment variables
        required_vars = [
            "POSTGRES_PASSWORD", "OPENAI_API_KEY", "TWILIO_ACCOUNT_SID", 
            "TWILIO_AUTH_TOKEN", "JWT_SECRET_KEY"
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                errors.append(f"Required environment variable {var} not set")
        
        # Validate SSL configuration in production
        if self.environment == Environment.PRODUCTION and self.security.ssl_enabled:
            if not os.path.exists(self.security.ssl_cert_path):
                errors.append(f"SSL certificate not found: {self.security.ssl_cert_path}")
            if not os.path.exists(self.security.ssl_key_path):
                errors.append(f"SSL private key not found: {self.security.ssl_key_path}")
        
        # Validate resource limits
        if self.resource_limits["compute"]["max_memory_gb"] < 4:
            errors.append("Minimum 4GB memory required for production")
        
        # Validate monitoring configuration
        if self.environment == Environment.PRODUCTION:
            if not self.monitoring.alerting_config["pagerduty_key"]:
                errors.append("PagerDuty integration key required for production")
        
        return len(errors) == 0, errors

# Global production settings instance
current_env = Environment(os.getenv("ENVIRONMENT", "production").lower())
production_settings = ProductionSettings(current_env)

# Validate configuration on import
is_valid, config_errors = production_settings.validate_configuration()
if not is_valid:
    logger.error(f"Production configuration validation failed: {config_errors}")
    if current_env == Environment.PRODUCTION:
        raise RuntimeError("Production configuration validation failed")
    else:
        logger.warning("Non-production environment, continuing with warnings")

logger.info(f"Production settings initialized for {current_env.value} environment")