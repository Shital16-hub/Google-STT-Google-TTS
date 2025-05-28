#!/bin/bash
# Enhanced RunPod Environment Setup Script for Multi-Agent Voice AI System
# Fixed version with proper error handling and service management

set -e  # Exit on any error

echo "🚀 Setting up Multi-Agent Voice AI System on RunPod..."
echo "📅 $(date)"
echo "🖥️ System: $(uname -a)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=${3:-30}
    local attempt=0
    
    log "Waiting for $service_name to start..."
    
    while [ $attempt -lt $max_attempts ]; do
        if eval "$check_command" >/dev/null 2>&1; then
            log "✅ $service_name is ready"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    error "❌ $service_name failed to start after $((max_attempts * 2)) seconds"
    return 1
}

# Update system packages
log "📦 Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y || warn "Package update had some issues, continuing..."

# Install essential system dependencies
log "🔧 Installing essential system dependencies..."
apt-get install -y \
    build-essential \
    cmake \
    curl \
    wget \
    git \
    htop \
    nano \
    vim \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libffi-dev \
    libssl-dev \
    python3-dev \
    python3-pip \
    supervisor \
    || warn "Some system packages failed to install, continuing..."

# Install Redis with multiple approaches
log "📊 Installing and configuring Redis..."

install_redis() {
    if command_exists redis-server; then
        log "✅ Redis already installed"
        return 0
    fi
    
    # Try official Redis installation
    log "Installing Redis via apt..."
    apt-get install -y redis-server redis-tools || {
        warn "Official Redis installation failed, trying alternative..."
        
        # Try building from source as fallback
        log "Building Redis from source..."
        cd /tmp
        wget http://download.redis.io/redis-stable.tar.gz || return 1
        tar xzf redis-stable.tar.gz || return 1
        cd redis-stable
        make && make install || return 1
        
        # Create Redis user and directories
        useradd --system --home /var/lib/redis --shell /bin/false redis || true
        mkdir -p /var/lib/redis /var/log/redis /etc/redis
        chown redis:redis /var/lib/redis /var/log/redis
        
        # Create basic config
        cat > /etc/redis/redis.conf << 'EOF'
bind 127.0.0.1
port 6379
daemonize yes
supervised systemd
pidfile /var/run/redis/redis-server.pid
logfile /var/log/redis/redis-server.log
dir /var/lib/redis
maxmemory 2gb
maxmemory-policy allkeys-lru
EOF
        
        log "✅ Redis built and configured from source"
    }
}

install_redis

# Configure and start Redis
configure_redis() {
    log "⚙️ Configuring Redis for optimal performance..."
    
    # Create Redis directories
    mkdir -p /var/lib/redis /var/log/redis /var/run/redis /etc/redis
    chown redis:redis /var/lib/redis /var/log/redis /var/run/redis 2>/dev/null || true
    
    # Create optimized Redis configuration
    cat > /etc/redis/redis.conf << 'EOF'
# Redis Configuration for Voice AI System
bind 127.0.0.1
port 6379
daemonize yes
supervised auto
pidfile /var/run/redis/redis-server.pid
logfile /var/log/redis/redis-server.log
dir /var/lib/redis

# Memory optimization
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Performance settings
tcp-keepalive 300
timeout 0
tcp-backlog 511
databases 16

# Persistence settings
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename voice-ai-cache.rdb

# AOF settings
appendonly yes
appendfilename "voice-ai-cache.aof"
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Client settings
maxclients 10000
timeout 0

# Security
protected-mode yes
EOF

    # Fix permissions
    chown redis:redis /etc/redis/redis.conf 2>/dev/null || true
    chmod 640 /etc/redis/redis.conf 2>/dev/null || true
}

configure_redis

# Start Redis with multiple methods
start_redis() {
    log "🚀 Starting Redis server..."
    
    # Stop any existing Redis processes
    pkill redis-server 2>/dev/null || true
    sleep 2
    
    # Try different startup methods
    if systemctl is-active --quiet redis-server 2>/dev/null; then
        log "Redis already running via systemctl"
        return 0
    fi
    
    # Method 1: systemctl
    if command_exists systemctl && systemctl start redis-server 2>/dev/null; then
        log "Started Redis via systemctl"
        return 0
    fi
    
    # Method 2: service command
    if command_exists service && service redis-server start 2>/dev/null; then
        log "Started Redis via service command"
        return 0
    fi
    
    # Method 3: direct redis-server
    if command_exists redis-server; then
        redis-server /etc/redis/redis.conf 2>/dev/null || \
        redis-server --daemonize yes --port 6379 --bind 127.0.0.1 2>/dev/null || \
        redis-server --daemonize yes 2>/dev/null || {
            warn "All Redis startup methods failed, starting in background..."
            nohup redis-server > /var/log/redis/redis.log 2>&1 &
        }
    else
        error "Redis server not found!"
        return 1
    fi
    
    sleep 3
}

start_redis

# Verify Redis is working
if wait_for_service "Redis" "redis-cli ping | grep -q PONG"; then
    log "✅ Redis is running successfully"
    redis-cli info server | head -10 || true
else
    warn "⚠️ Redis may not be responding properly"
fi

# Install Qdrant
log "🗄️ Setting up Qdrant vector database..."

setup_qdrant() {
    local qdrant_setup_dir="/workspace/qdrant-setup"
    
    # Create Qdrant setup directory
    mkdir -p "$qdrant_setup_dir"/{config,storage}
    cd "$qdrant_setup_dir"
    
    # Download Qdrant binary if not present
    if [ ! -f "qdrant" ]; then
        log "📦 Downloading Qdrant binary..."
        
        # Determine architecture
        local arch=$(uname -m)
        local qdrant_url
        
        case $arch in
            x86_64)
                qdrant_url="https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz"
                ;;
            aarch64|arm64)
                qdrant_url="https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-aarch64-unknown-linux-gnu.tar.gz"
                ;;
            *)
                error "Unsupported architecture: $arch"
                return 1
                ;;
        esac
        
        if wget -q --timeout=60 "$qdrant_url" -O qdrant.tar.gz; then
            tar -xzf qdrant.tar.gz || return 1
            chmod +x qdrant
            rm qdrant.tar.gz
            log "✅ Qdrant binary downloaded and extracted"
        else
            warn "Failed to download Qdrant binary, will use fallback mode"
            return 1
        fi
    fi
    
    # Create Qdrant configuration
    cat > config/production.yaml << 'EOF'
service:
  host: "0.0.0.0"
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  max_request_size_mb: 32
  log_level: "INFO"

storage:
  storage_path: "./storage"
  performance:
    max_search_threads: 4
    max_optimization_threads: 2
  optimizers:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000
    default_segment_number: 2
    max_segment_size: 20000
    flush_interval_sec: 2

telemetry:
  enabled: false
EOF
    
    log "✅ Qdrant configuration created"
    return 0
}

setup_qdrant

# Install Python dependencies
log "🐍 Installing Python dependencies..."

# Upgrade pip first
python3 -m pip install --upgrade pip setuptools wheel || warn "Pip upgrade had issues"

# Create workspace directory if needed
mkdir -p /workspace/logs
cd /workspace

# Install dependencies with better error handling
install_python_deps() {
    local requirements_file="/workspace/requirements.txt"
    
    if [ -f "$requirements_file" ]; then
        log "Installing from requirements.txt..."
        python3 -m pip install -r "$requirements_file" || {
            warn "Full requirements install failed, trying essential packages..."
            
            # Install essential packages one by one
            local essential_packages=(
                "fastapi"
                "uvicorn[standard]"
                "redis"
                "numpy"
                "scipy"
                "pydantic"
                "python-dotenv"
                "pyyaml"
                "aiofiles"
                "asyncio"
                "httpx"
                "requests"
            )
            
            for package in "${essential_packages[@]}"; do
                python3 -m pip install "$package" || warn "Failed to install $package"
            done
        }
    else
        warn "requirements.txt not found, installing essential packages..."
        python3 -m pip install fastapi uvicorn redis numpy pydantic python-dotenv pyyaml
    fi
}

install_python_deps

# Set up process management with Supervisor
log "🔧 Setting up process management..."

# Create supervisor config for Redis
cat > /etc/supervisor/conf.d/redis.conf << 'EOF'
[program:redis]
command=/usr/bin/redis-server /etc/redis/redis.conf
autostart=true
autorestart=true
user=redis
stdout_logfile=/var/log/redis/redis-supervisor.log
stderr_logfile=/var/log/redis/redis-supervisor-error.log
EOF

# Create supervisor config for Qdrant
cat > /etc/supervisor/conf.d/qdrant.conf << 'EOF'
[program:qdrant]
command=/workspace/qdrant-setup/qdrant --config-path /workspace/qdrant-setup/config/production.yaml
directory=/workspace/qdrant-setup
autostart=true
autorestart=true
user=root
stdout_logfile=/workspace/qdrant-setup/qdrant-supervisor.log
stderr_logfile=/workspace/qdrant-setup/qdrant-supervisor-error.log
EOF

# Update supervisor and start services
if command_exists supervisorctl; then
    supervisorctl reread || true
    supervisorctl update || true
    supervisorctl start redis || warn "Failed to start Redis via supervisor"
    sleep 2
    supervisorctl start qdrant || warn "Failed to start Qdrant via supervisor"
else
    warn "Supervisor not available, services started manually"
fi

# Create startup script
log "📝 Creating system startup script..."

cat > /workspace/start_services.sh << 'EOF'
#!/bin/bash
# Service startup script for Voice AI System

echo "🚀 Starting Voice AI System Services..."

# Start Redis
if ! pgrep -f redis-server > /dev/null; then
    echo "Starting Redis..."
    redis-server /etc/redis/redis.conf 2>/dev/null || \
    redis-server --daemonize yes --port 6379 &
    sleep 2
fi

# Verify Redis
if redis-cli ping | grep -q PONG; then
    echo "✅ Redis is running"
else
    echo "⚠️ Redis may not be responding"
fi

# Start Qdrant
if ! pgrep -f qdrant > /dev/null; then
    echo "Starting Qdrant..."
    cd /workspace/qdrant-setup
    nohup ./qdrant --config-path config/production.yaml > qdrant.log 2>&1 &
    echo $! > qdrant.pid
    sleep 3
fi

# Verify Qdrant
if curl -s http://localhost:6333/health > /dev/null; then
    echo "✅ Qdrant is running"
else
    echo "⚠️ Qdrant may not be responding"
fi

echo "🎯 Service startup completed"
EOF

chmod +x /workspace/start_services.sh

# Create shutdown script
cat > /workspace/stop_services.sh << 'EOF'
#!/bin/bash
# Service shutdown script for Voice AI System

echo "🛑 Stopping Voice AI System Services..."

# Stop application first
if pgrep -f "python.*main.py" > /dev/null; then
    echo "Stopping application..."
    pkill -f "python.*main.py"
    sleep 2
fi

# Stop Qdrant
if pgrep -f qdrant > /dev/null; then
    echo "Stopping Qdrant..."
    pkill -f qdrant
    sleep 2
fi

# Stop Redis
if pgrep -f redis-server > /dev/null; then
    echo "Stopping Redis..."
    redis-cli shutdown 2>/dev/null || pkill redis-server
    sleep 2
fi

echo "✅ All services stopped"
EOF

chmod +x /workspace/stop_services.sh

# Set up system limits and optimization
log "⚡ Optimizing system settings..."

# Increase file descriptor limits
cat >> /etc/security/limits.conf << 'EOF'
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF

# Optimize kernel parameters for network performance
cat >> /etc/sysctl.conf << 'EOF'
# Network optimizations for Voice AI System
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000
EOF

sysctl -p || warn "Failed to apply sysctl settings"

# Set environment variables
log "🔧 Setting up environment variables..."

cat > /workspace/.env << 'EOF'
# Voice AI System Environment Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000
BASE_URL=http://localhost:8000

# Redis Configuration
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# Qdrant Configuration
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333

# Logging
LOG_LEVEL=INFO
LOG_FILE=/workspace/logs/voice-ai.log
EOF

# Test services
log "🧪 Testing services..."

# Test Redis
if redis-cli ping | grep -q PONG; then
    log "✅ Redis test successful"
    redis-cli info memory | grep used_memory_human || true
else
    warn "❌ Redis test failed"
fi

# Test Qdrant (start it first if needed)
if ! pgrep -f qdrant > /dev/null; then
    log "Starting Qdrant for testing..."
    cd /workspace/qdrant-setup
    nohup ./qdrant --config-path config/production.yaml > qdrant-test.log 2>&1 &
    sleep 5
fi

if curl -s http://localhost:6333/health > /dev/null; then
    log "✅ Qdrant test successful"
    curl -s http://localhost:6333/health | head -5 || true
else
    warn "❌ Qdrant test failed"
fi

# Create helpful aliases
log "📋 Creating helpful aliases..."

cat >> ~/.bashrc << 'EOF'
# Voice AI System aliases
alias va-start='/workspace/start_services.sh'
alias va-stop='/workspace/stop_services.sh'
alias va-status='echo "Redis: $(redis-cli ping 2>/dev/null || echo "DOWN")" && echo "Qdrant: $(curl -s http://localhost:6333/health > /dev/null && echo "UP" || echo "DOWN")"'
alias va-logs='tail -f /workspace/logs/voice-ai.log'
alias va-redis='redis-cli'
alias va-qdrant='curl -s http://localhost:6333/health | jq .'
EOF

# Final setup summary
log "🎉 RunPod setup completed successfully!"
echo ""
echo "📊 Setup Summary:"
echo "=================="
echo "🔴 Redis: $(redis-cli ping 2>/dev/null || echo "Not responding")"
echo "🟢 Qdrant: $(curl -s http://localhost:6333/health > /dev/null && echo "Running" || echo "Not responding")"
echo ""
echo "🚀 Quick Start Commands:"
echo "========================"
echo "• Start services: /workspace/start_services.sh"
echo "• Stop services: /workspace/stop_services.sh"
echo "• Check status: va-status (after sourcing ~/.bashrc)"
echo "• View logs: va-logs"
echo ""
echo "📁 Important Paths:"
echo "==================="
echo "• Application: /workspace/"
echo "• Logs: /workspace/logs/"
echo "• Redis Config: /etc/redis/redis.conf"
echo "• Qdrant Setup: /workspace/qdrant-setup/"
echo ""
echo "🔧 Environment file created: /workspace/.env"
echo ""
echo "⚡ System optimizations applied:"
echo "• File descriptor limits increased"
echo "• Network parameters optimized"
echo "• Process management configured"
echo ""

# Start services by default
log "🚀 Starting all services..."
/workspace/start_services.sh

echo ""
echo "✅ Setup complete! You can now run your Voice AI application."
echo "💡 Tip: Source your bashrc for aliases: source ~/.bashrc"
echo ""