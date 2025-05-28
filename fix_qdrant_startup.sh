#!/bin/bash
# Fix Qdrant startup issues and improve reliability

echo "🔧 Fixing Qdrant startup issues..."

# Stop any existing Qdrant processes
echo "Stopping existing Qdrant processes..."
pkill -f qdrant 2>/dev/null || true
sleep 3

# Create Qdrant setup directory with proper structure
QDRANT_DIR="/workspace/qdrant-setup"
mkdir -p "$QDRANT_DIR"/{config,storage,logs}
cd "$QDRANT_DIR"

# Download Qdrant binary if not present or corrupted
if [ ! -f "qdrant" ] || [ ! -x "qdrant" ]; then
    echo "📦 Downloading fresh Qdrant binary..."
    
    # Remove any existing corrupted binary
    rm -f qdrant* 2>/dev/null || true
    
    # Determine architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            QDRANT_URL="https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz"
            ;;
        aarch64|arm64)
            QDRANT_URL="https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-aarch64-unknown-linux-gnu.tar.gz"
            ;;
        *)
            echo "❌ Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    # Download with retry logic
    for attempt in 1 2 3; do
        echo "Download attempt $attempt/3..."
        if wget -q --timeout=60 --tries=3 "$QDRANT_URL" -O qdrant.tar.gz; then
            echo "✅ Downloaded successfully"
            break
        elif [ $attempt -eq 3 ]; then
            echo "❌ Failed to download Qdrant after 3 attempts"
            exit 1
        else
            echo "⚠️ Download failed, retrying..."
            sleep 5
        fi
    done
    
    # Extract and verify
    if tar -xzf qdrant.tar.gz; then
        chmod +x qdrant
        rm qdrant.tar.gz
        echo "✅ Qdrant binary extracted and made executable"
    else
        echo "❌ Failed to extract Qdrant binary"
        exit 1
    fi
fi

# Create optimized Qdrant configuration
cat > config/production.yaml << 'EOF'
service:
  host: "127.0.0.1"
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  max_request_size_mb: 16
  log_level: "INFO"

storage:
  storage_path: "./storage"
  
  # Optimized for RunPod environment
  performance:
    max_search_threads: 2
    max_optimization_threads: 1
    search_batch_size: 50
    max_concurrent_searches: 100
    max_indexing_threads: 1
    
  # Memory-optimized settings
  optimizers:
    deleted_threshold: 0.3
    vacuum_min_vector_number: 500
    default_segment_number: 1
    max_segment_size: 10000
    memmap_threshold: 10000
    flush_interval_sec: 5
    max_optimization_threads: 1
    
  # HNSW settings for smaller memory footprint
  hnsw:
    m: 8
    ef_construct: 64
    full_scan_threshold: 5000
    max_indexing_threads: 1
    on_disk: false
    ef: 32

# Disable telemetry for faster startup
telemetry:
  enabled: false

# Disable clustering for single instance
cluster:
  enabled: false

# Enable health checks
web_ui: false
EOF

echo "✅ Created optimized Qdrant configuration"

# Create startup script with better error handling
cat > start_qdrant.sh << 'EOF'
#!/bin/bash
# Qdrant startup script with health checking

QDRANT_DIR="/workspace/qdrant-setup"
cd "$QDRANT_DIR"

# Function to check if Qdrant is running
check_qdrant() {
    curl -s http://127.0.0.1:6333/health >/dev/null 2>&1
}

# Function to wait for Qdrant to start
wait_for_qdrant() {
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for Qdrant to start..."
    while [ $attempt -lt $max_attempts ]; do
        if check_qdrant; then
            echo "✅ Qdrant is ready"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    echo "❌ Qdrant failed to start after ${max_attempts} attempts"
    return 1
}

# Stop any existing Qdrant
pkill -f qdrant 2>/dev/null || true
sleep 2

# Clean up old log files
rm -f logs/qdrant.log logs/qdrant.err 2>/dev/null || true

# Start Qdrant with proper logging
echo "🚀 Starting Qdrant..."
nohup ./qdrant --config-path config/production.yaml > logs/qdrant.log 2> logs/qdrant.err &
QDRANT_PID=$!

# Save PID
echo $QDRANT_PID > qdrant.pid
echo "Started Qdrant with PID: $QDRANT_PID"

# Wait for startup and verify
if wait_for_qdrant; then
    echo "✅ Qdrant started successfully"
    
    # Test basic functionality
    if curl -s http://127.0.0.1:6333/collections >/dev/null; then
        echo "✅ Qdrant API responding correctly"
        exit 0
    else
        echo "⚠️ Qdrant API not responding properly"
        exit 1
    fi
else
    echo "❌ Qdrant startup failed"
    echo "Last 20 lines of error log:"
    tail -20 logs/qdrant.err 2>/dev/null || echo "No error log available"
    exit 1
fi
EOF

chmod +x start_qdrant.sh

# Create stop script
cat > stop_qdrant.sh << 'EOF'
#!/bin/bash
# Qdrant stop script

QDRANT_DIR="/workspace/qdrant-setup"
cd "$QDRANT_DIR"

echo "🛑 Stopping Qdrant..."

# Try graceful shutdown first
if [ -f qdrant.pid ]; then
    PID=$(cat qdrant.pid)
    if ps -p $PID > /dev/null; then
        echo "Sending TERM signal to PID $PID"
        kill -TERM $PID
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! ps -p $PID > /dev/null; then
                echo "✅ Qdrant stopped gracefully"
                rm -f qdrant.pid
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if needed
        echo "Forcing kill of PID $PID"
        kill -KILL $PID
        rm -f qdrant.pid
    fi
fi

# Kill any remaining qdrant processes
pkill -f qdrant 2>/dev/null || true
sleep 2

echo "✅ Qdrant stopped"
EOF

chmod +x stop_qdrant.sh

# Try to start Qdrant now
echo "🚀 Attempting to start Qdrant..."
if ./start_qdrant.sh; then
    echo "✅ Qdrant is now running successfully"
    
    # Show status
    echo "📊 Qdrant Status:"
    curl -s http://127.0.0.1:6333/health | head -3 || echo "Health check failed"
    
else
    echo "❌ Qdrant startup failed"
    echo "📋 Checking logs for errors..."
    
    if [ -f logs/qdrant.err ]; then
        echo "Error log contents:"
        cat logs/qdrant.err
    fi
    
    if [ -f logs/qdrant.log ]; then
        echo "Main log contents:"
        tail -20 logs/qdrant.log
    fi
    
    echo "⚠️ System will continue with in-memory vector storage"
fi

echo "🔧 Qdrant setup completed"
echo "   Start: /workspace/qdrant-setup/start_qdrant.sh"
echo "   Stop:  /workspace/qdrant-setup/stop_qdrant.sh"
echo "   Logs:  /workspace/qdrant-setup/logs/"