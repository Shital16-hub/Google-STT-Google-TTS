#!/bin/bash
# Enhanced Qdrant debugging and restart script

echo "🔍 Deep Qdrant Debugging..."

# Check process details
echo "📋 Qdrant process info:"
ps aux | grep qdrant | grep -v grep

# Check what's listening on port 6333
echo "🔌 Port 6333 listeners:"
netstat -tlnp | grep 6333 || ss -tlnp | grep 6333

# Check if it's actually Qdrant responding
echo "🧪 Testing raw HTTP connection:"
timeout 5 telnet localhost 6333 << EOF
GET /health HTTP/1.1
Host: localhost
Connection: close

EOF

echo "📋 Checking Qdrant logs..."
if [ -f "/workspace/qdrant-setup/qdrant.log" ]; then
    echo "Recent Qdrant logs:"
    tail -30 /workspace/qdrant-setup/qdrant.log
else
    echo "No qdrant.log found in /workspace/qdrant-setup/"
    # Check other possible locations
    find /workspace -name "*.log" -path "*qdrant*" 2>/dev/null
fi

echo "🛑 Stopping all Qdrant processes..."
pkill -f qdrant
sleep 3

# Make sure nothing is still running
if pgrep -f qdrant > /dev/null; then
    echo "⚠️ Qdrant still running, force killing..."
    pkill -9 -f qdrant
    sleep 2
fi

echo "🧹 Cleaning up any stale processes on port 6333..."
# Kill anything using port 6333
lsof -ti:6333 | xargs -r kill -9

echo "🚀 Starting fresh Qdrant instance..."

# Go to setup directory
cd /workspace/qdrant-setup

# Check if Qdrant binary exists
if [ ! -f "./qdrant" ]; then
    echo "❌ Qdrant binary not found! Let's reinstall..."
    
    # Re-download Qdrant
    QDRANT_VERSION="v1.7.0"
    echo "📦 Downloading Qdrant $QDRANT_VERSION..."
    wget -q https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz
    
    if [ $? -eq 0 ]; then
        echo "📂 Extracting Qdrant..."
        tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz --no-same-owner 2>/dev/null || tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
        chmod +x qdrant
    else
        echo "❌ Failed to download Qdrant"
        exit 1
    fi
fi

# Ensure config exists
if [ ! -f "config/production.yaml" ]; then
    echo "📝 Creating Qdrant configuration..."
    mkdir -p config storage/snapshots storage/temp
    
    cat > config/production.yaml << 'EOF'
service:
  host: "0.0.0.0"
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  cors_allow_origin: "*"
  max_request_size_mb: 32
  max_workers: 4
  log_level: "INFO"

storage:
  storage_path: "./storage"
  snapshots_path: "./storage/snapshots"
  temp_path: "./storage/temp"
  
  performance:
    max_search_threads: 2
    max_optimization_threads: 1
    search_batch_size: 100
    max_concurrent_searches: 100

telemetry:
  enabled: false

cluster:
  enabled: false
EOF
fi

echo "🚀 Starting Qdrant with verbose logging..."
# Start with explicit logging
./qdrant --config-path config/production.yaml 2>&1 | tee qdrant.log &
QDRANT_PID=$!
echo $QDRANT_PID > qdrant.pid

echo "⏳ Waiting for Qdrant to start (PID: $QDRANT_PID)..."
sleep 5

# Check if process is still running
if kill -0 $QDRANT_PID 2>/dev/null; then
    echo "✅ Qdrant process is running (PID: $QDRANT_PID)"
else
    echo "❌ Qdrant process died! Checking logs..."
    tail -20 qdrant.log
    exit 1
fi

# Test connection multiple times
echo "🧪 Testing Qdrant connection..."
for i in {1..10}; do
    echo "Attempt $i/10..."
    
    if curl -f -s --connect-timeout 2 http://localhost:6333/health > /dev/null; then
        echo "✅ Qdrant is responding!"
        
        # Get full health info
        echo "📊 Qdrant health status:"
        curl -s http://localhost:6333/health | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || curl -s http://localhost:6333/health
        
        echo "📊 Qdrant collections:"
        curl -s http://localhost:6333/collections | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || curl -s http://localhost:6333/collections
        
        echo "✅ Qdrant is fully operational!"
        exit 0
    else
        echo "⏳ Not ready yet, waiting..."
        sleep 2
    fi
done

echo "❌ Qdrant failed to respond after 10 attempts"
echo "📋 Process status:"
ps aux | grep qdrant | grep -v grep

echo "📋 Latest logs:"
tail -30 qdrant.log

echo "🔌 Network status:"
netstat -tlnp | grep 6333 || ss -tlnp | grep 6333

exit 1
EOF