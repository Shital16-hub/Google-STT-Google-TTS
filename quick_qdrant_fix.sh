#!/bin/bash
# Quick Qdrant fix for existing setup

echo "🔧 Quick Qdrant fix for existing configuration..."

# Check current directory structure
echo "📋 Current setup:"
ls -la /workspace/qdrant-setup/ 2>/dev/null || echo "Qdrant setup directory not found"

# Go to the qdrant setup directory
cd /workspace/qdrant-setup || {
    echo "❌ Qdrant setup directory doesn't exist, creating it..."
    mkdir -p /workspace/qdrant-setup
    cd /workspace/qdrant-setup
}

# Check if binary exists and is executable
if [ -f "qdrant" ] && [ -x "qdrant" ]; then
    echo "✅ Qdrant binary found and executable"
    ./qdrant --version 2>/dev/null || echo "⚠️ Binary may be corrupted"
else
    echo "❌ Qdrant binary missing or not executable"
    echo "📦 Downloading Qdrant binary..."
    
    # Clean up any partial downloads
    rm -f qdrant* 2>/dev/null
    
    # Download fresh binary
    wget -q --timeout=30 "https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz" -O qdrant.tar.gz
    
    if [ $? -eq 0 ]; then
        tar -xzf qdrant.tar.gz
        chmod +x qdrant
        rm qdrant.tar.gz
        echo "✅ Qdrant binary downloaded and extracted"
    else
        echo "❌ Failed to download Qdrant binary"
        echo "🔄 System will continue with in-memory mode"
        exit 1
    fi
fi

# Create config directory if it doesn't exist
mkdir -p config storage logs

# Create a simple, working config
cat > config/simple.yaml << 'EOF'
service:
  host: "0.0.0.0"
  http_port: 6333
  grpc_port: 6334
  enable_cors: true

storage:
  storage_path: "./storage"

telemetry:
  enabled: false
EOF

echo "✅ Created simple Qdrant config"

# Stop any running Qdrant processes
echo "🛑 Stopping existing Qdrant processes..."
pkill -f qdrant 2>/dev/null || true
sleep 3

# Check if port is free
if netstat -tuln 2>/dev/null | grep -q :6333; then
    echo "⚠️ Port 6333 still in use, waiting..."
    sleep 5
fi

# Start Qdrant with simple config
echo "🚀 Starting Qdrant with simple config..."
nohup ./qdrant --config-path config/simple.yaml > logs/qdrant.log 2>&1 &
QDRANT_PID=$!

echo "Started Qdrant with PID: $QDRANT_PID"
echo $QDRANT_PID > qdrant.pid

# Wait for Qdrant to start (with shorter timeout)
echo "⏳ Waiting for Qdrant to start..."
for i in {1..15}; do
    if curl -s http://localhost:6333/health >/dev/null 2>&1; then
        echo "✅ Qdrant is responding on port 6333"
        
        # Test basic functionality
        if curl -s http://localhost:6333/collections >/dev/null 2>&1; then
            echo "✅ Qdrant API is working correctly"
            echo "🎯 Qdrant startup successful!"
            exit 0
        else
            echo "⚠️ Qdrant API not fully ready yet..."
        fi
    fi
    
    echo -n "."
    sleep 2
done

echo ""
echo "❌ Qdrant failed to start properly"

# Check what went wrong
echo "📋 Debugging information:"
echo "Process status:"
ps aux | grep qdrant | grep -v grep || echo "No qdrant processes found"

echo ""
echo "Port check:"
netstat -tuln 2>/dev/null | grep 6333 || echo "Port 6333 not listening"

echo ""
echo "Log file contents:"
if [ -f logs/qdrant.log ]; then
    echo "--- Last 10 lines of qdrant.log ---"
    tail -10 logs/qdrant.log
else
    echo "No log file found"
fi

echo ""
echo "🔄 System will continue with in-memory vector storage"
echo "💡 This won't affect functionality, just means vectors won't persist between restarts"

exit 1