#!/bin/bash
# Simplified Qdrant startup - sometimes less is more

echo "🔥 Simple Qdrant Restart"

# Kill everything qdrant-related
echo "🛑 Stopping all Qdrant processes..."
pkill -f qdrant
pkill -9 -f qdrant 2>/dev/null
sleep 2

# Clear port 6333
echo "🧹 Clearing port 6333..."
fuser -k 6333/tcp 2>/dev/null

# Go to workspace
cd /workspace

# Create minimal setup
echo "📁 Creating minimal Qdrant setup..."
mkdir -p qdrant-simple
cd qdrant-simple

# Download if needed
if [ ! -f "qdrant" ]; then
    echo "📦 Downloading Qdrant..."
    wget -q -O qdrant-linux.tar.gz "https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz"
    tar -xzf qdrant-linux.tar.gz
    chmod +x qdrant
    rm qdrant-linux.tar.gz
fi

# Create ultra-simple config
echo "📝 Creating simple config..."
mkdir -p storage
cat > qdrant.yaml << 'EOF'
service:
  host: 0.0.0.0
  http_port: 6333
  max_request_size_mb: 32

storage:
  storage_path: ./storage

telemetry:
  enabled: false
EOF

echo "🚀 Starting Qdrant with minimal config..."
./qdrant --config-path qdrant.yaml > qdrant.log 2>&1 &
QDRANT_PID=$!
echo $QDRANT_PID > qdrant.pid

echo "⏳ Waiting 10 seconds for startup..."
sleep 10

# Simple test
echo "🧪 Testing connection..."
if curl -s --connect-timeout 5 "http://localhost:6333/" | grep -q "qdrant"; then
    echo "✅ SUCCESS! Qdrant is responding"
    curl -s "http://localhost:6333/health"
    echo ""
    echo "🎉 Qdrant is ready at http://localhost:6333"
else
    echo "❌ Still not working. Here's what we can see:"
    echo "Process status:"
    ps aux | grep qdrant | head -5
    echo "Port status:"
    netstat -tlnp | grep 6333
    echo "Recent logs:"
    tail -10 qdrant.log
fi