#!/bin/bash

# RunPod-specific Qdrant setup (No systemd, direct process management)
# This works on RunPod without system service dependencies

echo "🚀 Setting up Qdrant for RunPod (no systemd)..."

# Create working directory
cd /workspace
mkdir -p qdrant-setup
cd qdrant-setup

echo "📦 Downloading Qdrant..."

# Download Qdrant binary directly
QDRANT_VERSION="v1.7.0"
wget -q https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz

if [ $? -ne 0 ]; then
    echo "❌ Failed to download Qdrant. Trying alternative method..."
    # Try alternative download
    curl -L -o qdrant-x86_64-unknown-linux-gnu.tar.gz https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz
fi

echo "📂 Extracting Qdrant..."
# Extract with different approach to avoid permission issues
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz --no-same-owner 2>/dev/null || tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz

# Make sure the binary is executable
chmod +x qdrant

echo "⚙️ Creating Qdrant configuration..."

# Create directories
mkdir -p storage/snapshots
mkdir -p storage/temp
mkdir -p config

# Create optimized config for RunPod
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
    max_search_threads: 4
    max_optimization_threads: 2
    search_batch_size: 100
    max_concurrent_searches: 500
    search_queue_size: 5000
    max_indexing_threads: 2
    indexing_queue_size: 1000

  on_disk_payload: false
  
  quantization:
    binary:
      always_ram: true

  optimizers:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000
    default_segment_number: 2
    max_segment_size: 20000
    memmap_threshold: 20000
    indexing_threshold: 10000
    flush_interval_sec: 2
    max_optimization_threads: 2

  hnsw:
    m: 16
    ef_construct: 128
    full_scan_threshold: 10000
    max_indexing_threads: 2
    ef: 64
    on_disk: false

telemetry:
  enabled: false  # Disable for privacy
  
cluster:
  enabled: false
EOF

echo "🚀 Creating startup scripts..."

# Create simple startup script
cat > start_qdrant.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Qdrant..."
cd /workspace/qdrant-setup
./qdrant --config-path config/production.yaml > qdrant.log 2>&1 &
QDRANT_PID=$!
echo $QDRANT_PID > qdrant.pid
echo "✅ Qdrant started with PID: $QDRANT_PID"
echo "📋 Log file: /workspace/qdrant-setup/qdrant.log"
EOF

chmod +x start_qdrant.sh

# Create stop script
cat > stop_qdrant.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping Qdrant..."
if [ -f qdrant.pid ]; then
    PID=$(cat qdrant.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "✅ Qdrant stopped (PID: $PID)"
    else
        echo "⚠️ Qdrant process not found"
    fi
    rm -f qdrant.pid
else
    echo "⚠️ No PID file found"
    # Try to kill any running qdrant processes
    pkill -f qdrant || echo "No qdrant processes found"
fi
EOF

chmod +x stop_qdrant.sh

# Create status check script
cat > check_qdrant.sh << 'EOF'
#!/bin/bash
echo "🔍 Checking Qdrant status..."

# Check if process is running
if [ -f qdrant.pid ]; then
    PID=$(cat qdrant.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Qdrant process running (PID: $PID)"
    else
        echo "❌ Qdrant process not running"
        rm -f qdrant.pid
        exit 1
    fi
else
    echo "❌ No PID file found"
    exit 1
fi

# Check HTTP endpoint
echo "🌐 Testing HTTP endpoint..."
if curl -f -s http://localhost:6333/health > /dev/null; then
    echo "✅ HTTP endpoint healthy"
else
    echo "❌ HTTP endpoint not responding"
    exit 1
fi

# Check gRPC endpoint (basic connection test)
echo "🔌 Testing gRPC endpoint..."
if nc -z localhost 6334 2>/dev/null; then
    echo "✅ gRPC port accessible"
else
    echo "❌ gRPC port not accessible"
    exit 1
fi

echo "🎉 Qdrant is fully operational!"
EOF

chmod +x check_qdrant.sh

echo "📦 Installing Redis..."
# Install Redis if not present
if ! command -v redis-server &> /dev/null; then
    apt-get update -qq
    apt-get install -y redis-server
fi

# Start Redis in background (no systemd)
echo "🚀 Starting Redis..."
redis-server --daemonize yes --bind 0.0.0.0 --port 6379 --save 900 1 --save 300 10 --save 60 10000

# Wait a moment for Redis to start
sleep 2

# Test Redis
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis failed to start"
    exit 1
fi

echo "🚀 Starting Qdrant..."
./start_qdrant.sh

# Wait for Qdrant to start
echo "⏳ Waiting for Qdrant to initialize..."
sleep 5

# Check if Qdrant started successfully
for i in {1..10}; do
    if curl -f -s http://localhost:6333/health > /dev/null; then
        echo "✅ Qdrant is running successfully!"
        break
    else
        echo "⏳ Waiting for Qdrant... (attempt $i/10)"
        sleep 2
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ Qdrant failed to start. Checking logs..."
        echo "📋 Last 20 lines of log:"
        tail -20 qdrant.log
        exit 1
    fi
done

echo "🧪 Running comprehensive health check..."
./check_qdrant.sh

# Create Python health checker
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
"""Complete system health check for RunPod deployment."""
import asyncio
import sys
import json
import time

async def check_system():
    print("🏥 RunPod System Health Check")
    print("=" * 40)
    
    all_healthy = True
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis: Healthy")
    except Exception as e:
        print(f"❌ Redis: {e}")
        all_healthy = False
    
    # Check Qdrant HTTP
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:6333/health') as response:
                if response.status == 200:
                    print("✅ Qdrant HTTP: Healthy")
                else:
                    print(f"❌ Qdrant HTTP: Status {response.status}")
                    all_healthy = False
    except Exception as e:
        print(f"❌ Qdrant HTTP: {e}")
        all_healthy = False
    
    # Check Qdrant client
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"✅ Qdrant Client: Healthy ({len(collections.collections)} collections)")
    except Exception as e:
        print(f"❌ Qdrant Client: {e}")
        all_healthy = False
    
    # Check Qdrant gRPC
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
        collections = client.get_collections()
        print(f"✅ Qdrant gRPC: Healthy")
    except Exception as e:
        print(f"❌ Qdrant gRPC: {e}")
        all_healthy = False
    
    print("=" * 40)
    if all_healthy:
        print("🎉 All systems healthy!")
        return True
    else:
        print("⚠️ Some systems have issues")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(check_system())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n👋 Health check interrupted")
        sys.exit(1)
EOF

chmod +x health_check.py

# Install required Python packages
echo "🐍 Installing Python dependencies..."
pip install qdrant-client redis aiohttp

echo "🧪 Running Python health check..."
python health_check.py

# Create management script for easy control
cat > manage_qdrant.sh << 'EOF'
#!/bin/bash

case "$1" in
    start)
        echo "🚀 Starting Qdrant..."
        cd /workspace/qdrant-setup
        ./start_qdrant.sh
        sleep 3
        ./check_qdrant.sh
        ;;
    stop)
        echo "🛑 Stopping Qdrant..."
        cd /workspace/qdrant-setup
        ./stop_qdrant.sh
        ;;
    status)
        echo "🔍 Checking Qdrant status..."
        cd /workspace/qdrant-setup
        ./check_qdrant.sh
        ;;
    logs)
        echo "📋 Qdrant logs (last 50 lines):"
        cd /workspace/qdrant-setup
        tail -50 qdrant.log
        ;;
    health)
        echo "🏥 Complete health check..."
        cd /workspace/qdrant-setup
        python health_check.py
        ;;
    restart)
        echo "🔄 Restarting Qdrant..."
        cd /workspace/qdrant-setup
        ./stop_qdrant.sh
        sleep 2
        ./start_qdrant.sh
        sleep 3
        ./check_qdrant.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|health|restart}"
        echo ""
        echo "Commands:"
        echo "  start   - Start Qdrant"
        echo "  stop    - Stop Qdrant"
        echo "  status  - Check if Qdrant is running"
        echo "  logs    - Show recent Qdrant logs"
        echo "  health  - Run complete health check"
        echo "  restart - Restart Qdrant"
        exit 1
        ;;
esac
EOF

chmod +x manage_qdrant.sh

# Move management script to a global location
cp manage_qdrant.sh /usr/local/bin/qdrant-manage
chmod +x /usr/local/bin/qdrant-manage

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Quick Commands:"
echo "  qdrant-manage start   - Start Qdrant"
echo "  qdrant-manage stop    - Stop Qdrant"
echo "  qdrant-manage status  - Check status"
echo "  qdrant-manage health  - Full health check"
echo "  qdrant-manage logs    - View logs"
echo ""
echo "📍 Qdrant is running at:"
echo "  HTTP: http://localhost:6333"
echo "  gRPC: localhost:6334"
echo ""
echo "📂 Files location: /workspace/qdrant-setup/"
echo ""
echo "✅ Your existing Voice AI code should now work!"
EOF