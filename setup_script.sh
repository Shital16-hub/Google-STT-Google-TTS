#!/bin/bash

# Docker-Free Qdrant Setup for RunPod
echo "🚀 Setting up Qdrant without Docker..."

cd /workspace/Google-STT-Google-TTS

# Method 1: Try installing Qdrant via package manager
echo "📦 Method 1: Package manager installation..."

# Update package lists
apt-get update -qq

# Try installing Qdrant from official repository
if ! command -v qdrant &> /dev/null; then
    echo "📥 Installing Qdrant via package manager..."
    
    # Add Qdrant repository key and source
    curl -fsSL https://packages.qdrant.tech/gpg | apt-key add - 2>/dev/null || {
        # Fallback method for adding key
        wget -qO - https://packages.qdrant.tech/gpg | apt-key add -
    }
    
    # Add repository
    echo "deb https://packages.qdrant.tech/deb/ stable main" | tee /etc/apt/sources.list.d/qdrant.list
    
    # Update and install
    apt-get update -qq
    apt-get install -y qdrant || {
        echo "⚠️ Package installation failed, trying binary method..."
    }
fi

# Check if installation was successful
if command -v qdrant &> /dev/null; then
    echo "✅ Qdrant installed via package manager!"
    
    # Create configuration
    mkdir -p /etc/qdrant /var/lib/qdrant /var/log/qdrant
    
    cat > /etc/qdrant/config.yaml << 'EOF'
service:
  host: "0.0.0.0"
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  max_request_size_mb: 32

storage:
  storage_path: "/var/lib/qdrant"
  snapshots_path: "/var/lib/qdrant/snapshots"
  on_disk_payload: false

telemetry:
  enabled: false

log_level: INFO
EOF
    
    # Start Qdrant as a service
    qdrant --config-path /etc/qdrant/config.yaml &
    QDRANT_PID=$!
    
    echo "🚀 Started Qdrant with PID: $QDRANT_PID"
    
else
    echo "📦 Method 2: Binary download and installation..."
    
    # Create Qdrant directory
    mkdir -p /workspace/qdrant-setup
    cd /workspace/qdrant-setup
    
    # Download Qdrant binary
    echo "📥 Downloading Qdrant binary..."
    
    # Try multiple download methods
    DOWNLOAD_SUCCESS=false
    
    # Method 2a: wget
    if command -v wget &> /dev/null; then
        echo "Using wget..."
        wget -q --timeout=60 --tries=3 \
            https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz \
            -O qdrant.tar.gz && DOWNLOAD_SUCCESS=true
    fi
    
    # Method 2b: curl fallback
    if [ "$DOWNLOAD_SUCCESS" = false ] && command -v curl &> /dev/null; then
        echo "Using curl..."
        curl -L --max-time 60 --retry 3 \
            https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz \
            -o qdrant.tar.gz && DOWNLOAD_SUCCESS=true
    fi
    
    # Method 2c: Try older version if new one fails
    if [ "$DOWNLOAD_SUCCESS" = false ]; then
        echo "Trying older version..."
        if command -v wget &> /dev/null; then
            wget -q --timeout=60 \
                https://github.com/qdrant/qdrant/releases/download/v1.6.1/qdrant-x86_64-unknown-linux-gnu.tar.gz \
                -O qdrant.tar.gz && DOWNLOAD_SUCCESS=true
        fi
    fi
    
    if [ "$DOWNLOAD_SUCCESS" = true ]; then
        echo "✅ Download successful!"
        
        # Extract binary
        tar -xzf qdrant.tar.gz
        chmod +x qdrant
        
        # Create directories
        mkdir -p storage config logs
        
        # Create configuration file
        cat > config/production.yaml << 'EOF'
service:
  host: "127.0.0.1"
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  max_request_size_mb: 16

storage:
  storage_path: "./storage"
  snapshots_path: "./storage/snapshots"
  on_disk_payload: false
  performance:
    max_search_threads: 2

telemetry:
  enabled: false

log_level: INFO

cluster:
  enabled: false
EOF
        
        # Start Qdrant
        echo "🚀 Starting Qdrant binary..."
        nohup ./qdrant --config-path config/production.yaml > logs/qdrant.log 2>&1 &
        QDRANT_PID=$!
        
        echo "Started Qdrant with PID: $QDRANT_PID"
        echo $QDRANT_PID > qdrant.pid
        
    else
        echo "❌ Failed to download Qdrant binary"
        echo "🔄 Will use in-memory mode"
        cd /workspace/Google-STT-Google-TTS
        exit 0
    fi
    
    cd /workspace/Google-STT-Google-TTS
fi

# Wait for Qdrant to start
echo "⏳ Waiting for Qdrant to start..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "✅ Qdrant is now running!"
        curl -s http://localhost:6333/health | head -1
        break
    fi
    sleep 2
    echo "Attempt $i/30..."
    
    if [ $i -eq 30 ]; then
        echo "⚠️ Qdrant didn't start in time, checking logs..."
        if [ -f /workspace/qdrant-setup/logs/qdrant.log ]; then
            echo "Last 10 lines of Qdrant log:"
            tail -10 /workspace/qdrant-setup/logs/qdrant.log
        fi
        echo "🔄 System will use in-memory mode"
    fi
done

echo "✅ Setup complete!"