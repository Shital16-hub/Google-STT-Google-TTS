#!/bin/bash
# Quick script to restart and check Qdrant on RunPod

echo "🔧 Checking Qdrant status..."

# Check if Qdrant is running
if pgrep -f "qdrant" > /dev/null; then
    echo "✅ Qdrant process is running"
else
    echo "❌ Qdrant process not found"
fi

# Check if Qdrant port is accessible
if nc -z localhost 6333; then
    echo "✅ Qdrant HTTP port (6333) is accessible"
else
    echo "❌ Qdrant HTTP port (6333) not accessible"
fi

# Test Qdrant HTTP endpoint
echo "🧪 Testing Qdrant HTTP endpoint..."
if curl -f -s http://localhost:6333/health > /dev/null; then
    echo "✅ Qdrant HTTP endpoint is healthy"
    echo "📊 Qdrant collections:"
    curl -s http://localhost:6333/collections | python3 -m json.tool
else
    echo "❌ Qdrant HTTP endpoint not responding"
    echo "🔄 Attempting to restart Qdrant..."
    
    # Use the management script if available
    if [ -f "/usr/local/bin/qdrant-manage" ]; then
        echo "Using qdrant-manage script..."
        qdrant-manage restart
    elif [ -f "/workspace/qdrant-setup/manage_qdrant.sh" ]; then
        echo "Using local management script..."
        cd /workspace/qdrant-setup
        ./manage_qdrant.sh restart
    else
        echo "Manual restart attempt..."
        # Kill any existing Qdrant processes
        pkill -f qdrant
        sleep 2
        
        # Try to start Qdrant if setup exists
        if [ -d "/workspace/qdrant-setup" ]; then
            cd /workspace/qdrant-setup
            if [ -f "./qdrant" ]; then
                echo "Starting Qdrant..."
                ./qdrant --config-path config/production.yaml > qdrant.log 2>&1 &
                echo $! > qdrant.pid
                sleep 3
                
                # Test again
                if curl -f -s http://localhost:6333/health > /dev/null; then
                    echo "✅ Qdrant restarted successfully"
                else
                    echo "❌ Qdrant restart failed"
                    echo "📋 Recent logs:"
                    tail -20 qdrant.log
                fi
            else
                echo "❌ Qdrant binary not found in /workspace/qdrant-setup"
            fi
        else
            echo "❌ Qdrant setup directory not found"
        fi
    fi
fi

echo "🏥 Final health check..."
python3 -c "
import requests
try:
    response = requests.get('http://localhost:6333/health', timeout=5)
    if response.status_code == 200:
        print('✅ Qdrant is healthy')
    else:
        print(f'⚠️ Qdrant returned status: {response.status_code}')
        print(f'Response: {response.text}')
except Exception as e:
    print(f'❌ Qdrant health check failed: {e}')
"