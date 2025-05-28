#!/bin/bash
# Check existing agent configurations and fix path issues

echo "🔍 Checking existing agent configurations..."

# Check if the config directory exists
if [ -d "/workspace/app/config" ]; then
    echo "✅ Found app/config directory"
    ls -la /workspace/app/config/
else
    echo "❌ app/config directory not found"
fi

# Check if agents subdirectory exists
if [ -d "/workspace/app/config/agents" ]; then
    echo "✅ Found app/config/agents directory"
    echo "📄 Agent config files:"
    ls -la /workspace/app/config/agents/
    
    # Check each config file
    for config_file in /workspace/app/config/agents/*.yaml; do
        if [ -f "$config_file" ]; then
            echo ""
            echo "📋 Checking $(basename $config_file):"
            
            # Check if it's a valid YAML and has required fields
            if python3 -c "
import yaml
import sys
try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['agent_id', 'version', 'specialization', 'voice_settings', 'tools', 'routing']
    missing_fields = []
    
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        print(f'❌ Missing required fields: {missing_fields}')
        sys.exit(1)
    else:
        print(f'✅ Valid config with agent_id: {config.get(\"agent_id\", \"unknown\")}')
        sys.exit(0)
        
except Exception as e:
    print(f'❌ Invalid YAML or error: {e}')
    sys.exit(1)
" 2>/dev/null; then
                echo "   Config is valid"
            else
                echo "   Config has issues"
            fi
        fi
    done
    
else
    echo "❌ app/config/agents directory not found"
    echo "🔧 Creating it now..."
    
    mkdir -p /workspace/app/config/agents
    
    echo "📁 Created directory structure:"
    ls -la /workspace/app/config/
fi

# Check what the main.py is looking for
echo ""
echo "🔍 Checking what main.py expects..."

# Look at the error in the logs - it was looking for app/config/agents
echo "Based on the logs, main.py is looking for: app/config/agents"

# Check current working directory when app runs
echo "Current working directory when running: $(pwd)"

# Check if we're in the right place
if [ -f "/workspace/main.py" ]; then
    echo "✅ Found main.py in /workspace"
    
    # Check the ConfigurationManager path
    echo "🔍 Checking ConfigurationManager configuration..."
    grep -n "config_base_path" /workspace/main.py || echo "Could not find config_base_path in main.py"
    
else
    echo "❌ main.py not found in expected location"
fi

echo ""
echo "📊 Summary:"
echo "- Working directory: $(pwd)"
echo "- Config expected at: app/config/agents"
echo "- Actual path: /workspace/app/config/agents"

# Test if the path resolution works
if [ -d "app/config/agents" ]; then
    echo "✅ Relative path 'app/config/agents' resolves correctly"
    echo "📄 Files found:"
    ls -la app/config/agents/
else
    echo "❌ Relative path 'app/config/agents' does not resolve"
    echo "💡 Need to run from /workspace directory"
fi