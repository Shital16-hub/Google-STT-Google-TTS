#!/bin/bash
# Fix directory structure and path issues

echo "🔧 Fixing directory structure and paths..."

# Current situation analysis
echo "📊 Current situation:"
echo "- You're in: $(pwd)"
echo "- main.py is at: /workspace/Google-STT-Google-TTS/main.py"
echo "- Configs are at: /workspace/Google-STT-Google-TTS/app/config/agents/"
echo ""

# Option 1: Move everything to /workspace (recommended)
echo "🚀 Option 1: Move everything to /workspace (recommended)"
echo "This will make paths consistent with what main.py expects"

read -p "Do you want to move everything to /workspace? (y/n): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Moving files to /workspace..."
    
    # Create backup just in case
    echo "Creating backup..."
    cp -r /workspace/Google-STT-Google-TTS /workspace/backup_$(date +%Y%m%d_%H%M%S)
    
    # Move all files from Google-STT-Google-TTS to workspace root
    echo "Moving files..."
    cd /workspace/Google-STT-Google-TTS
    
    # Move all files and directories to parent
    for item in *; do
        if [[ "$item" != "." && "$item" != ".." ]]; then
            echo "Moving $item..."
            mv "$item" "/workspace/"
        fi
    done
    
    # Move hidden files too
    for item in .*; do
        if [[ "$item" != "." && "$item" != ".." && "$item" != ".git" ]]; then
            echo "Moving hidden file $item..."
            mv "$item" "/workspace/" 2>/dev/null || true
        fi
    done
    
    # Remove empty directory
    cd /workspace
    rmdir /workspace/Google-STT-Google-TTS 2>/dev/null || true
    
    echo "✅ Files moved successfully"
    echo "New structure:"
    ls -la /workspace/ | head -10
    
else
    echo "🔧 Option 2: Update config paths in main.py"
    echo "This will modify main.py to look in the correct directory"
    
    # Update the ConfigurationManager to use the correct path
    cd /workspace/Google-STT-Google-TTS
    
    # Backup main.py
    cp main.py main.py.backup
    
    # Update the config path
    sed -i 's|config_base_path: str = "app/config"|config_base_path: str = "/workspace/Google-STT-Google-TTS/app/config"|g' main.py
    
    echo "✅ Updated config path in main.py"
    echo "📋 Verification:"
    grep "config_base_path" main.py || echo "Could not verify change"
fi

echo ""
echo "🔍 Final verification:"
if [ -f "/workspace/main.py" ]; then
    echo "✅ main.py found at /workspace/main.py"
    if [ -d "/workspace/app/config/agents" ]; then
        echo "✅ Config directory found at /workspace/app/config/agents"
        echo "📄 Agent configs:"
        ls -la /workspace/app/config/agents/*.yaml 2>/dev/null || echo "No YAML files found"
    else
        echo "❌ Config directory not found at /workspace/app/config/agents"
    fi
else
    echo "📍 main.py is at: /workspace/Google-STT-Google-TTS/main.py"
    if [ -d "/workspace/Google-STT-Google-TTS/app/config/agents" ]; then
        echo "✅ Config directory found at /workspace/Google-STT-Google-TTS/app/config/agents"
        echo "📄 Agent configs:"
        ls -la /workspace/Google-STT-Google-TTS/app/config/agents/*.yaml 2>/dev/null || echo "No YAML files found"
    else
        echo "❌ Config directory not found"
    fi
fi

echo ""
echo "🎯 Next steps:"
echo "1. cd to the directory with main.py"
echo "2. Start Qdrant: /workspace/qdrant-setup/start_qdrant.sh"
echo "3. Run: python main.py"