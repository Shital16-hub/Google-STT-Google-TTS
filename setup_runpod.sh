#!/bin/bash
# RunPod Environment Setup Script for Multi-Agent Voice AI System

set -e  # Exit on any error

echo "🚀 Setting up Multi-Agent Voice AI System on RunPod..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update -y

# Install Redis
echo "📊 Installing Redis..."
apt-get install -y redis-server redis-tools

# Configure Redis for performance
echo "⚙️ Configuring Redis..."
sed -i 's/^daemonize no/daemonize yes/' /etc/redis/redis.conf
sed -i 's/^# maxmemory <bytes>/maxmemory 2gb/' /etc/redis/redis.conf
sed -i 's/^# maxmemory-policy noeviction/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf

# Install system dependencies for vector operations
echo "🔧 Installing system dependencies..."
apt-get install -y \
    build-essential \
    cmake \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    wget \
    curl \
    htop

# Install Python dependencies (if not already installed)
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install redis faiss-cpu qdrant-client numpy scipy

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspace/logs
mkdir -p /workspace/qdrant-setup/{config,storage}
mkdir -p /workspace/app/config/agents

# Set permissions
chmod -R 755 /workspace/logs
chmod -R 755 /workspace/qdrant-setup

# Download and setup Qdrant binary
echo "🗄️ Setting up Qdrant..."
cd /workspace/qdrant-setup
if [ ! -f "qdrant" ]; then
    wget -q https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
    tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
    chmod +x qdrant
    rm qdrant-x86_64-unknown-linux-gnu.tar.gz
fi

# Start Redis service
echo "🚀 Starting Redis..."
redis-server --daemonize yes --maxmemory 2gb --maxmemory-policy allkeys-lru

# Test Redis connection
echo "🧪 Testing Redis connection..."
if redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis is running successfully"
else
    echo "❌ Redis failed to start"
    exit 1
fi

echo "✅ RunPod setup completed successfully!"
echo "📊 Redis: ✅ Running"
echo "🗄️ Qdrant: ✅ Binary ready"
echo "📁 Directories: ✅ Created"
echo ""
echo "🚀 You can now start your application with: python main.py"