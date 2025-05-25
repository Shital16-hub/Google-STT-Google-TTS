#!/bin/bash

# Smart Dependency Installation for Runpod
# Handles version conflicts and system dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🚀 Installing Dependencies for Multi-Agent Voice AI System"
echo "========================================================"

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Found Python version: $PYTHON_VERSION"

# Update system packages
print_status "Updating system packages..."
apt-get update -qq

# Install system dependencies
print_status "Installing system dependencies..."
apt-get install -y \
    gcc g++ make \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    curl wget git \
    build-essential \
    python3-dev \
    > /dev/null 2>&1

print_success "System dependencies installed"

# Upgrade pip and essential tools
print_status "Upgrading pip and build tools..."
python3 -m pip install --upgrade pip setuptools wheel build

# Install dependencies in stages to handle conflicts
print_status "Installing Python dependencies in stages..."

# Stage 1: Core framework
print_status "Stage 1: Installing core framework..."
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    websockets==12.0 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0

# Stage 2: Database and storage
print_status "Stage 2: Installing database libraries..."
pip install \
    asyncpg==0.29.0 \
    sqlalchemy[asyncio]==2.0.23 \
    redis[hiredis]==5.0.1 \
    qdrant-client==1.7.0

# Stage 3: AI and ML (this might take longer)
print_status "Stage 3: Installing AI/ML libraries (this may take a while)..."
pip install \
    openai==1.6.1 \
    google-cloud-speech==2.21.0 \
    google-cloud-texttospeech==2.16.3

# Stage 4: Audio processing (potential conflicts here)
print_status "Stage 4: Installing audio processing libraries..."
pip install \
    pydub==0.25.1 \
    SoundFile==0.12.1 \
    webrtcvad==2.0.10

# Try to install librosa separately as it can be problematic
print_status "Installing librosa (may take time)..."
pip install librosa==0.10.1 || {
    print_warning "librosa failed, trying without version pin..."
    pip install librosa || print_warning "librosa installation failed, continuing..."
}

# Stage 5: Vector databases
print_status "Stage 5: Installing vector databases..."
pip install faiss-cpu==1.7.4 || {
    print_warning "faiss-cpu specific version failed, trying latest..."
    pip install faiss-cpu || print_warning "faiss-cpu installation failed, continuing..."
}

# Stage 6: LangChain components
print_status "Stage 6: Installing LangChain components..."
pip install \
    langchain==0.1.0 \
    langchain-community==0.0.13 \
    langchain-openai==0.0.2 \
    langgraph==0.0.20 || {
    print_warning "LangChain versions failed, trying without version pins..."
    pip install langchain langchain-community langchain-openai langgraph
}

# Stage 7: Data processing
print_status "Stage 7: Installing data processing libraries..."
pip install \
    pandas==2.1.4 \
    numpy==1.24.4 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

# Stage 8: External integrations
print_status "Stage 8: Installing external service integrations..."
pip install \
    twilio==8.10.0 \
    requests==2.31.0 \
    httpx==0.25.2 \
    stripe==7.8.0

# Stage 9: Security (fix the cryptography version)
print_status "Stage 9: Installing security libraries..."
pip install cryptography==41.0.7 || {
    print_warning "cryptography 41.0.7 failed, trying available version..."
    pip install "cryptography>=40.0.0,<42.0.0" || {
        print_warning "Installing latest cryptography..."
        pip install cryptography
    }
}

pip install \
    pyjwt==2.8.0 \
    passlib[bcrypt]==1.7.4 \
    python-multipart==0.0.6

# Stage 10: Utilities and remaining packages
print_status "Stage 10: Installing utilities..."
pip install \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    jsonschema==4.20.0 \
    structlog==23.2.0 \
    prometheus-client==0.19.0 \
    psutil==5.9.6 \
    jinja2==3.1.2 \
    click==8.1.7 \
    rich==13.7.0

# Stage 11: Try sentence-transformers (can be problematic)
print_status "Stage 11: Installing sentence transformers..."
pip install sentence-transformers==2.2.2 || {
    print_warning "sentence-transformers 2.2.2 failed, trying latest..."
    pip install sentence-transformers || print_warning "sentence-transformers failed, continuing..."
}

# Stage 12: Production tools
print_status "Stage 12: Installing production tools..."
pip install \
    gunicorn==21.2.0 \
    setproctitle==1.3.3

print_success "All dependencies installed successfully!"

# Verify critical imports
print_status "Verifying critical imports..."
python3 -c "
import sys
print(f'✅ Python {sys.version}')

try:
    import fastapi
    print('✅ FastAPI imported successfully')
except ImportError as e:
    print(f'❌ FastAPI import failed: {e}')

try:
    import openai
    print('✅ OpenAI imported successfully')  
except ImportError as e:
    print(f'❌ OpenAI import failed: {e}')

try:
    import redis
    print('✅ Redis imported successfully')
except ImportError as e:
    print(f'❌ Redis import failed: {e}')

try:
    import qdrant_client
    print('✅ Qdrant client imported successfully')
except ImportError as e:
    print(f'❌ Qdrant import failed: {e}')

try:
    import google.cloud.speech
    print('✅ Google Cloud Speech imported successfully')
except ImportError as e:
    print(f'❌ Google Cloud Speech import failed: {e}')

try:
    import sqlalchemy
    print('✅ SQLAlchemy imported successfully')
except ImportError as e:
    print(f'❌ SQLAlchemy import failed: {e}')

print('\\n🎉 Import verification completed!')
"

print_success "Dependency installation completed!"
print_status "You can now start your Voice AI system."

echo ""
echo "📋 Next steps:"
echo "1. Configure your .env file with API keys"
echo "2. Add Google Cloud credentials JSON file"  
echo "3. Run: ./start-voice-ai.sh"
echo ""
print_success "Ready to launch your Multi-Agent Voice AI system! 🚀"