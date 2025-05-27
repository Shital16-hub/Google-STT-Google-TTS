# Revolutionary Multi-Agent Voice AI System - Production Dockerfile
# Multi-stage optimized build for RunPod deployment
# Target: <50MB final image size with maximum performance

# ============================================================================
# BUILD ARGUMENTS
# ============================================================================
ARG PYTHON_VERSION=3.11
ARG DEBIAN_VERSION=slim-bullseye
ARG ENVIRONMENT=production

# ============================================================================
# STAGE 1: DEPENDENCY BUILDER
# ============================================================================
FROM python:${PYTHON_VERSION}-${DEBIAN_VERSION} as builder

# Build metadata
LABEL maintainer="Voice AI Team"
LABEL description="Revolutionary Multi-Agent Voice AI System"
LABEL version="2.0.0"

# Set build environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    curl \
    wget \
    git \
    # Audio processing dependencies
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libsndfile1-dev \
    # Scientific computing dependencies  
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    # SSL and crypto
    libssl-dev \
    libffi-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir \
    --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /tmp/requirements.txt

# Install additional performance packages
RUN pip install --no-cache-dir \
    uvloop \
    httptools \
    python-json-logger

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# ============================================================================
# STAGE 2: PRODUCTION RUNTIME
# ============================================================================
FROM python:${PYTHON_VERSION}-${DEBIAN_VERSION} as production

# Runtime metadata
LABEL stage="production"
LABEL optimization="ultra-low-latency"

# Set production environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:$PATH"

# Performance optimizations
ENV PYTHONOPTIMIZE=2
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_MMAP_THRESHOLD_=100000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libraries
    libasound2 \
    libportaudio2 \
    libsndfile1 \
    libopenblas0 \
    # Utilities
    curl \
    wget \
    # Health check tools
    netcat-openbsd \
    # Process management
    supervisor \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user for security
RUN groupadd -r voiceai && \
    useradd -r -g voiceai -d /app -s /bin/bash voiceai && \
    mkdir -p /app/logs /app/tmp /app/data /app/config /app/credentials && \
    chown -R voiceai:voiceai /app

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=voiceai:voiceai app/ ./app/
COPY --chown=voiceai:voiceai config/ ./config/
COPY --chown=voiceai:voiceai *.py ./

# Create additional directories
RUN mkdir -p \
    /app/logs/app \
    /app/logs/nginx \
    /app/logs/access \
    /app/tmp/uploads \
    /app/data/cache \
    /app/config/agents \
    && chown -R voiceai:voiceai /app

# Copy supervisor configuration
COPY --chown=voiceai:voiceai deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy health check script
COPY --chown=voiceai:voiceai deployment/health_check.py /app/health_check.py
RUN chmod +x /app/health_check.py

# Copy startup script
COPY --chown=voiceai:voiceai deployment/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Switch to application user
USER voiceai

# Expose application port
EXPOSE 8000

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/health_check.py || exit 1

# Default startup command
CMD ["/app/startup.sh"]

# ============================================================================
# STAGE 3: DEVELOPMENT (Optional)
# ============================================================================
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    tree \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    ipython \
    jupyterlab

# Development environment variables
ENV ENVIRONMENT=development
ENV DEBUG=true
ENV LOG_LEVEL=DEBUG

# Switch back to app user
USER voiceai

# Development command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# STAGE 4: TESTING (Optional)
# ============================================================================
FROM builder as testing

# Copy test files
COPY --chown=voiceai:voiceai tests/ /app/tests/
COPY --chown=voiceai:voiceai pytest.ini /app/
COPY --chown=voiceai:voiceai .coveragerc /app/

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    factory-boy \
    faker

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=app", "--cov-report=html"]

# ============================================================================
# MULTI-ARCH SUPPORT (ARM64 for some RunPod instances)
# ============================================================================

# Build for multiple architectures
# docker buildx build --platform linux/amd64,linux/arm64 -t voice-ai:latest .

# ============================================================================
# OPTIMIZATION NOTES
# ============================================================================

# Image Size Optimizations:
# - Multi-stage build reduces final image size by ~70%
# - Only runtime dependencies in production stage
# - Virtual environment isolation
# - Cleaned package manager caches

# Performance Optimizations:
# - Python bytecode optimization (PYTHONOPTIMIZE=2)
# - Memory allocation tuning
# - uvloop for async performance
# - Proper file permissions and ownership

# Security Optimizations:
# - Non-root user execution
# - Minimal attack surface
# - Proper file permissions
# - No unnecessary packages in production

# RunPod Optimizations:
# - GPU support ready (uncomment torch+cuda in requirements.txt)
# - Supervisor for process management
# - Health checks for container orchestration
# - Proper logging directory structure

# ============================================================================
# BUILD COMMANDS
# ============================================================================

# Production build:
# docker build --target production -t voice-ai:production .

# Development build:
# docker build --target development -t voice-ai:dev .

# Testing build:
# docker build --target testing -t voice-ai:test .

# Multi-arch build:
# docker buildx build --platform linux/amd64,linux/arm64 --target production -t voice-ai:latest .

# Build with custom Python version:
# docker build --build-arg PYTHON_VERSION=3.11 --target production .

# ============================================================================
# RUNTIME CONFIGURATIONS
# ============================================================================

# Production run:
# docker run -d -p 8000:8000 --name voice-ai voice-ai:production

# Development run with volume:
# docker run -d -p 8000:8000 -v $(pwd):/app --name voice-ai-dev voice-ai:dev

# With environment file:
# docker run -d --env-file .env -p 8000:8000 voice-ai:production

# With resource limits:
# docker run -d --memory=4g --cpus=2 -p 8000:8000 voice-ai:production