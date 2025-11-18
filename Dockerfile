# Multi-stage Dockerfile for Satellite Terrain Classification
# Production-ready containerization with optimization

# Stage 1: Base image with dependencies
FROM tensorflow/tensorflow:2.15.0-gpu as base

LABEL maintainer="Satellite Terrain Classification Team"
LABEL description="Cutting-edge satellite terrain classification with deep learning"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Copy source code
COPY . .

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    ipython

# Expose ports
EXPOSE 8000 8888 6006

# Default command
CMD ["python", "api/serve.py"]

# Stage 3: Production image
FROM base as production

# Copy only necessary files
COPY models/ /app/models/
COPY api/ /app/api/
COPY interpretability/ /app/interpretability/
COPY config/ /app/config/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Production command with optimized settings
CMD ["uvicorn", "api.serve:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--limit-concurrency", "100", \
     "--timeout-keep-alive", "30"]
