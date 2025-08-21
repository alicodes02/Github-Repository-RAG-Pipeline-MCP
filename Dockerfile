# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

# Avoid interactive prompts, speed up pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Optional build-time token (not recommended for secrets). Will be promoted to ENV for runtime use.
ARG GITHUB_TOKEN=""
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

ENV TRANSFORMERS_CACHE=/app/hf_cache


WORKDIR /app

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
    && rm -rf /var/lib/apt/lists/*

# Only copy requirements first to leverage Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default DB path is handled by code (chroma_db_hf under project root).
# Create it so users can volume-mount over it if desired.
RUN mkdir -p /app/chroma_db_hf

# Expose the HTTP port
EXPOSE 8010

# Run the server using fastmcp with HTTP transport
CMD ["python", "rag_mcp/server.py"]
