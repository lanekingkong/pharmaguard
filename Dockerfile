# Multi-stage Docker build for PharmaGuard
# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files for layer caching
COPY pyproject.toml requirements.txt ./

# Install PyTorch CPU first, then remaining dependencies
RUN pip install --no-cache-dir --user \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

LABEL maintainer="PharmaGuard Team" \
      description="Intelligent Medication Safety & DDI Prediction System" \
      version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash pharmaguard

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/pharmaguard/.local

# Copy source code
COPY --chown=pharmaguard:pharmaguard . .

ENV PATH="/home/pharmaguard/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PHARMAGUARD_ENV=production

RUN pip install --no-cache-dir -e .

USER pharmaguard

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["pharmaguard"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]