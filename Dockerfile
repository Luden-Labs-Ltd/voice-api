FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY readscore/ ./readscore/
COPY tests/ ./tests/

# Install package with all dependencies
RUN pip install --no-cache-dir -e ".[full,dev]"

# Default command: show help
CMD ["readscore", "--help"]
