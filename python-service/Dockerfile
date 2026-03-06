FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the readscore package and its runtime dependencies.
# Build context is the repo root so we can reference readscore/ and pyproject.toml.
COPY pyproject.toml README.md ./
COPY readscore/ ./readscore/

RUN pip install --no-cache-dir -e ".[full]" \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

# Copy the FastAPI entry point last so layer cache is reused on code changes.
COPY python-service/main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
