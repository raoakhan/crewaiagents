# Minimal production Dockerfile for call_analytics service
FROM python:3.11-slim

WORKDIR /app

# Install system deps (if Whisper large audio or ffmpeg needed add here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY call_analytics /app/call_analytics
COPY call_analytics/pyproject.toml /app/call_analytics/pyproject.toml

# Install deps
RUN pip install --upgrade pip && \
    pip install uv[virtualenv] && \
    uv pip install -e ./call_analytics

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "call_analytics.api:app", "--host", "0.0.0.0", "--port", "8000"]
