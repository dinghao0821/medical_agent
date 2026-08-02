# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # Image processing dependencies
    libpng-dev \
    libjpeg-dev \
    # For lxml
    libxml2-dev \
    libxslt1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories (include brain_tumor_output for segmentation results)
RUN mkdir -p uploads/backend uploads/frontend uploads/skin_lesion_output uploads/brain_tumor_output uploads/speech data

# Expose port
EXPOSE 8000

# Set environment variable for Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1
# Number of gunicorn workers (override at runtime, e.g. -e WORKERS=4).
ENV WORKERS=2

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run under gunicorn with Uvicorn workers for multi-process, multi-core serving.
# Stateless app: per-session state lives in the (Redis) checkpointer, so multiple
# workers are safe. Timeout is generous to accommodate slow LLM/inference calls.
CMD ["sh", "-c", "gunicorn app:app --workers ${WORKERS:-2} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300 --graceful-timeout 30 --keep-alive 5"]