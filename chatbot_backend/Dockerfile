# Use Python 3.11 slim image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY src/requirements.txt .
# Install numpy and scipy first to ensure wheels are built before other packages
RUN pip install --no-cache-dir --retries 10 --timeout 120 -i https://pypi.org/simple numpy scipy
# Install CPU-only torch (no CUDA, much smaller)
RUN pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir --retries 10 --timeout 120 -i https://pypi.org/simple -r requirements.txt

# Copy the entire application
COPY . .

# Debug: Show file structure
RUN echo "=== File structure ===" && \
    find /app -name "*.py" -type f | head -20

# Expose port
EXPOSE $PORT

# Start command pointing to src/main.py
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]