FROM python:3.11-slim

WORKDIR /app

## #Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/ || exit 1

# Run application (production mode)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090", "--proxy-headers"]
