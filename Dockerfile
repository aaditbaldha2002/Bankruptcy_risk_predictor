# Base image: lightweight, Python 3.10+ (adjust per your needs)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML, ZenML, and your pipeline
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies with pinned versions
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the entire project source
COPY . .

# Environment variables (example: disable cache if needed)
ENV PYTHONUNBUFFERED=1

# Expose ports if you have a server or API (example 8080)
# EXPOSE 8080

# Run your ZenML pipeline or entrypoint script
CMD ["python", "-m", "src.main"]  # Adjust to your main script or entrypoint

