FROM python:3.12-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]"

# Copy source
COPY src/ src/
COPY tests/ tests/

# Expose API port
EXPOSE 8082

# Default: run API server
CMD ["python", "-m", "video_lora.server", "--host", "0.0.0.0", "--port", "8082"]
