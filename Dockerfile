# Use Python 3.10 (TensorFlow compatible)
FROM python:3.10-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement file
COPY requirements.txt .

# Install dependencies (CPU TensorFlow)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose Render internal port
ENV PORT=10000

# Start app with gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app
