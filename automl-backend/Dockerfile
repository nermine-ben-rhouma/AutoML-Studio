FROM python:3.10-slim

# Configure non-buffered Python output & ignore bytecode creation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies (curl for healthchecks, compiler tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose backend port
EXPOSE 8000

# Start server (binding to 0.0.0.0 to enable external scraper/client access)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
