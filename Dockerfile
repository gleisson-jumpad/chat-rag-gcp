FROM python:3.9-slim

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Cloud SQL Auth Proxy
RUN wget https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.linux.amd64 -O /app/cloud-sql-proxy
RUN chmod +x /app/cloud-sql-proxy

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ app/
COPY start.sh .
RUN chmod +x start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1

# Service account credentials will be mounted by Cloud Run
EXPOSE 8501

CMD ["./start.sh"]
