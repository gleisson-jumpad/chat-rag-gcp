#!/bin/bash

# Set your project ID and region
PROJECT_ID="chat-rag-v1"
REGION="us-east4"
SERVICE_NAME="chat-rag"
INSTANCE_NAME="chat-rag-db"
INSTANCE_CONNECTION_NAME="${PROJECT_ID}:${REGION}:${INSTANCE_NAME}"

echo "Using connection name: $INSTANCE_CONNECTION_NAME"

# Build the container
echo "🔧 Building container image..."
gcloud builds submit --tag us-east4-docker.pkg.dev/$PROJECT_ID/chat-rag/image:latest

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image us-east4-docker.pkg.dev/$PROJECT_ID/chat-rag/image:latest \
  --platform managed \
  --region $REGION \
  --set-env-vars="PG_DB=postgres,PG_USER=postgres,PG_PASSWORD=AMGQk@50l9Eu,DB_PUBLIC_IP=34.48.95.143" \
  --allow-unauthenticated \
  --vpc-connector=cloudrun-vpc-connector \
  --vpc-egress=private-ranges-only \
  --service-account="682048092511-compute@developer.gserviceaccount.com"

echo "✅ Deployment complete!"