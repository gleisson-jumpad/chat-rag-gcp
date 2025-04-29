#!/bin/bash

# Set your project ID and region
PROJECT_ID="chat-rag-v1"  # Your actual GCP project ID
REGION="us-east4"
SERVICE_NAME="postgres-connection-test"

# Build the container
echo "Building container..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run with necessary environment variables and service account
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --set-env-vars="INSTANCE_CONNECTION_NAME=chat-rag-v1:us-east4:chat-rag-db" \
  --set-env-vars="PG_DB=postgres" \
  --set-env-vars="PG_USER=postgres" \
  --set-env-vars="PG_PASSWORD=your_password_here" \
  --set-env-vars="DB_PUBLIC_IP=34.72.123.456" \
  --set-env-vars="VPC_CONNECTOR=cloudrun-vpc-connector" \
  --allow-unauthenticated \
  --add-cloudsql-instances=chat-rag-v1:us-east4:chat-rag-db \
  --service-account="682048092511-compute@developer.gserviceaccount.com" \
  --vpc-connector=cloudrun-vpc-connector \
  --vpc-egress=private-ranges-only

echo "Deployment complete!" 