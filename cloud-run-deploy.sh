#!/bin/bash

# Set your project ID and region
PROJECT_ID="chat-rag-v1"  # Your actual GCP project ID
REGION="us-east4"
SERVICE_NAME="chat-rag"  # Updated to match your actual service name
INSTANCE_NAME="chat-rag-db"
INSTANCE_CONNECTION_NAME="${PROJECT_ID}:${REGION}:${INSTANCE_NAME}"

echo "Using connection name: $INSTANCE_CONNECTION_NAME"

# Build the container
echo "Building container..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run with necessary environment variables and service account
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --set-env-vars="INSTANCE_CONNECTION_NAME=${INSTANCE_CONNECTION_NAME}" \
  --set-env-vars="PG_DB=postgres" \
  --set-env-vars="PG_USER=postgres" \
  --set-env-vars="PG_PASSWORD=AMGQk@50l9Eu" \
  --set-env-vars="VPC_CONNECTOR=cloudrun-vpc-connector" \
  --set-env-vars="PG_HOST=/cloudsql/${INSTANCE_CONNECTION_NAME}" \
  --allow-unauthenticated \
  --add-cloudsql-instances=${INSTANCE_CONNECTION_NAME} \
  --service-account="682048092511-compute@developer.gserviceaccount.com" \
  --vpc-connector=cloudrun-vpc-connector \
  --vpc-egress=private-ranges-only

echo "Deployment complete!" 