# RAG System with LlamaIndex and OpenAI

This application implements a Retrieval-Augmented Generation (RAG) system using LlamaIndex with OpenAI integration, deployed on Google Cloud Run with PostgreSQL vector storage.

## Features

- Document upload and processing (PDF, TXT, DOCX, PPTX, MD, CSV)
- Vector embedding generation with OpenAI
- Vector storage in PostgreSQL database with pgvector
- Document retrieval and Q&A capabilities

## Setup and Deployment

### Local Testing

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export INSTANCE_CONNECTION_NAME="your-project:region:instance-name"
   export PG_DB="your-database-name"
   export PG_USER="your-database-user"
   export PG_PASSWORD="your-database-password"
   export DB_PUBLIC_IP="your-postgresql-public-ip"  # Only if using direct connection
   ```

3. **Run the application:**
   ```bash
   streamlit run app/main.py
   ```

### Cloud Run Deployment

1. **Store your OpenAI API key in Secret Manager:**
   ```bash
   gcloud secrets create openai-api-key --data-file=- <<< "your-openai-api-key"
   ```

2. **Build and deploy with Cloud Build:**
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

3. **Grant Secret Manager access to the service account:**
   ```bash
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

4. **Manual deployment (alternative):**
   ```bash
   gcloud run deploy chat-rag \
     --image gcr.io/PROJECT_ID/chat-rag \
     --platform managed \
     --region REGION \
     --set-env-vars="INSTANCE_CONNECTION_NAME=PROJECT:REGION:INSTANCE" \
     --set-env-vars="PG_DB=postgres" \
     --set-env-vars="PG_USER=postgres" \
     --set-env-vars="PG_PASSWORD=your_password_here" \
     --set-secrets="OPENAI_API_KEY=openai-api-key:latest" \
     --allow-unauthenticated \
     --add-cloudsql-instances=PROJECT:REGION:INSTANCE
   ```

## PostgreSQL Vector Database

The application uses pgvector for storing and querying document embeddings. To set up:

1. Ensure you have a PostgreSQL instance with pgvector extension enabled
2. The application will automatically create the necessary vector tables when documents are processed

## Architecture

- **Streamlit**: Web interface for document upload and Q&A
- **LlamaIndex**: Document processing and retrieval framework
- **OpenAI**: Embedding and LLM capabilities
- **PostgreSQL with pgvector**: Vector storage and similarity search

## Troubleshooting

If you encounter issues:

1. Check that all environment variables are properly set
2. Verify PostgreSQL connectivity and pgvector extension installation
3. Ensure your OpenAI API key is valid and has sufficient quota
4. For Cloud Run deployments, check IAM permissions for Secret Manager and Cloud SQL access 