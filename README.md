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

## PostgreSQL Vector Store Enhancements

This codebase includes several improvements to the PostgreSQL vector database integration:

### Connection Handling Improvements
- Added connection pooling for better performance and reliability
- Implemented context managers for safer resource handling
- Added automatic reconnection logic for more robust database access

### Vector Search Enhancements
- Added support for HNSW indices for faster approximate nearest neighbor search
- Improved hybrid search capabilities (combining semantic and keyword search)
- Added advanced filtering options for more precise query results
- Implemented similarity thresholds to filter out low-relevance results

### Database Management Tools
- Added `pgvector_admin.py` utility for managing vector tables:
  - List all vector tables in the database
  - Verify table structures and indices
  - Create and optimize HNSW indices
  - Vacuum tables to improve performance
  - Check database configuration and pgvector extension

### Reliability Improvements
- Enhanced error handling with fallback mechanisms
  - Graceful degradation when optimal features aren't available
  - Detailed logging for easier troubleshooting
- Improved connection cleanup to prevent resource leaks
- Added verification of pgvector extension and automatic configuration

### Performance Optimizations
- Increased embedding batch sizes for faster processing
- Optimized index creation with proper HNSW parameters
- Implemented progress tracking for long-running operations
- Added table statistics gathering for better query planning

To use the PostgreSQL vector store improvements:
1. Ensure your PostgreSQL instance has pgvector extension installed
2. Run the database check: `python app/pgvector_admin.py check-db`
3. Optimize existing tables: `python app/pgvector_admin.py create-indices`
4. Refer to the multi_table_rag.py and postgres_rag_tool.py files for usage examples

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

### RAG System Not Finding Documents

If your RAG system reports "Nenhum documento disponível para RAG":

1. **Check database connection**: 
   - Navigate to the "Teste de Conexão com PostgreSQL" page to verify database connectivity
   - Use the Diagnóstico Avançado page to check both connection and environment variables

2. **Environment variables**: 
   - Ensure all database environment variables are correctly set:
     ```bash
     export DB_PUBLIC_IP=your_postgresql_ip
     export PG_PORT=5432
     export PG_DB=postgres
     export PG_USER=llamaindex
     export PG_PASSWORD=your_password
     ```

3. **Reset session state**: 
   - On the Diagnóstico Avançado page, click "Limpar e Recarregar Sessão" to reset the application state
   - This will force the app to re-check for documents in the database

4. **Database tables**:
   - Check if vector tables exist in your PostgreSQL database
   - Tables should be named with the pattern `vectors_*`
   - You can verify this in the Diagnóstico Avançado page

5. **Create a test document**:
   - Upload a simple text or PDF file to verify the processing pipeline
   - Monitor the console logs for any errors during document processing 