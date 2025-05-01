# Vector Search RAG Implementation

This repository contains a Retrieval-Augmented Generation (RAG) implementation that uses PostgreSQL with pgvector for semantic search.

## Overview

The application uses LlamaIndex and OpenAI embeddings to build a robust RAG system with the following features:

- Vector search using PostgreSQL with pgvector extension
- Multi-table search capability to find information across various document collections
- Hybrid search combining vector similarity and keyword matching
- Fallback mechanisms to ensure reliable information retrieval
- Direct document retrieval when vector search fails

## Key Components

- `multi_table_rag.py`: Core RAG implementation with vector search capabilities
- `direct_query.py`: A simple standalone script for direct RAG queries
- `Tests/test_rag.py`: Test script to verify RAG functionality
- `Tests/test_vector_search.py`: Performance benchmark for vector search functionality
- `db_config.py`: Database configuration utilities

## Testing with test_rag.py

The `Tests/test_rag.py` script is a critical diagnostic tool for verifying your RAG implementation. It performs the following functions:

1. **Connection Verification**: Tests if the system can properly connect to the PostgreSQL database
2. **Table Discovery**: Identifies available vector tables in the database
3. **Document Discovery**: Lists all documents stored in the vector database
4. **Query Testing**: Runs multiple pre-defined queries against the RAG system to test retrieval capabilities

### Running the Test Script

```bash
python Tests/test_rag.py
```

### Sample Test Queries

The script includes several test queries to evaluate different aspects of the RAG system:

1. **Document Signatories**: "quem assinou o contrato entre Coentro e Jumpad?"
2. **Payment Terms**: "what are the payment terms in the contract between Coentro and Jumpad?"
3. **Signing Date**: "what was the signing date of the contract?"
4. **General Terms**: "explain the general terms of the contract"

### Output Format

For each query, the script outputs:
- The original query
- The RAG system's answer
- The best matching table that provided the information
- The number of sources used in generating the answer

### Interpreting Results

- **Successful Results**: Should display coherent answers with relevant source information
- **Failed Results**: Will report errors such as "No relevant information found" if the system cannot retrieve useful content

### Usage Example

This test script is invaluable when:
- Setting up a new RAG deployment
- Troubleshooting retrieval issues
- Testing after database or document updates
- Validating fallback mechanisms in the RAG pipeline

### Detailed Example: Diagnosing RAG Issues

Below is an example workflow for diagnosing issues with your RAG implementation:

1. **Run the test script to get baseline diagnostics**:
   ```bash
   python Tests/test_rag.py
   ```

2. **Check for database connectivity**:
   The test script will first verify your database connection and report details like:
   ```
   ✅ Database connection successful!
   PostgreSQL version: PostgreSQL 15.12
   pgvector extension: Installed
   pgvector version: 0.8.0
   Vector tables found: 2
   ```

3. **Examine document and table details**:
   The script will list all tables and documents found:
   ```
   Found 2 vector tables
     Table 1: data_vectors_472971c1_4265_4aba_a6cf_c3b633115fe1
       Description: Contains 1 documents: llamaindex.pdf
       Documents: 1
       Chunks: 14
       HNSW Index: Yes
   ```

4. **Review query results**:
   For each test query, examine the response quality. Successful queries will show:
   ```
   ✅ Query successful!
   Retrieved from table: data_vectors_cdc82293_1cb1_4986_806b_4f57459e57e3
   Sources used: 1
   Source documents: Coentro e Jumpad contract
   ```

5. **Check the log file for detailed diagnostics**:
   The script saves all logs to `test_rag_results.log` for more detailed review:
   ```bash
   cat test_rag_results.log
   ```

6. **Troubleshooting common issues**:
   - If database connection fails, check your environment variables
   - If no documents are found, ensure documents are properly indexed
   - If queries return no results, check that your RAG pipeline is configured correctly
   - If results lack source attribution, verify that metadata is properly stored

By using this script regularly, you can maintain confidence in your RAG system's functionality and quickly diagnose any issues that arise.

## Benchmarking with test_vector_search.py

The `Tests/test_vector_search.py` script is a comprehensive benchmarking tool that evaluates the quality and performance of the vector search capabilities within the RAG system. This script is particularly valuable for:

1. **Performance Evaluation**: Measures query execution time and response latency
2. **Retrieval Quality Assessment**: Tests the system's ability to find relevant information across documents
3. **Multi-domain Testing**: Evaluates both document-specific queries and general knowledge questions
4. **Source Relevance Analysis**: Reports on the relevance scores of retrieved sources

### Running the Benchmark Script

```bash
python Tests/test_vector_search.py
```

### Benchmark Query Types

The script tests the vector search system with two categories of queries:

1. **Document-specific Queries**:
   - Contract value inquiries
   - Document signatory identification
   - Date and duration information
   - These validate the system's ability to retrieve precise factual information

2. **Conceptual Knowledge Queries**:
   - Technical explanations about RAG systems
   - LlamaIndex and OpenAI integration details
   - Vector search implementation concepts
   - These test broader knowledge retrieval capabilities

### Output Metrics

For each test query, the script reports:
- Query execution time in seconds
- Table that provided the best matching results
- Full response text
- Source documents used with their relevance scores

### Usage Benefits

This benchmarking tool is invaluable when:
- Optimizing vector database configuration
- Evaluating search performance after adding new documents
- Testing system behavior with different query formulations
- Comparing the effectiveness of different indexing strategies
- Troubleshooting specific retrieval failures

By running this script regularly, you can track and improve your RAG system's search quality over time.

## How It Works

1. Documents are stored in PostgreSQL tables with vector embeddings
2. Queries are converted to embeddings and used for similarity search
3. Results are retrieved, ranked, and processed for relevance
4. A language model (OpenAI) generates coherent responses using retrieved context
5. Multiple fallback mechanisms ensure reliable information retrieval

## Environment Variables

The following environment variables should be set:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DB_PUBLIC_IP`: PostgreSQL server IP
- `PG_PORT`: PostgreSQL port (default: 5432)
- `PG_USER`: PostgreSQL username
- `PG_PASSWORD`: PostgreSQL password
- `PG_DATABASE`: PostgreSQL database name

## Example Usage

```python
from app.multi_table_rag import MultiTableRAGTool

# Initialize the RAG tool
rag_tool = MultiTableRAGTool()

# Query the system
result = rag_tool.query("Who signed the contract between Coentro and Jumpad?")

# Print the result
print(result["answer"])
```

## Troubleshooting

If you encounter issues with the RAG system:

1. Verify that the pgvector extension is properly installed in PostgreSQL
2. Check that document vectors are correctly stored in the database
3. Verify OpenAI API key and connection parameters
4. Run the `Tests/test_rag.py` script to diagnose any issues

## Dependencies

- LlamaIndex
- OpenAI
- PostgreSQL with pgvector extension
- psycopg2

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