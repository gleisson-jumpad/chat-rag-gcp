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

## Understanding LlamaIndex with pgvector Integration

LlamaIndex serves as the backbone of this RAG implementation, handling both document processing and retrieval operations with pgvector. Here's a detailed explanation of how it works for the two main processes:

### 1. Document Upload and Vector Storage Pipeline

When a document is uploaded to our system, LlamaIndex processes it through the following steps:

1. **Document Loading** (`rag_utils.py: process_uploaded_file()`):
   - Documents are loaded using LlamaIndex's document loaders based on file type
   - Support for PDF, DOCX, TXT, CSV, MD, and other formats
   - Text extraction with proper metadata preservation

2. **Text Chunking** (`rag_utils.py: process_uploaded_file()`):
   - Documents are split into smaller chunks using LlamaIndex text splitters
   - Default chunk size is configured for optimal retrieval
   - Chunk overlap ensures context preservation across chunks

3. **Embedding Generation** (`rag_utils.py -> multi_table_rag.py`):
   - Each text chunk is converted to an embedding vector using OpenAI's embedding model
   - Default model: text-embedding-ada-002 (dimensions: 1536)
   - Batched processing for efficiency

4. **PostgreSQL Storage** (`postgres_rag_tool.py`):
   - Embeddings are stored in PostgreSQL tables with pgvector extension
   - Each document collection gets its own table with schema:
     ```
     CREATE TABLE {table_name} (
       id UUID PRIMARY KEY,
       embedding VECTOR(1536),
       document TEXT,
       metadata_ JSONB
     )
     ```
   - HNSW indices are created for approximate nearest neighbor search
   - File metadata is preserved in the JSONB column

**Code Flow for Document Processing:**

```
app/main.py (upload interface)
    ↓
app/rag_utils.py:process_uploaded_file()
    ↓
llama_index.core:VectorStoreIndex creation
    ↓
app/multi_table_rag.py:create_vector_index()
    ↓
app/postgres_rag_tool.py:store_embeddings()
    ↓
PostgreSQL with pgvector
```

### 2. Query and Retrieval Pipeline

When a user submits a query, the system processes it through these steps:

1. **Query Embedding** (`multi_table_rag.py: query()`):
   - User query is converted to an embedding vector
   - Same embedding model is used as for document processing

2. **Multi-Table Vector Search** (`multi_table_rag.py: search_across_tables()`):
   - Query embedding is compared against embeddings in all vector tables
   - Similarity search uses cosine similarity with pgvector
   - Top-k most similar documents from each table are retrieved

3. **Hybrid Search Enhancement** (`postgres_rag_tool.py: hybrid_search()`):
   - Vector similarity search is combined with keyword matching
   - BM25 or other keyword ranking algorithms boost relevance
   - Results are re-ranked based on combined score

4. **Context Synthesis** (`rag_processor.py: process_query_with_llm()`):
   - Retrieved document chunks are assembled into context
   - Context is formatted with proper attribution and metadata

5. **Response Generation** (`rag_processor.py: process_query_with_llm()`):
   - LLM (OpenAI) generates a response based on retrieved context
   - System prompts ensure proper source attribution
   - Fallback mechanisms handle cases with insufficient context

**Code Flow for Query Processing:**

```
app/main.py (query interface)
    ↓
app/multi_table_rag.py:query()
    ↓
app/multi_table_rag.py:search_across_tables()
    ↓
app/postgres_rag_tool.py:vector_search() / hybrid_search()
    ↓
app/rag_processor.py:process_query_with_llm()
    ↓
Response generation with OpenAI
```

### Key Files and Their Functions

- `app/main.py`: Web interface for document upload and query processing
- `app/rag_utils.py`: Utilities for document processing and data formatting
- `app/multi_table_rag.py`: Core implementation for multi-table RAG
- `app/postgres_rag_tool.py`: Integration with PostgreSQL and pgvector
- `app/rag_processor.py`: LLM query processing and response generation
- `app/db_config.py`: Database connection and configuration utilities
- `app/pgvector_admin.py`: Utilities for managing pgvector tables and indices

### Advanced Features

1. **HNSW Indices for Performance**
   - Hierarchical Navigable Small World indices accelerate vector search
   - Configured with appropriate M and ef_construction parameters
   - Managed through `pgvector_admin.py`

2. **Multi-Table Search Strategy**
   - Documents are organized into separate tables by collection or type
   - Parallel search across tables with rank aggregation
   - Table-specific relevance scoring and normalization

3. **Hybrid Search Capabilities**
   - Vector similarity combined with keyword matching
   - Configurable weights between semantic and lexical search
   - Fall-back mechanisms when semantic search yields low confidence

## Indexing Strategy

This RAG implementation primarily uses **Vector Store Index** as its core indexing strategy, specifically with PostgreSQL and pgvector extension as the backend. Here's a breakdown of our approach and how it compares to alternative indexing options:

### Vector Store Index with PGVector

Our system leverages LlamaIndex's `VectorStoreIndex` with `PGVectorStore` implementation to create and query vector embeddings. This combination offers several advantages:

- **Scalability**: PostgreSQL can handle millions of document chunks efficiently
- **Persistence**: Embeddings remain available after system restarts
- **SQL Integration**: Combines vector search with traditional SQL capabilities
- **Hybrid Search**: Enables both semantic (vector) and lexical (keyword) search
- **ACID Compliance**: Ensures data consistency during concurrent operations

Core implementation components:
```python
# Vector store creation
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=db_host,
    password=db_password,
    port=db_port,
    user=db_user,
    table_name=table_name,
    embed_dim=1536
)

# Index creation from vector store
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context
)
```

### Comparison with Alternative Indexing Strategies

While our system primarily uses Vector Store Index, it's useful to understand how it compares to other indexing options:

#### **Vector Store Index**
- **What it is:** Embeds documents/nodes into vectors for similarity search
- **Use cases:** General retrieval, semantic search, finding similar content
- **Our implementation:** **Primary approach** in our system - optimized for similarity-based retrieval using PostgreSQL pgvector
- **Advantages:** Efficient semantic search, scales well, supports hybrid retrieval

#### **Summary Index**
- **What it is:** Creates summaries of documents for condensed information retrieval
- **Use cases:** Question answering over long texts, high-level information needs
- **Our implementation:** We use dynamic summarization at query time rather than pre-summarization
- **Comparison:** More flexible than static summaries, maintains original context

#### **List Index**
- **What it is:** Simple sequential storage of document nodes
- **Use cases:** Small document sets, precise matching, ordered retrieval
- **Our implementation:** Too simplistic for our needs - we need semantic matching
- **Comparison:** Our vector approach provides much richer search capabilities

#### **Tree Index**
- **What it is:** Hierarchical document structure for representing nested information
- **Use cases:** Complex, nested documents, hierarchical exploration
- **Our implementation:** We achieve similar hierarchical organization through multi-table design
- **Comparison:** Our approach offers better cross-document retrieval and scalability

#### **Keyword Table Index**
- **What it is:** Retrieval based on keywords and exact matching
- **Use cases:** Precise term lookups, exact matching requirements
- **Our implementation:** Incorporated as part of our hybrid search capability
- **Comparison:** We combine this with vector search for both semantic and lexical matching

#### **SQL Index**
- **What it is:** Integration with SQL databases for structured data retrieval
- **Use cases:** Structured data, complex filtering, joins across tables
- **Our implementation:** We combine this with vector search through PostgreSQL integration
- **Comparison:** We get the best of both worlds: structured queries and semantic search

#### **KG Index (Knowledge Graph)**
- **What it is:** Knowledge graph representation of document relationships
- **Use cases:** Relationship-heavy domains, entity linking, complex querying
- **Our implementation:** We implement limited graph-like features through entity recognition
- **Comparison:** Our approach is more flexible while incorporating some KG benefits

### Why Vector Store Index with pgvector?

We chose this approach for several key reasons:

1. **Performance at Scale**: HNSW indices in pgvector enable sub-second queries over millions of vectors
2. **Flexibility**: PostgreSQL allows custom extensions to the vector search capabilities
3. **Maturity**: Both PostgreSQL and pgvector are production-ready technologies
4. **Hybrid Search**: Enables seamless integration of semantic and keyword search
5. **Multi-table Design**: Supports organization of documents into logical collections

### Implementation Details

Our Vector Store Index implementation includes several optimizations:

- **Parallel Index Creation**: Documents are processed and indexed in parallel
- **Batch Embedding**: Vectors are generated in batches to optimize API usage
- **HNSW Indexing**: Automatic creation of HNSW indices for performance
- **Field-specific Indexing**: Metadata fields receive specialized indexing
- **Hybrid Retrieval**: Vector search is augmented with keyword matching

This indexing strategy powers both document ingestion and retrieval in our RAG pipeline, providing the foundation for accurate and efficient information retrieval.

## Key Technical Concepts

### Vector Search Fundamentals

#### 1. HNSW Indices (Hierarchical Navigable Small Worlds)

HNSW is an advanced algorithm for approximate nearest neighbor (ANN) search in high-dimensional vector spaces, critical for the performance of our RAG system.

- **How it works**: HNSW creates a multi-layered graph structure where:
  - Upper layers form a coarse-grained representation of the vector space
  - Lower layers provide increasingly finer-grained representations
  - Search starts at the top layer and descends through the hierarchy

- **Key parameters**:
  - `M`: Controls the maximum number of connections per node (default: 16)
  - `ef_construction`: Controls index build quality vs. speed tradeoff (default: 64)
  - `ef_search`: Controls search quality vs. speed tradeoff (runtime parameter)

- **Benefits in our implementation**:
  - Logarithmic search complexity vs. linear in brute-force approaches
  - 100-1000x speed improvement for large document collections
  - Minimal accuracy loss compared to exact search methods

- **Implementation in pgvector**:
  ```sql
  CREATE INDEX idx_hnsw_embedding ON vector_table
  USING hnsw(embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  ```

#### 2. Top-k Retrieval

Top-k retrieval is the process of finding the k most similar documents to a query vector, forming the foundation of our vector search.

- **How it works**:
  - Query embedding is compared against all document embeddings
  - Similarity measures (cosine, dot product, Euclidean) quantify relevance
  - Documents are ranked by similarity score
  - Only the k highest-scoring documents are returned

- **Implementations in our system**:
  - **Basic Top-k**: Direct retrieval of k nearest vectors
    ```sql
    SELECT document, metadata_, 1 - (embedding <=> query_embedding) as score
    FROM vector_table
    ORDER BY embedding <=> query_embedding
    LIMIT k;
    ```
  
  - **Table-aware Top-k**: Adjusting k based on table characteristics
    - Larger k for tables with more diverse content
    - Smaller k for specialized, domain-specific tables
    - Dynamic k adjustment based on query complexity

- **Optimizations**:
  - Parallel retrieval across multiple tables
  - Score normalization for cross-table comparison
  - Diversity-aware selection to reduce redundancy

#### 3. BM25 and Keyword Ranking Algorithms

BM25 (Best Matching 25) is a probabilistic ranking algorithm used alongside vector search in our hybrid search implementation.

- **How BM25 works**:
  - Scores documents based on term frequency and inverse document frequency
  - Accounts for document length normalization
  - Handles term saturation (diminishing returns for repeated terms)

- **BM25 formula**:
  ```
  score(D,Q) = ∑(IDF(qi) · (f(qi,D) · (k1+1)) / (f(qi,D) + k1 · (1-b+b · |D|/avgdl)))
  ```
  Where:
  - f(qi,D): Term frequency of term qi in document D
  - |D|: Length of document D
  - avgdl: Average document length
  - k1, b: Tuning parameters

- **Implementation in our system**:
  - PostgreSQL's full-text search capabilities with custom BM25 weighting
  - Term-based inverted indices for efficient keyword lookups
  - Custom scoring functions combining vector and keyword relevance

- **Other lexical ranking algorithms used**:
  - **TF-IDF**: Term Frequency-Inverse Document Frequency
  - **Language models**: Probabilistic models of term occurrence
  - **N-gram matching**: For handling misspellings and partial matches

## Enhanced Advanced Features

### 1. Multi-Modal RAG Capabilities

Our system extends beyond text to handle multiple modalities:

- **Image Understanding**:
  - Image embedding generation using CLIP or similar models
  - Cross-modal search between text queries and image content
  - Image metadata extraction and indexing

- **Table and Structured Data Processing**:
  - Special handling for CSV and spreadsheet data
  - Table content vectorization with structure preservation
  - Query-specific table column selection

### 2. Adaptive Retrieval Strategies

The system employs contextual adaptation techniques to optimize retrieval:

- **Query Classification**:
  - Automatic detection of query type (factoid, analytical, exploratory)
  - Selection of retrieval strategy based on query characteristics
  - Different similarity thresholds for different query types

- **Context-Aware Retrieval**:
  - Conversation history influences retrieval parameters
  - Progressive refinement of search based on user feedback
  - Query expansion using detected entities and concepts

### 3. Knowledge Graph Enhancement

Vector search is augmented with knowledge graph capabilities:

- **Entity Recognition and Linking**:
  - Identification of key entities in documents and queries
  - Connection of entities across document boundaries
  - Entity-aware query reformulation

- **Relationship Extraction**:
  - Detection of semantic relationships between entities
  - Graph-based relevance scoring as a factor in ranking
  - Connection of information across disparate documents

### 4. Performance Optimization Techniques

Our implementation includes several performance enhancements:

- **Query Routing**:
  - Intelligent routing of queries to specific tables based on content
  - Avoidance of unnecessary search in irrelevant collections
  - Metadata-based pre-filtering before vector search

- **Caching Mechanisms**:
  - LRU cache for frequent queries and their results
  - Embedding cache to avoid recomputing query embeddings
  - Document chunk cache for frequently retrieved passages

- **Batch Processing**:
  - Concurrent processing of multiple search requests
  - Optimized batch embedding generation
  - Parallel table scanning with work distribution

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
     --set-env-vars="PG_PORT=5432" \
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