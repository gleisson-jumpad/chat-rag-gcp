import os
import json
import logging
import time # Import time
import psycopg2
from psycopg2 import pool
import threading

# --- Add this line for debugging file loading ---
print(f"--- Loading app/multi_table_rag.py @ {time.time()} ---")
# -----------------------------------------------

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import openai

# Try importing with and without 'app.' prefix to handle different execution contexts
try:
    from db_config import get_pg_connection, check_postgres_connection
except ImportError:
    try:
        from app.db_config import get_pg_connection, check_postgres_connection
    except ImportError:
        raise ImportError("Could not import db_config module. Make sure it exists in the correct location.")

"""
IMPROVED RAG SYSTEM

This file contains improvements to the RAG system to make it more robust:

1. Enhanced document detection: Better recognition of document requests with special handling for accents and case sensitivity
2. Direct document access: Specialized methods for document summarization that bypass complex query logic
3. Improved error handling: More detailed logging and fallback mechanisms when things go wrong
4. Diagnostic tools: Verification functions to identify configuration issues
5. Multiple retrieval strategies: Trying different queries and direct database access when standard retrieval fails
6. Connection pooling: Using connection pools for better performance with PostgreSQL

The most important improvements are:
- The dedicated summarize_document method for direct document access
- Enhanced detect_document_request for better document recognition
- More logging throughout the codebase for easier debugging
- Connection pooling for better PostgreSQL performance
"""

class MultiTableRAGTool:
    def __init__(
        self,
        db_config: Dict[str, Any] = None,
        openai_model: str = "gpt-4o"
    ):
        """
        Initialize a RAG tool that can search across multiple PostgreSQL tables.
        
        Args:
            db_config: Database connection parameters (host, port, user, password, etc.)
            openai_model: The OpenAI model to use for queries
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MultiTableRAGTool")
        
        # Set default database config if not provided
        if not db_config:
            self.db_config = {
                "database": os.getenv("PG_DB", "postgres"),
                "host": os.getenv("DB_PUBLIC_IP", "34.150.190.157"),
                "password": os.getenv("PG_PASSWORD", "password123"),
                "port": int(os.getenv("PG_PORT", 5432)),
                "user": os.getenv("PG_USER", "llamaindex"),
            }
        else:
            self.db_config = db_config
            
        self.openai_model = openai_model
        self.llm = OpenAI(model=openai_model)
        
        # Initialize connection pool for better performance
        self._init_connection_pool()
        
        # Discover available tables and their metadata
        self.table_configs = self._discover_table_configs()
        self.logger.info(f"Found {len(self.table_configs)} vector tables")
        
        # Create available_tables attribute from table_configs for backward compatibility
        self.available_tables = [config["name"] for config in self.table_configs]
        
        # Initialize components
        self.vector_stores = {}
        self.indexes = {}
        self.query_engines = {}
        
        # Initialize vector stores and indexes
        self._initialize()
        
        # Verify PostgreSQL connection and pgvector extension
        self.check_postgres_connection()
    
    def _init_connection_pool(self, min_conn=2, max_conn=10):
        """Initialize a connection pool for better performance"""
        try:
            self.logger.info(f"Creating PostgreSQL connection pool with {min_conn}-{max_conn} connections")
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=self.db_config["host"],
                port=self.db_config["port"],
                dbname=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"]
            )
            self.logger.info("Connection pool created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {str(e)}")
            self.connection_pool = None
    
    def _get_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Get a connection from the pool or create a new direct connection if pool fails"""
        conn = None
        try:
            if self.connection_pool:
                conn = self.connection_pool.getconn()
                if conn:
                    self.logger.debug("Got connection from pool")
                    return conn
        except Exception as e:
            self.logger.warning(f"Error getting connection from pool: {str(e)}")
        
        # Fallback to direct connection
        try:
            self.logger.info("Falling back to direct connection")
            conn = get_pg_connection()
            return conn
        except Exception as e:
            self.logger.error(f"Failed to create direct connection: {str(e)}")
            return None
    
    def _return_connection(self, conn):
        """Return a connection to the pool if it came from there"""
        if not conn:
            return
            
        try:
            if self.connection_pool:
                self.connection_pool.putconn(conn)
        except Exception as e:
            self.logger.warning(f"Error returning connection to pool: {str(e)}")
            try:
                conn.close()
            except:
                pass
    
    def _discover_table_configs(self):
        """Automatically discover vector tables and extract their metadata"""
        table_configs = []
        conn = None
        
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("Could not establish database connection for table discovery")
                return []
                
            cursor = conn.cursor()
            
            # First verify pgvector extension
            try:
                cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
                pgvector_info = cursor.fetchone()
                if not pgvector_info:
                    self.logger.warning("pgvector extension not found - vector operations will not work")
            except Exception as e:
                self.logger.warning(f"Could not verify pgvector extension: {str(e)}")
            
            # Find all vector tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND 
                (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
            """)
            
            tables = [table[0] for table in cursor.fetchall()]
            
            if not tables:
                self.logger.warning("No vector tables found in database")
                return []
                
            self.logger.info(f"Found {len(tables)} vector tables: {', '.join(tables)}")
            
            # Extract metadata for each table
            for table_name in tables:
                try:
                    # Check if table has the required vector column
                    cursor.execute(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' AND column_name = 'embedding';
                    """)
                    vector_col = cursor.fetchone()
                    
                    if not vector_col:
                        self.logger.warning(f"Table {table_name} does not have an 'embedding' column")
                        continue
                        
                    # Get file names in this table
                    cursor.execute(f"""
                        SELECT DISTINCT metadata_->>'file_name' as file_name 
                        FROM {table_name}
                        WHERE metadata_->>'file_name' IS NOT NULL
                    """)
                    
                    file_names = [row[0] for row in cursor.fetchall() if row[0]]
                    files_description = ", ".join(file_names) if file_names else "Unknown documents"
                    
                    # Count documents and chunks
                    cursor.execute(f"""
                        SELECT COUNT(DISTINCT metadata_->>'file_name') as doc_count,
                               COUNT(*) as chunk_count
                        FROM {table_name}
                    """)
                    
                    stats = cursor.fetchone()
                    doc_count = stats[0] if stats else 0
                    chunk_count = stats[1] if stats else 0
                    
                    # Check for HNSW index on the table
                    cursor.execute(f"""
                        SELECT indexname, indexdef 
                        FROM pg_indexes 
                        WHERE tablename = '{table_name}' AND indexdef LIKE '%hnsw%'
                    """)
                    hnsw_indices = cursor.fetchall()
                    has_hnsw = len(hnsw_indices) > 0
                    
                    # Create table config
                    table_configs.append({
                        "name": table_name,
                        "description": f"Contains {doc_count} documents: {files_description}",
                        "embed_dim": 1536,
                        "top_k": 5,
                        "hybrid_search": True,
                        "language": "english",
                        "files": file_names,
                        "doc_count": doc_count,
                        "chunk_count": chunk_count,
                        "has_hnsw": has_hnsw,
                        "hnsw_indices": [idx[0] for idx in hnsw_indices] if has_hnsw else []
                    })
                    
                    self.logger.info(f"Table {table_name}: {doc_count} docs, {chunk_count} chunks, HNSW: {has_hnsw}")
                    
                except Exception as e:
                    self.logger.error(f"Error extracting metadata for table {table_name}: {str(e)}")
            
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Error discovering tables: {str(e)}")
        finally:
            if conn:
                self._return_connection(conn)
        
        return table_configs
    
    def _initialize(self):
        """Initialize vector stores and indexes for each table"""
        self.logger.info("Initializing vector stores and indexes for all tables")
        
        # Set up OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            self.logger.error("OPENAI_API_KEY not set in environment")
            raise ValueError("OpenAI API key is required")
            
        # Configure LlamaIndex settings with OpenAI embedding model
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        
        # Initialize each table
        for table_config in self.table_configs:
            table_name = table_config["name"]
            self.logger.info(f"Initializing table: {table_name}")
            
            try:
                # Create vector store with hybrid search capability
                vector_store = PGVectorStore.from_params(
                    database=self.db_config["database"],
                    host=self.db_config["host"],
                    password=self.db_config["password"],
                    port=self.db_config["port"],
                    user=self.db_config["user"],
                    table_name=table_name,
                    embed_dim=table_config.get("embed_dim", 1536),  # OpenAI embedding dimension
                    hybrid_search=table_config.get("hybrid_search", True),  # Enable hybrid search
                    text_search_config=table_config.get("language", "english")
                )
                
                self.vector_stores[table_name] = vector_store
                
                # Create storage context
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Create index from vector store
                self.logger.info(f"Loading index from vector store: {table_name}")
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model
                )
                
                self.indexes[table_name] = index
                
                # Create query engine
                self.query_engines[table_name] = self._create_query_engine(vector_store, self.llm)
                
                self.logger.info(f"Successfully initialized table: {table_name}")
                
            except Exception as e:
                self.logger.error(f"Error initializing table {table_name}: {str(e)}")
                # Continue with other tables instead of failing completely
    
    def __del__(self):
        """Cleanup connection pool on object destruction"""
        try:
            if hasattr(self, 'connection_pool') and self.connection_pool:
                self.connection_pool.closeall()
                self.logger.info("Connection pool closed")
        except Exception as e:
            self.logger.error(f"Error closing connection pool: {str(e)}")
    
    def check_postgres_connection(self):
        """Verify PostgreSQL connection and check for pgvector extension"""
        self.logger.info("Checking PostgreSQL connection and pgvector extension")
        
        try:
            conn = get_pg_connection()
            cursor = conn.cursor()
            
            # Check PostgreSQL version
            cursor.execute("SELECT version();")
            pg_version = cursor.fetchone()[0]
            self.logger.info(f"PostgreSQL connection successful. Version: {pg_version}")
            
            # Check if pgvector extension is installed
            cursor.execute("""
                SELECT extname, extversion 
                FROM pg_extension 
                WHERE extname = 'vector';
            """)
            
            pgvector_info = cursor.fetchone()
            
            if pgvector_info:
                self.logger.info(f"pgvector extension is installed. Version: {pgvector_info[1]}")
            else:
                self.logger.warning("pgvector extension is not installed. Vector operations will fail.")
                self.logger.warning("To install pgvector, run: CREATE EXTENSION vector;")
            
            # Check if there are any vector tables
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND 
                (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
            """)
            
            vector_table_count = cursor.fetchone()[0]
            self.logger.info(f"Found {vector_table_count} vector tables in the database")
            
            cursor.close()
            conn.close()
            
            return {
                "postgres_connection": True,
                "postgres_version": pg_version,
                "pgvector_installed": pgvector_info is not None,
                "pgvector_version": pgvector_info[1] if pgvector_info else None,
                "vector_table_count": vector_table_count
            }
            
        except Exception as e:
            self.logger.error(f"PostgreSQL connection check failed: {str(e)}")
            return {
                "postgres_connection": False,
                "error": str(e)
            }
    
    def query_single_table(self, table_name, query_text, filters=None):
        """
        Query a specific table by name
        
        This approach follows pg_rag_simple.py's pattern for querying and returning results
        with proper source attribution.
        """
        self.logger.info(f"Querying table '{table_name}' with: '{query_text}' (filters: {filters})")
        
        try:
            # Check if vector store exists
            if table_name not in self.vector_stores:
                self.logger.warning(f"No vector store found for table {table_name}")
                return {"error": "Table not found", "message": f"Table {table_name} is not available"}
            
            # Check if we have an index for this table
            if table_name not in self.indexes:
                self.logger.warning(f"No index found for table {table_name}")
                return {"error": "Index not found", "message": f"Index for table {table_name} is not available"}
            
            # Get or create query engine
            if table_name not in self.query_engines:
                self.logger.info(f"Creating new query engine for {table_name}")
                if table_name in self.vector_stores and table_name in self.indexes:
                    self.query_engines[table_name] = self._create_query_engine(
                        self.vector_stores[table_name], 
                        self.llm
                    )
            
            # Execute query with the engine
            self.logger.info(f"Executing query against table {table_name}")
            
            # Add filter if provided
            if filters:
                self.logger.info(f"Applying filters: {filters}")
                response = self.query_engines[table_name].query(query_text, filter=filters)
            else:
                response = self.query_engines[table_name].query(query_text)
            
            # Format the response with source nodes
            result = {
                "answer": str(response),
                "sources": []
            }
            
            # Include source nodes if available - following pg_rag_simple.py's pattern
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_info = {
                        "text": node.node.get_content()[:150] + "..." if len(node.node.get_content()) > 150 else node.node.get_content(),
                        "file_name": node.node.metadata.get("file_name", "unknown"),
                        "relevance_score": float(node.score) if hasattr(node, "score") else None
                    }
                    result["sources"].append(source_info)
                    
                # Debug print what we found
                self.logger.info(f"Found {len(result['sources'])} source nodes")
                for idx, source in enumerate(result["sources"]):
                    self.logger.info(f"Source {idx+1}: {source['file_name']} - Score: {source['relevance_score']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying table {table_name}: {str(e)}")
            return {"error": str(e), "message": f"Error querying table {table_name}"}
            
    def _format_response(self, response):
        """Format a response with source information"""
        if hasattr(response, 'response'):
            answer = response.response
        else:
            answer = str(response)
            
        source_info = ""
        if hasattr(response, 'source_nodes') and response.source_nodes:
            source_info = "\n\nSources:\n"
            for i, node in enumerate(response.source_nodes):
                source_info += f"\n{i+1}. {node.node.metadata.get('file_name', 'Unknown document')}"
                if hasattr(node, 'score'):
                    source_info += f" (Score: {node.score:.3f})"
                source_info += f"\n{node.node.get_content()[:150]}...\n"
        
        return answer + source_info
    
    def query_single_table_with_filters(self, table_name, query_text, filters=None):
        """
        Query a specific table with proper LlamaIndex MetadataFilters
        
        Args:
            table_name: Table to query
            query_text: Query text to run against the table
            filters: LlamaIndex MetadataFilters object for filtering results
        """
        self.logger.info(f"Querying table '{table_name}' with: '{query_text}' and filters: {filters}")
        
        try:
            if table_name not in self.indexes:
                self.logger.error(f"Table {table_name} is not in available indices: {list(self.indexes.keys())}")
                return {"error": f"Table {table_name} not found", "answer": ""}
            
            # Get the vector store index for this table
            index = self.indexes[table_name]
            
            # Create a custom query engine with the filters
            if filters:
                self.logger.info(f"Creating custom query engine with filters: {filters}")
                
                # Create LLM
                llm = OpenAI(model=self.openai_model)
                
                # Create a custom query engine with the given filters
                query_engine = index.as_query_engine(
                    llm=llm,
                    similarity_top_k=10,  # Higher k value for better recall
                    response_mode="compact",
                    filters=filters,  # Apply the proper MetadataFilters
                )
                
                # Execute query
                self.logger.info(f"Executing query with filters against table {table_name}")
                response = query_engine.query(query_text)
            else:
                # Use default query engine if no filters
                self.logger.info(f"Using default query engine for table {table_name}")
                query_engine = self.query_engines[table_name]
                response = query_engine.query(query_text)
            
            # Format response
            result = self._format_response(response)
            
            return result
            
        except Exception as e:
            error_msg = f"Error querying table {table_name} with filters: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "answer": ""}
            
    def query_all_tables(self, query_text):
        """Query across all available tables and combine results"""
        self.logger.info(f"Querying across all {len(self.table_configs)} tables with: '{query_text}'")
        
        all_results = {}
        
        for table_config in self.table_configs:
            table_name = table_config["name"]
            try:
                result = self.query_single_table(table_name, query_text)
                all_results[table_name] = result
            except Exception as e:
                self.logger.error(f"Error querying table {table_name}: {str(e)}")
                all_results[table_name] = {"error": str(e)}
        
        return all_results
    
    def determine_relevant_tables(self, query_text):
        """Determine which tables are most relevant for the given query"""
        # Default implementation - can be enhanced with table embedding/similarity
        # For now, we'll just use all tables
        return [config["name"] for config in self.table_configs]
    
    def query(self, query_text: str, model: str = "gpt-4") -> Dict[str, Any]:
        """
        Query across all vector stores using the provided query text.
        
        Args:
            query_text (str): The query text to search for
            model (str): The model to use for the query (e.g., "gpt-4", "gpt-3.5-turbo")
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            self.logger.info(f"Processing query: {query_text}")
            
            # Check if the query is about financial information
            financial_keywords = ["valor", "preço", "custo", "pagamento", "mensalidade", "reais", "R$", "preços", "valores"]
            is_financial_query = any(keyword in query_text.lower() for keyword in financial_keywords)
            
            # Update the LLM model for this query
            llm = OpenAI(model=model, temperature=0.1)
            
            # Initialize tracking variables
            best_table_name = None
            best_response = None
            all_sources = []
            
            # Try each table
            for table_name in self.available_tables:
                if table_name not in self.vector_stores or table_name not in self.indexes:
                    self.logger.warning(f"Skipping table {table_name} - not properly initialized")
                    continue
                
                try:
                    self.logger.info(f"Querying table: {table_name}")
                    
                    # Get the index for this table
                    index = self.indexes[table_name]
                    
                    # If this is a financial query, try direct SQL first
                    if is_financial_query:
                        self.logger.info(f"Financial query detected, trying direct SQL approach first")
                        conn = get_pg_connection()
                        cursor = conn.cursor()
                        
                        # Search conditions for financial information
                        search_conditions = """
                            text ILIKE '%valor%' OR 
                            text ILIKE '%R$%' OR 
                            text ILIKE '%reais%' OR
                            text ILIKE '%pagamento%' OR
                            text ILIKE '%mensalidade%'
                        """
                        
                        # Search for contract information
                        cursor.execute(f"""
                            SELECT text, metadata_->>'file_name' as filename, metadata_->>'page_label' as page 
                            FROM {table_name}
                            WHERE {search_conditions}
                            LIMIT 7
                        """)
                        
                        results = cursor.fetchall()
                        if results:
                            self.logger.info(f"Found {len(results)} chunks with contract information in {table_name}")
                            
                            # Build context from contract information
                            context = "\n\n".join([row[0] for row in results])
                            
                            # Create prompt for LLM
                            prompt = f"""
                            Answer the following question based on this contract information:
                            
                            Question: {query_text}
                            
                            Context:
                            {context}
                            
                            Answer the question using only information from the context. Use the exact values, names, and dates mentioned in the document.
                            If the information is not clearly stated in the context, indicate that it might not be available.
                            
                            For contract values, specify the exact amounts. 
                            For signing parties, mention full names, roles, and companies.
                            For signing dates, provide the precise date if available.
                            
                            Look carefully for lines like "Assinou como contratada em" or "Assinou como contratante em" as these indicate who signed and when.
                            Also look for emails associated with signatories.
                            """
                            
                            # Get answer from LLM
                            response = llm.complete(prompt)
                            answer_text = response.text if hasattr(response, 'text') else str(response)
                            
                            # Create synthetic sources
                            sources = []
                            for text, filename, page in results:
                                sources.append({
                                    "text": text[:150] + "..." if len(text) > 150 else text,
                                    "metadata": {"file_name": filename, "page_label": page},
                                    "score": 0.95,  # High score since this is direct match
                                    "table": table_name
                                })
                            
                            all_sources.extend(sources)
                            best_response = answer_text
                            best_table_name = table_name
                            
                            cursor.close()
                            conn.close()
                            
                            # Skip vector search for this table since we already have good results
                            continue
                        
                        cursor.close()
                        conn.close()
                    
                    # Configure a query engine with higher similarity_top_k for better recall
                    query_engine = index.as_query_engine(
                        llm=llm,
                        similarity_top_k=15,  # Increase from 10 to 15 for better recall
                        response_mode="compact",
                        use_hybrid_search=True  # Always use hybrid search for better results
                    )
                    
                    # Execute query
                    response = query_engine.query(query_text)
                    self.logger.info(f"Got response from {table_name}")
                    
                    # Check if we got source nodes
                    if hasattr(response, 'source_nodes') and len(response.source_nodes) > 0:
                        self.logger.info(f"Found {len(response.source_nodes)} source nodes from {table_name}")
                        
                        # Process source nodes
                        sources = []
                        for node in response.source_nodes:
                            try:
                                content = node.node.get_content() if hasattr(node.node, 'get_content') else node.text
                                metadata = node.node.metadata if hasattr(node.node, 'metadata') else node.metadata
                                source = {
                                    "text": content[:150] + "..." if len(content) > 150 else content,
                                    "metadata": metadata,
                                    "score": float(node.score) if hasattr(node, "score") else None,
                                    "table": table_name
                                }
                                sources.append(source)
                                self.logger.info(f"Source from {table_name}: {source['metadata'].get('file_name', 'Unknown')}")
                            except Exception as src_err:
                                self.logger.error(f"Error processing source node: {str(src_err)}")
                        
                        # Track all sources
                        all_sources.extend(sources)
                        
                        # If we got sources and no best response yet, use this one
                        if sources and (best_response is None or len(sources) > 0):
                            best_response = str(response)
                            best_table_name = table_name
                            self.logger.info(f"Setting best response from {table_name}")
                except Exception as e:
                    self.logger.error(f"Error querying table {table_name}: {str(e)}")
            
            # If we didn't find anything relevant
            if best_response is None:
                # Try one more approach - direct fetching of contract info
                try:
                    self.logger.info("No results from standard search, trying direct document lookup")
                    conn = get_pg_connection()
                    cursor = conn.cursor()
                    
                    # Search for any contract-related documents
                    for table_name in self.available_tables:
                        cursor.execute(f"""
                            SELECT text FROM {table_name}
                            WHERE 
                              metadata_->>'file_name' LIKE '%Coentro%' OR 
                              metadata_->>'file_name' LIKE '%Jumpad%' OR
                              metadata_->>'file_name' LIKE '%contrato%' OR
                              metadata_->>'file_name' LIKE '%contract%'
                            LIMIT 10
                        """)
                        
                        results = cursor.fetchall()
                        if results:
                            self.logger.info(f"Found direct document content in {table_name}")
                            
                            # Combine all text chunks
                            context = "\n\n".join([row[0] for row in results])
                            
                            # Create prompt for LLM
                            prompt = f"""
                            Answer the following question based on the provided context:
                            
                            Question: {query_text}
                            
                            Context:
                            {context}
                            
                            Answer the question using only the information from the context.
                            If the context doesn't contain the answer, say "I don't have information about that in my knowledge base."
                            """
                            
                            # Get answer directly from LLM
                            response = llm.complete(prompt)
                            best_response = response.text if hasattr(response, 'text') else str(response)
                            best_table_name = table_name
                            
                            # Create synthetic source
                            all_sources.append({
                                "text": "Direct document lookup result",
                                "metadata": {"file_name": "Coentro e Jumpad contract"},
                                "score": 1.0,
                                "table": table_name
                            })
                            
                            self.logger.info("Generated response from direct document lookup")
                            break
                            
                    cursor.close()
                    conn.close()
                    
                except Exception as direct_err:
                    self.logger.error(f"Error in direct document lookup: {str(direct_err)}")
            
            # If still no results
            if best_response is None:
                return {
                    "error": "No relevant information found",
                    "message": "Could not find relevant information in any knowledge base."
                }
            
            # Format the output
            output = best_response
            
            if all_sources:
                # Sort sources by score (highest first)
                sorted_sources = sorted(all_sources, key=lambda x: x.get("score", 0) if x.get("score") is not None else 0, reverse=True)
                
                output += "\n\nSources:"
                for i, source in enumerate(sorted_sources[:5], 1):  # Show top 5 sources
                    output += f"\n\n{i}. Score: {source.get('score', 'N/A'):.3f}" if source.get('score') is not None else f"\n\n{i}. Score: N/A"
                    output += f"\n   Document: {source['metadata'].get('file_name', 'Unknown')}"
                    output += f"\n   Text: {source['text']}"
            
            return {
                "answer": output,
                "sources": all_sources,
                "best_table": best_table_name
            }
            
        except Exception as e:
            self.logger.error(f"Error in query processing: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while processing your query."
            }
    
    def _create_synthesis_prompt(self, query, table_results):
        """Create a prompt to synthesize multiple table results"""
        prompt = f"The user asked: '{query}'\n\n"
        prompt += "I've found information from multiple data sources:\n\n"
        
        for result in table_results:
            prompt += f"From {result['table']} ({result['description']}):\n"
            prompt += f"{result['answer']}\n\n"
        
        prompt += "Please synthesize this information into a comprehensive, coherent answer. "
        prompt += "Combine the information where appropriate and note any contradictions if they exist. "
        prompt += "If different sources provide complementary information, integrate it into a complete picture. "
        prompt += "Answer as if you were directly responding to the user's question."
        
        return prompt

    def _create_doc_summary_prompt(self, document_name, table_results):
        """Create a prompt to synthesize document content from multiple tables"""
        prompt = f"I need a detailed summary of the document: '{document_name}'\n\n"
        prompt += "I've found information from multiple sources:\n\n"
        
        for result in table_results:
            prompt += f"From {result['table']} ({result['description']}):\n"
            prompt += f"{result['answer']}\n\n"
        
        prompt += f"Please create a comprehensive, well-structured summary of the document '{document_name}'. "
        prompt += "The summary should include the main sections, key points, and important information from the document. "
        prompt += "Organize the summary into logical sections with headings if appropriate. "
        prompt += "Integrate information from all sources to create a complete picture of the document content."
        
        return prompt

    def get_tables_info(self):
        """Get information about available tables"""
        return {
            "table_count": len(self.table_configs),
            "tables": self.table_configs
        }
    
    def get_files_in_database(self):
        """Get a list of all files stored in the vector tables"""
        files_by_table = {}
        
        for table_config in self.table_configs:
            table_name = table_config["name"]
            files = table_config.get("files", [])
            if files:
                files_by_table[table_name] = files
        
        # Create a flat list of all files
        all_files = []
        for files in files_by_table.values():
            all_files.extend(files)
        
        return {
            "files_by_table": files_by_table,
            "all_files": list(set(all_files))  # Remove duplicates
        }
        
    def verify_document_vectors(self, document_name):
        """Debug function to verify vectors for a specific document"""
        self.logger.info(f"Verifying vectors for document: '{document_name}'")
        
        results = {}
        vector_counts = {}
        sample_contents = {}
        
        try:
            conn = get_pg_connection()
            cursor = conn.cursor()
            
            # Check each table for this document
            for table_name, vector_store in self.vector_stores.items():
                try:
                    # Count vectors for this document
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM {table_name}
                        WHERE metadata_->>'file_name' = %s
                    """, (document_name,))
                    
                    count = cursor.fetchone()[0]
                    vector_counts[table_name] = count
                    
                    # Get sample content if vectors exist
                    if count > 0:
                        cursor.execute(f"""
                            SELECT id, SUBSTRING(text, 1, 300) as content_preview
                            FROM {table_name}
                            WHERE metadata_->>'file_name' = %s
                            LIMIT 3
                        """, (document_name,))
                        
                        samples = cursor.fetchall()
                        sample_contents[table_name] = [
                            {"id": sample[0], "content_preview": sample[1]} 
                            for sample in samples
                        ]
                except Exception as e:
                    self.logger.error(f"Error checking table {table_name}: {str(e)}")
            
            cursor.close()
            conn.close()
            
            # Compile results
            results = {
                "document_name": document_name,
                "vector_counts": vector_counts,
                "total_vectors": sum(vector_counts.values()),
                "sample_contents": sample_contents
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in verify_document_vectors: {str(e)}")
            return {"error": str(e)}

    def verify_configuration(self):
        """Verify that the RAG tool is properly configured and working"""
        self.logger.info("Verifying RAG tool configuration")
        
        results = {
            "status": "ok",
            "details": {},
            "errors": []
        }
        
        # 1. Check basic configuration
        results["details"]["openai_model"] = self.openai_model
        results["details"]["db_config"] = {
            "host": self.db_config.get("host"),
            "port": self.db_config.get("port"),
            "database": self.db_config.get("database"),
            "user": self.db_config.get("user"),
            "password": "********" if self.db_config.get("password") else None
        }
        
        # 2. Check if we have any tables
        if not self.table_configs:
            results["status"] = "error"
            results["errors"].append("No tables found in the database")
        else:
            results["details"]["table_count"] = len(self.table_configs)
            results["details"]["tables"] = [t["name"] for t in self.table_configs]
        
        # 3. Check if tables have files
        if self.table_configs:
            files_info = self.get_files_in_database()
            files_by_table = files_info["files_by_table"]
            all_files = files_info["all_files"]
            
            results["details"]["file_count"] = len(all_files)
            
            if not all_files:
                results["status"] = "warning"
                results["errors"].append("No files found in any table")
        
        # 4. Check if vector stores are initialized
        if not self.vector_stores:
            results["status"] = "error"
            results["errors"].append("Vector stores not initialized")
        else:
            results["details"]["vector_stores_count"] = len(self.vector_stores)
            results["details"]["vector_stores"] = list(self.vector_stores.keys())
        
        # 5. Check if indexes are initialized
        if not self.indexes:
            results["status"] = "error"
            results["errors"].append("Indexes not initialized")
        else:
            results["details"]["indexes_count"] = len(self.indexes)
            results["details"]["indexes"] = list(self.indexes.keys())
        
        # 6. Check if query engines are initialized
        if not self.query_engines:
            results["status"] = "error"
            results["errors"].append("Query engines not initialized")
        else:
            results["details"]["query_engines_count"] = len(self.query_engines)
            results["details"]["query_engines"] = list(self.query_engines.keys())
        
        # 7. Check database connection
        try:
            conn = get_pg_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            results["details"]["db_connection"] = "ok"
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(f"Database connection failed: {str(e)}")
            results["details"]["db_connection"] = "failed"
        
        # 8. Check OpenAI API
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=5
            )
            results["details"]["openai_api"] = "ok"
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(f"OpenAI API test failed: {str(e)}")
            results["details"]["openai_api"] = "failed"
        
        return results
    
    def detect_document_request(self, query_text):
        """
        Detect if a query is requesting information about a specific document
        Returns (is_document_request, document_name) tuple
        """
        self.logger.info(f"Checking if query is a document request: '{query_text}'")
        
        # Check if query contains document request indicators
        doc_request_terms = [
            "document", "documento", "file", "arquivo", "pdf", 
            "manual", "guide", "guia", "summary", "summarize", 
            "resumo", "summarise", "content", "conteúdo", 
            "explain", "explique", "details", "detalhes", 
            "information", "informação", "read", "tell me about", 
            "fale sobre", "text", "texto", "resumir",
            "resuma", "resumir o", "resuma o"
        ]
        
        # Log the query for debugging
        self.logger.info(f"Processing document request query: '{query_text}'")
        
        # Check if the query contains any document request terms
        has_doc_term = False
        for term in doc_request_terms:
            if term in query_text.lower():
                has_doc_term = True
                self.logger.info(f"Found document request term: '{term}'")
                break
        
        if not has_doc_term:
            self.logger.info("Query does not contain document request terms")
            return False, None
        
        # Get all available files
        files_info = self.get_files_in_database()
        all_files = files_info["all_files"]
        
        # Debug: Log all available files
        self.logger.info(f"Available files: {all_files}")
        
        if not all_files:
            self.logger.info("No documents available to check against")
            return False, None
            
        # First check for exact filename matches (case insensitive)
        for file in all_files:
            if file.lower() in query_text.lower():
                self.logger.info(f"Found exact document match: {file}")
                return True, file
        
        # Next check for filename without extension
        for file in all_files:
            file_name_without_ext = os.path.splitext(file)[0].lower()
            if file_name_without_ext in query_text.lower():
                self.logger.info(f"Found document match without extension: {file}")
                return True, file
            
            # Alternative check replacing accents
            file_normalized = file_name_without_ext.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
            query_normalized = query_text.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
            
            if file_normalized in query_normalized:
                self.logger.info(f"Found normalized document match: {file}")
                return True, file
        
        # Check for common document patterns in the query
        if "manual" in query_text.lower():
            for file in all_files:
                if "manual" in file.lower():
                    self.logger.info(f"Found 'manual' document match: {file}")
                    return True, file
        
        # Finally check for partial matches in multi-word filenames
        for file in all_files:
            # Split filename (without extension) into words
            file_words = os.path.splitext(file)[0].lower().split()
            
            # Count words from filename found in query (ignoring very short words)
            word_matches = sum(1 for word in file_words if len(word) > 3 and word in query_text.lower())
            
            # If we matched at least 1 substantial word for a document with a descriptive name, consider it a match
            if word_matches >= 1 and len(file_words) >= 2:
                self.logger.info(f"Found partial document match: {file} (matched {word_matches} words)")
                return True, file
        
        self.logger.info("Query seems to be document-related but no specific document was identified")
        
        # If we reached here, it's potentially a document request but we couldn't identify the specific document
        # Return True to indicate it's a document request, but None for the document name
        # This allows the main processing to handle general document requests appropriately
        return True, None
    
    def summarize_document(self, document_name):
        """
        Directly summarize a specific document by name without relying on complex query logic
        """
        self.logger.info(f"Summarizing document: '{document_name}'")
        
        if not document_name:
            return {
                "error": "No document name provided",
                "message": "Please provide a specific document name to summarize"
            }
            
        # Find tables containing this document
        doc_tables = []
        files_info = self.get_files_in_database()
        
        # Log all available files for debugging
        self.logger.info(f"Available files for summarization: {files_info['all_files']}")
        
        # Step 1: Try exact match
        for table_name, files in files_info["files_by_table"].items():
            self.logger.info(f"Checking table {table_name} for document '{document_name}'")
            if document_name in files:
                doc_tables.append(table_name)
                self.logger.info(f"Found document in table {table_name}")
        
        # Step 2: If no exact match, try case-insensitive match
        if not doc_tables:
            self.logger.info("Document not found with exact match, trying case-insensitive")
            for table_name, files in files_info["files_by_table"].items():
                for file in files:
                    if document_name.lower() == file.lower():
                        doc_tables.append(table_name)
                        document_name = file  # Use the actual filename with correct case
                        self.logger.info(f"Found document with case-insensitive match in table {table_name}")
                        break
        
        # Step 3: If still no match, try partial match
        if not doc_tables:
            self.logger.info("Document not found with exact or case-insensitive match, trying partial match")
            for table_name, files in files_info["files_by_table"].items():
                for file in files:
                    if document_name.lower() in file.lower() or file.lower() in document_name.lower():
                        doc_tables.append(table_name)
                        document_name = file  # Use the actual filename
                        self.logger.info(f"Found document with partial match in table {table_name}")
                        break
        
        # If no tables found, return error
        if not doc_tables:
            return {
                "error": f"Document '{document_name}' not found",
                "message": f"The document '{document_name}' was not found in any of the vector stores"
            }
        
        # Create queries to get document content - using multiple query variations
        # to increase chances of getting good results
        queries = [
            f"Provide a comprehensive summary of the document titled '{document_name}'. Include the main topics, key points, and important information.",
            f"What are the main sections and key information in the document '{document_name}'?",
            f"Summarize the content of '{document_name}' in detail, covering all important aspects."
        ]
        
        # Get results from relevant tables with multiple queries
        table_results = []
        all_sources = []
        
        # Prepare metadata filter for file_name
        filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=document_name)])
        
        # Try each table
        for table_name in doc_tables:
            # Find the table config
            table_config = next((config for config in self.table_configs if config["name"] == table_name), None)
            if table_config:
                # Try each query until we get a good result
                for query in queries:
                    self.logger.info(f"Querying table {table_name} with query: {query} and file_name filter")
                    result = self.query_single_table(table_name, query, filters=filters)
                    if "error" not in result:
                        result["description"] = table_config.get("description", "")
                        table_results.append(result)
                        
                        # Collect sources specific to this document
                        for source in result.get("sources", []):
                            # Match sources from this document (using flexible matching)
                            if (source.get("file_name") == document_name or 
                                document_name.lower() in source.get("file_name", "").lower() or
                                source.get("file_name", "").lower() in document_name.lower()):
                                source["table"] = table_name
                                all_sources.append(source)
                        
                        # If we got a good result, no need to try more queries for this table
                        break
        
        # If no results found, try direct database query as a fallback
        if not table_results:
            self.logger.warning(f"No results found through query engine, trying direct database query")
            try:
                conn = get_pg_connection()
                cursor = conn.cursor()
                
                # Get content directly from the database - first try exact match
                for table_name in doc_tables:
                    cursor.execute(f"""
                        SELECT text
                        FROM {table_name}
                        WHERE metadata_->>'file_name' = %s
                        LIMIT 10
                    """, (document_name,))
                    
                    rows = cursor.fetchall()
                    if not rows:
                        # If no exact match, try case-insensitive
                        self.logger.info(f"Trying case-insensitive match in direct query for table {table_name}")
                        cursor.execute(f"""
                            SELECT text
                            FROM {table_name}
                            WHERE LOWER(metadata_->>'file_name') = LOWER(%s)
                            LIMIT 10
                        """, (document_name,))
                        rows = cursor.fetchall()
                    
                    if not rows:
                        # If still no match, try partial match
                        self.logger.info(f"Trying partial match in direct query for table {table_name}")
                        cursor.execute(f"""
                            SELECT text
                            FROM {table_name}
                            WHERE LOWER(metadata_->>'file_name') LIKE %s
                            LIMIT 10
                        """, (f"%{document_name.lower()}%",))
                        rows = cursor.fetchall()
                    
                    if rows:
                        direct_content = "\n\n".join([row[0] for row in rows])
                        table_results.append({
                            "answer": f"Here is content from document '{document_name}':\n\n{direct_content}",
                            "table": table_name,
                            "description": "Direct database content retrieval",
                            "sources": []
                        })
                
                cursor.close()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Error in direct database query: {str(e)}")
        
        # If still no results found
        if not table_results:
            return {
                "error": f"No content found for document '{document_name}'",
                "message": f"The document was found but no content could be retrieved"
            }
        
        # For a single result, just return it directly
        if len(table_results) == 1:
            return {
                "answer": table_results[0]["answer"],
                "sources": all_sources
            }
        
        # For multiple tables/results, synthesize results
        try:
            synthesis_prompt = self._create_doc_summary_prompt(document_name, table_results)
            
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": f"You are an assistant tasked with creating a comprehensive summary of the document: {document_name}"},
                    {"role": "user", "content": synthesis_prompt}
                ]
            )
            
            synthesized_answer = response.choices[0].message.content
            
            return {
                "answer": synthesized_answer,
                "sources": all_sources
            }
        except Exception as e:
            self.logger.error(f"Error synthesizing document summary: {str(e)}")
            
            # Fall back to simple concatenation if synthesis fails
            combined_answer = f"Summary of '{document_name}' from multiple sources:\n\n"
            for result in table_results:
                combined_answer += f"From {result['table']}: {result['answer']}\n\n"
            
            return {
                "answer": combined_answer,
                "sources": all_sources,
                "synthesis_error": str(e)
            }

    def _create_query_engine(self, vector_store, llm):
        """
        Create a query engine for a given vector store
        
        This function follows the pattern in pg_rag_simple.py to configure
        the query engine with the right parameters for effective retrieval.
        """
        self.logger.info("Creating query engine from vector store")
        
        try:
            # Create index from vector store if necessary
            if vector_store not in self.vector_stores.values():
                self.logger.info("Creating index for new vector store")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model
                )
            else:
                # Find the index associated with this vector store
                for table_name, vs in self.vector_stores.items():
                    if vs == vector_store and table_name in self.indexes:
                        index = self.indexes[table_name]
                        self.logger.info(f"Using existing index for table {table_name}")
                        break
                else:
                    self.logger.warning("Could not find existing index, creating new one")
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_vector_store(
                        vector_store,
                        storage_context=storage_context,
                        embed_model=Settings.embed_model
                    )
            
            # Configure query engine with parameters aligned with pg_rag_simple.py
            query_engine = index.as_query_engine(
                llm=llm,
                similarity_top_k=10,  # Increased from 5 to retrieve more chunks
                response_mode="compact",  # How to format the response
                use_hybrid_search=True    # Use hybrid search for better results
            )
            
            self.logger.info("Query engine created successfully")
            return query_engine
        except Exception as e:
            self.logger.error(f"Error creating query engine: {str(e)}")
            raise

def create_multi_rag_tool_spec(available_files=None):
    """Create a specification for the multi-table RAG tool"""
    description = (
        "Search external knowledge bases for specific information that the model doesn't know. "
        "Use this ONLY when the user asks about information likely contained in specific documents or data sources "
        "that are not part of your general knowledge."
    )
    
    # Add information about available files if provided
    if available_files and len(available_files) > 0:
        files_list = ", ".join([f"'{f}'" for f in available_files[:10]]) # Quote filenames
        if len(available_files) > 10:
            files_list += f", and {len(available_files) - 10} more"
        description += (
            f"\n\nThe knowledge bases contain information primarily from the following documents: {files_list}. "
            "Consult these bases if the query refers to these documents or topics likely within them."
        )
    else:
        description += "\nCurrently, the specific documents in the knowledge base are unknown, but use this tool if the query seems to require external data retrieval."

    return {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific question to ask the knowledge base. Formulate this as precisely as possible to get the most relevant information."
                    }
                },
                "required": ["query"]
            }
        }
    }

def process_message_with_multi_rag(user_message, rag_tool, model="gpt-4o"):
    """Process a user message using the multi-table RAG tool"""
    import openai
    import logging
    
    # Configure detailed logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("process_message_rag")
    
    # Ensure the OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set in the environment")
        return "Error: OPENAI_API_KEY is not set in the environment"
    
    # Get list of available files
    files_info = rag_tool.get_files_in_database()
    available_files = files_info["all_files"]
    
    logger.info(f"Processing message with multi-RAG: '{user_message}'")
    
    # --- Start: Ensure direct document handling is commented out ---
    # # Check if this is a direct document request using our document detection
    # is_doc_request, target_document = rag_tool.detect_document_request(user_message)
    
    # # Handle document request where we identified a specific document
    # if is_doc_request and target_document:
    #     logger.info(f"Processing direct document request for: {target_document}")
    #     try:
    #         # Get document summary directly
    #         rag_response = rag_tool.summarize_document(target_document)
            
    #         if "error" in rag_response:
    #             logger.error(f"Error summarizing document: {rag_response['error']}")
    #             # Fall back to general query with the LLM (continue with standard processing below)
    #         else:
    #             # Create prompt for the final response
    #             final_prompt_messages = [
    #                 {"role": "system", "content": f"You are an assistant. The user asked about '{target_document}'. You have retrieved the following summary from the knowledge base."},
    #                 {"role": "user", "content": user_message},
    #                 {"role": "assistant", "content": f"I found information about the document '{target_document}'. Here is a summary:\n\n{rag_response['answer']}\n\nUse this information to answer the user's original question."},
    #             ]
                
    #             # Get the final response
    #             logger.info("Getting final response using direct document summary")
    #             final_response = openai.chat.completions.create(
    #                 model=model,
    #                 messages=final_prompt_messages
    #             )
                
    #             return final_response.choices[0].message.content
    #     except Exception as e:
    #         logger.error(f"Error in direct document handling: {str(e)}")
    #         logger.exception("Exception details:")
    #         # Continue with standard processing if direct document handling fails
    # --- End: Ensure direct document handling is commented out ---

    # Define system prompt
    system_prompt = (
        "You are a helpful assistant. Your primary goal is to answer the user's questions accurately. "
        "You have access to an external knowledge base via the 'query_knowledge_base' tool. "
        "Use this tool ONLY if the user's query asks for specific information that seems likely to be found in the documents mentioned in the tool description "
        "(e.g., manuals, specific reports) or if the query explicitly asks to consult external documents. "
        "Otherwise, answer based on your general knowledge."
    )

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        # First call to decide if RAG is needed
        logger.info("Making initial call to LLM to decide if RAG tool is needed")
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=[create_multi_rag_tool_spec(available_files)],
            tool_choice="auto"  # Let the model decide whether to use the tool
        )
        
        assistant_message = response.choices[0].message
        
        # If the model chose to use the RAG tool
        if assistant_message.tool_calls:
            logger.info("LLM decided to use the multi-RAG tool")
            
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            
            if function_name == "query_knowledge_base":
                # Parse the function arguments
                function_args = json.loads(tool_call.function.arguments)
                query = function_args.get("query")
                
                # Call the RAG tool
                logger.info(f"Calling multi-RAG tool with query: '{query}'")
                rag_response = rag_tool.query(query)
                
                # Add the assistant message and tool response to the conversation
                messages.append(assistant_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(rag_response)
                })
                
                # Get the final response
                logger.info("Getting final response after multi-RAG lookup")
                final_response = openai.chat.completions.create(
                    model=model,
                    messages=messages
                )
                
                return final_response.choices[0].message.content
        
        # If the model didn't need to use the RAG tool
        logger.info("LLM decided NOT to use the multi-RAG tool")
        return assistant_message.content
    
    except Exception as e:
        logger.error(f"Error in process_message_with_multi_rag: {str(e)}")
        logger.exception("Detailed error traceback:")
        return f"An error occurred: {str(e)}" 