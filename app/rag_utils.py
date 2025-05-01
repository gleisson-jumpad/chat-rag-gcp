import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
import tempfile
import logging
import psycopg2
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Try importing with and without 'app.' prefix to handle different execution contexts
try:
    from db_config import get_pg_connection, return_pg_connection, get_pg_cursor, verify_vector_table, ensure_pgvector_extension
except ImportError:
    try:
        from app.db_config import get_pg_connection, return_pg_connection, get_pg_cursor, verify_vector_table, ensure_pgvector_extension
    except ImportError:
        raise ImportError("Could not import db_config module. Make sure it exists in the correct location.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_Utils")

# Extensões suportadas por default
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".pptx", ".md", ".csv"]

def is_supported_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in SUPPORTED_EXTENSIONS

class PostgresRAGManager:
    """
    Class to manage RAG functionality with PostgreSQL and LlamaIndex
    """
    def __init__(self, session_id):
        self.session_id = session_id
        self.table_name = f"vectors_{session_id.replace('-', '_')}"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY não está definida no ambiente.")
        
        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        # Get database connection parameters
        self.host = os.getenv("DB_PUBLIC_IP", "34.150.190.157")
        self.port = int(os.getenv("PG_PORT", 5432))
        self.dbname = os.getenv("PG_DB", "postgres")
        self.user = os.getenv("PG_USER", "llamaindex")
        self.password = os.getenv("PG_PASSWORD", "password123")
        
        # Initialize components with improved settings
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            embed_batch_size=100,  # Increased batch size for better performance
            api_key=self.openai_api_key
        )
        self.vector_store = None
        self.index = None
        
        # Set up LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 200
        
        # Ensure pgvector extension is installed
        self._ensure_pgvector()
        
        logger.info(f"Initialized PostgresRAGManager with session ID: {session_id}")
        
    def _ensure_pgvector(self):
        """Ensure pgvector extension is installed"""
        try:
            if ensure_pgvector_extension():
                logger.info("pgvector extension is installed and ready")
            else:
                logger.warning("Could not ensure pgvector extension is installed - vector operations may fail")
        except Exception as e:
            logger.error(f"Error checking pgvector extension: {str(e)}")
            
    def setup_vector_store(self):
        """Set up the PostgreSQL vector store with advanced configuration"""
        try:
            logger.info(f"Creating PGVectorStore with params: host={self.host}, port={self.port}, db={self.dbname}, user={self.user}, table={self.table_name}")
            
            # First check if table exists and its configuration
            table_info = verify_vector_table(self.table_name)
            
            # Define PostgreSQL connection string
            connection_params = {
                "host": self.host,
                "port": self.port,
                "database": self.dbname,
                "user": self.user,
                "password": self.password,
                "table_name": self.table_name,
                "embed_dim": 1536,     # OpenAI embedding dimension
                "use_jsonb": True      # Use JSONB for better metadata handling
            }
            
            # Add advanced configuration options
            advanced_params = {
                "hybrid_search": True,  # Enable hybrid search capability
                "text_search_config": "english", # Text search configuration
            }
            
            # Check if table already has HNSW index
            hnsw_config = None
            if table_info["exists"] and table_info["has_index"]:
                # Check if any existing indices are HNSW
                has_hnsw = any(idx.get("type") == "hnsw" for idx in table_info.get("indices", []))
                if has_hnsw:
                    logger.info(f"Table {self.table_name} already has HNSW index")
                else:
                    # Set up HNSW configuration
                    hnsw_config = {
                        "hnsw_m": 16,
                        "hnsw_ef_construction": 64,
                        "hnsw_ef_search": 40,
                        "hnsw_dist_method": "vector_cosine_ops"
                    }
                    logger.info(f"Table {self.table_name} has index but not HNSW, will create HNSW index")
            else:
                # Table doesn't exist or has no index, set up HNSW config
                hnsw_config = {
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 64,
                    "hnsw_ef_search": 40,
                    "hnsw_dist_method": "vector_cosine_ops"
                }
                logger.info(f"Setting up HNSW configuration for new table {self.table_name}")
            
            # Create a special connection callback for getting connections
            def get_conn_callback():
                return get_pg_connection()
            
            # First try with connection callback
            try:
                logger.info("Attempting to create PGVectorStore with connection callback")
                
                if hnsw_config:
                    self.vector_store = PGVectorStore.from_params(
                        **connection_params,
                        **advanced_params,
                        hnsw_kwargs=hnsw_config,
                        connection_creator=get_conn_callback
                    )
                else:
                    self.vector_store = PGVectorStore.from_params(
                        **connection_params,
                        **advanced_params,
                        connection_creator=get_conn_callback
                    )
                
                logger.info(f"Successfully created PGVectorStore with connection callback")
            except Exception as callback_error:
                logger.warning(f"Failed to create PGVectorStore with connection callback: {callback_error}")
                logger.info("Falling back to standard connection")
                
                # If callback approach fails, try without it
                if hnsw_config:
                    self.vector_store = PGVectorStore.from_params(
                        **connection_params,
                        **advanced_params,
                        hnsw_kwargs=hnsw_config
                    )
                else:
                    self.vector_store = PGVectorStore.from_params(
                        **connection_params,
                        **advanced_params
                    )
                
                logger.info(f"Successfully created PGVectorStore with standard connection")
                
            return True
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}", exc_info=True)
            raise

    def check_table_schema(self):
        """Check if the vector table has the correct schema"""
        try:
            table_info = verify_vector_table(self.table_name)
            if not table_info["exists"]:
                logger.info(f"Table {self.table_name} does not exist yet, will be created")
                return False
                
            # Check for required columns
            if not table_info["vector_column"]:
                logger.warning(f"Table {self.table_name} exists but has no embedding column")
                return False
                
            if not table_info["metadata_column"]:
                logger.warning(f"Table {self.table_name} exists but has no metadata column")
                return False
                
            # Check for index
            if not table_info["has_index"]:
                logger.warning(f"Table {self.table_name} has no vector index, search will be slow")
            
            return True
        except Exception as e:
            logger.error(f"Error checking table schema: {str(e)}")
            return False

    def create_index_from_documents(self, docs):
        """Create a vector index from documents"""
        try:
            logger.info(f"Creating index from {len(docs)} documents")
            
            # Set up node parser with specific parameters
            node_parser = SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=200,
                paragraph_separator="\n\n",
                include_metadata=True,
                include_prev_next_rel=True  # Add relationships between chunks
            )
            logger.info("Configured SentenceSplitter with 1024 chunk size")
            
            # Set up storage context with the vector store
            if not self.vector_store:
                logger.error("Vector store not initialized before creating index")
                raise ValueError("Vector store is not initialized. Call setup_vector_store() first.")
                
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            logger.info("Storage context created with vector store")
            
            # Add progress tracking
            import threading
            import time
            
            class ProgressTracker(threading.Thread):
                def __init__(self, interval=5):
                    super().__init__()
                    self.interval = interval
                    self.running = True
                    self.daemon = True
                    
                def run(self):
                    start_time = time.time()
                    while self.running:
                        # Check table row count to track progress
                        try:
                            with get_pg_cursor() as cursor:
                                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                                count = cursor.fetchone()[0]
                                elapsed = time.time() - start_time
                                logger.info(f"Indexing progress: {count} vectors created in {elapsed:.2f}s")
                        except:
                            pass
                        time.sleep(self.interval)
                
                def stop(self):
                    self.running = False
            
            # Start a progress tracker if table exists
            progress_tracker = None
            try:
                with get_pg_cursor() as cursor:
                    cursor.execute(f"""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_schema = 'public' AND table_name = '{self.table_name}'
                        );
                    """)
                    if cursor.fetchone()[0]:
                        progress_tracker = ProgressTracker()
                        progress_tracker.start()
            except:
                pass
            
            # Create index with specific settings
            logger.info("Creating VectorStoreIndex from documents...")
            
            try:
                self.index = VectorStoreIndex.from_documents(
                    docs,
                    storage_context=storage_context,
                    transformations=[node_parser],
                    show_progress=True,
                    embed_model=self.embed_model
                )
                
                logger.info(f"Successfully created index in table {self.table_name}")
                
                # Stop progress tracker if it was started
                if progress_tracker:
                    progress_tracker.stop()
                
                # Verify the created table
                table_info = verify_vector_table(self.table_name)
                logger.info(f"Table {self.table_name} created with {table_info.get('row_count', 0)} rows")
                
                return self.index
            except Exception as idx_error:
                logger.error(f"Error creating index: {str(idx_error)}")
                if progress_tracker:
                    progress_tracker.stop()
                raise
                
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}", exc_info=True)
            raise

def process_uploaded_file(file, session_id):
    """Process an uploaded file and create a vector index"""
    logger.info(f"Processing uploaded file: {file.name}")
    
    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded file to the temporary directory
        filepath = os.path.join(temp_dir, file.name)
        try:
            with open(filepath, "wb") as f:
                f.write(file.getvalue())
            
            logger.info(f"File saved to temporary location: {filepath}")
            
            # Handle PDF files specially if it's a PDF
            file_name = os.path.basename(filepath)
            
            # Create Documents from file with metadata
            if filepath.lower().endswith('.pdf'):
                # PDF Processing logic using SimpleDirectoryReader which can handle PDFs
                logger.info(f"Processing PDF file: {file_name}")
                docs = SimpleDirectoryReader(
                    input_files=[filepath],
                    filename_as_id=True
                ).load_data()
                logger.info(f"Loaded PDF with {len(docs)} chunks")
            else:
                # Regular file processing
                docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
                logger.info(f"Loaded {len(docs)} documents from file")
            
            # Add metadata to all documents
            for doc in docs:
                doc.metadata["file_name"] = file_name
                doc.metadata["source"] = filepath
            
            # Initialize RAG manager
            logger.info(f"Initializing PostgresRAGManager with session ID: {session_id}")
            rag_manager = PostgresRAGManager(session_id)
            
            # Check if table already exists with proper schema
            logger.info("Checking if table already exists with proper schema")
            table_exists = rag_manager.check_table_schema()
            if table_exists:
                logger.info(f"Table {rag_manager.table_name} already exists with proper schema")
            
            # Set up vector store
            logger.info("Setting up vector store")
            rag_manager.setup_vector_store()
            
            # Configure LlamaIndex settings
            Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
            
            # Create node parser with smaller chunks for better retrieval
            # Similar to the approach in pg_rag_simple.py
            node_parser = SentenceSplitter(
                chunk_size=512,  # Smaller chunks for more precise retrieval
                chunk_overlap=50,  # Some overlap to maintain context
            )
            
            # Parse documents into nodes
            nodes = node_parser.get_nodes_from_documents(docs)
            
            # Ensure all nodes have the proper metadata for filtering
            for i, node in enumerate(nodes):
                node.metadata["doc_id"] = f"{file_name}_{i}"
                node.metadata["file_name"] = file_name
            
            logger.info(f"Created {len(nodes)} nodes from document")
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=rag_manager.vector_store)
            
            # Create index from nodes
            logger.info("Creating index from documents")
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info(f"Successfully processed file {file.name}")
            return index
            
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {str(e)}", exc_info=True)
            raise

def get_existing_tables():
    """Return a list of existing vector tables in the database"""
    try:
        logger.info("Retrieving existing vector tables from database")
        
        with get_pg_cursor() as cursor:
            # Query to find vector tables with both naming patterns
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND 
                (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
            """)
            
            vector_tables = [table[0] for table in cursor.fetchall()]
        
        logger.info(f"Found {len(vector_tables)} vector tables: {vector_tables}")
        return vector_tables
    except Exception as e:
        logger.error(f"Error retrieving existing tables: {str(e)}", exc_info=True)
        return []

def create_query_engine(model_id, table_name, similarity_top_k=5):
    """Create a query engine for a specific table
    
    This function aligns with pg_rag_simple.py's approach to:
    1. Create a query engine with appropriate configuration
    2. Set up similarity search with proper parameters
    3. Support returning source attribution
    """
    try:
        logger.info(f"Creating query engine for table: {table_name} with model: {model_id}")
        
        # Initialize components needed for query engine
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY não está definida no ambiente.")
        
        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Get database connection parameters
        host = os.getenv("DB_PUBLIC_IP", "34.150.190.157")
        port = int(os.getenv("PG_PORT", 5432))
        dbname = os.getenv("PG_DB", "postgres")
        user = os.getenv("PG_USER", "llamaindex")
        password = os.getenv("PG_PASSWORD", "password123")
        
        # Create LLM
        logger.info(f"Initializing OpenAI LLM with model: {model_id}")
        llm = OpenAI(model=model_id, temperature=0.1)
        
        # Create embedding model
        logger.info("Initializing OpenAI embedding model")
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Check if table exists with proper vector schema
        table_info = verify_vector_table(table_name)
        if not table_info["exists"] or not table_info["vector_column"]:
            logger.error(f"Table {table_name} does not exist or lacks proper vector column")
            raise ValueError(f"Table {table_name} is not properly configured for vector search")
            
        # Create a special connection callback
        def get_conn_callback():
            return get_pg_connection()
        
        # Create vector store with hybrid search
        logger.info(f"Creating PGVectorStore for table: {table_name}")
        
        # Create the vector store with connection callback and hybrid search
        try:
            vector_store = PGVectorStore.from_params(
                host=host,
                port=port,
                database=dbname,
                user=user,
                password=password,
                table_name=table_name,
                embed_dim=1536,
                hybrid_search=True,
                text_search_config="english",
                connection_creator=get_conn_callback
            )
            logger.info(f"Created PGVectorStore with connection callback for {table_name}")
        except Exception as callback_error:
            logger.warning(f"Failed to create PGVectorStore with connection callback: {callback_error}")
            
            # Try without connection callback
            vector_store = PGVectorStore.from_params(
                host=host,
                port=port,
                database=dbname,
                user=user,
                password=password,
                table_name=table_name,
                embed_dim=1536,
                hybrid_search=True,
                text_search_config="english"
            )
            logger.info(f"Created PGVectorStore with standard connection for {table_name}")
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from vector store
        logger.info(f"Creating VectorStoreIndex for {table_name}")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Configure query engine similar to pg_rag_simple.py
        logger.info(f"Creating QueryEngine with similarity_top_k={similarity_top_k}")
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=similarity_top_k,  # Number of chunks to retrieve
            response_mode="compact"  # Compact and useful response format
        )
        
        logger.info(f"Successfully created query engine for {table_name}")
        return query_engine
        
    except Exception as e:
        logger.error(f"Error creating query engine: {str(e)}", exc_info=True)
        raise

def query_document(query_engine, query_text):
    """
    Query the document using vector similarity search
    
    Similar to pg_rag_simple.py, this function:
    1. Converts the query to a vector embedding
    2. Finds similar vectors in the database
    3. Retrieves only the most relevant chunks (not the entire document)
    4. Sends these chunks to the LLM for answer generation
    
    Args:
        query_engine: Query engine to use
        query_text (str): Query text
        
    Returns:
        dict: Response with answer and source information
    """
    logger.info(f"Querying document with: {query_text}")
    
    try:
        # Execute query
        response = query_engine.query(query_text)
        
        # Get source nodes for context and attribution
        sources = []
        if hasattr(response, 'source_nodes'):
            for i, node in enumerate(response.source_nodes):
                source = {
                    "text": node.node.get_content()[:150] + "..." if len(node.node.get_content()) > 150 else node.node.get_content(),
                    "metadata": node.node.metadata,
                    "score": float(node.score) if hasattr(node, "score") else None
                }
                sources.append(source)
        
        return {
            "answer": str(response),
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        return {
            "answer": f"Error querying document: {str(e)}",
            "sources": []
        }
