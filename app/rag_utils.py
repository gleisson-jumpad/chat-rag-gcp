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
from db_config import get_pg_connection

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
        
        # Initialize components
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        self.vector_store = None
        self.index = None
        
    def setup_vector_store(self):
        """Set up the PostgreSQL vector store with advanced configuration"""
        try:
            logging.info(f"Creating PGVectorStore with params: host={self.host}, port={self.port}, db={self.dbname}, user={self.user}")
            
            # Create vector store with basic configuration (for testing schema creation)
            self.vector_store = PGVectorStore.from_params(
                host=self.host,
                port=self.port,
                database=self.dbname,
                user=self.user,
                password=self.password,
                table_name=self.table_name,
                embed_dim=1536, # OpenAI embedding dimension
                # Removed use_jsonb, hybrid_search, text_search_config, hnsw_kwargs for testing
            )
            
            logging.info(f"Successfully created PGVectorStore for table {self.table_name}")
            return True
        except Exception as e:
            logging.error(f"Error setting up vector store: {str(e)}")
            raise

    def create_index_from_documents(self, docs):
        """Create a vector index from documents"""
        try:
            # Set up node parser with specific parameters
            node_parser = SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=200,
                paragraph_separator="\n\n",
                include_metadata=True
            )
            
            # Set up storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Create index with specific settings
            self.index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=[node_parser],
                show_progress=True
            )
            
            return self.index
        except Exception as e:
            logging.error(f"Error creating index: {str(e)}")
            raise

def process_uploaded_file(file, session_id):
    """Process an uploaded file and create a vector index"""
    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded file to the temporary directory
        filepath = os.path.join(temp_dir, file.name)
        with open(filepath, "wb") as f:
            f.write(file.getvalue())
        
        # Load documents using LlamaIndex
        docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
        
        # Initialize RAG manager
        rag_manager = PostgresRAGManager(session_id)
        
        # Set up vector store
        rag_manager.setup_vector_store()
        
        # Create index from documents
        index = rag_manager.create_index_from_documents(docs)
        
        return index

def get_existing_tables():
    """Return a list of existing vector tables in the database"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Query to find vector tables with both naming patterns
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND 
            (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
        """)
        
        vector_tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        logging.info(f"Found {len(vector_tables)} vector tables: {vector_tables}")
        return vector_tables
    except Exception as e:
        logging.error(f"Error retrieving existing tables: {str(e)}")
        return []

def create_query_engine(model_id, table_name, similarity_top_k=3):
    """Create a query engine for a specific table"""
    try:
        # Initialize components needed for query engine
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY não está definida no ambiente.")
        
        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Get database connection parameters
        host = os.getenv("DB_PUBLIC_IP", "34.150.190.157")
        port = int(os.getenv("PG_PORT", 5432))
        dbname = os.getenv("PG_DB", "postgres")
        user = os.getenv("PG_USER", "llamaindex")
        password = os.getenv("PG_PASSWORD", "password123")
        
        logging.info(f"Creating query engine for table: {table_name} with model: {model_id}")
        
        # Create LLM
        llm = OpenAI(model=model_id)
        
        # Create embedding model
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Check if table exists
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = '{table_name}'
            );
        """)
        table_exists = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        if not table_exists:
            logging.error(f"Table {table_name} does not exist in the database!")
            raise ValueError(f"Tabela {table_name} não existe no banco de dados.")
        
        # Create vector store
        vector_store = PGVectorStore.from_params(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=password,
            table_name=table_name,
            embed_dim=1536,
            hybrid_search=True
        )
        
        logging.info(f"Created vector store for table: {table_name}")
        
        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
        
        logging.info(f"Created index from vector store")
        
        # Create query engine with specific parameters
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=similarity_top_k,
            streaming=True,
            response_mode="compact"
        )
        
        logging.info(f"Successfully created query engine")
        return query_engine
    except Exception as e:
        logging.error(f"Error creating query engine: {str(e)}")
        logging.exception("Detailed error information:")
        raise
