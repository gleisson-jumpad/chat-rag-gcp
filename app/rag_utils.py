import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.storage_context import StorageContext
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

def process_uploaded_file(file, session_id):
    # Ensure OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não está definida no ambiente.")
    
    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded file to the temporary directory
        filepath = os.path.join(temp_dir, file.name)
        with open(filepath, "wb") as f:
            f.write(file.getvalue())
        
        # Load documents using LlamaIndex
        docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
        
        # Get database connection parameters from environment variables
        host = os.getenv("DB_PUBLIC_IP", "34.150.190.157")
        port = int(os.getenv("PG_PORT", 5432))
        dbname = os.getenv("PG_DB", "postgres")
        user = os.getenv("PG_USER", "llamaindex") 
        password = os.getenv("PG_PASSWORD", "password123")
        
        # Create table name with session ID to prevent collisions
        table_name = f"vectors_{session_id.replace('-', '_')}"
        
        try:
            # Log connection details (without password)
            logging.info(f"Creating PGVectorStore with params: host={host}, port={port}, db={dbname}, user={user}")
            
            # Simplest approach: Create PGVectorStore with basic parameters
            vector_store = PGVectorStore.from_params(
                host=host,
                port=port,
                database=dbname,
                user=user,
                password=password,
                table_name=table_name,
                embed_dim=1536,  # OpenAI dimension
                use_jsonb=False  # Use simpler storage format
            )
            
            logging.info(f"Successfully created PGVectorStore for table {table_name}")
            
        except Exception as e:
            logging.error(f"ERROR setting up database: {str(e)}")
            raise
        
        # Set the OpenAI API key as environment variable
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Create embedding model with explicit model name
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Create storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from documents
        index = VectorStoreIndex.from_documents(
            docs, 
            storage_context=storage_context,
            embed_model=embed_model
        )
        
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
