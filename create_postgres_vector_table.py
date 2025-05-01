#!/usr/bin/env python3
"""
Create PostgreSQL Vector Table for LlamaIndex
Following the tutorial in 'Postgres Vector Store - LlamaIndex.pdf'
"""
import os
import sys
import logging
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("postgres_setup")

def load_environment():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', '.env')
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    except ImportError:
        logger.warning("dotenv not installed, skipping .env loading")

def get_connection():
    """Get a connection to the PostgreSQL database"""
    pg_host = os.getenv("PG_HOST", "34.150.190.157")
    pg_port = int(os.getenv("PG_PORT", "5432"))
    pg_user = os.getenv("PG_USER", "llamaindex")
    pg_password = os.getenv("PG_PASSWORD", "password123")
    pg_db = os.getenv("PG_DATABASE", "postgres")
    
    logger.info(f"Connecting to PostgreSQL: {pg_host}:{pg_port}/{pg_db}")
    
    # Establish connection
    conn = psycopg2.connect(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        database=pg_db
    )
    
    return conn

def create_table(conn, table_name="vector_tutorial_table"):
    """Create a vector table in PostgreSQL with proper structure"""
    logger.info(f"Creating table: {table_name}")
    
    cursor = conn.cursor()
    
    # First, ensure pgvector extension is installed
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    logger.info("Ensured pgvector extension is installed")
    
    # Drop table if it exists (for clean setup)
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    logger.info(f"Dropped table {table_name} if it existed")
    
    # Create table according to the tutorial format
    create_table_sql = f"""
    CREATE TABLE {table_name} (
        id SERIAL PRIMARY KEY,
        text TEXT,
        embedding VECTOR(1536),
        metadata_ JSONB
    );
    """
    cursor.execute(create_table_sql)
    logger.info(f"Created table {table_name}")
    
    # Create HNSW index for faster vector search (as per tutorial)
    create_index_sql = f"""
    CREATE INDEX ON {table_name} 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);
    """
    cursor.execute(create_index_sql)
    logger.info(f"Created HNSW index on {table_name}")
    
    # Create text search index for hybrid search (as per tutorial)
    create_text_index_sql = f"""
    CREATE INDEX ON {table_name} 
    USING GIN (to_tsvector('english', text));
    """
    cursor.execute(create_text_index_sql)
    logger.info(f"Created text search index on {table_name}")
    
    # Commit changes
    conn.commit()
    cursor.close()
    
    logger.info(f"Table {table_name} created successfully with all required indexes")
    return table_name

def register_vector_adapter():
    """Register adapter for NumPy arrays to PostgreSQL vectors"""
    def adapt_numpy_array(numpy_array):
        return AsIs("'[%s]'::vector" % ','.join(map(str, numpy_array)))
    
    register_adapter(np.ndarray, adapt_numpy_array)
    logger.info("Registered NumPy adapter for PostgreSQL vectors")

def main():
    """Main function"""
    # Process command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Create PostgreSQL vector table for LlamaIndex")
    parser.add_argument("--table", "-t", type=str, default="vector_tutorial_table",
                        help="Name of the table to create")
    args = parser.parse_args()
    
    # Load environment variables
    load_environment()
    
    # Register vector adapter for NumPy arrays
    register_vector_adapter()
    
    # Create table
    try:
        conn = get_connection()
        table_name = create_table(conn, args.table)
        logger.info(f"Successfully created PostgreSQL vector table: {table_name}")
        
        # Show example of usage
        print("\n" + "="*80)
        print("EXAMPLE USAGE WITH LLAMAINDEX:")
        print("="*80)
        print(f"""
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# Set up vector store
vector_store = PGVectorStore.from_params(
    database="postgres",
    host="34.150.190.157",
    password="password123",
    port=5432,
    user="llamaindex",
    table_name="{table_name}",
    embed_dim=1536,
    hybrid_search=True,
    text_search_config="english"
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# Now you can query with proper vector search
query_engine = index.as_query_engine()
response = query_engine.query("Your query here")
        """)
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Closed database connection")

if __name__ == "__main__":
    main() 