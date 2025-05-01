import os
import sys
import logging
import json
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DirectQuery")

# Set up paths
sys.path.append('.')

# Import LlamaIndex
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import database config
from db_config import get_pg_connection

def direct_llamaindex_summary(document_name="llamaindex.pdf", table_name=None):
    """Directly query the llamaindex.pdf document content and generate a summary"""
    try:
        logger.info(f"Starting direct summary of document: {document_name}")
        
        # First, determine which table contains the document
        if table_name is None:
            conn = get_pg_connection()
            cursor = conn.cursor()
            
            # Find tables that might contain vectors
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND 
                (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
            """)
            
            vector_tables = [table[0] for table in cursor.fetchall()]
            logger.info(f"Found vector tables: {vector_tables}")
            
            # Check each table for the document
            target_table = None
            for table in vector_tables:
                cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM {table} 
                    WHERE metadata_->>'file_name' = %s
                """, (document_name,))
                
                count = cursor.fetchone()[0]
                if count > 0:
                    target_table = table
                    logger.info(f"Found document in table: {target_table}")
                    break
            
            cursor.close()
            conn.close()
            
            if target_table is None:
                logger.error(f"Document {document_name} not found in any vector table")
                return f"ERROR: Document {document_name} not found in any vector table"
        else:
            target_table = table_name
        
        # Get database configuration
        host = os.getenv("DB_PUBLIC_IP", "34.150.190.157")
        port = int(os.getenv("PG_PORT", 5432))
        dbname = os.getenv("PG_DB", "postgres")
        user = os.getenv("PG_USER", "llamaindex")
        password = os.getenv("PG_PASSWORD", "password123")
        
        # Initialize OpenAI API
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        # Set up embedding model
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            embed_batch_size=100,
            api_key=openai_api_key
        )
        
        # Set up vector store with exact table
        vector_store = PGVectorStore.from_params(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=password,
            table_name=target_table,
            embed_dim=1536,
            hybrid_search=True,
            text_search_config="english"
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create LLM
        llm = OpenAI(model="gpt-4o", api_key=openai_api_key)
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Create query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=10,
            vector_store_query_mode="hybrid",
            alpha=0.5,
            response_mode="compact",
            # Use proper MetadataFilters format
            filters=None  # We'll handle filtering in our query
        )
        
        # Query for comprehensive summary
        logger.info(f"Querying for comprehensive summary of {document_name}")
        
        # Explicitly include the document name in the query for better context
        query = f"""
        Please provide a comprehensive summary of the document titled '{document_name}'.
        Include the main topics, key points, and important information.
        Format the summary with clear sections and bullet points where appropriate.
        This document is about implementing LlamaIndex with Direct OpenAI API Integration.
        """
        
        response = query_engine.query(query)
        
        # Log completion
        logger.info(f"Summary generated successfully")
        
        return str(response)
    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    # Get the table name from command line if provided
    table_name = None
    if len(sys.argv) > 2:
        table_name = sys.argv[2]
    
    summary = direct_llamaindex_summary(document_name, table_name)
    
    print("\n" + "="*80)
    print(f"SUMMARY OF {document_name}:")
    print("="*80)
    print(summary) 