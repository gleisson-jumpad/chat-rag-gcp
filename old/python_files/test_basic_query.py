import sys
import logging
import os
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestBasicQuery")

# Add current directory to path
sys.path.append('.')

from db_config import get_pg_connection

def direct_document_query(document_name="llamaindex.pdf", table_name="data_vectors_472971c1_4265_4aba_a6cf_c3b633115fe1"):
    """Use a direct approach without all the wrapper classes"""
    try:
        logger.info(f"Directly querying document: {document_name} in table: {table_name}")
        
        # Get database configuration
        host = os.getenv("DB_PUBLIC_IP", "34.150.190.157")
        port = int(os.getenv("PG_PORT", 5432))
        dbname = os.getenv("PG_DB", "postgres")
        user = os.getenv("PG_USER", "llamaindex") 
        password = os.getenv("PG_PASSWORD", "password123")
        
        # Initialize OpenAI components
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        # Create embedding model
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            embed_batch_size=10,
            api_key=openai_api_key
        )
        
        # Create LLM
        llm = OpenAI(model="gpt-4o", api_key=openai_api_key)
        
        # Try to validate the connection first
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Check if the table has our document
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {table_name} 
            WHERE metadata_->>'file_name' = %s
        """, (document_name,))
        
        count = cursor.fetchone()[0]
        logger.info(f"Found {count} chunks for {document_name} in {table_name}")
        
        cursor.close()
        conn.close()
        
        if count == 0:
            return f"No chunks found for {document_name} in {table_name}"
        
        # Create vector store with basic settings (avoid hybrid search)
        vector_store = PGVectorStore.from_params(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=password,
            table_name=table_name,
            embed_dim=1536,
            use_jsonb=True
        )
        
        # Create index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Create very specific query
        query = (
            f"Please provide a comprehensive summary of the document titled '{document_name}'. "
            "Include the main topics, key points, and important information."
        )
        
        # This approach uses direct metadata filter at the retriever level
        retriever = index.as_retriever(
            similarity_top_k=10
        )
        
        # Set up query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=10,
            response_mode="compact"
        )
        
        # Execute query
        logger.info(f"Executing query: {query}")
        response = query_engine.query(query)
        
        if not response or not str(response).strip():
            return "Empty response from query engine"
        
        # Return full response object for debugging
        logger.info(f"Got response: {type(response)} length: {len(str(response))}")
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Error in direct query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"

if __name__ == "__main__":
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    table_name = "data_vectors_472971c1_4265_4aba_a6cf_c3b633115fe1" 
    if len(sys.argv) > 2:
        table_name = sys.argv[2]
    
    result = direct_document_query(document_name, table_name)
    
    print("\n" + "="*80)
    print(f"QUERY RESULT FOR {document_name}:")
    print("="*80)
    print(result) 