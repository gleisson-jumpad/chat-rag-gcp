#!/usr/bin/env python3
"""
Direct RAG Query Script Based on pg_rag_simple.py
"""
import os
import argparse
import logging
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from db_config import get_pg_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("direct_query")

def setup_vector_store(table_name):
    """
    Setup PostgreSQL vector store with pgvector extension
    """
    logger.info(f"Setting up PostgreSQL vector store for table: {table_name}")
    
    # PostgreSQL connection parameters 
    pg_host = os.getenv("DB_PUBLIC_IP", "34.150.190.157")
    pg_port = int(os.getenv("PG_PORT", "5432"))
    pg_user = os.getenv("PG_USER", "llamaindex")
    pg_password = os.getenv("PG_PASSWORD", "password123")
    pg_db = os.getenv("PG_DATABASE", "postgres")
    
    # Create PGVectorStore with hybrid search capability
    vector_store = PGVectorStore.from_params(
        database=pg_db,
        host=pg_host,
        password=pg_password,
        port=pg_port,
        user=pg_user,
        table_name=table_name,
        embed_dim=1536,  # OpenAI embedding dimension
        hybrid_search=True,  # Enable hybrid search for better results
        text_search_config="english"
    )
    
    logger.info(f"PostgreSQL vector store initialized with table: {table_name}")
    return vector_store

def get_index_from_vector_store(vector_store):
    """
    Load an index from an existing vector store
    """
    logger.info(f"Loading index from vector store: {vector_store.table_name}")
    
    # Configure LlamaIndex settings
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    llm = OpenAI(model="gpt-4o", temperature=0.1)
    
    # Create index from existing vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    logger.info("Index loaded successfully")
    return index

def query_document(index, query_text):
    """
    Query the document using vector similarity search
    """
    logger.info(f"Querying document with: {query_text}")
    
    # Create query engine with similarity search
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # Number of chunks to retrieve
        response_mode="compact"  # How to format the response
    )
    
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
    
    result = {
        "answer": str(response),
        "sources": sources
    }
    
    # Format the output similar to pg_rag_simple.py
    output = result["answer"]
    
    if result["sources"]:
        output += "\n\nSources:"
        for i, source in enumerate(result["sources"], 1):
            output += f"\n\n{i}. Score: {source['score']:.3f}"
            output += f"\n   Document: {source['metadata'].get('file_name', 'Unknown')}"
            output += f"\n   Text: {source['text']}"
    
    return output

def find_contract_table():
    """Find the table containing the contract document"""
    conn = get_pg_connection()
    cursor = conn.cursor()
    
    # Find all vector tables
    cursor.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' AND 
        (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
    """)
    
    tables = [row[0] for row in cursor.fetchall()]
    contract_table = None
    
    # Check each table for the contract document
    for table in tables:
        cursor.execute(f"""
            SELECT 1 FROM {table}
            WHERE metadata_->>'file_name' LIKE '%Coentro%' OR metadata_->>'file_name' LIKE '%Jumpad%'
            LIMIT 1
        """)
        
        if cursor.fetchone():
            contract_table = table
            break
    
    cursor.close()
    conn.close()
    
    return contract_table

def main():
    parser = argparse.ArgumentParser(description="Direct RAG Query")
    parser.add_argument("--query", type=str, help="Query to run against the indexed document")
    
    args = parser.parse_args()
    
    if not args.query:
        args.query = "quem assinou o contrato entre Coentro e Jumpad?"
    
    # Find the table containing the contract
    table_name = find_contract_table()
    
    if not table_name:
        print("Could not find a table containing the contract document.")
        return
    
    print(f"Found contract in table: {table_name}")
    
    # Setup vector store
    vector_store = setup_vector_store(table_name)
    
    # Get index from vector store
    index = get_index_from_vector_store(vector_store)
    
    # Query document
    result = query_document(index, args.query)
    
    print("\n" + "="*80)
    print(f"QUERY: {args.query}")
    print("="*80 + "\n")
    print(result)

if __name__ == "__main__":
    main() 