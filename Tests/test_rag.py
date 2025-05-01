#!/usr/bin/env python3
"""
Test and Diagnostic Script for MultiTableRAG Implementation
----------------------------------------------------------

This script tests and verifies the RAG implementation's ability to retrieve 
information from vector-stored documents in PostgreSQL. It's designed to:

1. Test database connectivity
2. Discover and list available vector tables
3. Identify documents stored in the vector database
4. Execute test queries to validate information retrieval
5. Format and display results with source attribution

Usage:
    python test_rag.py

Output:
    Detailed logging of the RAG process and formatted query results
"""
import os
import logging
import sys
from app.multi_table_rag import MultiTableRAGTool

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_rag_results.log')  # Also save to file
    ]
)
logger = logging.getLogger("test_rag")

def test_db_connection():
    """Verify database connection and configuration"""
    logger.info("Verifying database connection and configuration...")
    
    # Log environment variables (without exposing sensitive values)
    db_host = os.getenv("DB_PUBLIC_IP", "Not set")
    db_port = os.getenv("PG_PORT", "Not set")
    db_name = os.getenv("PG_DB", "Not set")
    db_user = os.getenv("PG_USER", "Not set")
    
    # Log connection parameters (with masked password)
    logger.info(f"Database host: {db_host}")
    logger.info(f"Database port: {db_port}")
    logger.info(f"Database name: {db_name}")
    logger.info(f"Database user: {db_user}")
    logger.info(f"Password: {'*****' if os.getenv('PG_PASSWORD') else 'Not set'}")
    
    # Initialize the RAG tool (which tests connection)
    try:
        rag_tool = MultiTableRAGTool()
        
        # Run specific connection test
        result = rag_tool.check_postgres_connection()
        
        if result.get("postgres_connection") == True:
            logger.info(f"✅ Database connection successful!")
            logger.info(f"PostgreSQL version: {result.get('postgres_version', 'Unknown')}")
            logger.info(f"pgvector extension: {'Installed' if result.get('pgvector_installed') else 'Missing'}")
            if result.get("pgvector_installed"):
                logger.info(f"pgvector version: {result.get('pgvector_version', 'Unknown')}")
            logger.info(f"Vector tables found: {result.get('vector_table_count', 0)}")
            return rag_tool
        else:
            logger.error(f"❌ Database connection failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error establishing database connection: {str(e)}")
        return None

def test_queries():
    """Test different queries to see if they return proper results"""
    # Initialize the RAG tool with connection test
    rag_tool = test_db_connection()
    
    if not rag_tool:
        logger.error("Cannot proceed with testing due to database connection failure")
        return
    
    # Get available tables and log them
    tables_info = rag_tool.get_tables_info()
    logger.info(f"Found {len(tables_info['tables'])} vector tables")
    for i, table in enumerate(tables_info['tables'], 1):
        logger.info(f"  Table {i}: {table['name']}")
        logger.info(f"    Description: {table.get('description', 'No description')}")
        logger.info(f"    Documents: {table.get('doc_count', 'Unknown')}")
        logger.info(f"    Chunks: {table.get('chunk_count', 'Unknown')}")
        logger.info(f"    HNSW Index: {'Yes' if table.get('has_hnsw') else 'No'}")
    
    # Get available files
    files_info = rag_tool.get_files_in_database()
    
    if not files_info['all_files']:
        logger.error("❌ No documents found in the database. Cannot proceed with testing.")
        return
        
    logger.info(f"Found {len(files_info['all_files'])} document(s) in the database")
    logger.info(f"Documents: {', '.join(files_info['all_files'])}")
    
    # Run test queries
    test_questions = [
        "quem assinou o contrato entre Coentro e Jumpad?",
        "what are the payment terms in the contract between Coentro and Jumpad?",
        "what was the signing date of the contract?",
        "explain the general terms of the contract"
    ]
    
    # Execute each test query
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n\nRUNNING TEST QUERY {i}/{len(test_questions)}: '{question}'")
        
        # Query the RAG system
        result = rag_tool.query(question)
        
        # Print the result
        if "error" in result:
            logger.error(f"❌ Query failed: {result['error']}")
            logger.error(f"Message: {result.get('message', 'No message provided')}")
        else:
            logger.info(f"✅ Query successful!")
            
            # Format and display the output
            print("\n" + "="*80)
            print(f"QUERY #{i}: {question}")
            print("="*80 + "\n")
            print(result["answer"])
            
            # Display source information
            sources = result.get('sources', [])
            best_table = result.get('best_table', 'Unknown')
            
            print(f"\nRetrieved from: {best_table}")
            print(f"Sources: {len(sources)}")
            
            # Log additional diagnostic information
            logger.info(f"Retrieved from table: {best_table}")
            logger.info(f"Sources used: {len(sources)}")
            
            # Log source documents
            if sources:
                source_docs = set()
                for source in sources:
                    if 'metadata' in source and 'file_name' in source['metadata']:
                        source_docs.add(source['metadata']['file_name'])
                        
                logger.info(f"Source documents: {', '.join(source_docs)}")

def main():
    """Main function to run all tests"""
    logger.info("="*50)
    logger.info("STARTING RAG SYSTEM DIAGNOSTIC TESTS")
    logger.info("="*50)
    
    # Run the query tests
    test_queries()
    
    logger.info("="*50)
    logger.info("RAG SYSTEM TESTS COMPLETED")
    logger.info("="*50)

if __name__ == "__main__":
    main() 