#!/usr/bin/env python3
"""
Direct DB Query to Extract Streaming Information
This script directly extracts information about streaming from the vector database
"""
import os
import logging
import psycopg2
import json

from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streaming_query")

def get_document_content(document_name="OpenAI - LlamaIndex.pdf"):
    """Get document content directly from database"""
    logger.info(f"Getting content for document: {document_name}")
    
    # Connect to the database
    pg_host = os.getenv("PG_HOST", "34.150.190.157")
    pg_port = int(os.getenv("PG_PORT", "5432"))
    pg_user = os.getenv("PG_USER", "llamaindex")
    pg_password = os.getenv("PG_PASSWORD", "password123")
    pg_db = os.getenv("PG_DATABASE", "postgres")
    
    conn = psycopg2.connect(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        database=pg_db
    )
    
    cursor = conn.cursor()
    table_name = "data_llamaindex_vectors"
    
    # Get all chunks for the document
    cursor.execute(f"""
        SELECT text FROM {table_name} 
        WHERE metadata_::json->>'file_name' = %s
        ORDER BY id
    """, (document_name,))
    
    chunks = cursor.fetchall()
    logger.info(f"Found {len(chunks)} chunks for document {document_name}")
    
    # Combine all chunks
    all_text = "\n\n".join([chunk[0] for chunk in chunks])
    logger.info(f"Combined document text length: {len(all_text)}")
    
    cursor.close()
    conn.close()
    
    return all_text

def query_streaming_info(document_content):
    """Query OpenAI about streaming in LlamaIndex"""
    logger.info("Querying OpenAI about streaming functionality")
    
    client = OpenAI()
    
    prompt = """
    Based on the document content provided, explain in detail how to use streaming with 
    OpenAI in LlamaIndex. Include:
    
    1. How to set up streaming
    2. The different streaming methods available
    3. Example code showing how to use streaming
    4. Any important considerations or limitations
    
    Document content:
    
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that provides accurate information about using LlamaIndex with OpenAI."},
            {"role": "user", "content": f"{prompt}\n\n{document_content[:30000]}"}
        ]
    )
    
    streaming_info = response.choices[0].message.content
    logger.info(f"Generated response of length {len(streaming_info)}")
    
    return streaming_info

def main():
    """Main function"""
    logger.info("Starting streaming query script")
    
    try:
        # Get document content
        document_content = get_document_content()
        
        # Query about streaming
        streaming_info = query_streaming_info(document_content)
        
        # Print the results
        print("\n" + "="*80)
        print("STREAMING WITH OPENAI IN LLAMAINDEX")
        print("="*80 + "\n")
        print(streaming_info)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 