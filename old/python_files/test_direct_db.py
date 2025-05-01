import sys
import logging
import os
import json
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI as DirectOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestDirectDB")

# Add current directory to path
sys.path.append('.')

from db_config import get_pg_connection

def analyze_document_chunks(document_name="llamaindex.pdf", table_name="data_vectors_472971c1_4265_4aba_a6cf_c3b633115fe1"):
    """Analyze document chunks and try a direct OpenAI approach"""
    try:
        # Get all document chunks
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Get chunks with all metadata
        cursor.execute(f"""
            SELECT id, text, metadata_
            FROM {table_name} 
            WHERE metadata_->>'file_name' = %s
            ORDER BY id
        """, (document_name,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Check if we got results
        if not rows:
            logger.error(f"No chunks found for {document_name} in {table_name}")
            return None
        
        logger.info(f"Found {len(rows)} chunks for {document_name}")
        
        # Analyze each chunk
        chunks = []
        for row in rows:
            chunk_id, text, metadata = row
            chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": metadata
            })
        
        # Sort chunks by page number if available
        try:
            chunks.sort(key=lambda x: int(x['metadata'].get('page_label', '0')))
            logger.info(f"Sorted chunks by page number")
        except Exception as e:
            logger.warning(f"Could not sort by page number: {e}")
        
        # Combine all document content
        document_text = "\n\n".join([chunk["text"] for chunk in chunks])
        logger.info(f"Combined document text length: {len(document_text)}")
        
        # Check if the text seems complete
        if len(document_text) < 50:
            logger.warning(f"Document text seems too short: '{document_text}'")
        
        # Create direct OpenAI query
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        # Try direct OpenAI query with the text
        client = DirectOpenAI(api_key=openai_api_key)
        
        summary_prompt = f"""
        Based on the following document content from '{document_name}', please provide a comprehensive summary.
        Include main topics, key points, and important information.
        
        DOCUMENT CONTENT:
        {document_text}
        """
        
        logger.info("Sending direct query to OpenAI API")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear document summaries."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        
        summary = response.choices[0].message.content
        logger.info(f"Got summary of length: {len(summary)}")
        
        return {
            "chunks": chunks,
            "document_text": document_text,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_test():
    """Run the test and display results"""
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    table_name = "data_vectors_472971c1_4265_4aba_a6cf_c3b633115fe1" 
    if len(sys.argv) > 2:
        table_name = sys.argv[2]
    
    results = analyze_document_chunks(document_name, table_name)
    
    if not results:
        print(f"Error analyzing document {document_name}")
        return
    
    # Print summary info
    print("\n" + "="*80)
    print(f"DOCUMENT ANALYSIS FOR: {document_name}")
    print("="*80)
    
    # Print chunk stats
    print(f"Found {len(results['chunks'])} chunks")
    
    # Print first few characters of document text
    doc_preview = results['document_text'][:500] + "..." if len(results['document_text']) > 500 else results['document_text']
    print(f"\nDocument Text Preview:\n{doc_preview}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"SUMMARY:")
    print("="*80)
    print(results['summary'])

if __name__ == "__main__":
    run_test() 