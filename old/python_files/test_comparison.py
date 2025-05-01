import sys
import logging
import os
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestComparison")

# Add current directory to path
sys.path.append('.')

# Import RAG tools
from app.multi_table_rag import MultiTableRAGTool
from app.postgres_rag_tool import process_message_with_selective_rag
from openai import OpenAI
from db_config import get_pg_connection

def direct_db_summary(document_name="llamaindex.pdf", table_name=None):
    """Get document summary directly from database chunks"""
    start_time = time.time()
    
    try:
        # Get connection
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Find the right table if not provided
        if table_name is None:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name LIKE 'data_vectors_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                logger.error("No vector tables found")
                return None
                
            # Default to first table
            table_name = tables[0]
        
        # Get chunks for document
        cursor.execute(f"""
            SELECT text, metadata_->>'page_label' as page
            FROM {table_name} 
            WHERE metadata_->>'file_name' = %s
            ORDER BY id
        """, (document_name,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            logger.error(f"No chunks found for {document_name}")
            return None
            
        # Process chunks by page
        chunks_by_page = {}
        for text, page in rows:
            if page not in chunks_by_page:
                chunks_by_page[page] = []
            chunks_by_page[page].append(text)
        
        # Sort pages and combine text
        sorted_pages = sorted(chunks_by_page.keys(), key=lambda p: int(p) if p and p.isdigit() else 0)
        document_chunks = []
        
        for page in sorted_pages:
            page_text = "\n\n".join(chunks_by_page[page])
            document_chunks.append(f"[Page {page}] {page_text}")
        
        document_text = "\n\n".join(document_chunks)
        logger.info(f"Retrieved document text length: {len(document_text)}")
        
        # Summarize with OpenAI
        client = OpenAI()
        
        logger.info("Generating summary with direct database approach")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear document summaries."},
                {"role": "user", "content": f"Summarize this document: {document_name}\n\n{document_text}"}
            ]
        )
        
        summary = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        
        return {
            "method": "direct_db",
            "document": document_name,
            "summary": summary,
            "time_seconds": elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Error in direct DB summary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def vector_rag_summary(document_name="llamaindex.pdf"):
    """Get document summary using the process_message_with_selective_rag function"""
    start_time = time.time()
    
    try:
        # Initialize RAG tool
        tool = MultiTableRAGTool()
        
        # Verify document exists
        files_info = tool.get_files_in_database()
        if document_name not in files_info['all_files']:
            # Try case-insensitive match
            matches = [f for f in files_info['all_files'] if f.lower() == document_name.lower()]
            if matches:
                document_name = matches[0]
            else:
                logger.error(f"Document {document_name} not found")
                return None
        
        # Process document summary request
        query = f"resuma o arquivo {document_name}"
        logger.info(f"Requesting summary with query: '{query}'")
        
        # Process with our RAG approach
        summary = process_message_with_selective_rag(query, tool, model="gpt-4o")
        elapsed_time = time.time() - start_time
        
        return {
            "method": "vector_rag",
            "document": document_name,
            "summary": summary,
            "time_seconds": elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Error in vector RAG summary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_comparison(document_name="llamaindex.pdf"):
    """Run both methods and compare results"""
    logger.info(f"Running comparison for document: {document_name}")
    
    # Direct DB approach
    logger.info("Running direct DB approach")
    direct_result = direct_db_summary(document_name)
    
    # Vector RAG approach
    logger.info("Running vector RAG approach")
    vector_result = vector_rag_summary(document_name)
    
    # Display results
    print("\n" + "="*80)
    print(f"COMPARISON FOR: '{document_name}'")
    print("="*80)
    
    if direct_result:
        print(f"\nDIRECT DB APPROACH (Time: {direct_result['time_seconds']:.2f} seconds):")
        print("-"*40)
        print(direct_result["summary"])
    else:
        print("\nDIRECT DB APPROACH: Failed")
    
    print("\n" + "="*80)
    
    if vector_result:
        print(f"\nVECTOR RAG APPROACH (Time: {vector_result['time_seconds']:.2f} seconds):")
        print("-"*40)
        print(vector_result["summary"])
    else:
        print("\nVECTOR RAG APPROACH: Failed")
    
    # Return both results
    return {
        "direct_db": direct_result,
        "vector_rag": vector_result
    }

if __name__ == "__main__":
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    results = run_comparison(document_name) 