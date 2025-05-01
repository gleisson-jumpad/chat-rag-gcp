import os
import sys
import logging
import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GPTSummary")

# Set up paths
sys.path.append('.')

# Import db config and extract document functions
from db_config import get_pg_connection

def extract_document_content(document_name="llamaindex.pdf"):
    """Directly extract document content from the PostgreSQL database"""
    try:
        logger.info(f"Extracting content for document: {document_name}")
        
        # Create database connection
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # First find tables that might contain vectors
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND 
            (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
        """)
        
        vector_tables = [table[0] for table in cursor.fetchall()]
        logger.info(f"Found vector tables: {vector_tables}")
        
        # Search for the document in each table
        document_chunks = []
        
        for table in vector_tables:
            # Check if the table has the expected structure
            try:
                # Check if table contains document
                cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM {table} 
                    WHERE metadata_->>'file_name' = %s
                """, (document_name,))
                
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.info(f"Found {count} chunks for {document_name} in table {table}")
                    
                    # Extract the actual text and metadata
                    cursor.execute(f"""
                        SELECT id, text, metadata_
                        FROM {table}
                        WHERE metadata_->>'file_name' = %s
                        ORDER BY id
                    """, (document_name,))
                    
                    rows = cursor.fetchall()
                    for row in rows:
                        document_chunks.append({
                            "id": row[0],
                            "text": row[1],
                            "metadata": row[2]
                        })
            except Exception as e:
                logger.error(f"Error checking table {table}: {e}")
                continue
        
        cursor.close()
        conn.close()
        
        if not document_chunks:
            logger.error(f"No content found for document {document_name}")
            return None
        
        # Return raw text content from document
        logger.info(f"Found {len(document_chunks)} chunks for document {document_name}")
        
        # Sort chunks by page number if available
        try:
            document_chunks.sort(key=lambda x: int(x['metadata'].get('page_label', '0')))
        except Exception as e:
            logger.warning(f"Could not sort by page number: {e}")
        
        # Extract just the text
        content = "\n\n".join([chunk["text"] for chunk in document_chunks])
        return content
        
    except Exception as e:
        logger.error(f"Error extracting document content: {e}")
        return None

def summarize_with_gpt(document_name="llamaindex.pdf"):
    """Generate a summary of the document using GPT-4o directly"""
    try:
        # Extract document content
        content = extract_document_content(document_name)
        if not content:
            return f"Error: Could not extract content for document {document_name}"
        
        logger.info(f"Extracted content with {len(content)} characters")
        
        # Check OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Generate summary with OpenAI
        logger.info("Generating summary with GPT-4o...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that provides comprehensive document summaries. Your summaries should be well-structured, highlighting main topics, key points, and important details from the document."},
                {"role": "user", "content": f"""Please summarize the following document titled '{document_name}'. 
                Format the summary with clear sections and bullet points where appropriate.
                
                DOCUMENT CONTENT:
                {content}"""}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        summary = response.choices[0].message.content
        logger.info("Summary generated successfully")
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    summary = summarize_with_gpt(document_name)
    
    print("\n" + "="*80)
    print(f"SUMMARY OF {document_name}:")
    print("="*80)
    print(summary) 