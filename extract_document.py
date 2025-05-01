import sys
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExtractDocument")

# Add current directory to path
sys.path.append('.')

# Import database config
from db_config import get_pg_connection

def extract_document_content(document_name="llamaindex.pdf"):
    """Directly extract document content from the PostgreSQL database"""
    try:
        logger.info(f"Attempting to extract content for document: {document_name}")
        
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
                else:
                    # Try case-insensitive search
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM {table} 
                        WHERE LOWER(metadata_->>'file_name') = LOWER(%s)
                    """, (document_name,))
                    
                    count = cursor.fetchone()[0]
                    if count > 0:
                        logger.info(f"Found {count} chunks with case-insensitive match for {document_name} in table {table}")
                        
                        # Extract the actual text and metadata
                        cursor.execute(f"""
                            SELECT id, text, metadata_
                            FROM {table}
                            WHERE LOWER(metadata_->>'file_name') = LOWER(%s)
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
            return f"No content found for document {document_name} in any vector table"
        
        # Format the extracted content
        logger.info(f"Found {len(document_chunks)} chunks for document {document_name}")
        formatted_content = f"Content from document '{document_name}':\n\n"
        
        for i, chunk in enumerate(document_chunks, 1):
            formatted_content += f"Chunk {i}:\n"
            formatted_content += f"ID: {chunk['id']}\n"
            formatted_content += f"Text: {chunk['text'][:300]}...\n"  # Show first 300 chars only
            formatted_content += f"Metadata: {chunk['metadata']}\n\n"
        
        return formatted_content
        
    except Exception as e:
        logger.error(f"Error extracting document content: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    content = extract_document_content(document_name)
    print("\n" + "="*80)
    print(f"CONTENT FOR {document_name}:")
    print("="*80)
    print(content) 