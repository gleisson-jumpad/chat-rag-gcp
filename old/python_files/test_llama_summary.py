import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test")

# Add current directory to path
sys.path.append('.')

# Import RAG tools
from app.multi_table_rag import MultiTableRAGTool
from app.postgres_rag_tool import process_message_with_selective_rag

def summarize_document(document_name="llamaindex.pdf"):
    """Try to summarize a specific document"""
    try:
        logger.info(f"Attempting to summarize document: {document_name}")
        tool = MultiTableRAGTool()
        
        # Get files in the database
        files_info = tool.get_files_in_database()
        logger.info(f"Available files in database: {files_info['all_files']}")
        
        # Check if our document is in the database
        if document_name not in files_info['all_files']:
            # Try case-insensitive matching
            matches = [f for f in files_info['all_files'] if f.lower() == document_name.lower()]
            if matches:
                document_name = matches[0]  # Use exact filename with correct capitalization
                logger.info(f"Found document with case-insensitive match: {document_name}")
            else:
                # Try partial matching
                matches = [f for f in files_info['all_files'] if document_name.lower() in f.lower()]
                if matches:
                    document_name = matches[0]
                    logger.info(f"Found document with partial match: {document_name}")
                else:
                    logger.error(f"Document {document_name} not found in the database")
                    return f"Document {document_name} not found in the database"
        
        # First approach: use process_message_with_selective_rag
        logger.info(f"Using process_message_with_selective_rag for {document_name}")
        query = f"Please summarize the document '{document_name}' in detail. What are the main topics, key points and important information covered in this document?"
        
        result = process_message_with_selective_rag(query, tool, model="gpt-4o")
        if result and len(result.strip()) > 0:
            return result
        
        # Second approach: Use directly the summarize_document method if available
        if hasattr(tool, 'summarize_document'):
            logger.info(f"Using summarize_document method for {document_name}")
            result = tool.summarize_document(document_name)
            answer = result.get("answer", "")
            if answer and len(answer.strip()) > 0:
                return answer
        
        # Third approach: Fall back to query method with specific table
        logger.info(f"Falling back to query method for {document_name}")
        query = f"Please provide a comprehensive summary of the document {document_name}."
        
        # Try to query relevant tables for this document
        relevant_tables = []
        for table, files in files_info['files_by_table'].items():
            if document_name in files:
                relevant_tables.append(table)
        
        if relevant_tables:
            table = relevant_tables[0]
            logger.info(f"Querying table {table} for document {document_name}")
            result = tool.query_single_table(table, query)
            return result.get("answer", "No answer found")
        else:
            logger.info("No specific table found, using general query")
            result = tool.query(query)
            return result.get("answer", "No answer found")
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    summary = summarize_document(document_name)
    print("\n" + "="*80)
    print(f"SUMMARY OF {document_name}:")
    print("="*80)
    print(summary) 