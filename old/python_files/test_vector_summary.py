import sys
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestVectorSummary")

# Add current directory to path
sys.path.append('.')

# Import RAG tools
from app.multi_table_rag import MultiTableRAGTool
from app.postgres_rag_tool import process_message_with_selective_rag

def test_vector_summary(document_name="llamaindex.pdf"):
    """Test document summary using vector search with metadata filters"""
    try:
        logger.info(f"Testing vector-based summary for document: {document_name}")
        
        # Initialize the RAG tool
        tool = MultiTableRAGTool()
        
        # Get available files
        files_info = tool.get_files_in_database()
        logger.info(f"Available files: {files_info['all_files']}")
        
        if document_name not in files_info['all_files']:
            # Try case-insensitive matching
            matches = [f for f in files_info['all_files'] if f.lower() == document_name.lower()]
            if matches:
                document_name = matches[0]  # Use exact filename with correct case
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
        
        # Find which table contains this document
        doc_table = None
        for table, files in files_info["files_by_table"].items():
            if document_name in files:
                doc_table = table
                break
        
        # Log available indexes
        if hasattr(tool, 'indexes'):
            logger.info(f"Available indexes: {list(tool.indexes.keys())}")
        
        # Get the index for this table
        index = None
        if doc_table and hasattr(tool, 'indexes') and doc_table in tool.indexes:
            index = tool.indexes[doc_table]
            logger.info(f"Found index for table {doc_table}")
        else:
            logger.warning(f"No index found for document {document_name} in table {doc_table}")
        
        # First approach: Using process_message_with_selective_rag
        logger.info("Testing with process_message_with_selective_rag")
        summary_query = f"resuma o arquivo {document_name}"
        response = process_message_with_selective_rag(summary_query, tool, model="gpt-4o")
        
        return {
            "document_name": document_name,
            "table": doc_table,
            "has_index": index is not None,
            "summary": response
        }
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"

if __name__ == "__main__":
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    result = test_vector_summary(document_name)
    
    if isinstance(result, dict) and "summary" in result:
        print("\n" + "="*80)
        print(f"SUMMARY FOR: '{result['document_name']}' (Table: {result['table']}, Has Index: {result['has_index']})")
        print("="*80)
        print(result["summary"])
    else:
        print(result) 