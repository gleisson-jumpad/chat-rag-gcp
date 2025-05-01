import sys
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestDocumentSummary")

# Add current directory to path
sys.path.append('.')

# Import RAG tools
from app.multi_table_rag import MultiTableRAGTool
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

def test_document_summary(document_name="llamaindex.pdf"):
    """Test document summary with proper metadata filtering"""
    try:
        logger.info(f"Testing document summary for: {document_name}")
        tool = MultiTableRAGTool()
        
        # Get files in the database
        files_info = tool.get_files_in_database()
        logger.info(f"Available files in database: {files_info['all_files']}")
        
        # Check if our document is in the database
        if document_name not in files_info['all_files']:
            logger.error(f"Document {document_name} not found in the database")
            return f"Document {document_name} not found in the database"
        
        # Find table containing this document
        doc_table = None
        for table, files in files_info['files_by_table'].items():
            if document_name in files:
                doc_table = table
                break
        
        if not doc_table:
            logger.error(f"Could not find table containing document {document_name}")
            return f"Could not find table containing document {document_name}"
        
        logger.info(f"Found document in table: {doc_table}")
        
        # Create query for document summary
        query = f"Please provide a comprehensive summary of the document '{document_name}'. Include all main topics, key points, and important information. Format your response with clear sections and bullet points where appropriate."
        
        # Create proper metadata filter
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="file_name", value=document_name)],
            condition="and"
        )
        
        # Use the query_single_table_with_filters method
        logger.info(f"Querying with filters: {filters}")
        result = tool.query_single_table_with_filters(doc_table, query, filters=filters)
        
        if "error" in result and result["error"]:
            logger.error(f"Error in query: {result['error']}")
            return f"Error: {result['error']}"
        
        return result["answer"]
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    document_name = "llamaindex.pdf"
    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    
    summary = test_document_summary(document_name)
    print("\n" + "="*80)
    print(f"SUMMARY OF {document_name}:")
    print("="*80)
    print(summary) 