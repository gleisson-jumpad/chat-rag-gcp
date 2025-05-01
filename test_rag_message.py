import sys
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestRAGMessage")

# Add current directory to path
sys.path.append('.')

# Import RAG tools
from app.multi_table_rag import MultiTableRAGTool
from app.postgres_rag_tool import process_message_with_selective_rag

def test_process_message(message="resuma o arquivo llamaindex.pdf"):
    """Test process_message_with_selective_rag with a document summary request"""
    try:
        logger.info(f"Testing process_message_with_selective_rag with: '{message}'")
        
        # Initialize the RAG tool
        tool = MultiTableRAGTool()
        
        # Get available files
        files_info = tool.get_files_in_database()
        logger.info(f"Available files: {files_info['all_files']}")
        
        # Process the message
        response = process_message_with_selective_rag(message, tool, model="gpt-4o")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"

if __name__ == "__main__":
    message = "resuma o arquivo llamaindex.pdf"
    if len(sys.argv) > 1:
        message = sys.argv[1]
    
    response = test_process_message(message)
    
    print("\n" + "="*80)
    print(f"RESPONSE FOR: '{message}'")
    print("="*80)
    print(response) 