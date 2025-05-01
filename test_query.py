import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestQuery")

# Add current directory to path
sys.path.append('.')

# Import RAG utils
from app.rag_utils import create_query_engine
from app.multi_table_rag import MultiTableRAGTool

# Test create_query_engine function
def test_create_query_engine():
    try:
        logger.info("Testing create_query_engine function...")
        engine = create_query_engine('gpt-4o', 'data_vectors_472971c1_4265_4aba_a6cf_c3b633115fe1')
        logger.info(f'Successfully created query engine with post-processors: {engine.retriever.node_postprocessors}')
        return True
    except Exception as e:
        logger.error(f'Error creating query engine: {e}')
        return False

# Test MultiTableRAGTool
def test_multi_table_rag():
    try:
        logger.info("Testing MultiTableRAGTool...")
        tool = MultiTableRAGTool()
        logger.info(f'Created MultiTableRAGTool with {len(tool.table_configs)} tables')
        logger.info(f'Available tables: {tool.available_tables}')
        
        # Test query
        if tool.available_tables:
            table_name = tool.available_tables[0]
            response = tool.query_single_table(table_name, "What is LlamaIndex?")
            logger.info(f'Query response: {response["answer"][:100]}...')
        return True
    except Exception as e:
        logger.error(f'Error testing MultiTableRAGTool: {e}')
        return False

if __name__ == "__main__":
    logger.info("Starting tests...")
    success = test_create_query_engine() and test_multi_table_rag()
    logger.info(f"Tests completed. Success: {success}")
    sys.exit(0 if success else 1) 