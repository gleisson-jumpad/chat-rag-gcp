#!/usr/bin/env python3
"""
Test script for signatory and date information in RAG queries
"""
import os
import sys
import logging

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from multi_table_rag import MultiTableRAGTool
from rag_processor import process_query, process_query_with_llm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_signatories")

def test_signatory_queries():
    """Test various queries about signatories and dates"""
    # Initialize the RAG tool
    logger.info("Initializing MultiTableRAGTool...")
    rag_tool = MultiTableRAGTool()
    
    # Signatory and date queries
    test_queries = [
        "Quem assinou o contrato entre a Jumpad e a Coentro?",
        "Quem assinou como contratada no contrato da Coentro?",
        "Quem assinou como contratante no contrato da Coentro?",
        "Quando foi assinado o contrato entre a Coentro e a Jumpad?",
        "Em que data a Monique assinou o contrato?",
        "Qual a data de assinatura do contrato da Coentro?",
        "Quem são os signatários do contrato Coentro e Jumpad?",
        "Quando o Gleisson assinou o contrato?",
        "Quem assinou o contrato da Coentro e quando foi assinado?"
    ]
    
    # Test with direct querying
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n=== QUERY {i}: {query} ===\n")
        
        # Use direct query method
        try:
            result = rag_tool.query(query)
            
            # Print response
            print(f"\nDIRECT QUERY: {query}")
            answer = result.get("answer", "No answer")
            print(f"\nRESPONSE:\n{answer[:500]}...")
            print("\n" + "-" * 80)
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
    
    # Test with enhanced LLM processing
    logger.info("\n\n=== TESTING WITH LLM ENHANCEMENT ===\n")
    for i, query in enumerate(test_queries[:3], 1):  # Test fewer queries to save time
        logger.info(f"LLM QUERY {i}: {query}")
        
        try:
            # Process with LLM enhancement
            result = process_query_with_llm(query, rag_tool)
            
            # Print response
            print(f"\nLLM ENHANCED: {query}")
            print(f"\nRESPONSE:\n{result}")
            print("\n" + "-" * 80)
        except Exception as e:
            logger.error(f"Error in LLM processing: {str(e)}")

if __name__ == "__main__":
    test_signatory_queries() 