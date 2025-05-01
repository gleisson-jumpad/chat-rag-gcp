#!/usr/bin/env python3
"""
Quick script to check queries through our improved rag_processor
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
logger = logging.getLogger("check_queries")

def check_queries():
    """Check various contract-related queries through our improved processor"""
    # Initialize the RAG tool
    logger.info("Initializing MultiTableRAGTool...")
    rag_tool = MultiTableRAGTool()
    
    # Test queries
    test_queries = [
        "qual o valor do contrato da jumpad com a coentro?",
        "quem assinou o contrato com a coentro?",
        "quando foi assinado o contrato entre a coentro e a jumpad?",
        "qual o valor mensal do contrato com a coentro?",
        "qual o valor do contrato da jumpad com a coentro, quando foi assinado e quem assinou?"
    ]
    
    # Test direct processing
    logger.info("\n=== TESTING DIRECT PROCESSING ===")
    for i, query in enumerate(test_queries, 1):
        logger.info(f"QUERY {i}: {query}")
        
        # Process the query
        result = process_query(query, rag_tool)
        print(f"\nQuery: {query}")
        print(f"RESPONSE:\n{result}\n")
        print("-" * 80)
    
    # Test processing with LLM enhancement
    logger.info("\n=== TESTING PROCESSING WITH LLM ENHANCEMENT ===")
    for i, query in enumerate(test_queries, 1):
        logger.info(f"QUERY {i}: {query}")
        
        # Process the query with LLM enhancement
        result = process_query_with_llm(query, rag_tool)
        print(f"\nQuery: {query}")
        print(f"RESPONSE:\n{result}\n")
        print("-" * 80)
    
    logger.info("Query checks completed.")

if __name__ == "__main__":
    check_queries() 