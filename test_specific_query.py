#!/usr/bin/env python3
"""
Test script for specific queries to debug signatories information retrieval
"""
import os
import sys
import logging
import time

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import our RAG tools
from multi_table_rag import MultiTableRAGTool
from rag_processor import process_query_with_llm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_specific_query")

def test_specific_query():
    """Test specific contract-related queries that are failing in the main app"""
    logger.info("Initializing MultiTableRAGTool...")
    rag_tool = MultiTableRAGTool()
    
    # Query about contract signatories
    query = "Quem assinou o contrato entre Coentro e Jumpad?"
    
    # First try direct query
    logger.info(f"\n{'='*80}\nTesting direct query with MultiTableRAGTool: {query}\n{'='*80}")
    
    start_time = time.time()
    direct_result = rag_tool.query(query)
    execution_time = time.time() - start_time
    
    answer = direct_result.get("answer", "No answer found")
    sources = direct_result.get("sources", [])
    
    print(f"\nDIRECT QUERY: {query}")
    print(f"EXECUTION TIME: {execution_time:.2f} seconds")
    print(f"\nRESPONSE:")
    print(answer)
    
    if sources:
        print(f"\nSOURCES ({len(sources)}):")
        for i, source in enumerate(sources[:3], 1):
            doc_name = source.get("metadata", {}).get("file_name", "Unknown")
            score = source.get("score", "N/A")
            print(f"  Source {i}: {doc_name} (Score: {score})")
    
    # Then try with the processor used by the frontend
    logger.info(f"\n{'='*80}\nTesting with process_query_with_llm: {query}\n{'='*80}")
    
    start_time = time.time()
    processor_result = process_query_with_llm(query, rag_tool)
    execution_time = time.time() - start_time
    
    print(f"\nPROCESSOR QUERY: {query}")
    print(f"EXECUTION TIME: {execution_time:.2f} seconds")
    print(f"\nRESPONSE:")
    print(processor_result)
    
    # Test another query for verification
    query2 = "Qual o valor do contrato da Jumpad com a Coentro?"
    logger.info(f"\n{'='*80}\nTesting with process_query_with_llm: {query2}\n{'='*80}")
    
    start_time = time.time()
    processor_result2 = process_query_with_llm(query2, rag_tool)
    execution_time = time.time() - start_time
    
    print(f"\nPROCESSOR QUERY 2: {query2}")
    print(f"EXECUTION TIME: {execution_time:.2f} seconds")
    print(f"\nRESPONSE:")
    print(processor_result2)

if __name__ == "__main__":
    test_specific_query() 