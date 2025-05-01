#!/usr/bin/env python3
"""
Test Vector Search Functionality for RAG System
-----------------------------------------------

This script performs comprehensive testing of the vector search capabilities 
in our RAG implementation. It evaluates search quality, response accuracy, 
and retrieval performance across various query types.

Key functions:
1. Test different query types (factual, conceptual, document-specific)
2. Evaluate performance metrics (response time, source relevance)
3. Validate multi-table search capabilities
4. Benchmark retrieval quality across different types of questions

This tool is particularly useful for:
- Evaluating search quality after adding new documents
- Benchmarking system performance with different query strategies
- Troubleshooting retrieval issues with specific document types
- Testing system response to both specific and general knowledge queries
"""
import os
import sys
import logging
import time
from typing import Dict, Any

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from multi_table_rag import MultiTableRAGTool
from rag_processor import process_query

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_vector_search")

def test_vector_search():
    """
    Test the vector search capabilities with various query types
    
    This function:
    1. Initializes the RAG system with MultiTableRAGTool
    2. Gathers information about available documents in the database
    3. Runs a series of test queries covering different knowledge domains
    4. Measures and reports performance metrics for each query
    5. Displays detailed information about retrieved sources and relevance
    
    Both document-specific queries (e.g., contract details) and general 
    knowledge queries are tested to validate the system's ability to
    retrieve information from different contexts.
    """
    logger.info("Initializing MultiTableRAGTool...")
    rag_tool = MultiTableRAGTool()
    
    # Get info about available documents
    files_info = rag_tool.get_files_in_database()
    logger.info(f"Available documents: {files_info.get('all_files', [])}")
    
    # Test queries - including both informational and document-specific ones
    test_queries = [
        # Contract details - testing document-specific retrieval
        "What is the monthly value of the contract?",
        "Who signed the contract?",
        "When was the contract signed?",
        "What is the contract duration?",
        
        # General information - testing conceptual understanding
        "How does LlamaIndex work with OpenAI API?",
        "What is RAG and how does it work?",
        "Tell me about vector search implementation in LlamaIndex"
    ]
    
    # Run the test queries
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*80}\nTEST QUERY {i}: {query}\n{'='*80}")
        
        try:
            # Time the query execution for performance benchmarking
            start_time = time.time()
            
            # Process the query through the RAG system
            result = rag_tool.query(query)
            
            # Calculate execution time - important performance metric
            execution_time = time.time() - start_time
            
            # Extract information from the result
            answer = result.get("answer", "No answer found")
            sources = result.get("sources", [])
            best_table = result.get("best_table", "None")
            
            # Print the results with performance data
            print(f"\nQUERY: {query}")
            print(f"EXECUTION TIME: {execution_time:.2f} seconds")
            print(f"BEST TABLE: {best_table}")
            print(f"\nRESPONSE:")
            print(f"{answer[:500]}..." if len(answer) > 500 else answer)
            
            # Print source information - critical for evaluating retrieval quality
            if sources:
                print(f"\nSOURCES ({len(sources)}):")
                for i, source in enumerate(sources[:3], 1):  # Show just the first 3 sources
                    doc_name = source.get("metadata", {}).get("file_name", "Unknown")
                    score = source.get("score", "N/A")
                    print(f"  Source {i}: {doc_name} (Score: {score})")
            else:
                print("\nNo sources found")
                
            print("\n" + "-"*80)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
    
    logger.info("Vector search tests completed.")

if __name__ == "__main__":
    test_vector_search() 