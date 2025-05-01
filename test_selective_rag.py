#!/usr/bin/env python3
"""
Test script for selective RAG processing
"""
import os
import sys
from app.multi_table_rag import MultiTableRAGTool
from app.postgres_rag_tool import process_message_with_selective_rag

def test_rag_query(query_text):
    """Test a RAG query using process_message_with_selective_rag"""
    print(f"Testing query: '{query_text}'")
    
    # Initialize RAG tool
    tool = MultiTableRAGTool()
    
    # Get available files in database
    files_info = tool.get_files_in_database()
    print(f"Available files: {files_info['all_files']}")
    
    # Process the query
    print("\nProcessing query...")
    response = process_message_with_selective_rag(
        query_text,
        tool,
        model="gpt-4o"
    )
    
    print("\nResponse:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    # Also try direct query method for comparison
    print("\nTrying direct query for comparison...")
    direct_result = tool.query(query_text)
    
    print("\nDirect query result:")
    print("-" * 80)
    print(direct_result)
    print("-" * 80)

if __name__ == "__main__":
    # Get query from command line arguments or use default
    query = "me explique o contrato entre Coentro e Jumpad" if len(sys.argv) < 2 else sys.argv[1]
    test_rag_query(query) 