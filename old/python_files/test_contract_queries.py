#!/usr/bin/env python3
"""
Test script for contract-related queries to verify our improvements
"""
import os
import sys
import logging

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from multi_table_rag import MultiTableRAGTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_contract_queries")

def test_contract_queries():
    """Test various contract-related queries"""
    logger.info("Initializing the RAG tool...")
    rag_tool = MultiTableRAGTool()
    
    # Define test queries
    test_queries = [
        "Qual o valor mensal do contrato com a Coentro?",
        "Quem assinou o contrato entre Coentro e Jumpad?",
        "Quando o contrato entre Coentro e Jumpad foi assinado?",
        "Qual a duração do contrato com a Coentro?",
        "Quais são as condições de pagamento no contrato da Coentro?",
        "qual o valor do contrato da jumpad com a coentro, quando foi assinado e quem assinou?"
    ]
    
    # Run test queries
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*80}\nTEST QUERY {i}: {query}\n{'='*80}")
        
        try:
            # Execute query
            result = rag_tool.query(query)
            
            # Print results
            print(f"\nQUERY: {query}")
            print(f"\nRESPONSE:")
            print(result.get("answer", "No answer found"))
            
            # Print details about sources
            sources = result.get("sources", [])
            if sources:
                print(f"\nFound {len(sources)} sources")
                for i, source in enumerate(sources[:3], 1):  # Show just the first 3 sources
                    print(f"Source {i}: {source.get('metadata', {}).get('file_name', 'Unknown')}")
            else:
                print("\nNo sources found")
                
            print("\n" + "-"*80)
                
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
    
    logger.info("Contract query tests completed.")

if __name__ == "__main__":
    test_contract_queries() 