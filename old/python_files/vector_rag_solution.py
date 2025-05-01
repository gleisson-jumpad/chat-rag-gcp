#!/usr/bin/env python3
"""
Vector RAG Solution with PostgreSQL and LlamaIndex
This script demonstrates the proper implementation of a RAG (Retrieval-Augmented Generation) system
using the existing vector database table with LlamaIndex.
"""
import os
import sys
import logging
import argparse
from typing import List, Dict, Any
import json
import re

# Import LlamaIndex components
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.postprocessor import SimilarityPostprocessor
from openai import OpenAI as DirectOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vector_rag")

def setup_vector_store(table_name="data_llamaindex_vectors") -> PGVectorStore:
    """Set up a vector store for the existing table"""
    logger.info(f"Setting up PostgreSQL vector store for table: {table_name}")
    
    # PostgreSQL connection parameters
    pg_host = os.getenv("PG_HOST", "34.150.190.157")
    pg_port = int(os.getenv("PG_PORT", "5432"))
    pg_user = os.getenv("PG_USER", "llamaindex")
    pg_password = os.getenv("PG_PASSWORD", "password123")
    pg_db = os.getenv("PG_DATABASE", "postgres")
    
    # Set up vector store with the existing table
    vector_store = PGVectorStore.from_params(
        database=pg_db,
        host=pg_host,
        password=pg_password,
        port=pg_port,
        user=pg_user,
        table_name=table_name,
        embed_dim=1536,
        hybrid_search=True, 
        text_search_config="english"
    )
    
    logger.info(f"PostgreSQL vector store initialized with table: {table_name}")
    return vector_store

def get_available_documents(vector_store) -> List[str]:
    """Get a list of available documents in the vector store"""
    logger.info("Getting list of available documents")
    
    try:
        import psycopg2
        
        # Connect to the database
        pg_host = os.getenv("PG_HOST", "34.150.190.157")
        pg_port = int(os.getenv("PG_PORT", "5432"))
        pg_user = os.getenv("PG_USER", "llamaindex")
        pg_password = os.getenv("PG_PASSWORD", "password123")
        pg_db = os.getenv("PG_DATABASE", "postgres")
        
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            database=pg_db
        )
        
        cursor = conn.cursor()
        table_name = "data_llamaindex_vectors"
        
        # Query to get distinct file names from the metadata
        cursor.execute(f"""
            SELECT DISTINCT json_extract_path_text(metadata_::json, 'file_name') AS document
            FROM {table_name}
            WHERE metadata_ IS NOT NULL
        """)
        
        docs = [row[0] for row in cursor.fetchall() if row[0]]
        logger.info(f"Found {len(docs)} documents: {docs}")
        
        cursor.close()
        conn.close()
        
        return docs
    except Exception as e:
        logger.error(f"Error getting document list: {e}")
        return []

def get_document_summary(vector_store, document_name) -> str:
    """Generate a comprehensive summary of a document"""
    logger.info(f"Generating summary for document: {document_name}")
    
    # HYBRID APPROACH: Try vector search first, then fall back to direct DB access
    try:
        # 1. Direct database access to get all document chunks
        import psycopg2
        
        pg_host = os.getenv("PG_HOST", "34.150.190.157")
        pg_port = int(os.getenv("PG_PORT", "5432"))
        pg_user = os.getenv("PG_USER", "llamaindex")
        pg_password = os.getenv("PG_PASSWORD", "password123")
        pg_db = os.getenv("PG_DATABASE", "postgres")
        
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            database=pg_db
        )
        
        cursor = conn.cursor()
        table_name = "data_llamaindex_vectors"
        
        # Get document chunks
        cursor.execute(f"""
            SELECT text, metadata_::json->>'file_name' as file_name 
            FROM {table_name}
            WHERE metadata_::json->>'file_name' = %s
            ORDER BY id
        """, (document_name,))
        
        chunks = cursor.fetchall()
        logger.info(f"Found {len(chunks)} chunks directly from database")
        
        # Check if we got content
        if chunks:
            # Combine all chunks into a single document text
            all_text = "\n\n".join([chunk[0] for chunk in chunks])
            logger.info(f"Combined document text length: {len(all_text)}")
            
            # Generate a summary with OpenAI
            client = DirectOpenAI()
            
            prompt = f"""
            You are tasked with creating a comprehensive summary of the document titled "{document_name}".
            
            Here is the full content of the document:
            
            {all_text[:30000]}
            
            Please create a comprehensive, well-structured summary that:
            1. Covers the main topics and key points
            2. Is organized in a logical flow
            3. Highlights important technical details and examples
            4. Is comprehensive while remaining concise
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear, comprehensive document summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            final_summary = response.choices[0].message.content
            logger.info(f"Generated direct summary of length {len(final_summary)}")
            
            return final_summary
        
        cursor.close()
        conn.close()
    
    except Exception as e:
        logger.error(f"Error in direct database access: {e}")
    
    # 2. Fall back to vector search approach - this is now the backup
    logger.info("Falling back to vector search for document summary")
    
    # Configure LlamaIndex settings
    Settings.llm = LlamaOpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    
    # Create an index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Set up metadata filter for the specific document
    filters = MetadataFilters(filters=[
        ExactMatchFilter(key="file_name", value=document_name)
    ])
    
    # Create a specialized query engine for summarization
    query_engine = index.as_query_engine(
        similarity_top_k=15,  # Retrieve more nodes for better coverage 
        filters=filters,      # Use metadata filtering to focus on the document
        response_mode="tree_summarize"  # Better for summarization
    )
    
    # Multiple queries for comprehensive coverage
    summary_queries = [
        f"What are the main topics and key points covered in the document '{document_name}'?",
        f"What is the overall structure and content of '{document_name}'?", 
        f"Provide a comprehensive summary of '{document_name}' covering all important aspects."
    ]
    
    # Run the queries and collect the responses
    all_responses = []
    
    for query in summary_queries:
        logger.info(f"Running summary query: {query}")
        try:
            response = query_engine.query(query)
            if response and hasattr(response, 'response'):
                all_responses.append(response.response)
                logger.info(f"Query returned response of length {len(response.response)}")
                
                # Log source nodes for debugging
                if hasattr(response, 'source_nodes'):
                    logger.info(f"Query used {len(response.source_nodes)} source nodes")
        except Exception as e:
            logger.error(f"Error running query '{query}': {e}")
    
    # If we have responses, combine them into a final summary
    if all_responses:
        logger.info("Generating combined summary")
        client = DirectOpenAI()
        
        combined_info = "\n\n".join(all_responses)
        
        prompt = f"""
        Create a comprehensive, well-structured summary of the document '{document_name}' 
        based on the following extracted information:
        
        {combined_info}
        
        Your summary should:
        1. Cover the main topics and key points presented in the document
        2. Be well-organized with logical flow
        3. Highlight important technical details and examples
        4. Be comprehensive while remaining concise
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates clear, comprehensive document summaries."},
                {"role": "user", "content": prompt}
            ]
        )
        
        final_summary = response.choices[0].message.content
        logger.info(f"Generated vector-based summary of length {len(final_summary)}")
        
        return final_summary
    else:
        return f"Could not generate a summary for {document_name}"

def answer_query(vector_store, query, document_name=None) -> Dict[str, Any]:
    """Answer a query using the vector store"""
    logger.info(f"Answering query: {query}")
    
    # Configure LlamaIndex settings
    Settings.llm = LlamaOpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    
    # Create an index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Check if this is a document summary request
    summary_pattern = r"(?i)(?:summarize|summary|summarization|resume|resumo).*?\b(.+?\.pdf)\b"
    summary_match = re.search(summary_pattern, query)
    
    if summary_match:
        # This is a summary request
        doc_name = summary_match.group(1)
        logger.info(f"Detected document summary request for: {doc_name}")
        
        # Get available documents to validate
        available_docs = get_available_documents(vector_store)
        
        if doc_name in available_docs:
            summary = get_document_summary(vector_store, doc_name)
            return {
                "answer": summary,
                "is_summary": True,
                "document": doc_name
            }
        else:
            closest_match = None
            for doc in available_docs:
                if doc_name.lower() in doc.lower():
                    closest_match = doc
                    break
            
            if closest_match:
                logger.info(f"Using closest document match: {closest_match}")
                summary = get_document_summary(vector_store, closest_match)
                return {
                    "answer": summary,
                    "is_summary": True,
                    "document": closest_match
                }
    
    # Standard query processing
    # Set up query engine with metadata filter if document_name is provided
    if document_name:
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key="file_name", value=document_name)
        ])
        
        query_engine = index.as_query_engine(
            similarity_top_k=8,
            filters=filters,
            response_mode="compact",
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
        logger.info(f"Created query engine with document filter: {document_name}")
    else:
        query_engine = index.as_query_engine(
            similarity_top_k=6,
            response_mode="compact"
        )
        logger.info("Created standard query engine")
    
    # Execute the query
    try:
        logger.info(f"Executing query: {query}")
        response = query_engine.query(query)
        
        result = {
            "answer": str(response),
            "is_summary": False,
            "sources": []
        }
        
        # Add source information
        if hasattr(response, "source_nodes"):
            logger.info(f"Query returned {len(response.source_nodes)} source nodes")
            
            for node in response.source_nodes:
                source = {
                    "text": node.node.get_content()[:150] + "..." if len(node.node.get_content()) > 150 else node.node.get_content(),
                    "metadata": node.node.metadata,
                    "score": float(node.score) if hasattr(node, "score") else None
                }
                result["sources"].append(source)
        
        return result
    
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "is_summary": False,
            "sources": []
        }

def process_user_message(message):
    """Process a user message and determine if it needs RAG"""
    # Check if message is requesting document information
    document_info_pattern = r"(?i)(?:list|show|what|which).*(?:documents|files|pdfs)"
    document_summary_pattern = r"(?i)(?:summarize|summary|summarization|resume|resumo).*?\b(.+?\.pdf)\b"
    specific_doc_pattern = r"(?i)(?:about|in|from|of).*?\b(.+?\.pdf)\b"
    
    # Set up vector store
    vector_store = setup_vector_store()
    
    # Check for document listing request
    if re.search(document_info_pattern, message):
        docs = get_available_documents(vector_store)
        if docs:
            doc_list = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(docs)])
            return f"Available documents:\n{doc_list}"
        else:
            return "No documents found in the vector store."
    
    # Check for document summary request
    summary_match = re.search(document_summary_pattern, message)
    if summary_match:
        doc_name = summary_match.group(1)
        result = answer_query(vector_store, message)
        if result["is_summary"]:
            return f"Summary of {result['document']}:\n\n{result['answer']}"
    
    # Check if query is about a specific document
    doc_match = re.search(specific_doc_pattern, message)
    if doc_match:
        doc_name = doc_match.group(1)
        docs = get_available_documents(vector_store)
        
        # Find exact or closest match
        target_doc = None
        for doc in docs:
            if doc_name.lower() in doc.lower():
                target_doc = doc
                break
        
        if target_doc:
            result = answer_query(vector_store, message, target_doc)
            source_info = ""
            if not result["is_summary"] and "sources" in result:
                source_count = len(result["sources"])
                source_info = f"\n\n(Answer based on {source_count} relevant passages from {target_doc})"
            return f"{result['answer']}{source_info}"
    
    # General query
    result = answer_query(vector_store, message)
    return result["answer"]

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Vector RAG Solution with PostgreSQL")
    parser.add_argument("--list", action="store_true", help="List available documents")
    parser.add_argument("--query", type=str, help="Query to answer")
    parser.add_argument("--summarize", type=str, help="Generate a summary of the specified document")
    parser.add_argument("--document", type=str, help="Specific document to query against")
    parser.add_argument("--message", type=str, help="Process a user message through the RAG system")
    parser.add_argument("--table", type=str, default="data_llamaindex_vectors", help="Vector table name")
    
    args = parser.parse_args()
    
    # Set up vector store
    vector_store = setup_vector_store(args.table)
    
    if args.list:
        docs = get_available_documents(vector_store)
        if docs:
            print("\nAvailable documents:")
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc}")
        else:
            print("\nNo documents found in the vector store.")
        return
    
    if args.summarize:
        summary = get_document_summary(vector_store, args.summarize)
        print("\n" + "="*80)
        print(f"SUMMARY OF: {args.summarize}")
        print("="*80 + "\n")
        print(summary)
        return
    
    if args.query:
        result = answer_query(vector_store, args.query, args.document)
        print("\n" + "="*80)
        print(f"QUERY: {args.query}")
        if args.document:
            print(f"(Filtered to document: {args.document})")
        print("="*80 + "\n")
        print(result["answer"])
        if not result["is_summary"] and "sources" in result:
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\n{i}. Score: {source['score']:.3f}")
                if "metadata" in source and "file_name" in source["metadata"]:
                    print(f"   Document: {source['metadata']['file_name']}")
                print(f"   Text: {source['text']}")
        return
    
    if args.message:
        response = process_user_message(args.message)
        print("\n" + "="*80)
        print(f"USER: {args.message}")
        print("="*80)
        print(f"AI: {response}")
        return
    
    # If no arguments provided, run interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE RAG CHAT")
    print("="*80)
    print("\nType 'exit' to quit, 'list' to show available documents.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if user_input.lower() == "list":
            docs = get_available_documents(vector_store)
            if docs:
                print("\nAvailable documents:")
                for i, doc in enumerate(docs, 1):
                    print(f"{i}. {doc}")
            else:
                print("\nNo documents found in the vector store.")
            continue
        
        response = process_user_message(user_input)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main() 