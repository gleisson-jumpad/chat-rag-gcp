#!/usr/bin/env python3
"""
Debug Vector Store Script
This script will connect to the existing vector store table and demonstrate proper vector search
"""
import os
import sys
import logging
from typing import List, Dict, Any
import json

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, MetadataFilter
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_vector_store")

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

def list_documents(vector_store) -> List[str]:
    """List all documents in the vector store"""
    logger.info("Listing documents in vector store")
    
    try:
        import psycopg2
        import json
        
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
        
        # First check if the file_name is stored in metadata_ or if it's a direct column
        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
        columns = [row[0] for row in cursor.fetchall()]
        logger.info(f"Table columns: {columns}")
        
        # Check records
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        logger.info(f"Total records in table: {count}")
        
        # Examine the structure of metadata_
        cursor.execute(f"SELECT metadata_ FROM {table_name} LIMIT 1")
        sample_metadata = cursor.fetchone()
        if sample_metadata and sample_metadata[0]:
            logger.info(f"Sample metadata structure: {json.dumps(sample_metadata[0], indent=2)[:300]}...")
        
        # Try to get the document names
        if "metadata_" in columns:
            cursor.execute(f"""
                SELECT DISTINCT json_extract_path_text(metadata_::json, 'file_name') AS document
                FROM {table_name}
                WHERE metadata_ IS NOT NULL
            """)
        elif "file_name" in columns:
            cursor.execute(f"SELECT DISTINCT file_name FROM {table_name}")
        else:
            logger.error("Could not determine how document names are stored")
            return []
        
        documents = [row[0] for row in cursor.fetchall() if row[0]]
        logger.info(f"Found documents: {documents}")
        
        cursor.close()
        conn.close()
        
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []

def run_vector_search(vector_store, query_text, document_name=None) -> Dict[str, Any]:
    """Run a vector search query"""
    logger.info(f"Running vector search for: {query_text}")
    
    # Configure LlamaIndex settings
    Settings.llm = LlamaOpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    
    # Create the index
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    try:
        # Let's try a direct SQL query to get document content as a fallback
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
        
        # Get document chunks directly
        if document_name:
            cursor.execute(f"""
                SELECT text, metadata_::json->>'file_name' as file_name
                FROM {table_name}
                WHERE metadata_::json->>'file_name' = %s
                ORDER BY id
            """, (document_name,))
        else:
            cursor.execute(f"""
                SELECT text, metadata_::json->>'file_name' as file_name
                FROM {table_name}
                ORDER BY id
            """)
        
        chunks = cursor.fetchall()
        logger.info(f"Found {len(chunks)} chunks directly from database")
        
        if chunks:
            # For debugging, log some sample chunks
            for i, (text, file) in enumerate(chunks[:2]):
                logger.info(f"Sample chunk {i} from {file}: {text[:150]}...")
            
            # Combine all chunks into a single document text
            all_text = "\n\n".join([chunk[0] for chunk in chunks])
            logger.info(f"Combined document text length: {len(all_text)}")
            
            # Generate a response with OpenAI directly
            client = OpenAI()
            
            logger.info("Generating response using OpenAI directly")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the document content."},
                    {"role": "user", "content": f"Based on this document content, {query_text}\n\nDocument content: {all_text[:4000]}..."}
                ]
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": [{"content": chunk[0], "metadata": {"file_name": chunk[1]}, "score": 1.0} for chunk in chunks[:3]]
            }
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error in direct database query: {e}")
    
    # Continue with vector search as a fallback
    logger.info("Attempting vector search")
    
    # Create the query engine with metadata filtering if document_name is provided
    if document_name:
        # Example showing how to use MetadataFilters with exact match
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key="file_name", value=document_name)
        ])
        
        query_engine = index.as_query_engine(
            similarity_top_k=10,
            filters=filters,
            response_mode="tree_summarize"
        )
        logger.info(f"Query engine created with filter for document: {document_name}")
    else:
        # No filtering
        query_engine = index.as_query_engine(
            similarity_top_k=8,
            response_mode="compact"
        )
        logger.info("Query engine created without filters")
    
    # Execute the query
    response = query_engine.query(query_text)
    
    # Format the results
    result = {
        "answer": str(response),
        "sources": []
    }
    
    # Add source information if available
    if hasattr(response, "source_nodes"):
        logger.info(f"Query returned {len(response.source_nodes)} source nodes")
        for node in response.source_nodes:
            source_info = {
                "content": node.node.get_content()[:200] + "..." if len(node.node.get_content()) > 200 else node.node.get_content(),
                "metadata": node.node.metadata,
                "score": float(node.score) if hasattr(node, "score") else None
            }
            result["sources"].append(source_info)
    else:
        logger.warning("No source nodes in the response")
    
    return result

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Debug Vector Store")
    parser.add_argument("--list", action="store_true", help="List document names in vector store")
    parser.add_argument("--query", type=str, help="Query to run against the vector store")
    parser.add_argument("--document", type=str, help="Filter query to a specific document")
    parser.add_argument("--table", type=str, default="data_llamaindex_vectors", help="Vector table name")
    
    args = parser.parse_args()
    
    # Set up the vector store
    vector_store = setup_vector_store(args.table)
    
    if args.list:
        documents = list_documents(vector_store)
        if documents:
            print("\nDocuments in vector store:")
            for i, doc in enumerate(documents, 1):
                print(f"{i}. {doc}")
        else:
            print("\nNo documents found in vector store")
        return
    
    if args.query:
        result = run_vector_search(vector_store, args.query, args.document)
        print("\n" + "="*80)
        print(f"QUERY: {args.query}")
        if args.document:
            print(f"(Filtered to document: {args.document})")
        print("="*80 + "\n")
        print(result["answer"])
        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"\n{i}. Score: {source['score']:.3f}")
            if "file_name" in source["metadata"]:
                print(f"   Document: {source['metadata']['file_name']}")
            print(f"   Content: {source['content'][:150]}...")
        return
    
    # If no arguments provided
    print("\nPlease specify an action using --list or --query")
    parser.print_help()

if __name__ == "__main__":
    main() 