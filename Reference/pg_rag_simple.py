#!/usr/bin/env python3
"""
Simple RAG implementation with PostgreSQL and LlamaIndex
-------------------------------------------------------

This script demonstrates a proper RAG (Retrieval-Augmented Generation) solution with the following features:
1. Upload text or PDF files to PostgreSQL vector database
2. Query the vector database using semantic search
3. Retrieve only the most relevant chunks rather than the entire document
4. Provide source attribution for transparency

Key Components:
--------------
1. Vector Storage: PostgreSQL with pgvector for efficient similarity search
2. Document Processing: Chunking with metadata preservation
3. Vector Search: Semantic search with hybrid text search capability
4. Source Attribution: Display source chunks with relevance scores

Usage:
------
# Install dependencies
pip install pypdf2 llama-index llama-index-vector-stores-postgres llama-index-llms-openai llama-index-embeddings-openai psycopg2-binary

# Index a document
python pg_rag_simple.py --file your_document.txt

# Index a PDF document
python pg_rag_simple.py --file your_document.pdf

# Query the indexed document
python pg_rag_simple.py --query "Your question about the document"

# List indexed documents
python pg_rag_simple.py --list

# Force reindexing of a document
python pg_rag_simple.py --file your_document.txt --force

"""
import os
import sys
import logging
import argparse
from typing import List, Dict, Any

# Import LlamaIndex components
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter

# Add PyPDF2 for PDF support
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not installed. To handle PDF files, run: pip install PyPDF2")
    PyPDF2 = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pg_rag_simple")

def setup_vector_store(table_name="simple_vector_table"):
    """
    Setup PostgreSQL vector store with pgvector extension
    
    This function configures the connection to PostgreSQL and creates
    a vector store with hybrid search capabilities (vector + keyword search)
    
    Args:
        table_name (str): Name of the PostgreSQL table to store vectors
        
    Returns:
        PGVectorStore: Configured vector store instance
    """
    logger.info(f"Setting up PostgreSQL vector store for table: {table_name}")
    
    # PostgreSQL connection parameters - modify these as needed or set environment variables
    pg_host = os.getenv("PG_HOST", "34.150.190.157")
    pg_port = int(os.getenv("PG_PORT", "5432"))
    pg_user = os.getenv("PG_USER", "llamaindex")
    pg_password = os.getenv("PG_PASSWORD", "password123")
    pg_db = os.getenv("PG_DATABASE", "postgres")
    
    # Create PGVectorStore with hybrid search capability
    vector_store = PGVectorStore.from_params(
        database=pg_db,
        host=pg_host,
        password=pg_password,
        port=pg_port,
        user=pg_user,
        table_name=table_name,
        embed_dim=1536,  # OpenAI embedding dimension
        hybrid_search=True,  # Enable hybrid search for better results
        text_search_config="english"
    )
    
    logger.info(f"PostgreSQL vector store initialized with table: {table_name}")
    return vector_store

def read_pdf(file_path):
    """
    Extract text from a PDF file
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF or None if extraction failed
    """
    if PyPDF2 is None:
        logger.error("PyPDF2 is required for reading PDF files. Please install it with: pip install PyPDF2")
        return None
    
    logger.info(f"Reading PDF file: {file_path}")
    text = ""
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        if not text.strip():
            logger.warning(f"Extracted empty text from PDF: {file_path}")
        
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return None

def load_and_index_document(file_path, vector_store, force_reload=False):
    """
    Load a document from file and index it in the vector database
    
    This function performs the following steps:
    1. Extract text from the file (PDF or text)
    2. Create a LlamaIndex document with metadata
    3. Split the document into smaller chunks
    4. Store the chunks in the vector database
    
    Args:
        file_path (str): Path to the document file
        vector_store (PGVectorStore): Vector store instance
        force_reload (bool): Whether to force reindexing if document exists
        
    Returns:
        VectorStoreIndex: Index created from the document or None if failed
    """
    logger.info(f"Loading document: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    # First check if there's existing data in the table
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "34.150.190.157"),
            port=int(os.getenv("PG_PORT", "5432")),
            user=os.getenv("PG_USER", "llamaindex"),
            password=os.getenv("PG_PASSWORD", "password123"),
            database=os.getenv("PG_DATABASE", "postgres")
        )
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {vector_store.table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        if count > 0 and not force_reload:
            logger.info(f"Table {vector_store.table_name} already has {count} records. Use --force to reload.")
            return get_index_from_vector_store(vector_store)
    except Exception as e:
        logger.warning(f"Error checking table content: {e}")
    
    # Load document content
    file_name = os.path.basename(file_path)
    
    # Read file content based on extension
    if file_path.lower().endswith('.pdf'):
        text = read_pdf(file_path)
        if text is None:
            return None
    else:
        # Read regular text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            logger.error(f"Unable to read {file_path} with UTF-8 encoding. Try using --pdf option for PDF files.")
            return None
    
    # Create LlamaIndex document with appropriate metadata
    document = Document(
        text=text,
        metadata={
            "file_name": file_name,
            "source": file_path
        }
    )
    logger.info(f"Loaded document with {len(text)} characters")
    
    # Configure LlamaIndex
    Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    
    # Parse document into nodes with smaller chunks for better retrieval
    # NOTE: Proper chunking is critical for effective RAG
    node_parser = SentenceSplitter(
        chunk_size=512,  # Smaller chunks for more precise retrieval
        chunk_overlap=50,  # Some overlap to maintain context
    )
    nodes = node_parser.get_nodes_from_documents([document])
    
    # Ensure all nodes have the proper metadata for filtering
    for i, node in enumerate(nodes):
        node.metadata["doc_id"] = f"{file_name}_{i}"
        node.metadata["file_name"] = file_name
    
    logger.info(f"Created {len(nodes)} nodes from document")
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    logger.info(f"Document indexed successfully in {vector_store.table_name}")
    return index

def get_index_from_vector_store(vector_store):
    """
    Load an index from an existing vector store
    
    Args:
        vector_store (PGVectorStore): Vector store instance
        
    Returns:
        VectorStoreIndex: Index loaded from the vector store
    """
    logger.info(f"Loading index from vector store: {vector_store.table_name}")
    
    # Configure LlamaIndex
    Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    
    # Create index from existing vector store
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    logger.info("Index loaded successfully")
    return index

def query_document(index, query_text):
    """
    Query the document using vector similarity search
    
    This is the core RAG functionality that:
    1. Converts the query to a vector embedding
    2. Finds similar vectors in the database
    3. Retrieves only the most relevant chunks (not the entire document)
    4. Sends these chunks to the LLM for answer generation
    
    Args:
        index (VectorStoreIndex): Index to query
        query_text (str): Query text
        
    Returns:
        dict: Response with answer and source information
    """
    logger.info(f"Querying document with: {query_text}")
    
    # Create query engine with similarity search
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # Number of chunks to retrieve (not the entire doc)
        response_mode="compact"  # How to format the response
    )
    
    # Execute query
    response = query_engine.query(query_text)
    
    # Get source nodes for context and attribution
    sources = []
    if hasattr(response, 'source_nodes'):
        for i, node in enumerate(response.source_nodes):
            source = {
                "text": node.node.get_content()[:150] + "..." if len(node.node.get_content()) > 150 else node.node.get_content(),
                "metadata": node.node.metadata,
                "score": float(node.score) if hasattr(node, "score") else None
            }
            sources.append(source)
    
    return {
        "answer": str(response),
        "sources": sources
    }

def list_documents(vector_store):
    """
    List documents in the vector store
    
    Args:
        vector_store (PGVectorStore): Vector store instance
        
    Returns:
        dict: Information about documents in the store
    """
    logger.info(f"Listing documents in vector store: {vector_store.table_name}")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "34.150.190.157"),
            port=int(os.getenv("PG_PORT", "5432")),
            user=os.getenv("PG_USER", "llamaindex"),
            password=os.getenv("PG_PASSWORD", "password123"),
            database=os.getenv("PG_DATABASE", "postgres")
        )
        
        cursor = conn.cursor()
        
        # Try to get distinct file names from metadata
        cursor.execute(f"""
            SELECT DISTINCT json_extract_path_text(metadata_::json, 'file_name') AS document
            FROM {vector_store.table_name}
            WHERE metadata_ IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        documents = [row[0] for row in rows if row[0]]
        
        # Get count of chunks per document
        doc_counts = {}
        for doc in documents:
            cursor.execute(f"""
                SELECT COUNT(*)
                FROM {vector_store.table_name}
                WHERE json_extract_path_text(metadata_::json, 'file_name') = %s
            """, (doc,))
            count = cursor.fetchone()[0]
            doc_counts[doc] = count
            
        cursor.close()
        conn.close()
        
        return {
            "documents": documents,
            "counts": doc_counts
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"documents": [], "counts": {}}

def main():
    """
    Main function to parse arguments and execute commands
    """
    parser = argparse.ArgumentParser(description="Simple RAG with PostgreSQL and LlamaIndex")
    parser.add_argument("--file", type=str, help="Path to file to load and index")
    parser.add_argument("--query", type=str, help="Query to run against the indexed document")
    parser.add_argument("--table", type=str, default="simple_vector_table", help="PostgreSQL table name")
    parser.add_argument("--list", action="store_true", help="List documents in the vector store")
    parser.add_argument("--force", action="store_true", help="Force reload document even if it's already indexed")
    
    args = parser.parse_args()
    
    # Setup vector store
    vector_store = setup_vector_store(args.table)
    
    # List documents
    if args.list:
        result = list_documents(vector_store)
        docs = result["documents"]
        counts = result["counts"]
        
        if docs:
            print("\nDocuments in vector store:")
            for i, doc in enumerate(docs, 1):
                count = counts.get(doc, 0)
                print(f"{i}. {doc} ({count} chunks)")
        else:
            print("\nNo documents found in vector store")
        return
    
    # Load and index document
    if args.file:
        index = load_and_index_document(args.file, vector_store, args.force)
        if not index:
            print(f"Failed to load document: {args.file}")
            return
    else:
        # Try to load index from existing vector store
        index = get_index_from_vector_store(vector_store)
    
    # Query document
    if args.query:
        if not index:
            print("No document indexed. Please provide a file to index first.")
            return
        
        result = query_document(index, args.query)
        
        print("\n" + "="*80)
        print(f"QUERY: {args.query}")
        print("="*80 + "\n")
        print(result["answer"])
        
        if result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\n{i}. Score: {source['score']:.3f}")
                if "file_name" in source["metadata"]:
                    print(f"   Document: {source['metadata']['file_name']}")
                print(f"   Text: {source['text']}")
    
    # If neither --list, --file, nor --query provided, show help
    if not (args.list or args.file or args.query):
        parser.print_help()

if __name__ == "__main__":
    main() 

# Make this file immutable to preserve this implementation
# This will be executed after the script is run:
# chmod 444 pg_rag_simple.py 