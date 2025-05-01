#!/usr/bin/env python3
"""
Simple Postgres Vector Search RAG Solution with LlamaIndex
This script demonstrates a proper implementation of RAG using LlamaIndex with PostgreSQL and pgvector
"""
import os
import sys
import logging
from typing import List, Dict, Any
import argparse

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, MetadataFilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llamaindex_pgvector")

def load_environment():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', '.env')
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    except ImportError:
        logger.warning("dotenv not installed, skipping .env loading")

def setup_pgvector_store() -> PGVectorStore:
    """Set up and return a PostgreSQL vector store"""
    logger.info("Setting up PostgreSQL vector store")
    
    # PostgreSQL connection parameters
    pg_host = os.getenv("PG_HOST", "34.150.190.157")
    pg_port = int(os.getenv("PG_PORT", "5432"))
    pg_user = os.getenv("PG_USER", "llamaindex")
    pg_password = os.getenv("PG_PASSWORD", "password123")
    pg_db = os.getenv("PG_DATABASE", "postgres")
    
    # Set up vector store with correct parameters - use existing table
    existing_table = "data_vectors_472971c1_4265_4aba_a6cf_c3b633115fe1"
    
    vector_store = PGVectorStore.from_params(
        database=pg_db,
        host=pg_host,
        password=pg_password,
        port=pg_port,
        user=pg_user,
        table_name=existing_table,  # Using the existing table
        embed_dim=1536,             # OpenAI's ada-002 dimension
        hybrid_search=True,         # Enable hybrid search
        text_search_config="english" # Configure for English text
    )
    
    logger.info(f"PostgreSQL vector store initialized with existing table: {existing_table}")
    return vector_store

def load_and_index_documents(vector_store: PGVectorStore, file_path: str) -> VectorStoreIndex:
    """Load documents, process them, and index them in the vector store"""
    logger.info(f"Loading document: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document {file_path} not found")
    
    # Extract file name from path
    file_name = os.path.basename(file_path)
    
    # Load document content - this is simplified for PDFs
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"[Page {page_num+1}]\n{page_text}\n\n"
                
        # Create LlamaIndex document with metadata
        document = Document(
            text=text,
            metadata={
                "file_name": file_name,
                "source": file_path
            }
        )
        
        logger.info(f"Loaded document with {len(text)} characters")
        
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise
    
    # Configure LlamaIndex settings
    Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    
    # Set up a node parser for text splitting
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Parse document into nodes with page metadata
    nodes = node_parser.get_nodes_from_documents([document])
    
    # Ensure each node has proper metadata
    for i, node in enumerate(nodes):
        node.metadata["doc_id"] = f"{file_name}_{i}"
        if "file_name" not in node.metadata:
            node.metadata["file_name"] = file_name
    
    logger.info(f"Created {len(nodes)} nodes from document")
    
    # Create storage context with the vector store
    from llama_index.core import StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the index from the nodes
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    logger.info(f"Successfully indexed document in PostgreSQL vector store")
    return index

def get_documents_in_store(vector_store: PGVectorStore) -> List[str]:
    """Get a list of documents in the vector store"""
    try:
        # Direct database query to get unique document names
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "34.150.190.157"),
            port=int(os.getenv("PG_PORT", "5432")),
            user=os.getenv("PG_USER", "llamaindex"),
            password=os.getenv("PG_PASSWORD", "password123"),
            database=os.getenv("PG_DATABASE", "postgres")
        )
        cursor = conn.cursor()
        
        # Get the table name from the vector store
        table_name = vector_store._table_name
        
        # Query to get distinct file names
        cursor.execute(f"""
            SELECT DISTINCT metadata_->>'file_name' AS file_name
            FROM {table_name}
            WHERE metadata_->>'file_name' IS NOT NULL
        """)
        
        files = [row[0] for row in cursor.fetchall() if row[0]]
        cursor.close()
        conn.close()
        
        return files
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

def query_document(index: VectorStoreIndex, query_text: str, document_name: str = None) -> Dict[str, Any]:
    """
    Query the vectorstore index with proper filtering by document name if provided
    Uses semantic search to find relevant passages
    """
    logger.info(f"Processing query: '{query_text}' {'for document: ' + document_name if document_name else ''}")
    
    # Create the query engine with higher k for better recall
    if document_name:
        # Create metadata filter for the specified document
        filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=document_name)])
        
        # Create filtered query engine
        query_engine = index.as_query_engine(
            similarity_top_k=10,  # Retrieve more nodes for better coverage
            filters=filters,      # Apply the document filter
            response_mode="tree_summarize",  # Use tree summarize for better summaries
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)  # Only use relevant results
            ]
        )
        logger.info(f"Created filtered query engine for document: {document_name}")
    else:
        # Standard query engine without document filtering
        query_engine = index.as_query_engine(
            similarity_top_k=8,
            response_mode="compact",
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
        logger.info("Created standard query engine (no document filter)")
    
    # Execute the query
    response = query_engine.query(query_text)
    
    # Format the response
    result = {
        "answer": str(response),
        "sources": []
    }
    
    # Add sources if available
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            source_info = {
                "text": node.node.get_content()[:250] + "..." if len(node.node.get_content()) > 250 else node.node.get_content(),
                "file_name": node.node.metadata.get("file_name", "unknown"),
                "score": float(node.score) if hasattr(node, "score") else None
            }
            result["sources"].append(source_info)
    
    logger.info(f"Query completed with {len(result.get('sources', []))} sources")
    return result

def summarize_document(index: VectorStoreIndex, document_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of a document using vector search
    This properly uses pgvector to retrieve relevant content
    """
    logger.info(f"Generating summary for document: {document_name}")
    
    # Create document filter
    filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=document_name)])
    
    # Create multiple summary queries to get comprehensive coverage
    summary_queries = [
        f"What are the main topics and key points covered in the document '{document_name}'?",
        f"What is the overall structure and content of '{document_name}'?",
        f"Provide a comprehensive summary of '{document_name}' covering all important aspects."
    ]
    
    all_responses = []
    all_sources = []
    
    # Create specialized query engine for summaries
    query_engine = index.as_query_engine(
        similarity_top_k=15,     # Higher k for better document coverage
        filters=filters,         # Apply document filter
        response_mode="tree_summarize"  # Better for summaries
    )
    
    # Execute multiple queries to get comprehensive coverage
    for query in summary_queries:
        logger.info(f"Running summary query: {query}")
        response = query_engine.query(query)
        
        if response and hasattr(response, 'response'):
            all_responses.append(response.response)
            
            # Collect sources
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source = {
                        "text": node.node.get_content()[:250] + "..." if len(node.node.get_content()) > 250 else node.node.get_content(),
                        "file_name": node.node.metadata.get("file_name", "unknown"),
                        "score": float(node.score) if hasattr(node, "score") else None
                    }
                    all_sources.append(source)
    
    # Create final synthesized summary from all responses
    if all_responses:
        combined_response = "\n\n".join(all_responses)
        
        # Use OpenAI to synthesize the final summary
        from openai import OpenAI as DirectOpenAI
        client = DirectOpenAI()
        
        synthesis_prompt = f"""
        Create a comprehensive, well-structured summary of the document '{document_name}' 
        based on the following extracted information:
        
        {combined_response}
        
        Your summary should:
        1. Cover all major topics and key points in the document
        2. Be well-organized with sections and subpoints where appropriate
        3. Present information in a logical flow
        4. Include any important details or conclusions from the document
        """
        
        logger.info("Generating final synthesized summary")
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates clear, comprehensive document summaries."},
                {"role": "user", "content": synthesis_prompt}
            ]
        )
        
        synthesized_summary = final_response.choices[0].message.content
        logger.info(f"Successfully generated synthesized summary of length {len(synthesized_summary)}")
        
        return {
            "answer": synthesized_summary,
            "sources": all_sources
        }
    else:
        logger.warning("No responses generated from vector queries")
        return {
            "answer": f"Could not generate a summary for {document_name}",
            "sources": []
        }

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="LlamaIndex with PostgreSQL Vector Search")
    parser.add_argument("--load", type=str, help="Path to document to load into vector store")
    parser.add_argument("--query", type=str, help="Query to run against the vector store")
    parser.add_argument("--document", type=str, help="Specific document to query (optional)")
    parser.add_argument("--summary", type=str, help="Generate summary for specified document")
    parser.add_argument("--list", action="store_true", help="List available documents in the vector store")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_environment()
    
    # Initialize the vector store
    vector_store = setup_pgvector_store()
    
    # Create the index from the vector store
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Process commands
    if args.list:
        files = get_documents_in_store(vector_store)
        print("\nDocuments in vector store:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        return
    
    if args.load:
        load_and_index_documents(vector_store, args.load)
        return
    
    if args.summary:
        result = summarize_document(index, args.summary)
        print("\n" + "="*80)
        print(f"SUMMARY OF: {args.summary}")
        print("="*80)
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"][:3]:  # Show first 3 sources
            print(f"- {source['file_name']} (Score: {source['score']:.2f})")
        return
    
    if args.query:
        result = query_document(index, args.query, args.document)
        print("\n" + "="*80)
        print(f"ANSWER TO: {args.query}")
        print("="*80)
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['file_name']} (Score: {source['score']:.2f})")
        return
    
    # If no arguments provided, show interactive mode
    print("\nInteractive RAG Query Mode (Ctrl+C to exit)")
    print("="*80)
    
    files = get_documents_in_store(vector_store)
    if files:
        print("\nAvailable documents:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
    
    try:
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ('exit', 'quit'):
                break
                
            doc_filter = None
            if files and len(files) > 1:
                use_filter = input("Filter by specific document? (y/n): ").lower() == 'y'
                if use_filter:
                    for i, file in enumerate(files, 1):
                        print(f"{i}. {file}")
                    try:
                        selection = int(input("Select document number: "))
                        if 1 <= selection <= len(files):
                            doc_filter = files[selection-1]
                    except ValueError:
                        print("Invalid selection, querying across all documents")
            
            result = query_document(index, query, doc_filter)
            print("\n" + "="*80)
            print("ANSWER:")
            print("="*80)
            print(result["answer"])
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['file_name']} (Score: {source['score']:.2f})")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 