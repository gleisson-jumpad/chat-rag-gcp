#!/usr/bin/env python3
"""
Proper RAG Query with PostgreSQL Vector Store and LlamaIndex
Demonstrating the correct approach to using vector search and metadata filtering
"""
import os
import sys
import logging
import argparse
from typing import List, Dict, Any

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proper_rag")

def load_environment():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', '.env')
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    except ImportError:
        logger.warning("dotenv not installed, skipping .env loading")

def setup_vector_store(table_name="vector_tutorial_table") -> PGVectorStore:
    """Set up and return the PostgreSQL vector store"""
    logger.info(f"Setting up PostgreSQL vector store for table: {table_name}")
    
    # PostgreSQL connection parameters
    pg_host = os.getenv("PG_HOST", "34.150.190.157")
    pg_port = int(os.getenv("PG_PORT", "5432"))
    pg_user = os.getenv("PG_USER", "llamaindex")
    pg_password = os.getenv("PG_PASSWORD", "password123")
    pg_db = os.getenv("PG_DATABASE", "postgres")
    
    # Set up vector store with specified table
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

def load_document(file_path: str) -> Document:
    """Load document from file path"""
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
                "source": file_path,
                "document_type": "pdf"
            }
        )
        
        logger.info(f"Loaded document with {len(text)} characters")
        return document
        
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise

def process_and_index_document(vector_store: PGVectorStore, document: Document) -> VectorStoreIndex:
    """Process and index the document in the vector store"""
    logger.info("Processing and indexing document")
    
    # Configure LlamaIndex settings
    Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    
    # Set up node parser for text splitting
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Parse document into nodes
    nodes = node_parser.get_nodes_from_documents([document])
    
    # Ensure all nodes have the proper metadata
    file_name = document.metadata["file_name"]
    for i, node in enumerate(nodes):
        node.metadata["doc_id"] = f"{file_name}_{i}"
        node.metadata["file_name"] = file_name
        # Add page number if available in the node text
        if "[Page " in node.text:
            try:
                page_match = node.text.split("[Page ")[1].split("]")[0]
                node.metadata["page_label"] = page_match
            except:
                pass
    
    logger.info(f"Created {len(nodes)} nodes from document")
    
    # Create storage context with the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the index from the nodes
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    logger.info(f"Successfully indexed document in PostgreSQL vector store")
    return index

def query_document(index: VectorStoreIndex, query_text: str, document_name: str = None) -> Dict[str, Any]:
    """
    Query the vectorstore index with proper metadata filtering
    This demonstrates the correct approach to document-specific search
    """
    logger.info(f"Processing query: '{query_text}' {'for document: ' + document_name if document_name else ''}")
    
    # Create the query engine with higher k for better recall
    if document_name:
        # Create metadata filter for the specified document
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key="file_name", value=document_name)
        ])
        
        # Create filtered query engine for document-specific queries
        query_engine = index.as_query_engine(
            similarity_top_k=10,  # Retrieve more nodes for better coverage
            filters=filters,      # Apply the document filter
            response_mode="tree_summarize",
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
        logger.info(f"Created filtered query engine for document: {document_name}")
    else:
        # Standard query engine without document filtering
        query_engine = index.as_query_engine(
            similarity_top_k=8,
            response_mode="compact"
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
                "page": node.node.metadata.get("page_label", "unknown"),
                "score": float(node.score) if hasattr(node, "score") else None
            }
            result["sources"].append(source_info)
    
    logger.info(f"Query completed with {len(result.get('sources', []))} sources")
    return result

def summarize_document(index: VectorStoreIndex, document_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of a document using vector search
    Uses the proper document metadata filtering approach
    """
    logger.info(f"Generating summary for document: {document_name}")
    
    # Create document filter using metadata
    filters = MetadataFilters(filters=[
        ExactMatchFilter(key="file_name", value=document_name)
    ])
    
    # Create multiple summary queries for comprehensive coverage
    summary_queries = [
        f"What are the main topics and key points covered in {document_name}?",
        f"What is the overall structure and content of {document_name}?",
        f"Provide a comprehensive summary of {document_name} covering all important aspects."
    ]
    
    all_responses = []
    all_sources = []
    
    # Create a specialized query engine for summaries with metadata filtering
    query_engine = index.as_query_engine(
        similarity_top_k=15,     # Higher k for better document coverage
        filters=filters,         # Apply document filter - THIS IS CRUCIAL
        response_mode="tree_summarize"
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
                        "page": node.node.metadata.get("page_label", "unknown"),
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
        1. Cover all major topics and key points
        2. Be well-organized with sections and subpoints where appropriate
        3. Present information in a logical flow
        4. Include any important details or conclusions
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

def list_documents(vector_store: PGVectorStore) -> List[str]:
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
        table_name = "vector_tutorial_table"
        
        cursor.execute(f"""
            SELECT DISTINCT metadata_->>'file_name' AS document
            FROM {table_name}
            WHERE metadata_->>'file_name' IS NOT NULL
        """)
        
        documents = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Proper RAG with PostgreSQL and LlamaIndex")
    parser.add_argument("--table", "-t", type=str, default="vector_tutorial_table",
                        help="PostgreSQL table to use")
    parser.add_argument("--load", "-l", type=str, help="Path to document to load into vector store")
    parser.add_argument("--query", "-q", type=str, help="Query to run against the vector store")
    parser.add_argument("--document", "-d", type=str, help="Specific document to query")
    parser.add_argument("--summary", "-s", type=str, help="Generate summary for specified document")
    parser.add_argument("--list", action="store_true", help="List available documents")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_environment()
    
    # Set up vector store
    vector_store = setup_vector_store(args.table)
    
    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Process commands
    if args.list:
        documents = list_documents(vector_store)
        if documents:
            print("\nDocuments in vector store:")
            for i, doc in enumerate(documents, 1):
                print(f"{i}. {doc}")
        else:
            print("\nNo documents found in vector store")
        return
    
    if args.load:
        document = load_document(args.load)
        index = process_and_index_document(vector_store, document)
        print(f"\nDocument '{document.metadata['file_name']}' successfully loaded and indexed")
        return
    
    if args.summary:
        result = summarize_document(index, args.summary)
        print("\n" + "="*80)
        print(f"SUMMARY OF: {args.summary}")
        print("="*80)
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"][:3]:  # Show first 3 sources
            print(f"- {source['file_name']}, Page {source['page']} (Score: {source['score']:.2f})")
        return
    
    if args.query:
        result = query_document(index, args.query, args.document)
        print("\n" + "="*80)
        print(f"ANSWER TO: {args.query}")
        if args.document:
            print(f"(Filtered to document: {args.document})")
        print("="*80)
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['file_name']}, Page {source['page']} (Score: {source['score']:.2f})")
        return
    
    # Interactive mode if no specific arguments provided
    print("\n" + "="*80)
    print("INTERACTIVE RAG DEMO")
    print("="*80)
    
    documents = list_documents(vector_store)
    if documents:
        print("\nAvailable documents:")
        for i, doc in enumerate(documents, 1):
            print(f"{i}. {doc}")
    else:
        print("\nNo documents found in vector store")
        if input("Would you like to load a document? (y/n): ").lower() == 'y':
            file_path = input("Enter the path to the document: ")
            try:
                document = load_document(file_path)
                index = process_and_index_document(vector_store, document)
                print(f"\nDocument '{document.metadata['file_name']}' successfully loaded and indexed")
                documents = list_documents(vector_store)
            except Exception as e:
                print(f"Error loading document: {e}")
                return
    
    # If we have documents, allow queries
    if documents:
        try:
            while True:
                print("\n" + "-"*40)
                print("1. Query with document filter")
                print("2. General query (no filter)")
                print("3. Generate document summary")
                print("4. Exit")
                choice = input("\nChoose an option (1-4): ")
                
                if choice == '1':
                    # Query with document filter
                    print("\nSelect document to query:")
                    for i, doc in enumerate(documents, 1):
                        print(f"{i}. {doc}")
                    try:
                        doc_idx = int(input("Enter document number: ")) - 1
                        if 0 <= doc_idx < len(documents):
                            document_name = documents[doc_idx]
                            query = input("\nEnter your query: ")
                            result = query_document(index, query, document_name)
                            print("\n" + "="*80)
                            print(f"ANSWER (Document: {document_name}):")
                            print("="*80)
                            print(result["answer"])
                            print("\nSources:")
                            for source in result["sources"]:
                                print(f"- Page {source['page']} (Score: {source['score']:.2f})")
                    except ValueError:
                        print("Invalid selection")
                
                elif choice == '2':
                    # General query without filter
                    query = input("\nEnter your query: ")
                    result = query_document(index, query)
                    print("\n" + "="*80)
                    print("ANSWER:")
                    print("="*80)
                    print(result["answer"])
                    print("\nSources:")
                    for source in result["sources"]:
                        print(f"- {source['file_name']}, Page {source['page']} (Score: {source['score']:.2f})")
                
                elif choice == '3':
                    # Generate document summary
                    print("\nSelect document to summarize:")
                    for i, doc in enumerate(documents, 1):
                        print(f"{i}. {doc}")
                    try:
                        doc_idx = int(input("Enter document number: ")) - 1
                        if 0 <= doc_idx < len(documents):
                            document_name = documents[doc_idx]
                            result = summarize_document(index, document_name)
                            print("\n" + "="*80)
                            print(f"SUMMARY OF: {document_name}")
                            print("="*80)
                            print(result["answer"])
                            print("\nSources used:")
                            source_counts = {}
                            for source in result["sources"]:
                                page = source.get("page", "unknown")
                                if page in source_counts:
                                    source_counts[page] += 1
                                else:
                                    source_counts[page] = 1
                            for page, count in source_counts.items():
                                print(f"- Page {page}: {count} chunks used")
                    except ValueError:
                        print("Invalid selection")
                
                elif choice == '4':
                    break
                
                else:
                    print("Invalid choice. Please select 1-4.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 