import os
import json
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Try importing with and without 'app.' prefix to handle different execution contexts
try:
    from db_config import get_pg_connection, get_pg_cursor, return_pg_connection, verify_vector_table, ensure_pgvector_extension
except ImportError:
    try:
        from app.db_config import get_pg_connection, get_pg_cursor, return_pg_connection, verify_vector_table, ensure_pgvector_extension
    except ImportError:
        raise ImportError("Could not import db_config module. Make sure it exists in the correct location.")

class PostgresRAGTool:
    def __init__(
        self, 
        db_name="postgres", 
        host="34.150.190.157", 
        password=None, 
        port=5432, 
        user="llamaindex",
        openai_model="gpt-4",
        embed_dim=1536,
        use_hybrid_search=True
    ):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PostgresRAGTool")
        
        # Database configuration
        self.db_name = db_name
        self.host = host
        self.password = password or os.getenv("PG_PASSWORD", "password123")
        self.port = port
        self.user = user
        self.embed_dim = embed_dim
        self.use_hybrid_search = use_hybrid_search
        
        # LLM configuration
        self.openai_model = openai_model
        self.llm = OpenAI(model=openai_model)
        
        # Initialize components
        self.vector_stores = {}
        self.indices = {}
        self.query_engines = {}
        
        # Make sure pgvector extension is installed
        self._ensure_pgvector()
        
        # Get available vector tables from database
        self.available_tables = self._get_available_tables()
        self.logger.info(f"Found {len(self.available_tables)} available vector tables")
    
    def _ensure_pgvector(self):
        """Ensure that pgvector extension is installed in the database"""
        try:
            if ensure_pgvector_extension():
                self.logger.info("pgvector extension is installed and ready")
            else:
                self.logger.warning("Could not ensure pgvector extension installation - vector operations may fail")
        except Exception as e:
            self.logger.error(f"Error checking pgvector extension: {str(e)}")
    
    def _get_available_tables(self):
        """Get a list of all available vector tables in the database"""
        try:
            with get_pg_cursor() as cursor:
                # Query to find vector tables
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND 
                    (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
                """)
                
                tables = [table[0] for table in cursor.fetchall()]
                
                # Get basic stats for each table
                table_stats = {}
                for table_name in tables:
                    try:
                        # Check if table has the required structure
                        table_info = verify_vector_table(table_name)
                        if table_info["exists"] and table_info["vector_column"]:
                            self.logger.info(f"Table {table_name} is a valid vector table with {table_info['row_count']} rows")
                            table_stats[table_name] = table_info
                        else:
                            self.logger.warning(f"Table {table_name} exists but is not a valid vector table")
                    except Exception as e:
                        self.logger.error(f"Error verifying table {table_name}: {str(e)}")
                
                self.table_stats = table_stats
                return tables
                
        except Exception as e:
            self.logger.error(f"Error getting available tables: {str(e)}")
            return []
    
    def _initialize_table_resources(self, table_name):
        """Initialize vector store, index and query engine for a specific table"""
        if table_name not in self.query_engines:
            self.logger.info(f"Initializing resources for table: {table_name}")
            
            try:
                # Check if table exists
                if table_name not in self.available_tables:
                    self.logger.error(f"Table {table_name} is not available")
                    raise ValueError(f"Table {table_name} does not exist in the database")
                
                # Verify table has proper structure
                table_info = verify_vector_table(table_name)
                if not table_info["exists"] or not table_info["vector_column"]:
                    self.logger.error(f"Table {table_name} does not have proper vector structure")
                    raise ValueError(f"Table {table_name} is not properly configured for vector search")
                
                # Create the connection callback
                def get_conn_callback():
                    return get_pg_connection()
                
                # Create vector store for this table
                vector_store_params = {
                    "database": self.db_name,
                    "host": self.host,
                    "password": self.password,
                    "port": self.port,
                    "user": self.user,
                    "table_name": table_name,
                    "embed_dim": self.embed_dim,
                    "use_jsonb": True,
                    "connection_creator": get_conn_callback
                }
                
                # Add hybrid search if enabled
                if self.use_hybrid_search:
                    self.logger.info("Enabling hybrid search with English text configuration")
                    vector_store_params.update({
                        "hybrid_search": True,
                        "text_search_config": "english",
                    })
                
                # Add HNSW configuration if needed
                if table_info.get("has_index", False):
                    # Check if any HNSW indices exist
                    has_hnsw = any(idx.get("type") == "hnsw" 
                                for idx in table_info.get("indices", []) 
                                if isinstance(idx, dict) and "type" in idx)
                    if has_hnsw:
                        self.logger.info(f"Table {table_name} already has HNSW index, using existing index")
                    else:
                        # Set up HNSW configuration since table has no HNSW index
                        vector_store_params["hnsw_kwargs"] = {
                            "hnsw_m": 16,
                            "hnsw_ef_construction": 64,
                            "hnsw_ef_search": 40,
                            "hnsw_dist_method": "vector_cosine_ops"
                        }
                        self.logger.info(f"Table {table_name} has no HNSW index, will try to create one")
                else:
                    # No index exists, create HNSW
                    vector_store_params["hnsw_kwargs"] = {
                        "hnsw_m": 16,
                        "hnsw_ef_construction": 64,
                        "hnsw_ef_search": 40,
                        "hnsw_dist_method": "vector_cosine_ops"
                    }
                    self.logger.info(f"Table {table_name} has no vector index, will try to create HNSW index")
                
                self.logger.info(f"Creating vector store with params: {vector_store_params}")
                
                # Create the vector store with connection callback first
                try:
                    self.vector_stores[table_name] = PGVectorStore.from_params(**vector_store_params)
                    self.logger.info(f"Successfully created vector store for {table_name} with connection callback")
                except Exception as vs_error:
                    self.logger.error(f"Failed to create vector store with connection callback: {vs_error}")
                    
                    # Try fallback without connection callback
                    self.logger.info("Trying fallback without connection callback")
                    vector_store_params.pop("connection_creator", None)
                    self.vector_stores[table_name] = PGVectorStore.from_params(**vector_store_params)
                    self.logger.info(f"Successfully created fallback vector store for {table_name}")
                
                # Create embedding model with larger batch size
                self.logger.info("Creating OpenAI embedding model")
                embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
                
                # Create index from vector store
                self.logger.info(f"Creating index from vector store for {table_name}")
                storage_context = StorageContext.from_defaults(vector_store=self.vector_stores[table_name])
                
                self.indices[table_name] = VectorStoreIndex.from_vector_store(
                    self.vector_stores[table_name],
                    embed_model=embed_model,
                    storage_context=storage_context
                )
                
                # Create query engine with appropriate configuration
                self.logger.info(f"Creating query engine for {table_name}")
                self.query_engines[table_name] = self.indices[table_name].as_query_engine(
                    llm=self.llm,
                    similarity_top_k=5,  # Increased from 3 for better recall
                    response_mode="compact",
                    # If hybrid search is enabled, use it in the query
                    vector_store_query_mode="hybrid" if self.use_hybrid_search else "default",
                    alpha=0.5 if self.use_hybrid_search else None,
                    # similarity_cutoff moved to post-processors
                    verbose=True
                )
                
                # Add similarity cutoff as a post-processor
                from llama_index.core.postprocessor import SimilarityPostprocessor
                self.query_engines[table_name].retriever.node_postprocessors = [
                    SimilarityPostprocessor(similarity_cutoff=0.7)  # Only include results above this threshold
                ]
                
                self.logger.info(f"Successfully initialized query engine for table: {table_name}")
                return True
            
            except Exception as e:
                self.logger.error(f"Error initializing resources for table {table_name}: {str(e)}")
                raise
    
    def query(self, query_text, table_name=None):
        """
        Query the knowledge base using vector similarity search
        
        This approach aligns with pg_rag_simple.py by:
        1. Converting the query to a vector embedding
        2. Finding similar vectors in the database
        3. Retrieving only the most relevant chunks
        4. Generating an answer based on these chunks
        5. Including source information for attribution
        """
        self.logger.info(f"RAG query received: '{query_text}'")
        
        # If no specific table provided, use the first available table
        if not table_name and self.available_tables:
            table_name = self.available_tables[0]
            self.logger.info(f"No table specified, using default: {table_name}")
        
        if not table_name:
            return {
                "error": "No tables available",
                "message": "No vector tables found in the database"
            }
        
        try:
            # Initialize resources for this table if not already done
            if table_name not in self.query_engines:
                self._initialize_table_resources(table_name)
            
            # Get the query engine
            query_engine = self.query_engines[table_name]
            
            # Execute the query
            self.logger.info(f"Executing query against table {table_name}")
            response = query_engine.query(query_text)
            
            # Format response with sources
            result = {
                "answer": str(response),
                "sources": []
            }
            
            # Add source information if available
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    source_info = {
                        "text": node.node.get_content()[:150] + "..." if len(node.node.get_content()) > 150 else node.node.get_content(),
                        "file_name": node.node.metadata.get("file_name", "unknown"),
                        "page_number": node.node.metadata.get("page_label", "unknown"),
                        "relevance_score": float(node.score) if hasattr(node, "score") else None
                    }
                    result["sources"].append(source_info)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            return {
                "error": str(e),
                "message": "Error querying the knowledge base"
            }

    def get_files_in_database(self):
        """Get a list of all files stored in the vector tables"""
        files_by_table = {}
        
        try:
            with get_pg_cursor() as cursor:
                # Handle both MultiTableRAGTool (which uses table_configs) and PostgresRAGTool classes
                table_names = []
                if hasattr(self, 'available_tables'):
                    table_names = self.available_tables
                elif hasattr(self, 'table_configs'):
                    table_names = [config["name"] for config in self.table_configs]
                else:
                    self.logger.error("No table information available")
                    return {"files_by_table": {}, "all_files": []}
                
                for table_name in table_names:
                    try:
                        # Query to extract file names from metadata
                        cursor.execute(f"""
                            SELECT DISTINCT metadata_->>'file_name' as file_name 
                            FROM {table_name}
                            WHERE metadata_->>'file_name' IS NOT NULL
                        """)
                        
                        # Get unique filenames
                        distinct_files = [file[0] for file in cursor.fetchall() if file[0]]
                        
                        if distinct_files:
                            files_by_table[table_name] = distinct_files
                    except Exception as e:
                        self.logger.error(f"Error getting files from table {table_name}: {str(e)}")
            
            # Also create a flat list of all files
            all_files = []
            for files in files_by_table.values():
                all_files.extend(files)
            
            return {
                "files_by_table": files_by_table,
                "all_files": list(set(all_files))  # Remove duplicates
            }
            
        except Exception as e:
            self.logger.error(f"Error getting files from database: {str(e)}")
            return {"files_by_table": {}, "all_files": []}

def process_message_with_selective_rag(user_message, rag_tool, model="gpt-4o"):
    """Process a user message with selective RAG (only use RAG when needed)"""
    import openai
    import logging
    import re
    import json
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    from openai import OpenAI as DirectOpenAI
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("selective_rag")
    
    logger.info(f"Processing message with selective RAG: '{user_message}'")
    
    # Ensure we have the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return "ERROR: OpenAI API key not set in environment variables"
    
    # Get a list of available documents
    files_info = rag_tool.get_files_in_database()
    available_docs = files_info["all_files"]
    
    # Check if this is a document summary request
    # Examples: "resumir o arquivo X.pdf", "summarize document Y.pdf", etc.
    doc_summary_pattern = re.compile(r'(resum[ae]|sum[aá]rio|sumarize|summarize).*?(documento|document|arquivo|file|pdf)[:\s]*([^?\s]+\.[a-zA-Z0-9]{2,4})', re.IGNORECASE)
    match = doc_summary_pattern.search(user_message)
    
    if match and available_docs:
        requested_doc = match.group(3).strip()
        logger.info(f"Document summary request detected for: {requested_doc}")
        
        # Try to find the exact document or closest match
        exact_match = None
        for doc in available_docs:
            if doc.lower() == requested_doc.lower():
                exact_match = doc
                break
        
        if not exact_match:
            # Try partial matching
            for doc in available_docs:
                if requested_doc.lower() in doc.lower() or doc.lower() in requested_doc.lower():
                    exact_match = doc
                    logger.info(f"Found partial match: {doc} for requested document: {requested_doc}")
                    break
        
        if exact_match:
            logger.info(f"Summarizing document: {exact_match}")
            
            # VECTOR SEARCH APPROACH: Use LlamaIndex's vector search with metadata filtering
            try:
                # Find the table containing this document
                doc_table = None
                for table, files in files_info["files_by_table"].items():
                    if exact_match in files:
                        doc_table = table
                        break
                
                if doc_table:
                    logger.info(f"Found document in table: {doc_table}")
                    
                    # HYBRID APPROACH: Use both vector search for relevant topics and direct retrieval for content
                    
                    # Step 1: Get document chunks directly from the database for comprehensive content
                    from db_config import get_pg_connection
                    conn = get_pg_connection()
                    cursor = conn.cursor()
                    
                    logger.info(f"Direct DB retrieval: Getting chunks for document {exact_match}")
                    cursor.execute(f"""
                        SELECT text, metadata_->>'page_label' as page
                        FROM {doc_table} 
                        WHERE metadata_->>'file_name' = %s
                        ORDER BY id
                    """, (exact_match,))
                    
                    rows = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    
                    document_chunks = []
                    if rows:
                        logger.info(f"Found {len(rows)} chunks directly from the database")
                        
                        # Arrange chunks by page number if available
                        chunks_by_page = {}
                        for text, page in rows:
                            if page not in chunks_by_page:
                                chunks_by_page[page] = []
                            chunks_by_page[page].append(text)
                        
                        # Get sorted pages
                        sorted_pages = sorted(chunks_by_page.keys(), key=lambda p: int(p) if p and p.isdigit() else 0)
                        
                        # Combine all text in order
                        for page in sorted_pages:
                            page_text = "\n\n".join(chunks_by_page[page])
                            document_chunks.append(f"[Page {page}] {page_text}")
                        
                        document_text = "\n\n".join(document_chunks)
                        logger.info(f"Combined document text length: {len(document_text)}")
                    else:
                        logger.warning(f"No chunks found directly for document {exact_match}")
                    
                    # Step 2: Use vector search to find the most relevant segments and main topics
                    vector_responses = []
                    
                    # Get the index for this table
                    index = None
                    if hasattr(rag_tool, 'indexes') and doc_table in rag_tool.indexes:
                        index = rag_tool.indexes[doc_table]
                        logger.info(f"Using existing index for table {doc_table}")
                    
                    if index:
                        # Create metadata filter for the document
                        filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=exact_match)])
                        
                        # Create LLM
                        llm = OpenAI(model=model)
                        
                        # Prepare a list of summary queries to retrieve comprehensive information
                        summary_queries = [
                            f"What are the main topics and key points in the document '{exact_match}'?",
                            f"What is the overall structure and content of '{exact_match}'?",
                            f"Provide a detailed summary of '{exact_match}' covering all important aspects."
                        ]
                        
                        # Create a custom query engine with higher top_k and metadata filters
                        try:
                            query_engine = index.as_query_engine(
                                llm=llm,
                                similarity_top_k=15,  # Higher k value for better document coverage
                                response_mode="tree_summarize",  # Use tree_summarize for better summaries of longer texts
                                filters=filters  # Apply the file_name filter
                            )
                            
                            # Run multiple queries to get comprehensive coverage
                            for query in summary_queries:
                                logger.info(f"Running vector query: {query}")
                                response = query_engine.query(query)
                                if response and hasattr(response, 'response'):
                                    vector_responses.append(f"Vector Search Result: {response.response}")
                        except Exception as ve:
                            logger.error(f"Error during vector query: {ve}")
                    
                    # Step 3: Generate the summary using both vector search results and direct document content
                    client = DirectOpenAI(api_key=openai_api_key)
                    
                    # Combine vector search results with direct document content
                    if vector_responses:
                        logger.info(f"Got {len(vector_responses)} vector search responses")
                        vector_content = "\n\n".join(vector_responses)
                    else:
                        vector_content = "No additional information from vector search available."
                    
                    if document_chunks:
                        if len(document_text) > 90000:  # If document is too large, only use the first 90K chars
                            document_text = document_text[:90000] + "... [truncated]"
                            logger.info(f"Document text truncated to 90000 characters")
                    else:
                        document_text = "No direct document content available."
                    
                    # Create synthesis prompt
                    synthesis_prompt = f"""
                    You are tasked with creating a comprehensive summary of the document '{exact_match}'.
                    
                    I am providing you with two sources of information:
                    
                    1. DIRECT DOCUMENT CONTENT:
                    {document_text}
                    
                    2. VECTOR SEARCH INSIGHTS (main topics and structure):
                    {vector_content}
                    
                    Please create a well-structured, comprehensive summary that:
                    1. Covers all major topics and key points from the document
                    2. Organizes information into logical sections with headings
                    3. Presents content in a clear, readable format
                    4. Highlights the most important details and conclusions
                    
                    Your summary should read as a cohesive document that accurately reflects the original content.
                    """
                    
                    logger.info("Generating final synthesized summary")
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that creates clear, comprehensive document summaries."},
                            {"role": "user", "content": synthesis_prompt}
                        ]
                    )
                    
                    summary = response.choices[0].message.content
                    logger.info(f"Successfully generated synthesized summary of length {len(summary)}")
                    
                    return summary
            
            except Exception as e:
                logger.error(f"Error during hybrid document summary approach: {e}")
                logger.info("Falling back to direct approach")
            
            # DIRECT APPROACH as fallback (only if hybrid approach fails)
            try:
                # Extract document content directly from the database
                from db_config import get_pg_connection
                conn = get_pg_connection()
                cursor = conn.cursor()
                
                # Get all chunks for this document
                cursor.execute(f"""
                    SELECT text, metadata_->>'page_label' as page
                    FROM {doc_table} 
                    WHERE metadata_->>'file_name' = %s
                    ORDER BY id
                """, (exact_match,))
                
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                
                if rows:
                    logger.info(f"Found {len(rows)} chunks for document {exact_match}")
                    
                    # Arrange chunks by page number if available
                    chunks_by_page = {}
                    for text, page in rows:
                        if page not in chunks_by_page:
                            chunks_by_page[page] = []
                        chunks_by_page[page].append(text)
                    
                    # Get sorted pages
                    sorted_pages = sorted(chunks_by_page.keys(), key=lambda p: int(p) if p and p.isdigit() else 0)
                    
                    # Combine all text in order
                    all_text = []
                    for page in sorted_pages:
                        page_text = "\n\n".join(chunks_by_page[page])
                        all_text.append(page_text)
                    
                    document_text = "\n\n".join(all_text)
                    logger.info(f"Combined document text length: {len(document_text)}")
                    
                    # Generate summary with OpenAI directly
                    client = DirectOpenAI(api_key=openai_api_key)
                    
                    summary_prompt = f"""
                    Based on the following document content from '{exact_match}', please provide a comprehensive summary.
                    Include main topics, key points, and important information.
                    Format your response with clear sections and bullet points where appropriate.
                    
                    DOCUMENT CONTENT:
                    {document_text}
                    """
                    
                    logger.info("Generating summary directly with OpenAI")
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides clear document summaries."},
                            {"role": "user", "content": summary_prompt}
                        ]
                    )
                    
                    summary = response.choices[0].message.content
                    logger.info(f"Successfully generated summary of length {len(summary)}")
                    
                    return summary
                else:
                    logger.error(f"No chunks found for document {exact_match}")
            except Exception as e:
                logger.error(f"Error during direct document summary: {e}")
    
    # Standard RAG evaluation for other queries
    # Create a prompt for assessing if RAG is needed
    rag_assessment_messages = [
        {"role": "system", "content": 
         f"You are an assistant that decides whether to use RAG (Retrieval Augmented Generation) for a user query. "
         f"The knowledge base contains the following documents: {', '.join(available_docs) if available_docs else 'No documents available'}. "
         f"Respond with 'RAG:YES' if the query likely requires information from these documents, "
         f"or 'RAG:NO' if the query can be answered without them."},
        {"role": "user", "content": user_message}
    ]
    
    try:
        logger.info("Making initial assessment of whether RAG is needed")
        assessment_response = openai.chat.completions.create(
            model=model,
            messages=rag_assessment_messages,
            max_tokens=50
        )
        
        assessment_text = assessment_response.choices[0].message.content
        logger.info(f"RAG assessment: {assessment_text}")
        
        use_rag = "RAG:YES" in assessment_text.upper()
        
        if use_rag and available_docs:
            logger.info("Decision: Use RAG for this query")
            
            # Get tables containing documents - handle different class types
            if hasattr(rag_tool, 'available_tables'):
                tables = list(rag_tool.available_tables)
            elif hasattr(rag_tool, 'table_configs'):
                tables = [config["name"] for config in rag_tool.table_configs]
            else:
                logger.error("RAG tool has neither 'available_tables' nor 'table_configs' attributes")
                tables = []
            
            # If multiple tables, try to identify the most relevant one
            table_to_use = tables[0] if tables else None
            
            if table_to_use:
                # Query RAG with the user message - handle different tool types
                logger.info(f"Querying RAG system with table: {table_to_use}")
                
                # Perform RAG query and get response with sources
                try:
                    # Query processing similar to pg_rag_simple.py
                    if hasattr(rag_tool, 'query_single_table'):
                        # Using MultiTableRAGTool
                        rag_result = rag_tool.query_single_table(table_to_use, user_message)
                    else:
                        # Using PostgresRAGTool
                        rag_result = rag_tool.query(user_message, table_to_use)
                    
                    # Check for errors in the result
                    if "error" in rag_result:
                        logger.error(f"Error querying RAG: {rag_result['error']}")
                        return f"I encountered an error while searching the knowledge base: {rag_result['error']}"
                    
                    # Format sources for better readability, similar to pg_rag_simple.py
                    sources_text = ""
                    if "sources" in rag_result and rag_result["sources"]:
                        sources_text = "\n\nSources:\n"
                        for i, source in enumerate(rag_result["sources"], 1):
                            score = source.get("relevance_score", 0)
                            if score:
                                score_str = f"Score: {score:.3f}"
                            else:
                                score_str = "No score available"
                                
                            file_name = source.get("file_name", "Unknown")
                            text_snippet = source.get("text", "")
                            
                            sources_text += f"\n{i}. {score_str}\n   Document: {file_name}\n   Text: {text_snippet}\n"
                    
                    # Create a response with attribution similar to the pg_rag_simple.py approach
                    answer = rag_result["answer"]
                    
                    # Debug print to stdout to help diagnose issues
                    print(f"DEBUG: Found answer: {answer[:100]}...")
                    print(f"DEBUG: Found {len(rag_result.get('sources', []))} sources")
                    
                    # Return the answer with source information, even if no sources found
                    formatted_response = answer
                    if sources_text:
                        formatted_response += sources_text
                        
                    # Don't return empty responses
                    if not formatted_response or formatted_response.strip() == "":
                        # Try to get direct database content if formatter fails
                        conn = None
                        try:
                            # Find the table containing this document
                            doc_table = tables[0] if tables else None
                            if doc_table:
                                conn = get_pg_connection()
                                cursor = conn.cursor()
                                # Get content directly
                                cursor.execute(f"""
                                    SELECT text FROM {doc_table} 
                                    ORDER BY id LIMIT 5
                                """)
                                chunks = [row[0] for row in cursor.fetchall()]
                                if chunks:
                                    return f"Based on the available documents, here's what I found:\n\n{chunks[0]}\n\n(This is a fallback response as normal processing failed.)"
                        except Exception as e:
                            print(f"DEBUG: Fallback retrieval error: {e}")
                        finally:
                            if conn:
                                conn.close()
                                
                        return "I cannot find any information about this in the documents. Please try a different query."
                    
                    return formatted_response
                
                except Exception as query_error:
                    logger.error(f"Error during RAG query: {str(query_error)}")
                    return f"I encountered an error while processing your query: {str(query_error)}"
            else:
                logger.warning("No table available for RAG, falling back to normal response")
                # Fall through to non-RAG response
        else:
            if not available_docs:
                logger.warning("No documents available in the knowledge base")
            else:
                logger.info("Decision: RAG not needed for this query")
        
        # Process without RAG
        logger.info("Generating response without RAG")
        direct_response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        
        return direct_response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error in process_message_with_selective_rag: {str(e)}")
        return f"I encountered an error while processing your message: {str(e)}" 