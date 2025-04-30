import os
import json
import logging
import time # Import time

# --- Add this line for debugging file loading ---
print(f"--- Loading app/multi_table_rag.py @ {time.time()} ---")
# -----------------------------------------------

from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
from db_config import get_pg_connection

"""
IMPROVED RAG SYSTEM

This file contains improvements to the RAG system to make it more robust:

1. Enhanced document detection: Better recognition of document requests with special handling for accents and case sensitivity
2. Direct document access: Specialized methods for document summarization that bypass complex query logic
3. Improved error handling: More detailed logging and fallback mechanisms when things go wrong
4. Diagnostic tools: Verification functions to identify configuration issues
5. Multiple retrieval strategies: Trying different queries and direct database access when standard retrieval fails

The most important improvements are:
- The dedicated summarize_document method for direct document access
- Enhanced detect_document_request for better document recognition
- More logging throughout the codebase for easier debugging
"""

class MultiTableRAGTool:
    def __init__(
        self,
        db_config: Dict[str, Any] = None,
        openai_model: str = "gpt-4o"
    ):
        """
        Initialize a RAG tool that can search across multiple PostgreSQL tables.
        
        Args:
            db_config: Database connection parameters (host, port, user, password, etc.)
            openai_model: The OpenAI model to use for queries
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MultiTableRAGTool")
        
        # Set default database config if not provided
        if not db_config:
            self.db_config = {
                "database": os.getenv("PG_DB", "postgres"),
                "host": os.getenv("DB_PUBLIC_IP", "34.150.190.157"),
                "password": os.getenv("PG_PASSWORD", "password123"),
                "port": int(os.getenv("PG_PORT", 5432)),
                "user": os.getenv("PG_USER", "llamaindex"),
            }
        else:
            self.db_config = db_config
            
        self.openai_model = openai_model
        self.llm = OpenAI(model=openai_model)
        
        # Discover available tables and their metadata
        self.table_configs = self._discover_table_configs()
        self.logger.info(f"Found {len(self.table_configs)} vector tables")
        
        # Initialize components
        self.vector_stores = {}
        self.indexes = {}
        self.query_engines = {}
        
        # Initialize vector stores and indexes
        self._initialize()
    
    def _discover_table_configs(self):
        """Automatically discover vector tables and extract their metadata"""
        table_configs = []
        
        try:
            conn = get_pg_connection()
            cursor = conn.cursor()
            
            # Find all vector tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND 
                (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
            """)
            
            tables = [table[0] for table in cursor.fetchall()]
            
            # Extract metadata for each table
            for table_name in tables:
                try:
                    # Get file names in this table
                    cursor.execute(f"""
                        SELECT DISTINCT metadata_->>'file_name' as file_name 
                        FROM {table_name}
                        WHERE metadata_->>'file_name' IS NOT NULL
                    """)
                    
                    file_names = [row[0] for row in cursor.fetchall() if row[0]]
                    files_description = ", ".join(file_names) if file_names else "Unknown documents"
                    
                    # Count documents and chunks
                    cursor.execute(f"""
                        SELECT COUNT(DISTINCT metadata_->>'file_name') as doc_count,
                               COUNT(*) as chunk_count
                        FROM {table_name}
                    """)
                    
                    stats = cursor.fetchone()
                    doc_count = stats[0] if stats else 0
                    chunk_count = stats[1] if stats else 0
                    
                    # Create table config
                    table_configs.append({
                        "name": table_name,
                        "description": f"Contains {doc_count} documents: {files_description}",
                        "embed_dim": 1536,
                        "top_k": 3,
                        "hybrid_search": True,
                        "language": "english",
                        "files": file_names,
                        "doc_count": doc_count,
                        "chunk_count": chunk_count
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error extracting metadata for table {table_name}: {str(e)}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error discovering tables: {str(e)}")
        
        return table_configs
    
    def _initialize(self):
        """Initialize all vector stores and query engines"""
        for table_config in self.table_configs:
            table_name = table_config["name"]
            
            try:
                self.logger.info(f"Initializing resources for table: {table_name}")
                
                # Create vector store for this table
                vector_store_params = {
                    **self.db_config,  # Include all database config parameters
                    "table_name": table_name,
                    "embed_dim": table_config.get("embed_dim", 1536),
                    "hybrid_search": False, # <-- Force disable hybrid search for querying test
                    "text_search_config": table_config.get("language", "english"), # Explicitly set config
                }
                
                # Create embedding model
                embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
                
                self.logger.info(f"Attempting PGVectorStore.from_params for {table_name}")
                # 1. Create PGVectorStore
                self.vector_stores[table_name] = PGVectorStore.from_params(**vector_store_params)
                self.logger.info(f"SUCCESS: PGVectorStore created for {table_name}")
                
                # Create the index
                try:
                    self.logger.info(f"Attempting VectorStoreIndex.from_vector_store for {table_name}")
                    self.indexes[table_name] = VectorStoreIndex.from_vector_store(
                        self.vector_stores[table_name],
                        embed_model=embed_model,
                        store_nodes_override=True
                    )
                    self.logger.info(f"SUCCESS: VectorStoreIndex created for {table_name}")
                except Exception as e:
                    self.logger.error(f"Error loading index from table '{table_name}': {e}")
                    continue
                
                # Create query engine with metadata
                self.query_engines[table_name] = self.indexes[table_name].as_query_engine(
                    llm=self.llm,
                    similarity_top_k=table_config.get("top_k", 5),
                    response_mode="compact"
                )
                self.logger.info(f"SUCCESS: QueryEngine created for {table_name}")
                
            except Exception as e:
                self.logger.error(f"FAILED Outer Initialization for table {table_name}: {str(e)}")
    
    def query_single_table(self, table_name, query_text):
        """Query a single table"""
        self.logger.info(f"Querying table '{table_name}' with: '{query_text}'")
        
        if table_name not in self.query_engines:
            return {
                "error": f"Table '{table_name}' not found or not initialized",
                "message": "The specified table does not exist or could not be loaded"
            }
        
        try:
            response = self.query_engines[table_name].query(query_text)
            
            # --- Debug: Log source nodes ---
            self.logger.info(f"Raw response object type from query engine for table '{table_name}': {type(response)}")
            if hasattr(response, 'source_nodes') and response.source_nodes:
                self.logger.info(f"Query engine for table '{table_name}' returned {len(response.source_nodes)} source nodes.")
                for i, node in enumerate(response.source_nodes):
                    self.logger.info(f"  Node {i+1} Score: {node.score}")
                    self.logger.info(f"  Node {i+1} Metadata: {node.node.metadata}")
                    self.logger.info(f"  Node {i+1} Content Preview: {node.node.get_content()[:100]}...") # Log preview
            else:
                self.logger.warning(f"Query engine for table '{table_name}' returned NO source nodes.")
            # --- End Debug ---
            
            # Get the table config for additional info
            table_config = next((conf for conf in self.table_configs if conf["name"] == table_name), {})
            
            # Check if the synthesized answer is empty
            final_answer = str(response)
            if not final_answer or final_answer.strip() == "Empty Response":
                self.logger.warning(f"Query engine for table '{table_name}' produced an empty final answer string, despite potentially finding nodes.")
                # Optionally, return a more informative message or error here if desired
                # return {"error": "LLM synthesis failed", "message": "Retrieved nodes but couldn't generate answer."} 
            
            result = {
                "answer": final_answer, # Use the stored final_answer
                "table": table_name,
                "description": table_config.get("description", ""),
                "sources": []
            }
            
            # Add source information if available
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    source_info = {
                        "text": node.node.get_content()[:250] + "..." if len(node.node.get_content()) > 250 else node.node.get_content(),
                        "file_name": node.node.metadata.get("file_name", "unknown"),
                        "page_number": node.node.metadata.get("page_label", "unknown"),
                        "score": float(node.score) if hasattr(node, "score") else None
                    }
                    result["sources"].append(source_info)
            
            return result
        except Exception as e:
            self.logger.error(f"Error querying table '{table_name}': {str(e)}")
            return {
                "error": str(e),
                "message": f"Failed to query table '{table_name}'"
            }
    
    def query_all_tables(self, query_text):
        """Query all tables and combine results"""
        self.logger.info(f"Querying all tables with: '{query_text}'")
        all_results = []
        
        for table_config in self.table_configs:
            table_name = table_config["name"]
            result = self.query_single_table(table_name, query_text)
            if "error" not in result:
                all_results.append(result)
        
        return all_results
    
    def determine_relevant_tables(self, query_text):
        """Determine which tables are most relevant to the query"""
        if not self.table_configs or len(self.table_configs) <= 1:
            # If there's only one table or no tables, just return all
            return [config["name"] for config in self.table_configs]
            
        self.logger.info(f"Determining relevant tables for query: '{query_text}'")
        
        try:
            # Using OpenAI to determine relevance
            table_descriptions = "\n".join([
                f"- {config['name']}: {config['description']}"
                for config in self.table_configs
            ])
            
            prompt = f"""
            Given the following question and available knowledge bases, which knowledge bases 
            should be queried to provide the most relevant answer? Return only the names of 
            the knowledge bases as a comma-separated list.
            
            Question: "{query_text}"
            
            Available knowledge bases:
            {table_descriptions}
            """
            
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a system that determines which knowledge bases are relevant to a query."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            
            relevant_tables = response.choices[0].message.content.strip().split(",")
            relevant_tables = [table.strip() for table in relevant_tables]
            
            self.logger.info(f"Selected relevant tables: {relevant_tables}")
            return relevant_tables
            
        except Exception as e:
            self.logger.error(f"Error determining relevant tables: {str(e)}")
            # Fall back to using all tables if there's an error
            return [config["name"] for config in self.table_configs]
    
    def query(self, query_text):
        """
        Query only relevant tables and synthesize a comprehensive answer
        """
        self.logger.info(f"RAG query received: '{query_text}'")
        
        # No tables available
        if not self.table_configs:
            self.logger.warning("No knowledge bases found to search")
            return {
                "answer": "I couldn't find any knowledge bases to search.",
                "sources": []
            }
        
        try:
            # --- Start: Comment out direct document handling INSIDE query method ---
            # # Check for document request in the simplified format
            # # This is a faster pre-check before going into the complex detection logic
            
            # document_request = False
            # target_document = None
            
            # # Get all available files
            # files_info = self.get_files_in_database()
            # all_files = files_info["all_files"]
            
            # # Quickly check if the query directly mentions any document names
            # for file in all_files:
            #     if file.lower() in query_text.lower():
            #         document_request = True
            #         target_document = file
            #         self.logger.info(f"Direct document mention found: {target_document}")
            #         break
            
            # # If found a direct document mention, use our specialized document function
            # if document_request and target_document:
            #     self.logger.info(f"Using specialized document function for '{target_document}'")
            #     return self.summarize_document(target_document)
            # --- End: Comment out direct document handling INSIDE query method ---
            
            # Standard query processing continues below
            # Determine which tables are relevant to this query
            self.logger.info("Determining relevant tables for the query")
            relevant_tables = self.determine_relevant_tables(query_text)
            self.logger.info(f"Relevant tables: {relevant_tables}")
            
            # Query only the relevant tables
            table_results = []
            for table_name in relevant_tables:
                # Find the table config
                table_config = next((config for config in self.table_configs if config["name"] == table_name), None)
                if table_config:
                    self.logger.info(f"Querying table: {table_name}")
                    result = self.query_single_table(table_name, query_text)
                    
                    if "error" in result:
                        self.logger.error(f"Error querying table {table_name}: {result['error']}")
                    else:
                        self.logger.info(f"Successfully queried table {table_name}")
                        result["description"] = table_config.get("description", "")
                        table_results.append(result)
            
            # If no results were found
            if not table_results:
                self.logger.warning("No results found in any table")
                return {
                    "answer": "I couldn't find relevant information in any of our knowledge bases.",
                    "sources": []
                }
            
            # Compile all sources for reference
            all_sources = []
            for result in table_results:
                for source in result.get("sources", []):
                    source["table"] = result["table"]
                    all_sources.append(source)
            
            # If we only found one result, just return that
            if len(table_results) == 1:
                self.logger.info("Only one result found, returning directly")
                return {
                    "answer": table_results[0]["answer"],
                    "sources": all_sources
                }
            
            # For multiple tables, synthesize the responses
            self.logger.info(f"Synthesizing information from {len(table_results)} tables")
            synthesis_prompt = self._create_synthesis_prompt(query_text, table_results)
            
            try:
                self.logger.info("Calling OpenAI for synthesis")
                response = openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an assistant tasked with synthesizing information from multiple knowledge bases."},
                        {"role": "user", "content": synthesis_prompt}
                    ]
                )
                
                synthesized_answer = response.choices[0].message.content
                self.logger.info("Successfully synthesized answer")
                
                return {
                    "answer": synthesized_answer,
                    "sources": all_sources
                }
            except Exception as synthesis_error:
                self.logger.error(f"Error during synthesis: {str(synthesis_error)}")
                
                # Fall back to simple concatenation
                self.logger.info("Falling back to simple answer concatenation")
                combined_answer = "I found information from multiple sources:\n\n"
                for result in table_results:
                    combined_answer += f"From {result['table']}: {result['answer']}\n\n"
                
                return {
                    "answer": combined_answer,
                    "sources": all_sources,
                    "synthesis_error": str(synthesis_error)
                }
            
        except Exception as e:
            self.logger.error(f"Error in multi-table query: {str(e)}")
            self.logger.exception("Detailed error traceback:")
            
            # If we have at least some results but another error occurred
            if 'table_results' in locals() and table_results:
                # Fall back to simple concatenation
                self.logger.info("Error occurred but we have some results - falling back to simple concatenation")
                combined_answer = "I found some information, but had trouble combining it properly. Here's what I found:\n\n"
                for result in table_results:
                    combined_answer += f"From {result['table']}: {result['answer']}\n\n"
                
                return {
                    "answer": combined_answer,
                    "sources": all_sources if 'all_sources' in locals() else [],
                    "error": str(e)
                }
            
            # Complete failure
            return {
                "error": str(e),
                "message": "Failed to query knowledge base"
            }
    
    def _create_synthesis_prompt(self, query, table_results):
        """Create a prompt to synthesize multiple table results"""
        prompt = f"The user asked: '{query}'\n\n"
        prompt += "I've found information from multiple data sources:\n\n"
        
        for result in table_results:
            prompt += f"From {result['table']} ({result['description']}):\n"
            prompt += f"{result['answer']}\n\n"
        
        prompt += "Please synthesize this information into a comprehensive, coherent answer. "
        prompt += "Combine the information where appropriate and note any contradictions if they exist. "
        prompt += "If different sources provide complementary information, integrate it into a complete picture. "
        prompt += "Answer as if you were directly responding to the user's question."
        
        return prompt

    def _create_doc_summary_prompt(self, document_name, table_results):
        """Create a prompt to synthesize document content from multiple tables"""
        prompt = f"I need a detailed summary of the document: '{document_name}'\n\n"
        prompt += "I've found information from multiple sources:\n\n"
        
        for result in table_results:
            prompt += f"From {result['table']} ({result['description']}):\n"
            prompt += f"{result['answer']}\n\n"
        
        prompt += f"Please create a comprehensive, well-structured summary of the document '{document_name}'. "
        prompt += "The summary should include the main sections, key points, and important information from the document. "
        prompt += "Organize the summary into logical sections with headings if appropriate. "
        prompt += "Integrate information from all sources to create a complete picture of the document content."
        
        return prompt

    def get_tables_info(self):
        """Get information about available tables"""
        return {
            "table_count": len(self.table_configs),
            "tables": self.table_configs
        }
    
    def get_files_in_database(self):
        """Get a list of all files stored in the vector tables"""
        files_by_table = {}
        
        for table_config in self.table_configs:
            table_name = table_config["name"]
            files = table_config.get("files", [])
            if files:
                files_by_table[table_name] = files
        
        # Create a flat list of all files
        all_files = []
        for files in files_by_table.values():
            all_files.extend(files)
        
        return {
            "files_by_table": files_by_table,
            "all_files": list(set(all_files))  # Remove duplicates
        }
        
    def verify_document_vectors(self, document_name):
        """Debug function to verify vectors for a specific document"""
        self.logger.info(f"Verifying vectors for document: '{document_name}'")
        
        results = {}
        vector_counts = {}
        sample_contents = {}
        
        try:
            conn = get_pg_connection()
            cursor = conn.cursor()
            
            # Check each table for this document
            for table_name, vector_store in self.vector_stores.items():
                try:
                    # Count vectors for this document
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM {table_name}
                        WHERE metadata_->>'file_name' = %s
                    """, (document_name,))
                    
                    count = cursor.fetchone()[0]
                    vector_counts[table_name] = count
                    
                    # Get sample content if vectors exist
                    if count > 0:
                        cursor.execute(f"""
                            SELECT id, SUBSTRING(content, 1, 300) as content_preview
                            FROM {table_name}
                            WHERE metadata_->>'file_name' = %s
                            LIMIT 3
                        """, (document_name,))
                        
                        samples = cursor.fetchall()
                        sample_contents[table_name] = [
                            {"id": sample[0], "content_preview": sample[1]} 
                            for sample in samples
                        ]
                except Exception as e:
                    self.logger.error(f"Error checking table {table_name}: {str(e)}")
            
            cursor.close()
            conn.close()
            
            # Compile results
            results = {
                "document_name": document_name,
                "vector_counts": vector_counts,
                "total_vectors": sum(vector_counts.values()),
                "sample_contents": sample_contents
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in verify_document_vectors: {str(e)}")
            return {"error": str(e)}

    def verify_configuration(self):
        """Verify that the RAG tool is properly configured and working"""
        self.logger.info("Verifying RAG tool configuration")
        
        results = {
            "status": "ok",
            "details": {},
            "errors": []
        }
        
        # 1. Check basic configuration
        results["details"]["openai_model"] = self.openai_model
        results["details"]["db_config"] = {
            "host": self.db_config.get("host"),
            "port": self.db_config.get("port"),
            "database": self.db_config.get("database"),
            "user": self.db_config.get("user"),
            "password": "********" if self.db_config.get("password") else None
        }
        
        # 2. Check if we have any tables
        if not self.table_configs:
            results["status"] = "error"
            results["errors"].append("No tables found in the database")
        else:
            results["details"]["table_count"] = len(self.table_configs)
            results["details"]["tables"] = [t["name"] for t in self.table_configs]
        
        # 3. Check if tables have files
        if self.table_configs:
            files_info = self.get_files_in_database()
            files_by_table = files_info["files_by_table"]
            all_files = files_info["all_files"]
            
            results["details"]["file_count"] = len(all_files)
            
            if not all_files:
                results["status"] = "warning"
                results["errors"].append("No files found in any table")
        
        # 4. Check if vector stores are initialized
        if not self.vector_stores:
            results["status"] = "error"
            results["errors"].append("Vector stores not initialized")
        else:
            results["details"]["vector_stores_count"] = len(self.vector_stores)
            results["details"]["vector_stores"] = list(self.vector_stores.keys())
        
        # 5. Check if indexes are initialized
        if not self.indexes:
            results["status"] = "error"
            results["errors"].append("Indexes not initialized")
        else:
            results["details"]["indexes_count"] = len(self.indexes)
            results["details"]["indexes"] = list(self.indexes.keys())
        
        # 6. Check if query engines are initialized
        if not self.query_engines:
            results["status"] = "error"
            results["errors"].append("Query engines not initialized")
        else:
            results["details"]["query_engines_count"] = len(self.query_engines)
            results["details"]["query_engines"] = list(self.query_engines.keys())
        
        # 7. Check database connection
        try:
            conn = get_pg_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            results["details"]["db_connection"] = "ok"
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(f"Database connection failed: {str(e)}")
            results["details"]["db_connection"] = "failed"
        
        # 8. Check OpenAI API
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=5
            )
            results["details"]["openai_api"] = "ok"
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(f"OpenAI API test failed: {str(e)}")
            results["details"]["openai_api"] = "failed"
        
        return results
    
    def detect_document_request(self, query_text):
        """
        Detect if a query is requesting information about a specific document
        Returns (is_document_request, document_name) tuple
        """
        self.logger.info(f"Checking if query is a document request: '{query_text}'")
        
        # Check if query contains document request indicators
        doc_request_terms = [
            "document", "documento", "file", "arquivo", "pdf", 
            "manual", "guide", "guia", "summary", "summarize", 
            "resumo", "summarise", "content", "conteúdo", 
            "explain", "explique", "details", "detalhes", 
            "information", "informação", "read", "tell me about", 
            "fale sobre", "text", "texto", "resumir",
            "resuma", "resumir o", "resuma o"
        ]
        
        # Log the query for debugging
        self.logger.info(f"Processing document request query: '{query_text}'")
        
        # Check if the query contains any document request terms
        has_doc_term = False
        for term in doc_request_terms:
            if term in query_text.lower():
                has_doc_term = True
                self.logger.info(f"Found document request term: '{term}'")
                break
        
        if not has_doc_term:
            self.logger.info("Query does not contain document request terms")
            return False, None
        
        # Get all available files
        files_info = self.get_files_in_database()
        all_files = files_info["all_files"]
        
        # Debug: Log all available files
        self.logger.info(f"Available files: {all_files}")
        
        if not all_files:
            self.logger.info("No documents available to check against")
            return False, None
            
        # First check for exact filename matches (case insensitive)
        for file in all_files:
            if file.lower() in query_text.lower():
                self.logger.info(f"Found exact document match: {file}")
                return True, file
        
        # Next check for filename without extension
        for file in all_files:
            file_name_without_ext = os.path.splitext(file)[0].lower()
            if file_name_without_ext in query_text.lower():
                self.logger.info(f"Found document match without extension: {file}")
                return True, file
            
            # Alternative check replacing accents
            file_normalized = file_name_without_ext.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
            query_normalized = query_text.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
            
            if file_normalized in query_normalized:
                self.logger.info(f"Found normalized document match: {file}")
                return True, file
        
        # Check for common document patterns in the query
        if "manual" in query_text.lower():
            for file in all_files:
                if "manual" in file.lower():
                    self.logger.info(f"Found 'manual' document match: {file}")
                    return True, file
        
        # Finally check for partial matches in multi-word filenames
        for file in all_files:
            # Split filename (without extension) into words
            file_words = os.path.splitext(file)[0].lower().split()
            
            # Count words from filename found in query (ignoring very short words)
            word_matches = sum(1 for word in file_words if len(word) > 3 and word in query_text.lower())
            
            # If we matched at least 1 substantial word for a document with a descriptive name, consider it a match
            if word_matches >= 1 and len(file_words) >= 2:
                self.logger.info(f"Found partial document match: {file} (matched {word_matches} words)")
                return True, file
        
        self.logger.info("Query seems to be document-related but no specific document was identified")
        
        # If we reached here, it's potentially a document request but we couldn't identify the specific document
        # Return True to indicate it's a document request, but None for the document name
        # This allows the main processing to handle general document requests appropriately
        return True, None
    
    def summarize_document(self, document_name):
        """
        Directly summarize a specific document by name without relying on complex query logic
        """
        self.logger.info(f"Summarizing document: '{document_name}'")
        
        if not document_name:
            return {
                "error": "No document name provided",
                "message": "Please provide a specific document name to summarize"
            }
            
        # Find tables containing this document
        doc_tables = []
        files_info = self.get_files_in_database()
        
        # Log all available files for debugging
        self.logger.info(f"Available files for summarization: {files_info['all_files']}")
        
        # Step 1: Try exact match
        for table_name, files in files_info["files_by_table"].items():
            self.logger.info(f"Checking table {table_name} for document '{document_name}'")
            if document_name in files:
                doc_tables.append(table_name)
                self.logger.info(f"Found document in table {table_name}")
        
        # Step 2: If no exact match, try case-insensitive match
        if not doc_tables:
            self.logger.info("Document not found with exact match, trying case-insensitive")
            for table_name, files in files_info["files_by_table"].items():
                for file in files:
                    if document_name.lower() == file.lower():
                        doc_tables.append(table_name)
                        document_name = file  # Use the actual filename with correct case
                        self.logger.info(f"Found document with case-insensitive match in table {table_name}")
                        break
        
        # Step 3: If still no match, try partial match
        if not doc_tables:
            self.logger.info("Document not found with exact or case-insensitive match, trying partial match")
            for table_name, files in files_info["files_by_table"].items():
                for file in files:
                    if document_name.lower() in file.lower() or file.lower() in document_name.lower():
                        doc_tables.append(table_name)
                        document_name = file  # Use the actual filename
                        self.logger.info(f"Found document with partial match in table {table_name}")
                        break
        
        # If no tables found, return error
        if not doc_tables:
            return {
                "error": f"Document '{document_name}' not found",
                "message": f"The document '{document_name}' was not found in any of the vector stores"
            }
        
        # Create queries to get document content - using multiple query variations
        # to increase chances of getting good results
        queries = [
            f"Provide a comprehensive summary of the document titled '{document_name}'. Include the main topics, key points, and important information.",
            f"What are the main sections and key information in the document '{document_name}'?",
            f"Summarize the content of '{document_name}' in detail, covering all important aspects."
        ]
        
        # Get results from relevant tables with multiple queries
        table_results = []
        all_sources = []
        
        # Try each table
        for table_name in doc_tables:
            # Find the table config
            table_config = next((config for config in self.table_configs if config["name"] == table_name), None)
            if table_config:
                # Try each query until we get a good result
                for query in queries:
                    self.logger.info(f"Querying table {table_name} with query: {query}")
                    result = self.query_single_table(table_name, query)
                    if "error" not in result:
                        result["description"] = table_config.get("description", "")
                        table_results.append(result)
                        
                        # Collect sources specific to this document
                        for source in result.get("sources", []):
                            # Match sources from this document (using flexible matching)
                            if (source.get("file_name") == document_name or 
                                document_name.lower() in source.get("file_name", "").lower() or
                                source.get("file_name", "").lower() in document_name.lower()):
                                source["table"] = table_name
                                all_sources.append(source)
                                
                        # If we got a good result, no need to try more queries for this table
                        break
        
        # If no results found, try direct database query as a fallback
        if not table_results:
            self.logger.warning(f"No results found through query engine, trying direct database query")
            try:
                conn = get_pg_connection()
                cursor = conn.cursor()
                
                # Get content directly from the database - first try exact match
                for table_name in doc_tables:
                    cursor.execute(f"""
                        SELECT content
                        FROM {table_name}
                        WHERE metadata_->>'file_name' = %s
                        LIMIT 10
                    """, (document_name,))
                    
                    rows = cursor.fetchall()
                    if not rows:
                        # If no exact match, try case-insensitive
                        self.logger.info(f"Trying case-insensitive match in direct query for table {table_name}")
                        cursor.execute(f"""
                            SELECT content
                            FROM {table_name}
                            WHERE LOWER(metadata_->>'file_name') = LOWER(%s)
                            LIMIT 10
                        """, (document_name,))
                        rows = cursor.fetchall()
                    
                    if not rows:
                        # If still no match, try partial match
                        self.logger.info(f"Trying partial match in direct query for table {table_name}")
                        cursor.execute(f"""
                            SELECT content
                            FROM {table_name}
                            WHERE LOWER(metadata_->>'file_name') LIKE %s
                            LIMIT 10
                        """, (f"%{document_name.lower()}%",))
                        rows = cursor.fetchall()
                    
                    if rows:
                        direct_content = "\n\n".join([row[0] for row in rows])
                        table_results.append({
                            "answer": f"Here is content from document '{document_name}':\n\n{direct_content}",
                            "table": table_name,
                            "description": "Direct database content retrieval",
                            "sources": []
                        })
                
                cursor.close()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Error in direct database query: {str(e)}")
        
        # If still no results found
        if not table_results:
            return {
                "error": f"No content found for document '{document_name}'",
                "message": f"The document was found but no content could be retrieved"
            }
        
        # For a single result, just return it directly
        if len(table_results) == 1:
            return {
                "answer": table_results[0]["answer"],
                "sources": all_sources
            }
        
        # For multiple tables/results, synthesize results
        try:
            synthesis_prompt = self._create_doc_summary_prompt(document_name, table_results)
            
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": f"You are an assistant tasked with creating a comprehensive summary of the document: {document_name}"},
                    {"role": "user", "content": synthesis_prompt}
                ]
            )
            
            synthesized_answer = response.choices[0].message.content
            
            return {
                "answer": synthesized_answer,
                "sources": all_sources
            }
        except Exception as e:
            self.logger.error(f"Error synthesizing document summary: {str(e)}")
            
            # Fall back to simple concatenation if synthesis fails
            combined_answer = f"Summary of '{document_name}' from multiple sources:\n\n"
            for result in table_results:
                combined_answer += f"From {result['table']}: {result['answer']}\n\n"
            
            return {
                "answer": combined_answer,
                "sources": all_sources,
                "synthesis_error": str(e)
            }

def create_multi_rag_tool_spec(available_files=None):
    """Create a specification for the multi-table RAG tool"""
    description = (
        "Search external knowledge bases for specific information that the model doesn't know. "
        "Use this ONLY when the user asks about information likely contained in specific documents or data sources "
        "that are not part of your general knowledge."
    )
    
    # Add information about available files if provided
    if available_files and len(available_files) > 0:
        files_list = ", ".join([f"'{f}'" for f in available_files[:10]]) # Quote filenames
        if len(available_files) > 10:
            files_list += f", and {len(available_files) - 10} more"
        description += (
            f"\n\nThe knowledge bases contain information primarily from the following documents: {files_list}. "
            "Consult these bases if the query refers to these documents or topics likely within them."
        )
    else:
        description += "\nCurrently, the specific documents in the knowledge base are unknown, but use this tool if the query seems to require external data retrieval."

    return {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific question to ask the knowledge base. Formulate this as precisely as possible to get the most relevant information."
                    }
                },
                "required": ["query"]
            }
        }
    }

def process_message_with_multi_rag(user_message, rag_tool, model="gpt-4o"):
    """Process a user message using the multi-table RAG tool"""
    import openai
    import logging
    
    # Configure detailed logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("process_message_rag")
    
    # Ensure the OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set in the environment")
        return "Error: OPENAI_API_KEY is not set in the environment"
    
    # Get list of available files
    files_info = rag_tool.get_files_in_database()
    available_files = files_info["all_files"]
    
    logger.info(f"Processing message with multi-RAG: '{user_message}'")
    
    # --- Start: Ensure direct document handling is commented out ---
    # # Check if this is a direct document request using our document detection
    # is_doc_request, target_document = rag_tool.detect_document_request(user_message)
    
    # # Handle document request where we identified a specific document
    # if is_doc_request and target_document:
    #     logger.info(f"Processing direct document request for: {target_document}")
    #     try:
    #         # Get document summary directly
    #         rag_response = rag_tool.summarize_document(target_document)
            
    #         if "error" in rag_response:
    #             logger.error(f"Error summarizing document: {rag_response['error']}")
    #             # Fall back to general query with the LLM (continue with standard processing below)
    #         else:
    #             # Create prompt for the final response
    #             final_prompt_messages = [
    #                 {"role": "system", "content": f"You are an assistant. The user asked about '{target_document}'. You have retrieved the following summary from the knowledge base."},
    #                 {"role": "user", "content": user_message},
    #                 {"role": "assistant", "content": f"I found information about the document '{target_document}'. Here is a summary:\n\n{rag_response['answer']}\n\nUse this information to answer the user's original question."},
    #             ]
                
    #             # Get the final response
    #             logger.info("Getting final response using direct document summary")
    #             final_response = openai.chat.completions.create(
    #                 model=model,
    #                 messages=final_prompt_messages
    #             )
                
    #             return final_response.choices[0].message.content
    #     except Exception as e:
    #         logger.error(f"Error in direct document handling: {str(e)}")
    #         logger.exception("Exception details:")
    #         # Continue with standard processing if direct document handling fails
    # --- End: Ensure direct document handling is commented out ---

    # Define system prompt
    system_prompt = (
        "You are a helpful assistant. Your primary goal is to answer the user's questions accurately. "
        "You have access to an external knowledge base via the 'query_knowledge_base' tool. "
        "Use this tool ONLY if the user's query asks for specific information that seems likely to be found in the documents mentioned in the tool description "
        "(e.g., manuals, specific reports) or if the query explicitly asks to consult external documents. "
        "Otherwise, answer based on your general knowledge."
    )

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        # First call to decide if RAG is needed
        logger.info("Making initial call to LLM to decide if RAG tool is needed")
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=[create_multi_rag_tool_spec(available_files)],
            tool_choice="auto"  # Let the model decide whether to use the tool
        )
        
        assistant_message = response.choices[0].message
        
        # If the model chose to use the RAG tool
        if assistant_message.tool_calls:
            logger.info("LLM decided to use the multi-RAG tool")
            
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            
            if function_name == "query_knowledge_base":
                # Parse the function arguments
                function_args = json.loads(tool_call.function.arguments)
                query = function_args.get("query")
                
                # Call the RAG tool
                logger.info(f"Calling multi-RAG tool with query: '{query}'")
                rag_response = rag_tool.query(query)
                
                # Add the assistant message and tool response to the conversation
                messages.append(assistant_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(rag_response)
                })
                
                # Get the final response
                logger.info("Getting final response after multi-RAG lookup")
                final_response = openai.chat.completions.create(
                    model=model,
                    messages=messages
                )
                
                return final_response.choices[0].message.content
        
        # If the model didn't need to use the RAG tool
        logger.info("LLM decided NOT to use the multi-RAG tool")
        return assistant_message.content
    
    except Exception as e:
        logger.error(f"Error in process_message_with_multi_rag: {str(e)}")
        logger.exception("Detailed error traceback:")
        return f"An error occurred: {str(e)}" 