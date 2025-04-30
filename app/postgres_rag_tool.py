import os
import json
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from db_config import get_pg_connection

class PostgresRAGTool:
    def __init__(
        self, 
        db_name="postgres", 
        host="34.150.190.157", 
        password=None, 
        port=5432, 
        user="llamaindex",
        openai_model="gpt-4o",
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
        
        # Get available vector tables from database
        self.available_tables = self._get_available_tables()
        self.logger.info(f"Found {len(self.available_tables)} available vector tables")
    
    def _get_available_tables(self):
        """Get a list of all available vector tables in the database"""
        try:
            conn = get_pg_connection()
            cursor = conn.cursor()
            
            # Query to find vector tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND 
                (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
            """)
            
            tables = [table[0] for table in cursor.fetchall()]
            cursor.close()
            conn.close()
            
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
                
                # Create vector store for this table
                vector_store_params = {
                    "database": self.db_name,
                    "host": self.host,
                    "password": self.password,
                    "port": self.port,
                    "user": self.user,
                    "table_name": table_name,
                    "embed_dim": self.embed_dim,
                }
                
                # Add hybrid search if enabled
                if self.use_hybrid_search:
                    vector_store_params["hybrid_search"] = True
                    vector_store_params["text_search_config"] = "english"
                
                # Create the vector store
                self.vector_stores[table_name] = PGVectorStore.from_params(**vector_store_params)
                
                # Create embedding model
                embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
                
                # Create index from vector store
                self.indices[table_name] = VectorStoreIndex.from_vector_store(
                    self.vector_stores[table_name],
                    embed_model=embed_model
                )
                
                # Create query engine
                self.query_engines[table_name] = self.indices[table_name].as_query_engine(
                    llm=self.llm,
                    similarity_top_k=3,
                    response_mode="compact"
                )
                
                self.logger.info(f"Successfully initialized query engine for table: {table_name}")
                return True
            
            except Exception as e:
                self.logger.error(f"Error initializing resources for table {table_name}: {str(e)}")
                raise
    
    def query(self, query_text, table_name=None):
        """Query the knowledge base"""
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
                        "text": node.node.get_content()[:250] + "..." if len(node.node.get_content()) > 250 else node.node.get_content(),
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
            conn = get_pg_connection()
            cursor = conn.cursor()
            
            for table_name in self.available_tables:
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
            
            cursor.close()
            conn.close()
            
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

def create_rag_tool_spec(available_files=None):
    """Create a specification for the RAG tool that the LLM can call when needed"""
    description = "Search the knowledge base for information that the model does not know. Only use this tool when you need specific information from documents or data that wouldn't be in your training data. Don't use this for general knowledge questions you can answer yourself."
    
    # Add information about available files if provided
    if available_files and len(available_files) > 0:
        files_list = ", ".join(available_files[:10])
        if len(available_files) > 10:
            files_list += f", and {len(available_files) - 10} more"
        description += f"\n\nThe knowledge base contains information from the following documents: {files_list}."
    
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
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Optional. The specific table to query. Leave blank to use the default table."
                    }
                },
                "required": ["query"]
            }
        }
    }

def process_message_with_selective_rag(user_message, rag_tool, model="gpt-4o"):
    """Process a user message, using RAG only when necessary based on LLM's decision"""
    import openai
    
    # Ensure the OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY is not set in the environment"
    
    # Get list of available files
    files_info = rag_tool.get_files_in_database()
    available_files = files_info["all_files"]
    
    logging.info(f"Processing message: '{user_message}'")
    messages = [{"role": "user", "content": user_message}]
    
    try:
        # First call to decide if RAG is needed
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=[create_rag_tool_spec(available_files)],
            tool_choice="auto"  # Let the model decide whether to use the tool
        )
        
        assistant_message = response.choices[0].message
        
        # If the model chose to use the RAG tool
        if assistant_message.tool_calls:
            logging.info("LLM decided to use the RAG tool")
            
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            
            if function_name == "query_knowledge_base":
                # Parse the function arguments
                function_args = json.loads(tool_call.function.arguments)
                query = function_args.get("query")
                table_name = function_args.get("table_name")
                
                # Call the RAG tool
                logging.info(f"Calling RAG tool with query: '{query}'")
                rag_response = rag_tool.query(query, table_name)
                
                # Add the assistant message and tool response to the conversation
                messages.append(assistant_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(rag_response)
                })
                
                # Get the final response
                logging.info("Getting final response after RAG lookup")
                final_response = openai.chat.completions.create(
                    model=model,
                    messages=messages
                )
                
                return final_response.choices[0].message.content
        
        # If the model didn't need to use the RAG tool
        logging.info("LLM decided NOT to use the RAG tool")
        return assistant_message.content
    
    except Exception as e:
        logging.error(f"Error in process_message_with_selective_rag: {str(e)}")
        return f"An error occurred: {str(e)}" 