import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.storage_context import StorageContext
import tempfile
from openai import OpenAI

# Extensões suportadas por default
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".pptx", ".md", ".csv"]

def is_supported_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in SUPPORTED_EXTENSIONS

def process_uploaded_file(file, session_id):
    # Ensure OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não está definida no ambiente.")
    
    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded file to the temporary directory
        filepath = os.path.join(temp_dir, file.name)
        with open(filepath, "wb") as f:
            f.write(file.getvalue())
        
        # Load documents using LlamaIndex
        docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
        
        # Connect to pgvector database
        db_public_ip = os.getenv("DB_PUBLIC_IP")
        
        # Determine connection method
        if os.path.exists("/cloudsql"):
            # In Cloud Run with private connectivity
            instance_connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
            if instance_connection_name:
                host = f"/cloudsql/{instance_connection_name}"
                port = None
            else:
                # Fallback to public IP if instance connection name is not set
                host = db_public_ip or "localhost"
                port = int(os.getenv("PG_PORT", 5432))
        else:
            # Local development or using public IP
            host = db_public_ip or "localhost"
            port = int(os.getenv("PG_PORT", 5432))
        
        # Create table name with session ID to prevent collisions
        table_name = f"vectors_{session_id.replace('-', '_')}"
        
        # Set up PGVectorStore
        vector_store = PGVectorStore.from_params(
            database=os.getenv("PG_DB", "postgres"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD"),
            host=host,
            port=port,
            table_name=table_name
        )
        
        # Create embedding model with the OpenAI API key
        # Create client first to avoid "proxies" error
        client = OpenAI(api_key=openai_api_key)
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Create service context with the embedding model
        service_context = ServiceContext.from_defaults(
            embed_model=embed_model
        )
        
        # Create storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from documents
        index = VectorStoreIndex.from_documents(
            docs, 
            storage_context=storage_context,
            service_context=service_context
        )
        
        return index
