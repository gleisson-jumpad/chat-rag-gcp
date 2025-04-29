import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.storage_context import StorageContext

# Extensões suportadas por default
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".pptx", ".md", ".csv"]

def is_supported_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in SUPPORTED_EXTENSIONS

def process_uploaded_file(file, session_id):
    # Salvar arquivo temporariamente
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.name)
    with open(filepath, "wb") as f:
        f.write(file.getvalue())

    # Ler documentos via LlamaIndex
    docs = SimpleDirectoryReader(input_files=[filepath]).load_data()

    # Conectar com pgvector
    vector_store = PGVectorStore.from_params(
        database=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        host=os.getenv("PG_HOST"),
        port=int(os.getenv("PG_PORT", 5432)),
        table_name=f"vectors_{session_id}"
    )

    # Criar contexto com embeddings da OpenAI
    service_context = ServiceContext.from_defaults(
        embed_model=OpenAIEmbedding()
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, service_context=service_context)

    return index
