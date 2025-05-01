import os
import logging
from dotenv import load_dotenv
from llama_index.core.vector_stores import VectorStoreQuery, ExactMatchFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from db_config import get_pg_connection # Import the connection function
import json # For pretty printing JSON

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("query_test")

# Load environment variables from .env file in the app directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
logger.info(f"Carregando variáveis do arquivo .env do diretório {os.path.dirname(__file__)}")
load_dotenv(dotenv_path=dotenv_path)

# Verify if OPENAI_API_KEY is loaded (optional, for debugging)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    logger.info(f"OPENAI_API_KEY está definida (primeiros 8 caracteres: {api_key[:8]}...)")
else:
    logger.error("OPENAI_API_KEY não encontrada nas variáveis de ambiente.")
    # Decide if you want to exit or continue
    # exit(1) 

# Import the RAG tool *after* loading environment variables
try:
    from multi_table_rag import MultiTableRAGTool
except ImportError as e:
    logger.error(f"Erro ao importar MultiTableRAGTool: {e}")
    logger.error("Certifique-se de que o script está sendo executado do diretório raiz do projeto ou ajuste o PYTHONPATH.")
    exit(1)

# --- Add function to execute direct SQL ---
def execute_direct_sql(query: str):
    logger.info(f"Executando SQL direto: {query[:100]}...")
    conn = None
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        
        if cursor.description:
            column_names = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            logger.info(f"SQL retornou {len(results)} linhas.")
            return results, column_names
        else:
            logger.info("SQL não retornou resultados (possivelmente um comando DDL/DML).")
            conn.commit() # Commit if it was an action query
            return [], None
            
    except Exception as e:
        logger.error(f"Erro ao executar SQL direto: {e}")
        if conn:
            conn.rollback() # Rollback on error
        return None, None
    finally:
        if conn:
            cursor.close()
            conn.close()
# -----------------------------------------

def run_test_query():
    logger.info("Inicializando MultiTableRAGTool...")
    try:
        # Initialize the RAG tool
        rag_tool = MultiTableRAGTool()
        logger.info("MultiTableRAGTool inicializado com sucesso.")

        # --- Define your test query here ---
        query_text = "Generative AI" # Simplified query
        # ---------------------------------

        logger.info(f"Executando consulta via Query Engine: '{query_text}'")

        # Execute the query using the high-level RAG tool's query method
        result = rag_tool.query(query_text)

        logger.info("Consulta via Query Engine concluída.")

        # Print the high-level results (we expect this to be empty based on previous runs)
        print("\n--- Resultado da Consulta (High-Level Query Engine) ---")
        if isinstance(result, dict):
            if "error" in result:
                print(f"Erro: {result.get('error')}")
                if "message" in result:
                    print(f"Mensagem: {result.get('message')}")
            else:
                print(f"Resposta:\n{result.get('answer', 'Nenhuma resposta gerada.')}")
                print("\n--- Fontes ---")
                sources = result.get('sources', [])
                if sources:
                    for i, source in enumerate(sources):
                        print(f"Fonte {i+1}:")
                        print(f"  Arquivo: {source.get('file_name', 'N/A')}")
                        print(f"  Tabela: {source.get('table', 'N/A')}")
                        print(f"  Score: {source.get('score', 'N/A')}")
                        print(f"  Trecho: {source.get('text', 'N/A')[:200]}...") # Show preview
                        print("---")
                else:
                    print("Nenhuma fonte encontrada.")
        else:
            print("Formato de resultado inesperado:")
            print(result)

        # --- Direct Vector Store Query Test ---
        logger.info("\n--- Iniciando teste de consulta direta ao Vector Store --- ")
        target_table = None
        # Find a table that likely contains the relevant doc (adjust if needed)
        for config in rag_tool.table_configs:
            if "Getting Access to Generative AI.pdf" in config.get("files", []):
                target_table = config["name"]
                break
            elif "llamaindex.pdf" in config.get("files", []): # Fallback
                target_table = config["name"]
                break

        if not target_table and rag_tool.table_configs:
            target_table = rag_tool.table_configs[0]["name"] # Just use the first table if no specific one found

        if target_table and target_table in rag_tool.vector_stores:
            logger.info(f"Testando consulta direta na tabela: {target_table}")
            vector_store = rag_tool.vector_stores[target_table]

            # Create embedding for the query
            logger.info(f"Gerando embedding para a consulta: '{query_text}'")
            embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
            query_embedding = embed_model.get_query_embedding(query_text)
            logger.info(f"Embedding gerado (primeiros 10 valores): {query_embedding[:10]}...")

            # Create a low-level vector store query
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=5, # Ask for top 5 directly
                mode="default" # Or try "hybrid" if enabled in your store
            )

            logger.info(f"Executando consulta direta ao vector_store para a tabela {target_table}")
            vs_result = vector_store.query(vector_store_query)

            print("\n--- Resultado da Consulta (Direct Vector Store) ---")
            if vs_result and vs_result.nodes:
                print(f"Consulta direta encontrou {len(vs_result.nodes)} nós.")
                for i, node in enumerate(vs_result.nodes):
                    print(f"Nó Direto {i+1}: Score={vs_result.similarities[i] if vs_result.similarities else 'N/A'}")
                    print(f"  Metadata: {node.metadata}")
                    print(f"  Conteúdo Preview: {node.get_content()[:100]}...")
            else:
                print("Consulta direta NÃO encontrou nós.")
                logger.warning(f"Consulta direta ao vector_store para {target_table} não retornou nós.")
                # Log the result object itself for inspection
                logger.info(f"Objeto de resultado da consulta direta: {vs_result}")

        else:
            logger.warning("Não foi possível encontrar uma tabela de destino ou vector store para o teste direto.")
        # --- End Direct Vector Store Query Test ---

        # --- Execute Direct SQL to Check Content ---
        logger.info("\n--- Iniciando teste de verificação de conteúdo via SQL --- ")
        target_table_sql = "data_vectors_06bacfb6_7a3d_404f_a461_73358b4dc1d5" # The new table name
        target_file_sql = "llamaindex.pdf"
        sql_query = f"""\
            SELECT \
                id, \
                metadata_, \
                content \
            FROM \
                {target_table_sql} \
            WHERE \
                metadata_->>'file_name' = '{target_file_sql}' \
            LIMIT 5;\
        """
        
        sql_results, column_names = execute_direct_sql(sql_query)
        
        print("\n--- Resultado da Verificação de Conteúdo SQL --- ")
        if sql_results is not None:
            if sql_results:
                print(f"Encontradas {len(sql_results)} linhas para '{target_file_sql}' na tabela '{target_table_sql}'.")
                print(f"Colunas: {column_names}")
                print("Amostra de Dados:")
                for i, row in enumerate(sql_results):
                    print(f"--- Linha {i+1} ---")
                    row_dict = dict(zip(column_names, row))
                    print(f"  ID: {row_dict.get('id')}")
                    # Pretty print metadata
                    metadata_json = row_dict.get('metadata_')
                    if metadata_json:
                         print(f"  Metadata: {json.dumps(metadata_json, indent=2)}")
                    else:
                         print(f"  Metadata: None")
                    print(f"  Content Preview: {str(row_dict.get('content', ''))[:200]}...") # Show preview
            else:
                print(f"Nenhuma linha encontrada para '{target_file_sql}' na tabela '{target_table_sql}'. A tabela pode estar vazia ou o nome do arquivo está incorreto no metadados.")
                # Check if the table exists at all
                check_table_sql = f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{target_table_sql}');"
                exists_result, _ = execute_direct_sql(check_table_sql)
                if exists_result and exists_result[0][0]:
                     print(f"(A tabela '{target_table_sql}' existe, mas não contém dados para o arquivo especificado.)")
                else:
                     print(f"(A tabela '{target_table_sql}' NÃO existe.)")
        else:
            print("Ocorreu um erro ao executar a consulta SQL. Verifique os logs.")
        # ------------------------------------------

        # --- Metadata Filter Retrieval Test ---
        logger.info("\n--- Iniciando teste de recuperação por filtro de metadados --- ")
        target_table_filter = "data_vectors_5286662f_dd5c_4d20_8a9c_f58d90669040" # The latest table name
        target_file_filter = "llamaindex.pdf"

        if target_table_filter in rag_tool.indexes:
            logger.info(f"Testando recuperação via filtro na tabela: {target_table_filter}")
            index = rag_tool.indexes[target_table_filter]
            
            # Create a retriever with a metadata filter
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="file_name", value=target_file_filter)]
            )
            retriever = index.as_retriever(
                similarity_top_k=5, # Keep this reasonably high
                filters=filters,
                vector_store_query_mode="default" # Ensure we aren't forcing hybrid here
            )
            
            # Retrieve nodes using the filter
            # Use a simple query text, as the filter should dominate
            logger.info(f"Executando retriever.retrieve para file_name='{target_file_filter}'")
            try:
                retrieved_nodes = retriever.retrieve("dummy query for filter") 
            except Exception as retrieve_err:
                 logger.error(f"Erro durante retriever.retrieve: {retrieve_err}")
                 retrieved_nodes = []

            print("\n--- Resultado da Recuperação por Filtro de Metadados --- ")
            if retrieved_nodes:
                print(f"Recuperação por filtro encontrou {len(retrieved_nodes)} nós.")
                for i, node in enumerate(retrieved_nodes):
                    print(f"Nó Filtrado {i+1}: Score={node.score}") # Score might be None for pure filter
                    print(f"  Metadata: {node.metadata}")
                    # Access text via node.text (VectorStoreIndex handles the column mapping internally now)
                    print(f"  Conteúdo Preview: {node.text[:100]}...")
            else:
                print(f"Recuperação por filtro NÃO encontrou nós para file_name='{target_file_filter}'.")
                logger.warning(f"Recuperação por filtro não retornou nós para {target_file_filter} na tabela {target_table_filter}")
        else:
            logger.warning(f"Índice para a tabela {target_table_filter} não encontrado para teste de filtro.")
        # --- End Metadata Filter Retrieval Test ---

    except Exception as e:
        logger.error(f"Erro durante a execução do teste de consulta: {e}")
        logger.exception("Traceback detalhado:")

if __name__ == "__main__":
    run_test_query() 