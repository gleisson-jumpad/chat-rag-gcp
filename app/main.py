import streamlit as st
import os
import uuid
from rag_utils import process_uploaded_file, is_supported_file, SUPPORTED_EXTENSIONS, get_existing_tables
import time
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.core.query_engine import RouterQueryEngine

st.set_page_config(page_title="Chat RAG", layout="wide")

st.title("💬 Chat RAG – Jumpad")

# Initialize session state for storing uploaded files info
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar menu
menu = st.sidebar.radio("Navegação", [
    "🏠 Início",
    "🔌 Teste de Conexão com PostgreSQL",
    "📥 Upload e Vetorização de Arquivos",
    "🔍 Consulta com RAG",
    "🧪 Diagnóstico Avançado",
])

# Páginas do menu
if menu == "🏠 Início":
    st.subheader("Bem-vindo ao sistema de RAG da Jumpad!")
    st.markdown("Use o menu lateral para navegar entre testes e funcionalidades do sistema.")
    
    # Add explanation about LlamaIndex and OpenAI API integration
    st.markdown("## Sobre a Integração LlamaIndex com OpenAI API")
    st.markdown("""
    Este sistema utiliza LlamaIndex com integração direta à API da OpenAI para:
    
    1. **Processamento de Documentos**: Convertemos seus documentos em chunks e criamos embeddings usando o modelo `text-embedding-ada-002` da OpenAI.
    
    2. **Armazenamento de Vetores**: Os embeddings são armazenados em um banco PostgreSQL com suporte a vetores.
    
    3. **Consulta Semântica**: Suas perguntas são convertidas em vetores e comparadas aos documentos através de busca por similaridade.
    
    4. **Geração Aumentada**: O LLM da OpenAI (GPT-4o ou GPT-3.5 Turbo) recebe os fragmentos relevantes de documento para gerar respostas precisas.
    """)

elif menu == "🔌 Teste de Conexão com PostgreSQL":
    st.subheader("🔌 Teste de Conexão com o Banco")
    try:
        from db_config import get_pg_connection
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        st.success("✅ Conexão estabelecida com sucesso!")
        st.code(version)
        cursor.close()
        conn.close()
    except Exception as e:
        st.error("❌ Erro ao conectar ao banco de dados:")
        st.code(str(e))

elif menu == "📥 Upload e Vetorização de Arquivos":
    st.subheader("📥 Upload e Vetorização de Arquivos")
    
    # Check if we need to import existing documents
    if len(st.session_state.uploaded_files) == 0:
        st.info("📋 Verificando documentos existentes no banco de dados...")
        try:
            # Get document tables using our utility function
            vector_tables = get_existing_tables()
            
            if vector_tables:
                # Connect to database to get document info
                from db_config import get_pg_connection
                conn = get_pg_connection()
                cursor = conn.cursor()
                
                doc_count = 0
                unique_files = set()
                
                for table_name in vector_tables:
                    try:
                        # Query to extract file names from metadata_
                        cursor.execute(f"""
                            SELECT DISTINCT metadata_->>'file_name' as file_name 
                            FROM {table_name}
                            WHERE metadata_->>'file_name' IS NOT NULL
                        """)
                        
                        # Get unique filenames
                        distinct_files = cursor.fetchall()
                        
                        if distinct_files:
                            for file_info in distinct_files:
                                file_name = file_info[0] if file_info[0] else f"Documento {doc_count+1}"
                                
                                # Only add if we haven't seen this file before
                                if file_name not in unique_files:
                                    unique_files.add(file_name)
                                    
                                    # Add document info to session state
                                    st.session_state.uploaded_files.append({
                                        'name': file_name,
                                        'size': 0,
                                        'table': table_name,
                                        'file_id': f"{table_name}_{doc_count}"
                                    })
                                    doc_count += 1
                    except Exception as e:
                        st.warning(f"Erro ao extrair informações da tabela {table_name}: {str(e)}")
                
                cursor.close()
                conn.close()
                
                st.success(f"✅ Encontrados {len(st.session_state.uploaded_files)} documentos processados!")
            else:
                st.warning("Nenhuma tabela de vetores encontrada no banco de dados.")
        except Exception as e:
            st.warning(f"Não foi possível verificar documentos existentes: {str(e)}")
    
    # Display supported file types
    st.info(f"Formatos suportados: {', '.join(SUPPORTED_EXTENSIONS)}")
    
    # File uploader
    uploaded_file = st.file_uploader("Escolha um arquivo para processar", type=[ext[1:] for ext in SUPPORTED_EXTENSIONS])
    
    if uploaded_file is not None:
        # Show file details
        st.write(f"**Arquivo:** {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Process button
        if st.button("Processar e Vetorizar"):
            with st.spinner("Processando documento e gerando vetores..."):
                try:
                    # Use the utility function from rag_utils.py
                    index = process_uploaded_file(uploaded_file, st.session_state.session_id)
                    
                    # Store file info in session state
                    if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                        st.session_state.uploaded_files.append({
                            'name': uploaded_file.name,
                            'size': uploaded_file.size,
                            'table': f"vectors_{st.session_state.session_id}"
                        })
                    
                    st.success(f"✅ Documento '{uploaded_file.name}' processado e vetorizado com sucesso!")
                    st.info(f"Vetores armazenados na tabela: vectors_{st.session_state.session_id}")
                except Exception as e:
                    st.error(f"❌ Erro ao processar o documento: {str(e)}")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("Arquivos Processados")
        for idx, file_info in enumerate(st.session_state.uploaded_files):
            st.write(f"{idx+1}. **{file_info['name']}** - Tabela: {file_info['table']}")

elif menu == "🔍 Consulta com RAG":
    st.subheader("💬 ChatRAG Multi-Tabela")
    
    # Initialize chat history if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    # Initialize multi-table RAG tool if not already done
    if 'multi_rag_tool' not in st.session_state:
        from multi_table_rag import MultiTableRAGTool
        with st.spinner("Inicializando ferramenta RAG multi-tabela..."):
            try:
                # Validate OpenAI API key first
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    st.error("❌ OPENAI_API_KEY não está definida no ambiente.")
                    st.info("Configure a chave API da OpenAI nas variáveis de ambiente ou no arquivo .env")
                    st.stop()
                
                # Initialize the RAG tool
                st.session_state.multi_rag_tool = MultiTableRAGTool()
                
                # Run database checks
                db_status = st.session_state.multi_rag_tool.check_postgres_connection()
                if not db_status.get("postgres_connection", False):
                    st.error(f"❌ Falha na conexão com PostgreSQL: {db_status.get('error', 'Erro desconhecido')}")
                    st.warning("Verifique as configurações do banco de dados e a conexão com a internet.")
                    st.stop()
                
                if not db_status.get("pgvector_installed", False):
                    st.warning("⚠️ Extensão pgvector não está instalada no PostgreSQL.")
                    st.info("As operações vetoriais não funcionarão sem essa extensão.")
                
                # Check for vector tables
                if db_status.get("vector_table_count", 0) == 0:
                    st.warning("⚠️ Nenhuma tabela de vetores encontrada no banco de dados.")
                    st.info("Faça upload de documentos na seção 'Upload e Vetorização de Arquivos' primeiro.")
                
            except Exception as init_error:
                st.error(f"❌ Erro ao inicializar MultiTableRAGTool: {init_error}")
                st.warning("Verifique as configurações do banco de dados e a chave da API OpenAI.")
                st.stop() # Stop execution if initialization fails
    
    # Display model selection in the sidebar
    with st.sidebar:
        st.subheader("Configurações do Chat")
        
        # --- Add Refresh Button ---
        if st.button("🔄 Recarregar Ferramenta RAG", help="Força a reinicialização da ferramenta RAG para descobrir novas tabelas/documentos."):
            if 'multi_rag_tool' in st.session_state:
                del st.session_state['multi_rag_tool'] # Remove old instance
            st.success("Ferramenta RAG reinicializada. Recarregando a página...")
            time.sleep(1) # Brief pause
            st.rerun()
        st.markdown("---")
        # ------------------------
        
        # Model selection with cleaner layout
        st.markdown("**Modelo de IA:**")
        model_options = {
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
            "GPT-4o": "gpt-4o",
            "GPT-4o-mini": "gpt-4o-mini"
        }
        selected_model = st.radio("", list(model_options.keys()), index=1)
        model_id = model_options[selected_model]
        
        st.markdown("---")
        
        # Get table information for display
        tables_info = []
        for table_config in st.session_state.multi_rag_tool.table_configs:
            tables_info.append({
                "name": table_config["name"],
                "docs": table_config.get("doc_count", "?"),
                "chunks": table_config.get("chunk_count", "?"),
                "files": table_config.get("files", [])
            })
        
        # Display tables information in sidebar
        if tables_info:
            st.markdown("**Tabelas de Conhecimento:**")
            for table in tables_info:
                with st.expander(f"{table['name']} ({table['docs']} docs)"):
                    st.write(f"Chunks: {table['chunks']}")
                    if table['files']:
                        st.markdown("**Arquivos:**")
                        for file in table['files']:
                            st.markdown(f"- {file}")
        else:
            st.info("Nenhuma tabela de dados encontrada")
            st.markdown("Faça o upload de documentos na seção **Upload e Vetorização de Arquivos**")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Digite sua pergunta..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Procurando nos documentos..."):
                try:
                    # Create a robust system prompt for better RAG
                    system_prompt = """
                    Você é um assistente de IA especializado em responder perguntas com base em documentos específicos. 
                    Analise cuidadosamente as informações fornecidas pela ferramenta RAG e utilize apenas essas informações ao responder.
                    
                    Diretrizes importantes:
                    - Responda SOMENTE com base nos documentos fornecidos
                    - Se a informação não estiver clara nos documentos, admita que não sabe
                    - Não invente ou infira informações além do que está explicitamente nos documentos
                    - Seja direto e conciso em suas respostas
                    """
                    
                    # Process the message with RAG integration
                    from postgres_rag_tool import process_message_with_selective_rag
                    
                    response = process_message_with_selective_rag(
                        prompt, 
                        st.session_state.multi_rag_tool, 
                        model=model_id
                    )
                    
                    # Check for empty responses
                    if not response or response.strip() == "":
                        response = "Não consegui encontrar informações relevantes nos documentos para responder essa pergunta."
                    
                    # Display the response
                    st.markdown(response)
                    
                except Exception as e:
                    error_msg = f"❌ Erro ao processar a consulta: {str(e)}"
                    st.error(error_msg)
                    response = error_msg
            
            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

elif menu == "🧪 Diagnóstico Avançado":
    st.subheader("🧪 Diagnóstico do Sistema RAG")
    
    # Check OpenAI API key
    api_key_tab, db_tab, vector_tab = st.tabs(["API OpenAI", "Banco de Dados", "Tabelas Vetoriais"])
    
    with api_key_tab:
        st.subheader("Verificação da API da OpenAI")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            # Check if key starts with "sk-"
            if openai_api_key.startswith("sk-"):
                # Mask the key
                masked_key = f"sk-...{openai_api_key[-4:]}"
                st.success(f"✅ Chave da API OpenAI configurada: {masked_key}")
                
                # Try a simple API call
                try:
                    import openai
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Test connection"}],
                        max_tokens=5
                    )
                    st.success("✅ Conexão com a API OpenAI testada com sucesso!")
                except Exception as e:
                    st.error(f"❌ Erro ao conectar à API OpenAI: {str(e)}")
            else:
                st.warning("⚠️ A chave da API OpenAI não parece estar em formato válido (deve começar com 'sk-')")
        else:
            st.error("❌ A chave da API OpenAI não está configurada!")
            st.info("Configure a variável de ambiente OPENAI_API_KEY ou adicione ao arquivo .env")
    
    with db_tab:
        st.subheader("Diagnóstico do Banco de Dados")
        
        with st.spinner("Verificando conexão com PostgreSQL..."):
            try:
                from db_config import get_pg_connection
                conn = get_pg_connection()
                cursor = conn.cursor()
                
                # Check PostgreSQL version
                cursor.execute("SELECT version();")
                pg_version = cursor.fetchone()[0]
                st.success(f"✅ Conexão PostgreSQL: {pg_version}")
                
                # Check pgvector extension
                cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
                pgvector_info = cursor.fetchone()
                
                if pgvector_info:
                    st.success(f"✅ Extensão pgvector instalada (versão {pgvector_info[1]})")
                    
                    # Check if the extension is working correctly
                    try:
                        cursor.execute("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector;")
                        distance = cursor.fetchone()[0]
                        st.success(f"✅ Operações vetoriais funcionando corretamente (distância teste: {distance:.4f})")
                    except Exception as ve:
                        st.error(f"❌ Erro ao testar operações vetoriais: {str(ve)}")
                else:
                    st.error("❌ Extensão pgvector NÃO está instalada!")
                    st.info("A extensão 'pgvector' é necessária para operações de pesquisa vetorial.")
                    st.code("CREATE EXTENSION vector;", language="sql")
                
                cursor.close()
                conn.close()
                
            except Exception as e:
                st.error(f"❌ Erro ao conectar ao banco de dados: {str(e)}")
                
    with vector_tab:
        st.subheader("Tabelas Vetoriais")
        
        with st.spinner("Verificando tabelas vetoriais..."):
            try:
                # Initialize RAG tool if not already done
                if 'multi_rag_tool' not in st.session_state:
                    from multi_table_rag import MultiTableRAGTool
                    try:
                        st.session_state.multi_rag_tool = MultiTableRAGTool()
                    except Exception as init_error:
                        st.error(f"❌ Erro ao inicializar MultiTableRAGTool: {init_error}")
                
                # Display tables information
                if hasattr(st.session_state, 'multi_rag_tool'):
                    tables = st.session_state.multi_rag_tool.table_configs
                    
                    if not tables:
                        st.warning("⚠️ Nenhuma tabela vetorial encontrada no banco de dados.")
                        st.info("Faça upload de documentos na seção 'Upload e Vetorização de Arquivos'.")
                    else:
                        st.success(f"✅ Encontradas {len(tables)} tabelas vetoriais")
                        
                        for table in tables:
                            with st.expander(f"Tabela: {table['name']}"):
                                st.write(f"**Descrição**: {table['description']}")
                                st.write(f"**Documentos**: {table.get('doc_count', '?')}")
                                st.write(f"**Chunks**: {table.get('chunk_count', '?')}")
                                st.write(f"**Dimensão dos vetores**: {table.get('embed_dim', 1536)}")
                                st.write(f"**Pesquisa híbrida**: {'Habilitada' if table.get('hybrid_search', False) else 'Desabilitada'}")
                                
                                if table.get('files'):
                                    st.write("**Arquivos:**")
                                    for file in table['files']:
                                        st.write(f"- {file}")
                
                # Option to run a test query
                if st.button("Executar consulta de teste"):
                    test_query = "O que é RAG e como funciona?"
                    
                    with st.spinner(f"Executando consulta de teste: '{test_query}'"):
                        try:
                            response = st.session_state.multi_rag_tool.query(test_query)
                            st.json(response)
                        except Exception as qe:
                            st.error(f"❌ Erro ao executar consulta de teste: {str(qe)}")
                
            except Exception as e:
                st.error(f"❌ Erro ao verificar tabelas vetoriais: {str(e)}")
