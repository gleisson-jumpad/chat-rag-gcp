import streamlit as st
import os
import uuid
import time
from rag_utils import process_uploaded_file, is_supported_file, SUPPORTED_EXTENSIONS, get_existing_tables
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.core.query_engine import RouterQueryEngine

st.set_page_config(page_title="Jumpad RAG", layout="wide")

# Custom CSS to widen the sidebar and improve file display
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 330px;
        max-width: 400px;
    }
    
    /* Improve document list appearance */
    .doc-item {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        padding: 5px;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    
    /* Auto-hide success messages */
    .auto-dismiss {
        animation: fadeOut 5s forwards;
        -webkit-animation: fadeOut 5s forwards;
    }
    
    @keyframes fadeOut {
        0% {opacity: 1;}
        80% {opacity: 1;}
        100% {opacity: 0; display: none;}
    }
    
    @-webkit-keyframes fadeOut {
        0% {opacity: 1;}
        80% {opacity: 1;}
        100% {opacity: 0; display: none;}
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 Jumpad RAG")

# Initialize session state for storing uploaded files info
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar menu - reorganized to make Consulta RAG the default
menu = st.sidebar.radio("Navegação", [
    "🏠 Jumpad RAG",
    "🛠️ Ferramentas",
], key="main_menu")

# Add a submenu for tools if that option is selected
if menu == "🛠️ Ferramentas":
    tools_menu = st.sidebar.selectbox("Selecione uma ferramenta:", [
        "📥 Upload e Vetorização de Arquivos",
        "🔌 Teste de Conexão com PostgreSQL",
        "🧪 Diagnóstico Avançado",
    ], key="tools_menu")

# Map the selected menu/submenu to the original page references
if menu == "🏠 Jumpad RAG":
    current_page = "🔍 Consulta com RAG"
elif menu == "🛠️ Ferramentas":
    current_page = tools_menu
else:
    current_page = menu

# Add a global status message area at the top of the page
def show_status_message():
    """Display status messages in the top right corner"""
    if 'global_status' in st.session_state and 'global_status_type' in st.session_state:
        # Create a container that positions the message at the top right
        container = st.container()
        with container:
            # Use columns to position the message at the right
            cols = st.columns([3, 1])
            with cols[1]:
                # Check if we're in a processing state
                is_processing = st.session_state.global_status_type == 'processing'
                
                if is_processing:
                    # Show a spinner icon with message for processing state
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; background-color: #004080; padding: 10px; border-radius: 5px; border: 1px solid #0066cc; max-width: 100%;">
                        <div class="stSpinner" style="margin-right: 10px;">
                            <svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <style>
                                    .spinner_9y7u {{
                                        transform-origin: center;
                                        animation: spinner_5meh .75s infinite linear;
                                    }}
                                    @keyframes spinner_5meh {{
                                        100% {{ transform: rotate(360deg) }}
                                    }}
                                </style>
                                <path class="spinner_9y7u" d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z" opacity=".25" fill="white"/>
                                <path class="spinner_9y7u" d="M12,4a8,8,0,0,1,7.89,6.7A1.53,1.53,0,0,0,21.38,12h0a1.5,1.5,0,0,0,1.48-1.75,11,11,0,0,0-21.72,0A1.5,1.5,0,0,0,2.62,12h0a1.53,1.53,0,0,0,1.49-1.3A8,8,0,0,1,12,4Z" fill="white"/>
                            </svg>
                        </div>
                        <span style="color: white; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{st.session_state.global_status}</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif st.session_state.global_status_type == 'success':
                    st.success(st.session_state.global_status)
                elif st.session_state.global_status_type == 'error':
                    st.error(st.session_state.global_status)
                elif st.session_state.global_status_type == 'info':
                    st.info(st.session_state.global_status)
                elif st.session_state.global_status_type == 'warning':
                    st.warning(st.session_state.global_status)
            
            # Auto-clear status after 5 seconds if requested
            if 'global_status_auto_clear' in st.session_state and st.session_state.global_status_auto_clear:
                # Add a simple CSS-based approach to auto-dismiss
                st.markdown("""
                <style>
                    .auto-dismiss {
                        animation: fadeOut 5s forwards;
                    }
                    @keyframes fadeOut {
                        0% {opacity: 1;}
                        80% {opacity: 1;}
                        100% {opacity: 0; display: none;}
                    }
                </style>
                <script>
                    setTimeout(function() {
                        const alerts = window.parent.document.querySelectorAll('.stAlert');
                        alerts.forEach(function(alert) {
                            alert.classList.add('auto-dismiss');
                        });
                    }, 500);
                </script>
                """, unsafe_allow_html=True)
                st.session_state.global_status_auto_clear = False

# Páginas do menu
if current_page == "🔍 Consulta com RAG":
    # Display status messages at the top
    show_status_message()
    
    # Initialize chat history if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Check if we need to refresh after file operations
    if 'needs_refresh' in st.session_state and st.session_state.needs_refresh:
        from multi_table_rag import MultiTableRAGTool
        with st.spinner("Reinicializando após atualização do banco de dados..."):
            try:
                # Reinitialize the RAG tool
                st.session_state.multi_rag_tool = MultiTableRAGTool()
                st.success("✅ Sistema atualizado com sucesso!")
                # Reset the refresh flag
                st.session_state.needs_refresh = False
            except Exception as e:
                st.error(f"❌ Erro ao reinicializar o sistema: {str(e)}")
    
    # Handle the file list refresh (less intrusive)
    elif 'file_list_refresh' in st.session_state and st.session_state.file_list_refresh:
        # Show notification about the removed file
        if 'last_removed_file' in st.session_state:
            file_name = st.session_state.last_removed_file
            rows_deleted = st.session_state.get('last_removed_count', 0)
            st.success(f"✅ Documento '{file_name}' removido com sucesso! ({rows_deleted} chunks)")
            # Clean up these session state variables
            del st.session_state['last_removed_file']
            if 'last_removed_count' in st.session_state:
                del st.session_state['last_removed_count']
        
        # Reset the file list refresh flag
        st.session_state.file_list_refresh = False
    
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
    
    # Display model selection and document list in the sidebar
    with st.sidebar:
        # Fixed list of models as specified
        st.markdown("**Modelo de IA:**")
        
        # Define the fixed models in order of preference
        fixed_models = [
            "gpt-4o-mini",
            "o4-mini",
            "o3-mini", 
            "o1-mini", 
            "gpt-4o", 
            "gpt-4.1"
        ]
        
        # Create the model selection
        selected_model = st.selectbox("Escolha o modelo:", fixed_models, index=0)
        model_id = selected_model
        
        st.markdown("---")
        
        # Add direct file upload in the sidebar with compact status display
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Adicionar Documentos:**")
        with col2:
            # Status indicator in top-right corner
            if 'status_message' in st.session_state and 'status_type' in st.session_state:
                if st.session_state.status_type == 'success':
                    st.markdown('✅', unsafe_allow_html=True)
                elif st.session_state.status_type == 'error':
                    st.markdown('❌', unsafe_allow_html=True)
                elif st.session_state.status_type == 'info':
                    st.markdown('ℹ️', unsafe_allow_html=True)
        
        # Simplified direct file uploader without form
        uploaded_files = st.file_uploader(
            "Arquivo", 
            type=[ext[1:] for ext in SUPPORTED_EXTENSIONS], 
            accept_multiple_files=True,
            key=f"direct_file_uploader"
        )
        
        # Display file list if any are selected
        if uploaded_files and len(uploaded_files) > 0:
            st.write(f"**{len(uploaded_files)} arquivo(s) selecionado(s)**")
            for f in uploaded_files:
                st.write(f"- {f.name}")
        
        # Simple direct upload button
        if st.button("Processar e Vetorizar", 
                    disabled=not uploaded_files or len(uploaded_files) == 0,
                    key="process_direct_btn"):
            if uploaded_files and len(uploaded_files) > 0:
                # Show processing status with spinner in top-right
                st.session_state.global_status = f"Iniciando processamento de {len(uploaded_files)} arquivo(s)..."
                st.session_state.global_status_type = 'processing'
                st.rerun()  # Refresh to show the status first
        
        # Handle file processing if triggered
        if ('global_status' in st.session_state and 
            st.session_state.global_status_type == 'processing' and 
            'Iniciando processamento' in st.session_state.get('global_status', '')):
            
            # Process files directly without showing local status
            processed_count = 0
            error_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update only the global status
                    st.session_state.global_status = f"Processando {i+1}/{len(uploaded_files)}: {uploaded_file.name}"
                    st.session_state.global_status_type = 'processing'
                    
                    # Use the utility function from rag_utils.py
                    index = process_uploaded_file(uploaded_file, st.session_state.session_id)
                    
                    # Store file info in session state
                    if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                        st.session_state.uploaded_files.append({
                            'name': uploaded_file.name,
                            'size': uploaded_file.size,
                            'table': f"vectors_{st.session_state.session_id}"
                        })
                    
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    # Errors will be shown in final status
            
            # Set global status for results
            if processed_count > 0:
                # Set global status for success
                st.session_state.global_status = f"✅ {processed_count} arquivo(s) processado(s) com sucesso!"
                st.session_state.global_status_type = 'success'
                st.session_state.global_status_auto_clear = True
            elif error_count > 0:
                st.session_state.global_status = f"❌ Erro no processamento dos arquivos."
                st.session_state.global_status_type = 'error'
            
            # Re-initialize the RAG tool with the new documents
            from multi_table_rag import MultiTableRAGTool
            st.session_state.multi_rag_tool = MultiTableRAGTool()
            
            # Force a rerun to refresh UI state
            st.rerun()
        
        st.markdown("---")
        
        # Simplified display of available documents with multi-select delete
        all_files = {}  # Change to dict to track file:table mapping
        
        # Use the lightweight file listing method instead of accessing table_configs
        file_listing = st.session_state.multi_rag_tool.get_file_listing()
        for table_name, files in file_listing.items():
            for file in files:
                all_files[file] = table_name
        
        # Display available files with multi-select delete
        if all_files:
            # Initialize selected files in session state if not present
            if 'selected_files_to_delete' not in st.session_state:
                st.session_state.selected_files_to_delete = []
            
            st.markdown("**Documentos Disponíveis:**")
            
            # Create a scrollable container for document list
            doc_list_container = st.container()
            with doc_list_container:
                # Create columns for title and delete button
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"Total: {len(all_files)} documento(s)")
                    
                    # Always show the selected count
                    selected_count = len(st.session_state.selected_files_to_delete)
                    if selected_count > 0:
                        st.markdown(f"**{selected_count} selecionado(s)**")
                
                with col2:
                    # Show delete button if files are selected
                    if selected_count > 0:
                        # Create a delete button that directly calls the delete function
                        if st.button("🗑️", key="direct_delete_btn"):
                            # Start deletion process directly
                            st.session_state.global_status = f"Iniciando exclusão de {selected_count} arquivo(s)..."
                            st.session_state.global_status_type = 'processing'
                            st.rerun()  # Refresh to show the status first
                
                # Handle deletion process if triggered
                if ('global_status' in st.session_state and 
                    st.session_state.global_status_type == 'processing' and 
                    'Iniciando exclusão' in st.session_state.get('global_status', '')):
                    try:
                        # Connect to database directly
                        from db_config import get_pg_connection
                        conn = get_pg_connection()
                        cursor = conn.cursor()
                        
                        deleted_count = 0
                        success_files = []
                        
                        # Process each file without showing individual UI elements
                        for i, file_name in enumerate(st.session_state.selected_files_to_delete):
                            # Update only the global status for progress
                            st.session_state.global_status = f"Excluindo {i+1}/{len(st.session_state.selected_files_to_delete)}: {file_name}"
                            st.session_state.global_status_type = 'processing'
                            
                            # Get table name
                            table_name = all_files.get(file_name)
                            if table_name:
                                try:
                                    # First check if the table exists
                                    cursor.execute("""
                                        SELECT EXISTS (
                                            SELECT FROM information_schema.tables 
                                            WHERE table_schema = 'public'
                                            AND table_name = %s
                                        );
                                    """, (table_name,))
                                    
                                    table_exists = cursor.fetchone()[0]
                                    
                                    if table_exists:
                                        # Delete from database with quoted table name
                                        cursor.execute(f"""
                                            DELETE FROM "{table_name}"
                                            WHERE metadata_->>'file_name' = %s
                                        """, (file_name,))
                                        
                                        # Track results
                                        rows = cursor.rowcount
                                        deleted_count += rows
                                        success_files.append((file_name, rows))
                                        
                                        # Also drop the entire table if it's a data_vectors table
                                        # and there are no more documents in it
                                        if table_name.startswith("data_vectors_"):
                                            # Check if table is now empty
                                            cursor.execute(f"""
                                                SELECT COUNT(*) FROM "{table_name}"
                                            """)
                                            remaining_rows = cursor.fetchone()[0]
                                            
                                            if remaining_rows == 0:
                                                try:
                                                    # Import and use the drop_vector_table function
                                                    from db_config import drop_vector_table
                                                    
                                                    # Update status before table drop
                                                    st.session_state.global_status = f"Excluindo {i+1}/{len(st.session_state.selected_files_to_delete)}: {file_name} (removendo tabela)"
                                                    
                                                    # Close current connection to avoid lock conflicts
                                                    cursor.close()
                                                    conn.close()
                                                    
                                                    # Drop table with separate connection
                                                    drop_success = drop_vector_table(table_name)
                                                    
                                                    # Reconnect for the rest of the operations
                                                    conn = get_pg_connection()
                                                    cursor = conn.cursor()
                                                    
                                                    if drop_success:
                                                        st.session_state.global_status = f"Excluindo {i+1}/{len(st.session_state.selected_files_to_delete)}: {file_name} (tabela removida)"
                                                except Exception as drop_error:
                                                    # Log the error but continue with file deletion
                                                    print(f"Error dropping table {table_name}: {drop_error}")
                                                    
                                                    # Reconnect if needed
                                                    if conn.closed:
                                                        conn = get_pg_connection()
                                                        cursor = conn.cursor()
                                    else:
                                        # Still track the file for removal, just no rows deleted
                                        success_files.append((file_name, 0))
                                    
                                    # Remove from session state
                                    st.session_state.uploaded_files = [
                                        f for f in st.session_state.uploaded_files if f['name'] != file_name
                                    ]
                                except Exception as e:
                                    # Just continue, we'll show overall status at the end
                                    pass
                        
                        # Commit changes
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        # Show results only in global status
                        if success_files:
                            # Set global status for success message
                            st.session_state.global_status = f"✅ {len(success_files)} arquivo(s) excluído(s) com sucesso! ({deleted_count} chunks)"
                            st.session_state.global_status_type = 'success'
                            st.session_state.global_status_auto_clear = True
                        
                        # Clear selected files
                        st.session_state.selected_files_to_delete = []
                        
                        # Reinitialize RAG tool
                        from multi_table_rag import MultiTableRAGTool
                        st.session_state.multi_rag_tool = MultiTableRAGTool()
                        
                        # Force a rerun to refresh the UI
                        st.rerun()
                    except Exception as e:
                        # Set global status for error display
                        st.session_state.global_status = f"❌ Erro na operação de exclusão: {str(e)}"
                        st.session_state.global_status_type = 'error'
                        st.rerun()
                
                # File list with checkboxes
                st.markdown('<div style="max-height: 300px; overflow-y: auto;">', unsafe_allow_html=True)
                
                # Create checkboxes for each file
                for i, (file_name, table_name) in enumerate(sorted(all_files.items())):
                    # Create a row for each file
                    cols = st.columns([0.5, 4.5])
                    
                    with cols[0]:
                        # Check if already selected
                        is_selected = file_name in st.session_state.selected_files_to_delete
                        
                        # Create a checkbox with a unique key
                        new_state = st.checkbox("", value=is_selected, key=f"select_file_{i}_{file_name[:10]}")
                        
                        # Update selection if changed
                        if new_state != is_selected:
                            if new_state:
                                # Add to selected files
                                if file_name not in st.session_state.selected_files_to_delete:
                                    st.session_state.selected_files_to_delete.append(file_name)
                                    st.rerun()
                            else:
                                # Remove from selected files
                                if file_name in st.session_state.selected_files_to_delete:
                                    st.session_state.selected_files_to_delete.remove(file_name)
                                    st.rerun()
                    
                    with cols[1]:
                        # Display file name with tooltip
                        if len(file_name) > 25:
                            display_name = file_name[:22] + "..."
                        else:
                            display_name = file_name
                            
                        st.markdown(f'<div class="doc-item" title="{file_name}">📄 {display_name}</div>',
                                   unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Nenhum documento encontrado")
            st.markdown("Adicione documentos usando o formulário acima")
    
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
            try:
                # Use RAG processing - LlamaIndex will determine when to use context
                with st.spinner("Processando consulta..."):
                    # Create a robust system prompt for better RAG
                    system_prompt = """
                    Você é um assistente de IA especializado em responder perguntas com base em documentos específicos. 
                    Analise cuidadosamente as informações fornecidas pela ferramenta RAG e utilize apenas essas informações ao responder.
                    
                    Diretrizes importantes:
                    - Responda com base nos documentos fornecidos quando relevante
                    - Se a informação não estiver clara nos documentos, admita que não sabe
                    - Para consultas conversacionais simples, responda naturalmente sem buscar informações nos documentos
                    - Seja direto e conciso em suas respostas
                    """
                    
                    # Process the message with our improved RAG processor
                    from rag_processor import process_query_with_llm
                    
                    response = process_query_with_llm(
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

elif current_page == "🏠 Início":
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

elif current_page == "🔌 Teste de Conexão com PostgreSQL":
    st.subheader("🔌 Teste de Conexão com o Banco")
    
    # Display status messages at the top
    show_status_message()
    
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

elif current_page == "📥 Upload e Vetorização de Arquivos":
    st.subheader("📥 Upload e Vetorização de Arquivos")
    
    # Display status messages at the top
    show_status_message()
    
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
    
    # Create a compact status display
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        st.markdown("**Selecionar Arquivos:**")
    with status_col2:
        # Status indicator
        if 'upload_status_message' in st.session_state and 'upload_status_type' in st.session_state:
            if st.session_state.upload_status_type == 'success':
                st.markdown('✅', unsafe_allow_html=True)
            elif st.session_state.upload_status_type == 'error':
                st.markdown('❌', unsafe_allow_html=True)
            elif st.session_state.upload_status_type == 'info':
                st.markdown('ℹ️', unsafe_allow_html=True)
    
    # Create a key in session state for the standalone form if it doesn't exist
    if 'standalone_form_key' not in st.session_state:
        st.session_state.standalone_form_key = 0
        
    # Create a container for status messages
    status_area = st.container()
    
    # File uploader with unique key
    uploaded_files = st.file_uploader(
        "Escolha arquivos para processar", 
        type=[ext[1:] for ext in SUPPORTED_EXTENSIONS], 
        accept_multiple_files=True,
        key=f"standalone_uploader_{st.session_state.standalone_form_key}"
    )
    
    # Show files details if any are selected
    if uploaded_files and len(uploaded_files) > 0:
        st.write(f"**{len(uploaded_files)} arquivo(s) selecionado(s):**")
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    # Use a button with clear visual indication
    col1, col2 = st.columns([3, 1])
    with col1:
        process_clicked = st.button(
            "Processar e Vetorizar", 
            key="process_standalone_btn",
            disabled=not (uploaded_files and len(uploaded_files) > 0),
            help="Processa os arquivos selecionados e cria embeddings vetoriais"
        )
    with col2:
        # Show status indicator
        if 'processing_files' in st.session_state and st.session_state.processing_files:
            st.markdown("⏳", unsafe_allow_html=True)
    
    # Only process if button was clicked and files were selected
    if process_clicked and uploaded_files and len(uploaded_files) > 0:
        # Mark that we're processing
        st.session_state.processing_files = True
        
        # Show processing status with spinner in top-right
        st.session_state.global_status = f"Processando {len(uploaded_files)} arquivo(s)..."
        st.session_state.global_status_type = 'processing'
        st.rerun()
        
        # Simple direct processing without status updates
        processed_count = 0
        error_count = 0
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update processing status with progress
            st.session_state.global_status = f"Processando arquivo {i+1}/{total_files}: {uploaded_file.name}"
            st.session_state.global_status_type = 'processing'
            st.rerun()
            
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
                
                processed_count += 1
            except Exception as e:
                error_count += 1
                # Set global status for error
                st.session_state.global_status = f"❌ Erro ao processar {uploaded_file.name}: {str(e)}"
                st.session_state.global_status_type = 'error'
        
        # Set status message in top-right
        if processed_count > 0:
            # Set global status for success
            st.session_state.global_status = f"✅ {processed_count} arquivo(s) processado(s) com sucesso!"
            st.session_state.global_status_type = 'success'
            st.session_state.global_status_auto_clear = True
        
        # Reset processing flags
        if 'processing_files' in st.session_state:
            st.session_state.processing_files = False
        
        # Re-initialize the RAG tool with the new documents
        from multi_table_rag import MultiTableRAGTool
        st.session_state.multi_rag_tool = MultiTableRAGTool()
        
        # Increment the form key to reset the form in the next render
        st.session_state.standalone_form_key += 1
        st.rerun()
    
    # Display uploaded files in a scrollable container with better formatting
    if st.session_state.uploaded_files:
        st.subheader("Arquivos Processados")
        
        # Create a scrollable container
        st.markdown('<div style="max-height: 400px; overflow-y: auto;">', unsafe_allow_html=True)
        
        for idx, file_info in enumerate(st.session_state.uploaded_files):
            # Create a card-like display for each file
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 8px;">
                <strong>{idx+1}. {file_info['name']}</strong>
                <br><small>Tabela: {file_info['table']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif current_page == "🧪 Diagnóstico Avançado":
    st.subheader("🧪 Diagnóstico do Sistema RAG")
    
    # Display status messages at the top
    show_status_message()
    
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
