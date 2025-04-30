import streamlit as st
import os
import uuid
from rag_utils import process_uploaded_file, is_supported_file, SUPPORTED_EXTENSIONS, get_existing_tables

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
                st.success(f"✅ Encontrados {len(vector_tables)} documentos previamente processados!")
                
                # Add existing documents to session state
                for table_name in vector_tables:
                    # Add to session state if not already there
                    if table_name not in [f['table'] for f in st.session_state.uploaded_files]:
                        st.session_state.uploaded_files.append({
                            'name': f"Documento {table_name.split('_')[1][:8]}",
                            'size': 0,  # Unknown size for existing documents
                            'table': table_name
                        })
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
    st.subheader("💬 ChatRAG")
    
    # Initialize chat history if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize session state for uploaded files if needed
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Default RAG setting
    use_rag = True
    selected_file_names = []
    selected_tables = []
    
    # Add OpenAI model selection in the sidebar
    with st.sidebar:
        st.subheader("Configurações do Chat")
        
        # Model selection with cleaner layout
        st.markdown("**Modelo de IA:**")
        model_options = {
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
            "GPT-4o": "gpt-4o",
            "GPT-4o-mini": "gpt-4o-mini"
        }
        selected_model = st.selectbox(
            "Selecione o modelo:",
            options=list(model_options.keys()),
            index=1,  # Default to GPT-4o
            label_visibility="collapsed"
        )
        model_id = model_options[selected_model]
        
        # RAG toggle with cleaner layout
        st.markdown("---")
        st.markdown("**Conhecimento dos Documentos:**")
        use_rag = st.toggle("Usar RAG", value=True)
    
    # Check for documents and load them if necessary (but don't show messages)
    if use_rag and len(st.session_state.uploaded_files) == 0:
        with st.spinner("Carregando documentos..."):
            try:
                from rag_utils import get_existing_tables
                vector_tables = get_existing_tables()
                
                if vector_tables:
                    # Connect to database to get document info
                    from db_config import get_pg_connection
                    conn = get_pg_connection()
                    cursor = conn.cursor()
                    
                    doc_count = 0
                    
                    for table_name in vector_tables:
                        try:
                            # Query to count distinct documents in the table
                            cursor.execute(f"""
                                SELECT DISTINCT metadata_->>'file_name' as file_name 
                                FROM {table_name}
                            """)
                            
                            # Get unique filenames
                            distinct_files = cursor.fetchall()
                            
                            if distinct_files:
                                for file_info in distinct_files:
                                    file_name = file_info[0] if file_info[0] else f"Documento {doc_count+1}"
                                    
                                    # Add document info to session state
                                    st.session_state.uploaded_files.append({
                                        'name': file_name,
                                        'size': 0,
                                        'table': table_name,
                                        'file_id': f"{table_name}_{doc_count}"
                                    })
                                    doc_count += 1
                        except Exception as e:
                            pass
                    
                    cursor.close()
                    conn.close()
                    
                    st.rerun()
                else:
                    use_rag = False
            except Exception as e:
                use_rag = False
    
    # Update sidebar with document selection if documents are available
    with st.sidebar:
        # Show documents if available
        if st.session_state.uploaded_files and use_rag:
            # Get unique files
            unique_files = []
            for f in st.session_state.uploaded_files:
                if f["name"] not in [uf["name"] for uf in unique_files]:
                    unique_files.append(f)
            
            st.markdown(f"📚 **Documentos Disponíveis:** {len(unique_files)}")
            
            # Multi-select for documents with cleaner styling
            file_options = [f["name"] for f in unique_files]
            
            # Use custom container for better styling
            selection_container = st.container(border=True)
            with selection_container:
                st.markdown("**Selecione os documentos:**")
                
                # Create checkboxes for each file
                selected_file_names = []
                selected_files = []
                for file_name in file_options:
                    if st.checkbox(file_name, value=True):
                        selected_file_names.append(file_name)
                        # Get all files with this name
                        selected_files.extend([f for f in st.session_state.uploaded_files if f["name"] == file_name])
                
                # Get tables from selected files        
                selected_tables = [f["table"] for f in selected_files]
            
            if not selected_file_names:
                st.warning("⚠️ Nenhum documento selecionado")
                use_rag = False
        elif use_rag:
            st.warning("⚠️ Nenhum documento disponível")
            use_rag = False
        
        # Add reset button at the bottom of sidebar
        st.markdown("---")
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat area - style with a light background
    chat_container = st.container(border=False)
    with chat_container:
        # Display chat messages in a scrollable area
        st.markdown("### Conversa")
        message_container = st.container(height=400, border=True)
        with message_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input at the bottom
        st.markdown("### Digite sua pergunta")
        prompt = st.chat_input("Converse com o sistema...")
        
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately 
            st.rerun()
    
    # Process the last message if it's from the user and hasn't been answered yet
    if (st.session_state.messages and 
        len(st.session_state.messages) % 2 == 1 and 
        st.session_state.messages[-1]["role"] == "user"):
        
        # Get the last message
        last_message = st.session_state.messages[-1]["content"]
        
        # Display assistant response in chat message container
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Show a spinner while processing
                with st.spinner("Gerando resposta..."):
                    try:
                        from llama_index.llms.openai import OpenAI
                        
                        # Initialize LLM with selected model
                        llm = OpenAI(model=model_id)
                        
                        # Define simple greeting patterns
                        simple_greetings = [
                            "oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", 
                            "hello", "hi", "hey", "good morning", "good afternoon", 
                            "good evening", "tudo bem", "como vai"
                        ]
                        
                        # Check if the query is a simple greeting
                        is_simple_greeting = False
                        clean_message = last_message.lower().strip()
                        
                        for greeting in simple_greetings:
                            if greeting in clean_message or clean_message == greeting:
                                is_simple_greeting = True
                                break
                        
                        # For simple greetings, don't use RAG
                        if is_simple_greeting or not use_rag or not selected_file_names:
                            # Use regular chat without RAG
                            from llama_index.core.llms import ChatMessage, MessageRole
                            
                            # Convert session history to format expected by LlamaIndex
                            chat_history = []
                            for msg in st.session_state.messages[:-1]:  # Exclude the current message
                                role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
                                chat_history.append(ChatMessage(role=role, content=msg["content"]))
                            
                            # Get streaming chat response
                            response = llm.stream_chat(
                                messages=[
                                    *chat_history,
                                    ChatMessage(role=MessageRole.USER, content=last_message)
                                ]
                            )
                            
                            # Stream the response
                            for chunk in response:
                                if chunk.delta:
                                    full_response += chunk.delta
                                    message_placeholder.markdown(full_response + "▌")
                        elif use_rag and st.session_state.uploaded_files and selected_file_names:
                            # Add a debugging message
                            message_placeholder.markdown("🔍 Consultando documentos... Por favor aguarde.")
                            
                            # First, attempt a direct SQL query to get context content
                            try:
                                from db_config import get_pg_connection
                                conn = get_pg_connection()
                                cursor = conn.cursor()
                                
                                # Execute a search query to get text context
                                search_terms = last_message.lower().split()
                                results = []
                                
                                for table_name in selected_tables:
                                    # Log the query attempt
                                    try:
                                        # Query to get document content with manual search
                                        cursor.execute(f"""
                                            SELECT content, metadata_->>'file_name' as filename
                                            FROM {table_name}
                                            LIMIT 5;
                                        """)
                                        
                                        # Get unique filenames
                                        query_results = cursor.fetchall()
                                        
                                        if query_results:
                                            for result in query_results:
                                                content = result[0] if result[0] else ""
                                                filename = result[1] if result[1] else "Unknown"
                                                results.append({"content": content, "filename": filename})
                                    except Exception as e:
                                        message_placeholder.markdown(f"⚠️ Erro ao consultar tabela {table_name}: {str(e)}")
                                
                                cursor.close()
                                conn.close()
                                
                                # If we have results, use them to generate response
                                if results:
                                    # Combine content to create context
                                    context = "\n\n".join([f"De {r['filename']}:\n{r['content']}" for r in results])
                                    
                                    # Format the context and question for the LLM
                                    formatted_prompt = f"""
                                    Baseado no seguinte contexto:
                                    
                                    {context}
                                    
                                    Por favor, responda esta pergunta: {last_message}
                                    
                                    Dê sua resposta baseada APENAS nas informações do contexto fornecido. Se a resposta não estiver no contexto, diga "Não tenho essa informação no contexto fornecido."
                                    """
                                    
                                    # Get response from LLM
                                    from llama_index.core.llms import ChatMessage, MessageRole
                                    chat_response = llm.complete(formatted_prompt)
                                    
                                    # Stream the response
                                    for char in chat_response.text:
                                        full_response += char
                                        message_placeholder.markdown(full_response + "▌")
                                    
                                    # Final update without cursor
                                    message_placeholder.markdown(full_response)
                                else:
                                    # No results found, try RAG approach
                                    message_placeholder.markdown("⚠️ Sem resultados diretos. Tentando approach RAG avançado...")
                                    
                                    # Use RAG for response with standard approach
                                    from llama_index.vector_stores.postgres import PGVectorStore
                                    from llama_index.core import VectorStoreIndex, StorageContext
                                    from llama_index.embeddings.openai import OpenAIEmbedding
                                    from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
                                    from llama_index.core.query_engine import RouterQueryEngine
                                    
                                    # Set up connection parameters
                                    conn_params = {
                                        "host": "34.150.190.157",
                                        "port": 5432, 
                                        "dbname": "postgres",
                                        "user": "llamaindex",
                                        "password": "password123"
                                    }
                                    
                                    # Create embedding model
                                    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
                                    
                                    try:
                                        # If only one document was selected
                                        if len(selected_file_names) == 1 and len(selected_tables) > 0:
                                            table_name = selected_tables[0]
                                            
                                            # Create vector store
                                            vector_store = PGVectorStore.from_params(
                                                host=conn_params["host"],
                                                port=conn_params["port"],
                                                database=conn_params["dbname"],
                                                user=conn_params["user"],
                                                password=conn_params["password"],
                                                table_name=table_name,
                                                embed_dim=1536
                                            )
                                            
                                            # Create storage context and index
                                            storage_context = StorageContext.from_defaults(vector_store=vector_store)
                                            index = VectorStoreIndex.from_vector_store(
                                                vector_store,
                                                embed_model=embed_model
                                            )
                                            
                                            # Create query engine with streaming
                                            query_engine = index.as_query_engine(
                                                llm=llm,
                                                streaming=True,
                                                similarity_top_k=3
                                            )
                                        else:
                                            # If multiple documents were selected, create a router
                                            # Create a query engine for each selected document
                                            query_engines = {}
                                            
                                            # Using unique combinations of file names and tables
                                            file_tables = list(zip(selected_file_names, selected_tables))
                                            for i, (file_name, table) in enumerate(file_tables):
                                                # Create vector store for this document
                                                vector_store = PGVectorStore.from_params(
                                                    host=conn_params["host"],
                                                    port=conn_params["port"],
                                                    database=conn_params["dbname"],
                                                    user=conn_params["user"],
                                                    password=conn_params["password"],
                                                    table_name=table,
                                                    embed_dim=1536
                                                )
                                                
                                                # Create index
                                                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                                                index = VectorStoreIndex.from_vector_store(
                                                    vector_store,
                                                    embed_model=embed_model
                                                )
                                                
                                                # Create query engine
                                                query_engines[f"{file_name}_{i}"] = index.as_query_engine(
                                                    llm=llm,
                                                    similarity_top_k=3
                                                )
                                            
                                            # Create step decompose transform
                                            step_decompose_transform = StepDecomposeQueryTransform(
                                                llm=llm,
                                                verbose=True
                                            )
                                            
                                            # Create router query engine
                                            query_engine = RouterQueryEngine(
                                                selector="llm",
                                                query_engines=query_engines,
                                                llm=llm,
                                                query_transform=step_decompose_transform
                                            )
                                        
                                        # Execute query
                                        response = query_engine.query(last_message)
                                        
                                        # Stream the response
                                        if hasattr(response, 'response_gen'):
                                            # For streaming response
                                            for token in response.response_gen:
                                                full_response += token
                                                message_placeholder.markdown(full_response + "▌")
                                        else:
                                            # For non-streaming response (like from RouterQueryEngine)
                                            full_response = str(response)
                                            message_placeholder.markdown(full_response)
                                            
                                    except Exception as e:
                                        # Fall back to regular chat without RAG
                                        from llama_index.core.llms import ChatMessage, MessageRole
                                        
                                        # Convert session history to format expected by LlamaIndex
                                        chat_history = []
                                        for msg in st.session_state.messages[:-1]:  # Exclude the current message
                                            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
                                            chat_history.append(ChatMessage(role=role, content=msg["content"]))
                                        
                                        # Get streaming chat response
                                        response = llm.stream_chat(
                                            messages=[
                                                *chat_history,
                                                ChatMessage(role=MessageRole.USER, content=last_message)
                                            ]
                                        )
                                        
                                        # Stream the response
                                        for chunk in response:
                                            if chunk.delta:
                                                full_response += chunk.delta
                                                message_placeholder.markdown(full_response + "▌")
                        
                        # Display final response without cursor
                        message_placeholder.markdown(full_response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        error_message = f"Desculpe, ocorreu um erro: {str(e)}"
                        message_placeholder.markdown(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        
                    # Rerun to update the UI
                    st.rerun()

elif menu == "🧪 Diagnóstico Avançado":
    st.subheader("🧪 Diagnóstico e Depuração")
    
    # Database connection test
    st.subheader("Teste de Conexão ao Banco de Dados")
    
    if st.button("Testar Conexão ao PostgreSQL"):
        try:
            from db_config import get_pg_connection
            conn = get_pg_connection()
            
            # Test the connection by executing a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            st.success("✅ Conexão estabelecida com sucesso!")
            st.code(version)
            
            cursor.close()
            conn.close()
        except Exception as e:
            st.error(f"❌ Erro ao conectar ao banco de dados: {str(e)}")
    
    # Check for vector tables
    st.subheader("Tabelas de Vetores no Banco")
    
    if st.button("Verificar Tabelas de Vetores"):
        try:
            from rag_utils import get_existing_tables
            vector_tables = get_existing_tables()
            
            if vector_tables:
                st.success(f"✅ Encontradas {len(vector_tables)} tabelas de vetores:")
                for table in vector_tables:
                    st.code(table)
            else:
                st.warning("⚠️ Não foram encontradas tabelas de vetores no banco de dados.")
        except Exception as e:
            st.error(f"❌ Erro ao verificar tabelas: {str(e)}")
    
    # Environment variables check
    st.subheader("Variáveis de Ambiente")
    
    if st.button("Verificar Variáveis de Ambiente"):
        env_vars = {
            "DB_PUBLIC_IP": os.getenv("DB_PUBLIC_IP"),
            "PG_PORT": os.getenv("PG_PORT"),
            "PG_DB": os.getenv("PG_DB"),
            "PG_USER": os.getenv("PG_USER"),
            "PG_PASSWORD": "*****" if os.getenv("PG_PASSWORD") else None,
            "OPENAI_API_KEY": "*****" if os.getenv("OPENAI_API_KEY") else None
        }
        
        st.json(env_vars)
    
    # Session state debug
    st.subheader("Estado da Sessão (Session State)")
    
    if st.button("Verificar Estado da Sessão"):
        debug_info = {
            "session_id": st.session_state.get("session_id", None),
            "uploaded_files_count": len(st.session_state.get("uploaded_files", [])),
            "uploaded_files": st.session_state.get("uploaded_files", []),
            "messages_count": len(st.session_state.get("messages", []))
        }
        
        st.json(debug_info)
    
    # Force refresh session
    if st.button("Limpar e Recarregar Sessão"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Sessão limpa com sucesso!")
        st.rerun()
    
    # Database query tool
    st.subheader("Ferramenta de Consulta SQL")
    
    # Predefined queries
    predefined_queries = {
        "Contar documentos distintos": "SELECT COUNT(DISTINCT metadata_->>'file_name') as documentos_distintos FROM data_vectors_5bed5a54_76f3_4a10_bb16_176d8fecc104;",
        "Listar arquivos distintos": "SELECT DISTINCT metadata_->>'file_name' as nome_arquivo FROM data_vectors_5bed5a54_76f3_4a10_bb16_176d8fecc104;",
        "Contar chunks por arquivo": "SELECT metadata_->>'file_name' as nome_arquivo, COUNT(*) as numero_chunks FROM data_vectors_5bed5a54_76f3_4a10_bb16_176d8fecc104 GROUP BY metadata_->>'file_name';",
        "Ver metadados dos documentos": "SELECT id, metadata_ FROM data_vectors_5bed5a54_76f3_4a10_bb16_176d8fecc104 LIMIT 10;",
        "Ver conteúdo dos documentos": "SELECT id, metadata_->>'file_name' as nome_arquivo, SUBSTRING(content, 1, 200) as preview_conteudo FROM data_vectors_5bed5a54_76f3_4a10_bb16_176d8fecc104 LIMIT 5;",
        "Listar todas as tabelas": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%');",
        "Ver estrutura da tabela": "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'data_vectors_5bed5a54_76f3_4a10_bb16_176d8fecc104';",
    }
    
    # Query selector
    selected_query = st.selectbox(
        "Consultas pré-definidas:",
        options=list(predefined_queries.keys())
    )
    
    # Text area for SQL query
    sql_query = st.text_area(
        "Digite sua consulta SQL:",
        predefined_queries[selected_query],
        height=100
    )
    
    # Execute button
    if st.button("Executar Consulta"):
        try:
            from db_config import get_pg_connection
            conn = get_pg_connection()
            cursor = conn.cursor()
            
            # Execute the query
            cursor.execute(sql_query)
            
            # Get the results
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
            # Display the results
            if results:
                st.success(f"✅ Consulta executada com sucesso: {len(results)} registros encontrados")
                
                # Create a DataFrame for display
                import pandas as pd
                df = pd.DataFrame(results, columns=column_names)
                st.dataframe(df)
                
                # Also show as raw text (useful for copying values)
                st.code(str(results))
            else:
                st.info("A consulta foi executada, mas não retornou resultados.")
            
            cursor.close()
            conn.close()
        except Exception as e:
            st.error(f"❌ Erro ao executar a consulta: {str(e)}")
