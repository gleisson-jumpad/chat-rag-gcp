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
                st.session_state.multi_rag_tool = MultiTableRAGTool()
            except Exception as init_error:
                st.error(f"Erro ao inicializar MultiTableRAGTool: {init_error}")
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
        selected_model = st.selectbox(
            "Selecione o modelo:",
            options=list(model_options.keys()),
            index=1,  # Default to GPT-4o
            label_visibility="collapsed"
        )
        model_id = model_options[selected_model]
        
        # Multi-RAG info
        st.markdown("---")
        st.info(
            "ℹ️ **Sobre o RAG Multi-Tabela:**\n\n"
            "Este modo avançado permite que a IA busque em múltiplas tabelas de vetores simultaneamente. "
            "O sistema decidirá automaticamente quais tabelas são relevantes para sua consulta "
            "e combinará as informações encontradas em uma resposta coerente."
        )
        
        # Get tables information
        tables_info = st.session_state.multi_rag_tool.get_tables_info()
        tables = tables_info["tables"]
        
        # Display all tables and their files
        if tables:
            st.markdown("---")
            st.markdown(f"🗃️ **Tabelas Disponíveis:** {len(tables)}")
            
            with st.expander("Ver detalhes das tabelas"):
                for idx, table in enumerate(tables):
                    st.markdown(f"**{idx+1}. {table['name']}**")
                    st.markdown(f"- Descrição: {table['description']}")
                    st.markdown(f"- Documentos: {table['doc_count']}")
                    st.markdown(f"- Chunks: {table['chunk_count']}")
                    
                    # Show files in this table
                    if 'files' in table and table['files']:
                        st.markdown("**Arquivos:**")
                        for file in table['files']:
                            st.markdown(f"- {file}")
                    
                    st.markdown("---")
        else:
            st.warning("⚠️ Nenhuma tabela de vetores encontrada")
        
        # Display files in database
        files_info = st.session_state.multi_rag_tool.get_files_in_database()
        all_files = files_info["all_files"]
        
        if all_files:
            st.markdown("---")
            st.markdown(f"📚 **Documentos Disponíveis:** {len(all_files)}")
            
            with st.expander("Ver todos os documentos"):
                for file in sorted(all_files):
                    st.markdown(f"- {file}")
        else:
            st.warning("⚠️ Nenhum documento encontrado na base de dados")
        
        # Add reset button at the bottom of sidebar
        st.markdown("---")
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    user_query = st.chat_input("Sua pergunta:")
    
    if user_query:
        # Import the multi-table RAG processing function
        from multi_table_rag import process_message_with_multi_rag
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Analisando sua pergunta...")
            
            try:
                # Process message with multi-table RAG
                with st.spinner("Consultando múltiplas bases de conhecimento..."):
                    response = process_message_with_multi_rag(
                        user_query, 
                        st.session_state.multi_rag_tool,
                        model=model_id
                    )
                
                # Display the response
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:
                error_message = f"❌ **Erro ao processar a mensagem:** {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

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

    # Add Direct Document Access Test
    st.subheader("Teste de Acesso Direto a Documentos")
    
    # Only show if the multi-RAG tool is initialized
    if 'multi_rag_tool' in st.session_state:
        try:
            # Get available documents
            files_info = st.session_state.multi_rag_tool.get_files_in_database()
            all_files = files_info["all_files"]
            
            if all_files:
                # Document selector
                selected_doc = st.selectbox(
                    "Selecione um documento para testar:",
                    options=sorted(all_files)
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Verify document vectors button
                    if st.button("Verificar Vetores do Documento"):
                        with st.spinner(f"Verificando vetores para '{selected_doc}'..."):
                            vector_info = st.session_state.multi_rag_tool.verify_document_vectors(selected_doc)
                            
                            if "error" in vector_info:
                                st.error(f"Erro ao verificar vetores: {vector_info['error']}")
                            else:
                                st.success(f"✅ Verificação completa para '{selected_doc}'")
                                
                                # Display vector counts
                                st.info(f"Total de vetores: {vector_info['total_vectors']}")
                                
                                # Display counts by table
                                for table, count in vector_info['vector_counts'].items():
                                    st.write(f"- **{table}**: {count} vetores")
                                
                                # Show sample content if available
                                if any(samples for samples in vector_info['sample_contents'].values()):
                                    with st.expander("Ver amostras de conteúdo"):
                                        for table, samples in vector_info['sample_contents'].items():
                                            if samples:
                                                st.write(f"**Amostras da tabela {table}:**")
                                                for sample in samples:
                                                    st.code(sample['content_preview'], language="text")
                
                with col2:
                    # Direct document summary button
                    if st.button("Resumir Documento Diretamente"):
                        with st.spinner(f"Gerando resumo para '{selected_doc}'..."):
                            try:
                                # Use our direct document summary method
                                summary_result = st.session_state.multi_rag_tool.summarize_document(selected_doc)
                                
                                if "error" in summary_result:
                                    st.error(f"Erro ao resumir documento: {summary_result['error']}")
                                    st.warning(summary_result.get("message", ""))
                                else:
                                    st.success(f"✅ Resumo gerado para '{selected_doc}'")
                                    
                                    # Display the summary
                                    st.markdown("### Resumo do Documento")
                                    st.markdown(summary_result["answer"])
                                    
                                    # Show sources if available
                                    if summary_result.get("sources"):
                                        with st.expander(f"Ver fontes ({len(summary_result['sources'])} trechos)"):
                                            for i, source in enumerate(summary_result["sources"]):
                                                st.markdown(f"**Trecho {i+1}** ({source.get('file_name', 'unknown')})")
                                                st.code(source.get('text', ''), language="text")
                            except Exception as e:
                                st.error(f"Erro ao resumir documento: {str(e)}")
            else:
                st.warning("⚠️ Nenhum documento encontrado na base de dados.")
        except Exception as e:
            st.error(f"Erro ao acessar documentos: {str(e)}")
    else:
        st.warning("⚠️ Ferramenta RAG não está inicializada. Navegue para a página 'Consulta com RAG' primeiro.")

    # Add RAG Configuration Verification
    st.subheader("Verificação da Configuração RAG")
    
    if 'multi_rag_tool' in st.session_state:
        if st.button("Verificar Configuração do RAG"):
            with st.spinner("Verificando configuração do RAG..."):
                try:
                    config_results = st.session_state.multi_rag_tool.verify_configuration()
                    
                    # Display status
                    if config_results["status"] == "ok":
                        st.success("✅ Todas as verificações passaram com sucesso!")
                    elif config_results["status"] == "warning":
                        st.warning("⚠️ Avisos detectados na configuração")
                    else:
                        st.error("❌ Erros encontrados na configuração")
                    
                    # Display errors if any
                    if config_results["errors"]:
                        st.error("Problemas encontrados:")
                        for error in config_results["errors"]:
                            st.markdown(f"- {error}")
                    
                    # Display configuration details
                    st.info("Detalhes da configuração:")
                    st.json(config_results["details"])
                except Exception as e:
                    st.error(f"Erro durante a verificação: {str(e)}")
    else:
        st.warning("⚠️ Ferramenta RAG não está inicializada. Navegue para a página 'Consulta com RAG' primeiro.")

    # Add Debug Document Summarization
    st.subheader("Debug de Resumo de Documentos")
    
    if 'multi_rag_tool' in st.session_state:
        st.info("Esta seção permite testar diretamente a funcionalidade de resumo de documentos. Útil para depuração quando os resumos via chat não funcionam corretamente.")
        
        # Get available documents
        files_info = {}
        try:
            files_info = st.session_state.multi_rag_tool.get_files_in_database()
            all_files = files_info["all_files"]
            
            if all_files:
                # Document selector
                selected_doc = st.selectbox(
                    "Selecione um documento para resumir:",
                    options=sorted(all_files)
                )
                
                if st.button("Resumir Documento Selecionado"):
                    with st.spinner(f"Gerando resumo para '{selected_doc}'..."):
                        try:
                            # Use our direct document summary method
                            summary_result = st.session_state.multi_rag_tool.summarize_document(selected_doc)
                            
                            if "error" in summary_result:
                                st.error(f"Erro ao resumir documento: {summary_result['error']}")
                                st.warning(summary_result.get("message", ""))
                            else:
                                st.success(f"✅ Resumo gerado para '{selected_doc}'")
                                
                                # Display the summary
                                st.markdown("### Resumo do Documento")
                                st.markdown(summary_result["answer"])
                                
                                # Show sources if available
                                if summary_result.get("sources"):
                                    with st.expander(f"Ver fontes ({len(summary_result['sources'])} trechos)"):
                                        for i, source in enumerate(summary_result["sources"]):
                                            st.markdown(f"**Trecho {i+1}** ({source.get('file_name', 'unknown')})")
                                            st.code(source.get('text', ''), language="text")
                        except Exception as e:
                            st.error(f"Erro ao resumir documento: {str(e)}")
            else:
                st.warning("Nenhum documento encontrado na base de dados.")
                
                # Provide diagnostic information
                st.markdown("### Informações de Diagnóstico")
                st.json(files_info)
                
                # Try to check database directly for documents
                try:
                    from db_config import get_pg_connection
                    conn = get_pg_connection()
                    cursor = conn.cursor()
                    
                    st.markdown("### Verificando Tabelas no Banco de Dados")
                    # List vector tables
                    cursor.execute("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'public' AND 
                        (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
                    """)
                    
                    tables = cursor.fetchall()
                    if tables:
                        st.success(f"Encontradas {len(tables)} tabelas de vetores:")
                        
                        for table in tables:
                            table_name = table[0]
                            st.write(f"**Tabela:** {table_name}")
                            
                            # Get document counts
                            cursor.execute(f"""
                                SELECT COUNT(DISTINCT metadata_->>'file_name') 
                                FROM {table_name}
                                WHERE metadata_->>'file_name' IS NOT NULL
                            """)
                            
                            count = cursor.fetchone()[0]
                            st.write(f"Documentos distintos: {count}")
                            
                            # Get document names
                            if count > 0:
                                cursor.execute(f"""
                                    SELECT DISTINCT metadata_->>'file_name' as file_name
                                    FROM {table_name}
                                    WHERE metadata_->>'file_name' IS NOT NULL
                                """)
                                
                                doc_names = cursor.fetchall()
                                if doc_names:
                                    st.write("Documentos encontrados:")
                                    for doc in doc_names:
                                        st.write(f"- {doc[0]}")
                    else:
                        st.warning("Nenhuma tabela de vetores encontrada no banco de dados.")
                    
                    cursor.close()
                    conn.close()
                    
                except Exception as e:
                    st.error(f"Erro ao consultar diretamente o banco: {str(e)}")
        except Exception as e:
            st.error(f"Erro ao acessar informações de documentos: {str(e)}")
    else:
        st.warning("⚠️ Ferramenta RAG não está inicializada. Navegue para a página 'Consulta com RAG' primeiro.")
