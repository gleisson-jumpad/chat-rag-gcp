import streamlit as st
import os
import uuid
from rag_utils import process_uploaded_file, is_supported_file, SUPPORTED_EXTENSIONS

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
    st.subheader("🔍 Consulta com RAG (a implementar)")
    st.info("Aqui você poderá fazer perguntas baseadas nos documentos indexados.")

elif menu == "🧪 Diagnóstico Avançado":
    st.subheader("🧪 Diagnóstico (a implementar)")
    st.info("Aqui serão exibidos dados de debug, conexão com socket e sessão.")
