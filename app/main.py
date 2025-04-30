import streamlit as st

st.set_page_config(page_title="Chat RAG", layout="wide")

st.title("💬 Chat RAG – Jumpad")

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
    st.subheader("📥 Upload de Arquivo e Vetorização (a implementar)")
    st.info("Este módulo será ativado no próximo passo.")

elif menu == "🔍 Consulta com RAG":
    st.subheader("🔍 Consulta com RAG (a implementar)")
    st.info("Aqui você poderá fazer perguntas baseadas nos documentos indexados.")

elif menu == "🧪 Diagnóstico Avançado":
    st.subheader("🧪 Diagnóstico (a implementar)")
    st.info("Aqui serão exibidos dados de debug, conexão com socket e sessão.")
