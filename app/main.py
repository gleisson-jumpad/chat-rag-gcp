import streamlit as st
import psycopg2
import os

st.set_page_config(page_title="Teste DB", page_icon="🐘")

st.title("🔌 Teste de Conexão com PostgreSQL (GCP)")

try:
    connection_name = os.getenv("INSTANCE_CONNECTION_NAME")  # ex: chat-rag-v1:us-east4:chat-rag-db
    socket_dir = "/cloudsql"  # Caminho fixo no Cloud Run

    conn = psycopg2.connect(
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        host=f"/cloudsql/{os.getenv('INSTANCE_CONNECTION_NAME')}"
    )
    socket_path = f"/cloudsql/{os.getenv('INSTANCE_CONNECTION_NAME')}"
    print("🔍 Conteúdo do /cloudsql:")
    print(os.listdir("/cloudsql"))
    print("🔍 Verificando se socket existe:", os.path.exists(socket_path))
    st.code(socket_path)

    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]

    st.success("✅ Conexão estabelecida com sucesso! Vai com tudo!!!!")
    st.code(version)

    cursor.close()
    conn.close()

except Exception as e:
    import traceback
    st.error("❌ Erro ao conectar no banco de dados:")
    st.code(str(e))
    # Mostrar traceback no log (importante para Cloud Logging)
    print("Erro ao conectar no banco:")
    traceback.print_exc()
