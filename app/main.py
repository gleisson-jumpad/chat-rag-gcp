import streamlit as st
import psycopg2
import os

st.set_page_config(page_title="Teste DB", page_icon="🐘")
st.title("🔌 Teste de Conexão com PostgreSQL (GCP)")

try:
    connection_name = os.getenv("INSTANCE_CONNECTION_NAME")  # ex: chat-rag-v1:us-east4:chat-rag-db
    socket_dir = "/cloudsql"
    socket_path = f"{socket_dir}/{connection_name}"

    # 🕵️ Verificar se o socket está realmente montado
    st.subheader("🔍 Verificando montagem do socket:")
    try:
        contents = os.listdir(socket_dir)
        st.code(f"Conteúdo de /cloudsql: {contents}")
    except Exception as dir_err:
        st.code(f"Erro ao listar /cloudsql: {dir_err}")

    st.code(f"Socket esperado: {socket_path}")
    st.code(f"Socket existe? {os.path.exists(socket_path)}")

    # ✅ Só tenta conectar se o socket existir
    if not os.path.exists(socket_path):
        raise RuntimeError("Socket não está montado dentro do container!")

    # Conexão via socket Unix
    conn = psycopg2.connect(
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        host=socket_path
    )

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
    print("Erro ao conectar no banco:")
    traceback.print_exc()
