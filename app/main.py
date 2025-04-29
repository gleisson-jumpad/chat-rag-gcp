import streamlit as st
import psycopg2
import os

st.set_page_config(page_title="Teste DB", page_icon="🐘")

st.title("🔌 Teste de Conexão com PostgreSQL (GCP)")

try:
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT", "5432"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        connect_timeout=5
    )
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]

    st.success("✅ Conexão estabelecida com sucesso!")
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
