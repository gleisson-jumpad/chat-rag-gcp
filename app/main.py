import streamlit as st
import psycopg2
import os

st.set_page_config(page_title="Teste DB", page_icon="🐘")

st.title("🔌 Teste de Conexão com PostgreSQL (GCP)")

try:
    conn = psycopg2.connect(
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        unix_sock=f"/cloudsql/{os.getenv('INSTANCE_CONNECTION_NAME')}/.s.PGSQL.5432"
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
