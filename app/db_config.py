import os
import psycopg2
from dotenv import load_dotenv

# Carrega variáveis do .env (para testes locais)
load_dotenv()

def get_pg_connection():
    """
    Cria conexão com o PostgreSQL via IP público (sem usar socket).
    Usa as variáveis de ambiente: DB_PUBLIC_IP, PG_PORT, PG_DB, PG_USER, PG_PASSWORD.
    """
    try:
        host = os.getenv("DB_PUBLIC_IP")
        port = int(os.getenv("PG_PORT", 5432))
        dbname = os.getenv("PG_DB", "postgres")
        user = os.getenv("PG_USER", "postgres")
        password = os.getenv("PG_PASSWORD")

        if not all([host, dbname, user, password]):
            raise ValueError("Alguma variável de ambiente necessária não foi definida.")

        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=5
        )

        return conn

    except Exception as e:
        raise RuntimeError(f"Erro ao conectar ao PostgreSQL via IP: {e}")
