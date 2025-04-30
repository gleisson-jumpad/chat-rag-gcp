import os
import psycopg2
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env local (para testes locais)
load_dotenv()

def get_pg_connection():
    """
    Cria uma conexão com o PostgreSQL usando IP público.
    Retorna uma conexão ativa.
    """
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", 5432),
        dbname=os.getenv("PG_DB", "postgres"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD"),
        connect_timeout=5
    )
    return conn
