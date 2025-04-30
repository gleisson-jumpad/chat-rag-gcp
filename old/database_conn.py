import os
import psycopg2
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()

def test_db_connection():
    try:
        connection = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            database=os.getenv("PG_DB"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            port=os.getenv("PG_PORT", 5432)
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        
        print(f"✅ Conexão com o banco estabelecida com sucesso!")
        print(f"Versão do banco: {db_version[0]}")
        
        cursor.close()
        connection.close()

    except Exception as e:
        print(f"❌ Erro ao conectar no banco de dados: {e}")

if __name__ == "__main__":
    test_db_connection()
