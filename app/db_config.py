import os
import psycopg2
import logging # É bom manter o logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Se você usa .env para testes locais, pode manter load_dotenv
# from dotenv import load_dotenv
# load_dotenv()

def get_pg_connection():
    """
    Cria conexão com o PostgreSQL via IP público.
    Usa as variáveis de ambiente: DB_PUBLIC_IP, PG_PORT, PG_DB, PG_USER, PG_PASSWORD.
    """
    try:
        # Variáveis de ambiente para conexão via IP Público
        host = os.getenv("DB_PUBLIC_IP")
        port = int(os.getenv("PG_PORT", 5432)) # Porta padrão do Postgres
        dbname = os.getenv("PG_DB")
        user = os.getenv("PG_USER")
        password = os.getenv("PG_PASSWORD")

        logging.info(f"Tentando conectar via IP público: {host}:{port} DB: {dbname} User: {user}")

        # Validação das variáveis necessárias para este método
        if not all([host, dbname, user, password]):
            logging.error("Erro Crítico: Variáveis DB_PUBLIC_IP, PG_DB, PG_USER ou PG_PASSWORD não definidas.")
            raise ValueError("Alguma variável de ambiente necessária para conexão via IP não foi definida (DB_PUBLIC_IP, PG_DB, PG_USER, PG_PASSWORD).")

        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=30 # Manter um timeout é bom
        )
        logging.info("✅ Conexão via IP público bem-sucedida!")
        return conn

    except psycopg2.OperationalError as e:
         logging.error(f"Erro operacional do Psycopg2 ao tentar conectar via IP: {e}")
         raise RuntimeError(f"Falha ao conectar ao PostgreSQL via IP. Verifique a rede e as credenciais. Erro: {e}") from e
    except Exception as e:
        logging.error(f"Erro inesperado ao configurar a conexão via IP: {e}")
        raise RuntimeError(f"Erro inesperado durante a conexão ao banco de dados via IP: {e}") from e