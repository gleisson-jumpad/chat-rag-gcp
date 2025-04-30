import os
import psycopg2
import logging # É bom manter o logging
import pathlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar variáveis do .env - verificar tanto no diretório da aplicação quanto no diretório raiz
from dotenv import load_dotenv

# Verificar primeiro no diretório da aplicação
env_app_path = pathlib.Path(__file__).parent / '.env'
# Verificar também no diretório raiz (um nível acima)
env_root_path = pathlib.Path(__file__).parent.parent / '.env'

if env_app_path.exists():
    logging.info(f"Carregando variáveis do arquivo .env do diretório app")
    load_dotenv(dotenv_path=env_app_path)
elif env_root_path.exists():
    logging.info(f"Carregando variáveis do arquivo .env do diretório raiz")
    load_dotenv(dotenv_path=env_root_path)
else:
    logging.info(f"Arquivo .env não encontrado, usando variáveis de ambiente existentes")

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
        
        # Debug - verifica OPENAI_API_KEY (sem mostrar o valor completo)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            logging.info(f"OPENAI_API_KEY está definida (primeiros 8 caracteres: {api_key[:8]}...)")
        else:
            logging.warning("OPENAI_API_KEY não está definida!")

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