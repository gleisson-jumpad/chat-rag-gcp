import os
import psycopg2
import logging

# Configura o logging para melhor depuração no Cloud Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load_dotenv() geralmente não é necessário/usado no ambiente Cloud Run,
# as variáveis são injetadas diretamente. Pode remover ou deixar comentado.
# from dotenv import load_dotenv
# load_dotenv()

def get_pg_connection():
    """
    Cria conexão com o PostgreSQL. Prioriza conexão via Cloud SQL socket UNIX
    se a variável de ambiente INSTANCE_CONNECTION_NAME estiver definida (método
    recomendado para Cloud Run com --add-cloudsql-instances).

    Variáveis de ambiente necessárias:
    - PG_DB: Nome do banco de dados.
    - PG_USER: Usuário do banco de dados.
    - PG_PASSWORD: Senha do banco de dados.
    - INSTANCE_CONNECTION_NAME: (Opcional, mas necessário para método de socket)
                                Formato: 'project:region:instance'
    """
    instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME")
    db_user = os.environ.get("PG_USER")
    db_password = os.environ.get("PG_PASSWORD")
    db_name = os.environ.get("PG_DB")

    # Validação inicial das variáveis essenciais
    if not db_user or not db_password or not db_name:
        logging.error("Erro Crítico: Variáveis PG_USER, PG_PASSWORD ou PG_DB não estão definidas.")
        raise ValueError("Variáveis de ambiente PG_USER, PG_PASSWORD, PG_DB devem ser definidas.")

    conn = None
    try:
        # --- Método 1: Conexão via Socket UNIX (Preferencial no Cloud Run) ---
        if instance_connection_name:
            # O caminho do socket é sempre /cloudsql/INSTANCE_CONNECTION_NAME
            socket_dir = '/cloudsql'
            db_socket_path = f"{socket_dir}/{instance_connection_name}"

            logging.info(f"Tentando conectar via Cloud SQL socket UNIX: {db_socket_path}")

            conn = psycopg2.connect(
                host=db_socket_path,  # IMPORTANTE: Usar o caminho do socket como 'host'
                dbname=db_name,
                user=db_user,
                password=db_password,
                # connect_timeout ainda pode ser útil
                connect_timeout=5
            )
            logging.info("✅ Conexão via Cloud SQL socket UNIX bem-sucedida!")
            return conn

        # --- Método 2: Fallback via IP (NÃO RECOMENDADO com vpc-egress=private-ranges-only) ---
        # Se INSTANCE_CONNECTION_NAME não for fornecido, você pode optar por lançar um erro
        # ou tentar conectar via IP (mas isso falhará com a configuração atual do Cloud Run)
        else:
            logging.warning("Aviso: INSTANCE_CONNECTION_NAME não definida. A conexão via socket não é possível.")
            # Você pode levantar um erro aqui se o socket for o único método esperado no Cloud Run
            raise ValueError("INSTANCE_CONNECTION_NAME não definida. Configure o deploy para usar conexão via socket.")

            # OU (NÃO RECOMENDADO COM A CONFIG ATUAL): Tentar conectar via IP como antes
            # host = os.getenv("DB_PUBLIC_IP")
            # port = int(os.getenv("PG_PORT", 5432))
            # if not host:
            #     raise ValueError("DB_PUBLIC_IP não definido para fallback de conexão via IP.")
            # logging.info(f"Tentando conectar via IP (fallback): {host}:{port}")
            # conn = psycopg2.connect(...)
            # return conn

    except psycopg2.OperationalError as e:
        # Erros operacionais são comuns para problemas de conexão (rede, autenticação, db não existe, permissão no socket)
        method = "socket UNIX" if instance_connection_name else "IP (fallback)"
        logging.error(f"Erro operacional do Psycopg2 ao tentar conectar via {method}: {e}")
        # Verificar permissões no diretório /cloudsql/ se usar socket.
        # Verificar se a instância Cloud SQL está rodando.
        # Verificar se o usuário/senha estão corretos.
        raise RuntimeError(f"Falha ao conectar ao PostgreSQL via {method}. Verifique os logs e a configuração. Erro: {e}") from e
    except Exception as e:
        # Captura outras exceções (ValueError, etc.)
        logging.error(f"Erro inesperado ao configurar a conexão: {e}")
        raise RuntimeError(f"Erro inesperado durante a conexão ao banco de dados: {e}") from e