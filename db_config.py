import os
import psycopg2
import psycopg2.pool
import threading
import logging # É bom manter o logging
import pathlib
import time
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager

# Configure logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carregar variáveis do .env - verificar tanto no diretório da aplicação quanto no diretório raiz
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

# Global connection pool
_connection_pool = None
_pool_lock = threading.Lock()

def get_connection_params() -> Dict[str, Any]:
    """
    Get database connection parameters from environment variables.
    
    Returns:
        Dict with connection parameters
    """
    # Get connection parameters from environment variables
    db_host = os.getenv("DB_PUBLIC_IP")
    db_port = os.getenv("PG_PORT")
    db_name = os.getenv("PG_DB")
    db_user = os.getenv("PG_USER")
    db_password = os.getenv("PG_PASSWORD")
    
    # Validate required parameters
    missing = []
    if not db_host: missing.append("DB_PUBLIC_IP")
    if not db_port: missing.append("PG_PORT")
    if not db_name: missing.append("PG_DB")
    if not db_user: missing.append("PG_USER")
    if not db_password: missing.append("PG_PASSWORD")
    
    if missing:
        error_msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Return the parameters as a dictionary
    return {
        "host": db_host,
        "port": int(db_port),
        "dbname": db_name,
        "user": db_user,
        "password": db_password,
    }

def initialize_connection_pool(min_connections: int = 2, max_connections: int = 10) -> None:
    """
    Initialize a global connection pool.
    
    Args:
        min_connections: Minimum number of connections in the pool
        max_connections: Maximum number of connections in the pool
    """
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            logger.info("Connection pool already initialized")
            return
            
        try:
            # Get connection parameters
            conn_params = get_connection_params()
            
            logger.info(f"Initializing connection pool with {min_connections}-{max_connections} connections")
            
            # Create the connection pool
            _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                **conn_params
            )
            
            # Test the pool by getting and returning a connection
            test_conn = _connection_pool.getconn()
            _connection_pool.putconn(test_conn)
            
            logger.info(f"PostgreSQL connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            _connection_pool = None
            raise

def get_pg_connection() -> psycopg2.extensions.connection:
    """
    Get a connection to the PostgreSQL database.
    Returns a connection from the pool if available, or creates a new one if not.
    
    Returns:
        PostgreSQL connection
    """
    global _connection_pool
    
    # Try to get a connection from the pool
    if _connection_pool is not None:
        try:
            conn = _connection_pool.getconn()
            logger.debug("Got connection from pool")
            return conn
        except Exception as e:
            logger.warning(f"Error getting connection from pool: {str(e)}")
            # Fall through to direct connection creation
    
    # If pool is not available or failed, create a direct connection
    try:
        logger.info("Creating direct database connection (pool not available)")
        conn_params = get_connection_params()
        connection = psycopg2.connect(**conn_params)
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
        raise

def return_pg_connection(conn: psycopg2.extensions.connection) -> None:
    """
    Return a connection to the pool if it came from the pool,
    otherwise close it.
    
    Args:
        conn: The connection to return or close
    """
    global _connection_pool
    
    if conn is None:
        return
        
    try:
        if _connection_pool is not None:
            _connection_pool.putconn(conn)
            logger.debug("Returned connection to pool")
        else:
            conn.close()
            logger.debug("Closed direct connection")
    except Exception as e:
        logger.warning(f"Error returning connection: {str(e)}")
        try:
            conn.close()
        except:
            pass

@contextmanager
def get_pg_cursor(commit: bool = False) -> psycopg2.extensions.cursor:
    """
    Context manager for getting a database cursor, automatically handling
    connection acquisition and release.
    
    Args:
        commit: Whether to commit the transaction before closing
        
    Yields:
        A database cursor
    """
    conn = None
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        yield cursor
        if commit:
            conn.commit()
    finally:
        if cursor:
            cursor.close()
        if conn:
            return_pg_connection(conn)

def check_postgres_connection() -> Dict[str, Any]:
    """
    Check PostgreSQL connection status and pgvector availability
    
    Returns:
        Dict with connection status information
    """
    result = {
        "postgres_connection": False,
        "pgvector_installed": False,
        "vector_table_count": 0,
        "error": None
    }
    
    conn = None
    start_time = time.time()
    
    try:
        # Try to establish connection
        conn = get_pg_connection()
        result["postgres_connection"] = True
        result["connection_time_ms"] = int((time.time() - start_time) * 1000)
        
        cursor = conn.cursor()
        
        # Get PostgreSQL version
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0]
        result["postgres_version"] = pg_version
        
        # Check pgvector extension
        try:
            cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            pgvector_info = cursor.fetchone()
            if pgvector_info:
                result["pgvector_installed"] = True
                result["pgvector_version"] = pgvector_info[1]
                
                # Check if pgvector is configured for HNSW
                try:
                    cursor.execute("SELECT setting FROM pg_settings WHERE name = 'vector.hnsw_ef_search';")
                    hnsw_setting = cursor.fetchone()
                    if hnsw_setting:
                        result["pgvector_hnsw_enabled"] = True
                        result["pgvector_hnsw_ef_search"] = hnsw_setting[0]
                except Exception as hnsw_error:
                    logger.warning(f"Error checking pgvector HNSW settings: {str(hnsw_error)}")
                    result["pgvector_hnsw_error"] = str(hnsw_error)
                    result["pgvector_hnsw_enabled"] = False
            else:
                result["pgvector_missing"] = True
                result["pgvector_installation_cmd"] = "CREATE EXTENSION vector;"
                
                # Check if the extension is available but not installed
                cursor.execute("SELECT * FROM pg_available_extensions WHERE name = 'vector';")
                if cursor.fetchone():
                    result["pgvector_available"] = True
                else:
                    result["pgvector_available"] = False
                    result["pgvector_install_instructions"] = "Install pgvector using your system package manager or build from source."
        except Exception as ve:
            logger.warning(f"Error checking pgvector extension: {str(ve)}")
            result["pgvector_error"] = str(ve)
        
        # Check vector tables
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' AND 
                (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
            """)
            vector_table_count = cursor.fetchone()[0]
            result["vector_table_count"] = vector_table_count
            
            # If tables exist, get their names and basic information
            if vector_table_count > 0:
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND 
                    (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
                """)
                tables = [table[0] for table in cursor.fetchall()]
                result["vector_tables"] = tables
                
                # Get row counts for each table
                table_stats = {}
                for table_name in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        row_count = cursor.fetchone()[0]
                        
                        # Check for unique file_names in this table
                        cursor.execute(f"""
                            SELECT COUNT(DISTINCT metadata_->>'file_name') 
                            FROM {table_name}
                            WHERE metadata_->>'file_name' IS NOT NULL
                        """)
                        file_count = cursor.fetchone()[0]
                        
                        # Check if table has a vector index
                        cursor.execute(f"""
                            SELECT indexname, indexdef 
                            FROM pg_indexes 
                            WHERE tablename = '{table_name}' AND indexdef LIKE '%embedding%'
                        """)
                        indices = cursor.fetchall()
                        has_index = len(indices) > 0
                        
                        table_stats[table_name] = {
                            "row_count": row_count,
                            "file_count": file_count,
                            "has_index": has_index,
                            "indices": [idx[0] for idx in indices] if has_index else []
                        }
                    except Exception as ts_error:
                        logger.warning(f"Error getting stats for table {table_name}: {str(ts_error)}")
                        table_stats[table_name] = {"error": str(ts_error)}
                
                result["table_stats"] = table_stats
        except Exception as te:
            logger.warning(f"Error checking vector tables: {str(te)}")
            result["table_error"] = str(te)
        
        # Close cursor
        cursor.close()
        
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        logger.error(f"Database connection check failed: {str(e)}")
    finally:
        # Return connection to pool or close it
        if conn:
            return_pg_connection(conn)
    
    return result

def ensure_pgvector_extension() -> bool:
    """
    Ensure the pgvector extension is installed in the database.
    
    Returns:
        True if the extension is installed or was successfully installed,
        False otherwise
    """
    with get_pg_cursor(commit=True) as cursor:
        try:
            # Check if pgvector is already installed
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            if cursor.fetchone():
                logger.info("pgvector extension is already installed")
                return True
                
            # Try to install the extension
            logger.info("Installing pgvector extension")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Verify installation
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            if cursor.fetchone():
                logger.info("Successfully installed pgvector extension")
                return True
            else:
                logger.error("Failed to verify pgvector installation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install pgvector extension: {str(e)}")
            return False

def verify_vector_table(table_name: str) -> Dict[str, Any]:
    """
    Verify a vector table and its configuration
    
    Args:
        table_name: The name of the table to verify
        
    Returns:
        Dict with table verification results
    """
    result = {
        "table_name": table_name,
        "exists": False,
        "vector_column": False,
        "metadata_column": False,
        "has_index": False,
        "row_count": 0
    }
    
    with get_pg_cursor() as cursor:
        try:
            # Check if table exists
            cursor.execute(f"""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                );
            """, (table_name,))
            
            if not cursor.fetchone()[0]:
                logger.warning(f"Table {table_name} does not exist")
                return result
                
            result["exists"] = True
            
            # Check for embedding vector column
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s AND column_name = 'embedding';
            """, (table_name,))
            
            embedding_col = cursor.fetchone()
            result["vector_column"] = embedding_col is not None
            if embedding_col:
                result["vector_column_type"] = embedding_col[1]
            
            # Check for metadata column
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s AND column_name = 'metadata_';
            """, (table_name,))
            
            metadata_col = cursor.fetchone()
            result["metadata_column"] = metadata_col is not None
            if metadata_col:
                result["metadata_column_type"] = metadata_col[1]
            
            # Check for vector index
            cursor.execute(f"""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = %s AND indexdef LIKE '%%embedding%%'
            """, (table_name,))
            
            indices = cursor.fetchall()
            result["has_index"] = len(indices) > 0
            if result["has_index"]:
                result["indices"] = []
                for idx in indices:
                    index_info = {"name": idx[0], "definition": idx[1]}
                    
                    # Check if it's an HNSW index
                    if "hnsw" in idx[1].lower():
                        index_info["type"] = "hnsw"
                    elif "ivfflat" in idx[1].lower():
                        index_info["type"] = "ivfflat"
                    else:
                        index_info["type"] = "standard"
                        
                    result["indices"].append(index_info)
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result["row_count"] = cursor.fetchone()[0]
            
            # Check for unique documents
            cursor.execute(f"""
                SELECT COUNT(DISTINCT metadata_->>'file_name') 
                FROM {table_name}
                WHERE metadata_->>'file_name' IS NOT NULL
            """)
            result["document_count"] = cursor.fetchone()[0]
            
            # Get sample document names
            cursor.execute(f"""
                SELECT DISTINCT metadata_->>'file_name' as file_name
                FROM {table_name}
                WHERE metadata_->>'file_name' IS NOT NULL
                LIMIT 10
            """)
            result["sample_documents"] = [row[0] for row in cursor.fetchall() if row[0]]
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying table {table_name}: {str(e)}")
            result["error"] = str(e)
            return result

# Initialize the connection pool when the module is imported
try:
    initialize_connection_pool()
except Exception as e:
    logger.warning(f"Failed to initialize connection pool on module import: {str(e)}")
    logger.info("Connections will be created directly when needed")