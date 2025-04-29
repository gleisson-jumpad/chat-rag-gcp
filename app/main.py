import streamlit as st
import psycopg2
import os
import platform
import socket
import subprocess
import time
import atexit
import traceback
import json
import urllib.request
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set your environment variables here if not already set
if not os.getenv("INSTANCE_CONNECTION_NAME"):
    # Replace these with your actual values
    os.environ["INSTANCE_CONNECTION_NAME"] = "chat-rag-v1:us-east4:chat-rag-db"
    os.environ["PG_DB"] = "postgres"  # or your database name
    os.environ["PG_USER"] = "postgres"  # or your username
    os.environ["PG_PASSWORD"] = "your_password_here"  # replace with your password
    # Add the public IP for direct connection testing
    # os.environ["DB_PUBLIC_IP"] = "34.72.123.456"  # Replace with your Cloud SQL public IP

st.set_page_config(page_title="PostgreSQL Connection Test", page_icon="🐘")
st.title("🔌 PostgreSQL Connection Test")

# Show system info
st.subheader("🖥️ System Information")
st.code(f"Platform: {platform.system()} {platform.release()}")
st.code(f"Python: {platform.python_version()}")
st.code(f"Instance Connection Name: {os.getenv('INSTANCE_CONNECTION_NAME')}")

# Detect environment
is_cloud_run = os.path.exists("/cloudsql")
st.code(f"Running in Cloud Run: {'✅' if is_cloud_run else '❌'}")
if is_cloud_run:
    st.code(f"VPC Connector: {os.getenv('VPC_CONNECTOR', 'Not configured')}")

# Function to start the proxy (only in local environment)
def start_cloud_sql_proxy():
    # Check if proxy exists
    proxy_exists = os.path.exists("./cloud-sql-proxy")
    if not proxy_exists:
        st.error("❌ Cloud SQL Auth Proxy not found in the current directory")
        st.info("Download it with: curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.darwin.arm64")
        return None, None
        
    connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
    socket_dir = "/tmp/cloudsql"
    
    # Ensure the directory exists
    os.makedirs(socket_dir, exist_ok=True)
    
    # Start the Cloud SQL Auth proxy
    cmd = f"./cloud-sql-proxy --unix-socket={socket_dir} {connection_name}"
    st.code(f"Running command: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    
    # Add a small delay to allow the proxy to start
    time.sleep(2)
    
    # Register the cleanup function
    atexit.register(lambda: process.terminate() if process else None)
    
    return process, socket_dir

# Check if running in Cloud Run
def is_running_in_cloud_run():
    try:
        # Cloud Run sets this environment variable
        return os.getenv("K_SERVICE") is not None
    except:
        return False

# Function to display VPC connector status when in Cloud Run
def check_vpc_connector():
    if is_running_in_cloud_run():
        try:
            vpc_connector = os.getenv("VPC_CONNECTOR")
            if vpc_connector:
                st.info(f"VPC Connector configured: {vpc_connector}")
            else:
                st.warning("No VPC Connector environment variable found")
                
            # Try to get Cloud Run metadata
            try:
                metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip"
                req = urllib.request.Request(metadata_url)
                req.add_header("Metadata-Flavor", "Google")
                ip = urllib.request.urlopen(req, timeout=2).read().decode('utf-8')
                st.info(f"Cloud Run internal IP: {ip}")
            except Exception as e:
                st.warning(f"Could not fetch Cloud Run metadata: {str(e)}")
        except Exception as e:
            st.error(f"Error checking VPC status: {str(e)}")

# Function to connect to Postgres and execute a query
def connect_to_postgres(in_cloud_run=False):
    # Connection details
    db_host = os.getenv("DB_PUBLIC_IP")
    db_name = os.getenv("PG_DB", "postgres")
    db_user = os.getenv("PG_USER", "postgres")
    db_password = os.getenv("PG_PASSWORD")
    instance_connection_name = os.getenv("INSTANCE_CONNECTION_NAME")

    # Validate instance_connection_name - a common issue
    if not instance_connection_name or instance_connection_name.count(":") != 2:
        st.error(f"Invalid INSTANCE_CONNECTION_NAME: {instance_connection_name}")
        st.warning("Format should be: PROJECT_ID:REGION:INSTANCE_NAME")
        
        # Try to fix if it's missing the instance name part
        if instance_connection_name and instance_connection_name.count(":") == 1:
            instance_connection_name = f"{instance_connection_name}:chat-rag-db"
            st.info(f"Attempting to fix by appending instance name: {instance_connection_name}")

    connection = None
    error_messages = []
    
    # Display environment for debugging
    st.subheader("Environment")
    st.write(f"INSTANCE_CONNECTION_NAME: {instance_connection_name}")
    st.write(f"DB_PUBLIC_IP: {db_host}")
    st.write(f"PG_DB: {db_name}")
    st.write(f"PG_USER: {db_user}")
    st.write(f"Running in Cloud Run: {in_cloud_run}")
    
    # Try different connection methods
    connection_methods = []
    
    # 1. First try the standard Cloud SQL Unix socket if in Cloud Run
    if in_cloud_run and instance_connection_name:
        # Standard Unix socket path in Cloud Run
        unix_socket = f"/cloudsql/{instance_connection_name}"
        
        # Debug information
        socket_exists = os.path.exists(unix_socket)
        socket_readable = os.access(unix_socket, os.R_OK) if socket_exists else False
        socket_writable = os.access(unix_socket, os.W_OK) if socket_exists else False
        socket_executable = os.access(unix_socket, os.X_OK) if socket_exists else False
        
        # Display the state of the socket
        st.write("Unix Socket Details:")
        st.json({
            "unix_socket_path": unix_socket,
            "exists": socket_exists,
            "readable": socket_readable,
            "writable": socket_writable,
            "executable": socket_executable
        })
        
        # List contents if possible
        if socket_exists:
            try:
                socket_contents = os.listdir(unix_socket)
                st.write(f"Contents of {unix_socket}: {socket_contents}")
            except Exception as e:
                st.write(f"Error listing {unix_socket}: {str(e)}")
        
        connection_methods.append({
            "method": "Cloud SQL Unix Socket",
            "details": {
                "socket_path": unix_socket,
                "socket_exists": socket_exists,
                "socket_readable": socket_readable,
                "socket_writable": socket_writable,
                "socket_executable": socket_executable
            }
        })
        
        try:
            st.info(f"Attempting connection via Unix socket: {unix_socket}")
            connection = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=unix_socket
            )
            return connection, None
        except Exception as e:
            error_detail = traceback.format_exc()
            error_messages.append(f"Unix socket connection failed: {str(e)}\n{error_detail}")
    
    # 2. Try PostgreSQL specific Unix socket path for Cloud Run
    if in_cloud_run and instance_connection_name:
        # PostgreSQL specific Unix socket path
        pg_socket = f"/cloudsql/{instance_connection_name}/.s.PGSQL.5432"
        pg_socket_dir = f"/cloudsql/{instance_connection_name}"
        
        # Check if directory exists first
        if os.path.exists(pg_socket_dir):
            st.write(f"PostgreSQL socket directory exists: {pg_socket_dir}")
            try:
                dir_contents = os.listdir(pg_socket_dir)
                st.write(f"Contents: {dir_contents}")
                
                # Check for the specific PostgreSQL socket file
                pg_socket_exists = '.s.PGSQL.5432' in dir_contents
                st.write(f"PostgreSQL socket file exists: {pg_socket_exists}")
            except Exception as e:
                st.write(f"Error listing PostgreSQL socket directory: {str(e)}")
        
        connection_methods.append({
            "method": "PostgreSQL Specific Socket",
            "details": {
                "socket_path": pg_socket,
                "directory_exists": os.path.exists(pg_socket_dir)
            }
        })
        
        try:
            st.info(f"Attempting connection via PostgreSQL specific socket")
            # For PostgreSQL, we still use the directory path but the driver will look for the .s.PGSQL.5432 file
            connection = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=pg_socket_dir  # Use the directory, not the actual socket file
            )
            return connection, None
        except Exception as e:
            error_detail = traceback.format_exc()
            error_messages.append(f"PostgreSQL specific socket connection failed: {str(e)}\n{error_detail}")

    # 3. Try private IP for Cloud Run with VPC connector
    if in_cloud_run and instance_connection_name:
        # Parse instance connection name to generate private IP hostname
        try:
            project, region, instance = instance_connection_name.split(":")
            private_host = f"{project}:{region}:{instance}"
            st.write(f"Private IP host: {private_host}")
            
            connection_methods.append({
                "method": "Private IP Connection",
                "details": {
                    "host": f"{project}-private.{region}.sql.gcp.internal",
                    "port": "5432"
                }
            })
            
            # First format with project-private
            try:
                st.info(f"Attempting connection via private IP: {project}-private.{region}.sql.gcp.internal")
                connection = psycopg2.connect(
                    dbname=db_name,
                    user=db_user,
                    password=db_password,
                    host=f"{project}-private.{region}.sql.gcp.internal",
                    port="5432"
                )
                return connection, None
            except Exception as e:
                st.warning(f"First private IP format failed: {str(e)}")
                
                # Try alternate format
                try:
                    alternate_host = f"{instance}-private.{region}.sql.gcp.internal"
                    st.info(f"Attempting alternate private IP format: {alternate_host}")
                    connection = psycopg2.connect(
                        dbname=db_name,
                        user=db_user,
                        password=db_password,
                        host=alternate_host,
                        port="5432"
                    )
                    return connection, None
                except Exception as alt_e:
                    error_detail = traceback.format_exc()
                    error_messages.append(f"Private IP connection failed: {str(e)} / {str(alt_e)}\n{error_detail}")
        except Exception as e:
            st.warning(f"Error parsing instance connection name: {str(e)}")

    # 4. Use public IP as fallback
    if db_host:
        connection_methods.append({
            "method": "Public IP Connection",
            "details": {
                "host": db_host,
                "port": "5432"
            }
        })
        
        try:
            st.info(f"Attempting connection via public IP: {db_host}")
            connection = psycopg2.connect(
                host=db_host,
                database=db_name,
                user=db_user,
                password=db_password
            )
            return connection, None
        except Exception as e:
            error_detail = traceback.format_exc()
            error_messages.append(f"Public IP connection failed: {str(e)}\n{error_detail}")
    
    # Display all connection methods tried
    st.subheader("Connection Methods Attempted")
    for method in connection_methods:
        st.write(f"Method: {method['method']}")
        st.json(method['details'])
    
    # If we're here, all connection methods failed
    combined_error = "\n\n".join(error_messages)
    return None, combined_error

# Main logic based on environment
try:
    if is_cloud_run:
        # In Cloud Run, the socket is already provided
        socket_dir = "/cloudsql"
        connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
        proxy_process = None
    else:
        # In local environment, start the proxy
        proxy_process, socket_dir = start_cloud_sql_proxy()
        if not socket_dir:
            st.error("Failed to start Cloud SQL Auth Proxy")
            st.stop()
        connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
    
    # Common path for both environments
    socket_path = f"{socket_dir}/{connection_name}"
    
    st.subheader("🔍 Connection Details:")
    st.code(f"Socket directory: {socket_dir}")
    st.code(f"Socket path: {socket_path}")
    
    # Check if directory exists
    if os.path.exists(socket_dir):
        try:
            contents = os.listdir(socket_dir)
            st.code(f"Contents of {socket_dir}: {contents}")
            
            # Check for PostgreSQL socket pattern
            postgresql_sockets = [f for f in contents if '.s.PGSQL.5432' in f]
            if postgresql_sockets:
                st.code(f"PostgreSQL sockets found: {postgresql_sockets}")
            
            if os.path.exists(socket_path):
                st.code(f"Socket exists: ✅")
                
                # Check for the socket file itself
                try:
                    subcontents = os.listdir(socket_path)
                    st.code(f"Contents of {socket_path}: {subcontents}")
                    
                    # Check for PostgreSQL socket file
                    if '.s.PGSQL.5432' in subcontents:
                        st.code(f"PostgreSQL socket file exists: ✅")
                    else:
                        st.code(f"PostgreSQL socket file missing: ❌")
                except Exception as subdir_err:
                    st.code(f"Error listing {socket_path}: {subdir_err}")
            else:
                st.code(f"Socket doesn't exist: ❌")
        except Exception as dir_err:
            st.code(f"Error listing {socket_dir}: {dir_err}")
    else:
        st.warning(f"Socket directory {socket_dir} doesn't exist.")
    
    # Try multiple connection approaches
    connection_methods = [
        {"name": "Standard Unix Socket", "path": socket_path, "cloud_run": is_cloud_run},
        # Add PostgreSQL specific socket path
        {"name": "With PostgreSQL Socket", "path": f"{socket_path}/.s.PGSQL.5432", "cloud_run": is_cloud_run},
    ]
    
    connected = False
    
    for method in connection_methods:
        if connected:
            break
            
        st.subheader(f"🔌 Trying: {method['name']}")
        try:
            conn = connect_to_postgres(method["cloud_run"])
            
            cursor = conn[0].cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            st.success(f"✅ Conexão estabelecida com sucesso via {method['name']}!")
            st.code(version)
            
            # Get some database stats
            cursor.execute("SELECT datname, pg_size_pretty(pg_database_size(datname)) FROM pg_database WHERE datistemplate = false;")
            databases = cursor.fetchall()
            
            st.subheader("📊 Database Sizes")
            for db in databases:
                st.code(f"{db[0]}: {db[1]}")
            
            cursor.close()
            conn[0].close()
            connected = True
            
        except Exception as method_err:
            st.error(f"❌ Falha ao conectar via {method['name']}:")
            st.code(str(method_err))
    
    # If still not connected, check for TCP connection as last resort
    if not connected and is_cloud_run:
        st.subheader("🔌 Trying: Private IP Connection")
        try:
            # Cloud Run should be able to connect via Private IP if VPC configured correctly
            conn = psycopg2.connect(
                dbname=os.getenv("PG_DB"),
                user=os.getenv("PG_USER"),
                password=os.getenv("PG_PASSWORD"),
                host=f"{connection_name.split(':')[0]}-private.{connection_name.split(':')[1]}.sql.gcp.internal",
                port="5432"
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            st.success("✅ Conexão estabelecida com sucesso via Private IP!")
            st.code(version)
            
            cursor.close()
            conn.close()
            connected = True
            
        except Exception as private_err:
            st.error("❌ Falha ao conectar via Private IP:")
            st.code(str(private_err))

except Exception as e:
    st.error("❌ Erro ao conectar no banco de dados:")
    st.code(str(e))
    
    # Show additional debug info
    st.subheader("📋 Debug Info")
    st.code(traceback.format_exc())
    
    # Check if we can connect to the instance's public IP as a fallback
    if os.getenv("DB_PUBLIC_IP"):
        st.subheader("🔄 Tentando conexão direta via IP público")
        try:
            conn = psycopg2.connect(
                dbname=os.getenv("PG_DB"),
                user=os.getenv("PG_USER"),
                password=os.getenv("PG_PASSWORD"),
                host=os.getenv("DB_PUBLIC_IP"),
                port="5432"
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            st.success("✅ Conexão direta estabelecida com sucesso!")
            st.code(version)
            
            cursor.close()
            conn.close()
        except Exception as direct_err:
            st.error("❌ Falha também na conexão direta:")
            st.code(str(direct_err))

# Important Notes for your specific GCP setup
st.subheader("📝 Troubleshooting Notes")
st.markdown("""
1. **VPC Connector**: Verifique se o VPC Connector `cloudrun-vpc-connector` está funcionando corretamente.

2. **Service Account**: Confirme que a Service Account `682048092511-compute@developer.gserviceaccount.com` tem o papel `Cloud SQL Client`.

3. **Socket no Cloud Run**: O PostgreSQL precisa do socket no formato `/cloudsql/INSTANCE_CONNECTION_NAME/.s.PGSQL.5432`.

4. **Redes Autorizadas**: Confirme que as redes autorizadas incluem `35.199.64.0/19` (faixa de IPs do Cloud Run).

5. **pgvector**: Embora configurado, a extensão ainda não foi testada.
""")

# Show current environment variables (Credentials masked)
st.subheader("🔐 Current Environment Variables")
st.code(f"INSTANCE_CONNECTION_NAME: {os.getenv('INSTANCE_CONNECTION_NAME')}")
st.code(f"PG_DB: {os.getenv('PG_DB')}")
st.code(f"PG_USER: {os.getenv('PG_USER')}")
st.code(f"PG_PASSWORD: {'*' * len(os.getenv('PG_PASSWORD')) if os.getenv('PG_PASSWORD') else 'Not set'}")
st.code(f"DB_PUBLIC_IP: {os.getenv('DB_PUBLIC_IP', 'Not set')}")

def main():
    st.title("PostgreSQL Connection Test")
    
    in_cloud_run = is_running_in_cloud_run()
    st.write(f"Running in Cloud Run: {in_cloud_run}")

    # Check VPC connector status if in Cloud Run
    if in_cloud_run:
        check_vpc_connector()
    
    # Cloud SQL directory check for Cloud Run
    if in_cloud_run:
        cloudsql_dir = "/cloudsql"
        st.subheader("Cloud SQL Directory Check")
        if os.path.exists(cloudsql_dir):
            st.success(f"{cloudsql_dir} directory exists")
            try:
                dir_contents = os.listdir(cloudsql_dir)
                if dir_contents:
                    st.json({"cloudsql_directory_contents": dir_contents})
                else:
                    st.warning(f"{cloudsql_dir} directory is empty")
                    st.error("Cloud SQL Auth Proxy may not be running correctly")
                    st.info("Check that --add-cloudsql-instances flag is set correctly in deployment")
                    
                    # Check if service account has proper permissions
                    st.subheader("Cloud SQL Service Account Check")
                    st.info("The service account needs the 'Cloud SQL Client' role")
                    st.code("gcloud projects add-iam-policy-binding PROJECT_ID --member=serviceAccount:SERVICE_ACCOUNT_EMAIL --role=roles/cloudsql.client")
                    
                    # List IAM permissions for this instance if possible
                    instance_name = os.getenv("INSTANCE_CONNECTION_NAME", "").split(":")[-1] if os.getenv("INSTANCE_CONNECTION_NAME") else ""
                    if instance_name:
                        st.info(f"Verify permissions for Cloud SQL instance: {instance_name}")
            except Exception as e:
                st.error(f"Error listing {cloudsql_dir} directory: {str(e)}")
        else:
            st.error(f"{cloudsql_dir} directory does not exist")
            
    # Add verification for Cloud SQL Admin API
    if in_cloud_run:
        st.subheader("Cloud SQL Admin API Check")
        st.info("Make sure the Cloud SQL Admin API is enabled in your project:")
        st.code("gcloud services enable sqladmin.googleapis.com")
        
    # Add proxy test section
    st.subheader("Cloud SQL Auth Proxy Test")
    if st.button("Test Auth Proxy Status"):
        if in_cloud_run:
            instance_connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
            if not instance_connection_name:
                st.error("INSTANCE_CONNECTION_NAME environment variable is not set")
            else:
                # Check if socket directory exists for this instance
                instance_socket_dir = f"/cloudsql/{instance_connection_name}"
                if os.path.exists(instance_socket_dir):
                    try:
                        socket_files = os.listdir(instance_socket_dir)
                        if socket_files:
                            st.success(f"Socket directory contains: {socket_files}")
                            if '.s.PGSQL.5432' in socket_files:
                                st.success("✅ PostgreSQL socket file exists!")
                            else:
                                st.warning("❌ PostgreSQL socket file (.s.PGSQL.5432) is missing")
                        else:
                            st.warning("Socket directory exists but is empty")
                            st.info("This suggests Cloud SQL Auth Proxy isn't creating socket files")
                    except Exception as e:
                        st.error(f"Error checking socket directory: {str(e)}")
                else:
                    st.error(f"Socket directory for this instance doesn't exist: {instance_socket_dir}")
                    st.info("This suggests that Cloud SQL Auth Proxy isn't correctly configured")
        else:
            # Local environment check
            st.info("In local environment, check Cloud SQL Auth Proxy manually")
            st.code("./cloud-sql-proxy --unix-socket=/tmp/cloudsql INSTANCE_CONNECTION_NAME")
    
    # Track connection results for summary
    connection_summary = {
        "unix_socket": {"tried": False, "success": False, "error": None},
        "tcp_socket": {"tried": False, "success": False, "error": None},
        "public_ip": {"tried": False, "success": False, "error": None},
        "private_ip": {"tried": False, "success": False, "error": None}
    }
    
    if st.button("Test Connection"):
        connection, error = connect_to_postgres(in_cloud_run)
        
        # First attempt - Unix socket
        connection_summary["unix_socket"]["tried"] = True
        
        if connection:
            try:
                cursor = connection[0].cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                cursor.close()
                connection[0].close()
                st.success("✅ Connection successful!")
                st.write(f"PostgreSQL Version: {version[0]}")
                connection_summary["unix_socket"]["success"] = True
            except Exception as e:
                st.error(f"❌ Query execution failed: {str(e)}")
                connection_summary["unix_socket"]["error"] = str(e)
        else:
            st.error("❌ Connection failed")
            st.error(error)
            connection_summary["unix_socket"]["error"] = error
            
            # If Unix socket failed, try public IP
            if os.getenv("DB_PUBLIC_IP"):
                connection_summary["public_ip"]["tried"] = True
                st.subheader("🔄 Trying direct connection via public IP")
                try:
                    conn = psycopg2.connect(
                        dbname=os.getenv("PG_DB"),
                        user=os.getenv("PG_USER"),
                        password=os.getenv("PG_PASSWORD"),
                        host=os.getenv("DB_PUBLIC_IP"),
                        port="5432"
                    )
                    
                    cursor = conn.cursor()
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                    
                    st.success("✅ Public IP connection successful!")
                    st.code(version)
                    
                    cursor.close()
                    conn.close()
                    connection_summary["public_ip"]["success"] = True
                except Exception as direct_err:
                    st.error("❌ Public IP connection failed:")
                    st.code(str(direct_err))
                    connection_summary["public_ip"]["error"] = str(direct_err)
            
            # If in Cloud Run, try private IP
            if in_cloud_run and os.getenv("INSTANCE_CONNECTION_NAME"):
                connection_summary["private_ip"]["tried"] = True
                st.subheader("🔄 Trying private IP connection")
                try:
                    connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
                    conn = psycopg2.connect(
                        dbname=os.getenv("PG_DB"),
                        user=os.getenv("PG_USER"),
                        password=os.getenv("PG_PASSWORD"),
                        host=f"{connection_name.split(':')[0]}-private.{connection_name.split(':')[1]}.sql.gcp.internal",
                        port="5432"
                    )
                    
                    cursor = conn.cursor()
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                    
                    st.success("✅ Private IP connection successful!")
                    st.code(version)
                    
                    cursor.close()
                    conn.close()
                    connection_summary["private_ip"]["success"] = True
                except Exception as private_err:
                    st.error("❌ Private IP connection failed:")
                    st.code(str(private_err))
                    connection_summary["private_ip"]["error"] = str(private_err)
            
            # Display additional troubleshooting information
            st.subheader("Troubleshooting Notes")
            st.markdown("""
            ### For Cloud Run deployments:
            
            1. **VPC Connector**: Ensure your Cloud Run service has a VPC connector configured correctly:
               - Verify that 'cloudrun-vpc-connector' is properly set up and attached to the service
               - Check that egress is set to 'private-ranges-only'
            
            2. **Service Account**: Make sure your service account (682048092511-compute@developer.gserviceaccount.com) has:
               - Cloud SQL Client role
               - Proper IAM permissions for the Cloud SQL instance
            
            3. **Socket Format**: Cloud SQL Unix sockets can be accessed via:
               - Standard path: `/cloudsql/[INSTANCE_CONNECTION_NAME]`
               - PostgreSQL specific format: `/cloudsql/[INSTANCE_CONNECTION_NAME]/.s.PGSQL.5432`
            
            4. **Cloud Run Configuration**:
               - Verify the `--add-cloudsql-instances` flag was used in deployment
               - Ensure all necessary environment variables are set
            
            ### General Checks:
            - Check if Cloud SQL instance is running and accessible
            - Verify network connectivity (firewall rules, IP addresses)
            - Confirm correct credentials (username/password)
            """)
        
        # Display comprehensive connection summary at the bottom
        st.subheader("📊 Connection Results Summary")
        
        # Create a formatted summary table
        summary_md = """
        | Connection Method | Attempted | Result | Error (if any) |
        |------------------|-----------|--------|----------------|
        """
        
        for method, details in connection_summary.items():
            method_name = method.replace("_", " ").title()
            attempted = "✅" if details["tried"] else "❌"
            result = "✅ Success" if details["success"] else ("❌ Failed" if details["tried"] else "Not Attempted")
            error = details["error"] if details["error"] else "-"
            
            if isinstance(error, str) and len(error) > 50:
                error = error[:50] + "..."
                
            summary_md += f"| {method_name} | {attempted} | {result} | {error} |\n"
        
        st.markdown(summary_md)
        
        # Add final recommendation
        if any(details["success"] for details in connection_summary.values()):
            st.success("✅ At least one connection method worked! You can proceed with your application.")
            
            # Show which method worked best
            working_methods = [method.replace("_", " ").title() for method, details in connection_summary.items() if details["success"]]
            st.info(f"Working connection methods: {', '.join(working_methods)}")
            
            if connection_summary["unix_socket"]["success"]:
                st.info("💡 Unix Socket connection is the recommended method for Cloud Run deployments.")
            elif connection_summary["private_ip"]["success"]:
                st.info("💡 Private IP connection is working. This is good for VPC networking.")
            elif connection_summary["public_ip"]["success"]:
                st.warning("⚠️ Only Public IP connection is working. Consider securing your connection with private networking.")
        else:
            st.error("❌ All connection methods failed. Please review the troubleshooting notes.")

if __name__ == "__main__":
    main()
