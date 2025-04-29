import streamlit as st
import psycopg2
import os
import json

st.set_page_config(page_title="Direct PostgreSQL Test", page_icon="🔌")
st.title("🔌 Direct PostgreSQL Connection Test")

# Get instance info from the command
with st.expander("Cloud SQL Instance Info"):
    instance_name = st.text_input("Instance Name", "chat-rag-db")
    region = st.text_input("Region", "us-east4")
    project_id = st.text_input("Project ID", "chat-rag-v1")
    public_ip = st.text_input("Public IP", "34.48.95.143")
    
    # Full connection string
    full_connection = f"{project_id}:{region}:{instance_name}"
    st.code(f"Full connection string: {full_connection}")
    
    # Generate proper connect commands
    st.subheader("Connection commands")
    st.code(f"gcloud sql connect {instance_name} --user=postgres")
    
    # Update environment variables
    if st.button("Update Environment Variables"):
        os.environ["INSTANCE_CONNECTION_NAME"] = full_connection
        os.environ["DB_PUBLIC_IP"] = public_ip
        st.success("Environment variables updated!")

# Define test connection options
st.subheader("Connection Test Options")

# Test public IP connection
if st.button("Test Direct Public IP Connection"):
    try:
        st.info(f"Attempting connection to public IP: {public_ip}")
        
        connection = psycopg2.connect(
            host=public_ip,
            database="postgres",
            user="postgres",
            password=st.secrets.get("db_password", os.environ.get("PG_PASSWORD", ""))
        )
        
        # Execute a simple query
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        st.success("✅ Public IP connection successful!")
        st.json({"version": version})
        
        # Get database info
        cursor.execute("SELECT datname FROM pg_database;")
        databases = cursor.fetchall()
        st.write("Available databases:")
        st.json({"databases": [db[0] for db in databases]})
        
        # Close connection
        cursor.close()
        connection.close()
        
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")
        
# Testing flags for Cloud Run
st.subheader("Cloud Run Configuration")
is_cloud_run = os.path.exists("/cloudsql")
st.write(f"Running in Cloud Run: {is_cloud_run}")

if is_cloud_run:
    # Check the instance connection name
    instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME", "")
    st.write(f"INSTANCE_CONNECTION_NAME: {instance_connection_name}")
    
    # Check if the socket directory exists
    socket_dir = f"/cloudsql/{instance_connection_name}"
    socket_exists = os.path.exists(socket_dir)
    st.write(f"Socket directory exists: {socket_exists}")
    
    if socket_exists:
        try:
            socket_contents = os.listdir(socket_dir)
            st.write(f"Socket directory contents: {socket_contents}")
        except Exception as e:
            st.error(f"Error listing socket directory: {str(e)}")
    
    # Test connection using unix socket
    if st.button("Test Unix Socket Connection"):
        try:
            socket_path = f"/cloudsql/{instance_connection_name}"
            st.info(f"Attempting connection via Unix socket: {socket_path}")
            
            connection = psycopg2.connect(
                host=socket_path,
                database="postgres",
                user="postgres",
                password=st.secrets.get("db_password", os.environ.get("PG_PASSWORD", ""))
            )
            
            cursor = connection.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            st.success("✅ Unix socket connection successful!")
            st.code(version)
            
            cursor.close()
            connection.close()
            
        except Exception as e:
            st.error(f"❌ Unix socket connection failed: {str(e)}")

# View current environment variables
st.subheader("Current Environment Variables")
env_vars = {
    "INSTANCE_CONNECTION_NAME": os.environ.get("INSTANCE_CONNECTION_NAME", "Not set"),
    "PG_DB": os.environ.get("PG_DB", "Not set"),
    "PG_USER": os.environ.get("PG_USER", "Not set"),
    "PG_PASSWORD": "*****" if os.environ.get("PG_PASSWORD") else "Not set",
    "PG_HOST": os.environ.get("PG_HOST", "Not set"),
    "DB_PUBLIC_IP": os.environ.get("DB_PUBLIC_IP", "Not set")
}
st.json(env_vars)

# Show connection options
st.subheader("Connection Options")

# Allow user to test different connection string formats
custom_connection_string = st.text_input(
    "Custom Connection String", 
    value=f"dbname=postgres user=postgres password=***** host={public_ip}"
)

if st.button("Test Custom Connection"):
    try:
        # Mask the displayed connection string if it contains a password
        display_string = custom_connection_string
        if "password" in custom_connection_string:
            parts = custom_connection_string.split()
            for i, part in enumerate(parts):
                if part.startswith("password="):
                    parts[i] = "password=*****"
            display_string = " ".join(parts)
            
        st.info(f"Attempting connection with: {display_string}")
        
        # Create the actual connection
        connection = psycopg2.connect(custom_connection_string.replace("*****", os.environ.get("PG_PASSWORD", "")))
        
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        st.success("✅ Custom connection successful!")
        st.code(version)
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        st.error(f"❌ Custom connection failed: {str(e)}") 