import streamlit as st
import psycopg2
import os
import sys
import subprocess

st.set_page_config(page_title="Simple DB Test", page_icon="🔍")
st.title("Simple Database Connection Test")

# Basic environment info
conn_name = os.environ.get("INSTANCE_CONNECTION_NAME", "Not set")
password = "***" if os.environ.get("PG_PASSWORD") else "Not set"

with st.expander("Environment"):
    st.write(f"**INSTANCE_CONNECTION_NAME**: {conn_name}")
    st.write(f"**PG_PASSWORD**: {password}")
    st.write(f"**Running in Cloud Run**: {os.path.exists('/cloudsql')}")

# Socket check
st.subheader("1. Unix Socket Check")
socket_path = f"/cloudsql/{conn_name}"

if os.path.exists(socket_path):
    st.success(f"✅ Socket path exists: {socket_path}")
    
    # Check contents
    try:
        contents = os.listdir(socket_path)
        if '.s.PGSQL.5432' in contents:
            st.success("✅ PostgreSQL socket file exists!")
        else:
            st.error(f"❌ PostgreSQL socket file not found. Contents: {contents}")
    except Exception as e:
        st.error(f"Error reading socket directory: {str(e)}")
else:
    st.error(f"❌ Socket path doesn't exist: {socket_path}")

# Simple connection tests
st.subheader("2. Connection Tests")

col1, col2 = st.columns(2)

with col1:
    st.write("**Unix Socket Connection**")
    
    # Add advanced socket options
    use_pg_socket = st.checkbox("Use PostgreSQL-specific socket format", value=True)
    
    if st.button("Test Socket Connection"):
        try:
            # Try different socket path formats
            if use_pg_socket:
                # Use standard socket directory but let psycopg2 find the .s.PGSQL.5432 file
                socket_to_use = socket_path
                st.info(f"Using socket path: {socket_to_use}")
            else:
                # Try exact socket path including .s.PGSQL.5432
                socket_to_use = f"{socket_path}/.s.PGSQL.5432"
                st.info(f"Using explicit socket file: {socket_to_use}")
            
            # Try connection with verbose logging
            st.write("Attempting connection...")
            
            conn = psycopg2.connect(
                host=socket_to_use,
                dbname="postgres",
                user="postgres",
                password=os.environ.get("PG_PASSWORD", "")
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            st.success(f"✅ Connected via socket!\n\n{version}")
        except Exception as e:
            st.error(f"❌ Socket connection failed:\n\n{str(e)}")
            
            # Show additional debugging options
            st.warning("Cloud SQL Auth Proxy might not be properly configured")
            st.info("Try checking Cloud Run logs for proxy errors:")
            st.code("gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=chat-rag\" --limit=20")

with col2:
    st.write("**Public IP Connection**")
    ip = st.text_input("Database IP", "34.48.95.143")
    
    if st.button("Test IP Connection"):
        try:
            conn = psycopg2.connect(
                host=ip,
                hostaddr=ip,  # Force TCP
                dbname="postgres",
                user="postgres",
                password=os.environ.get("PG_PASSWORD", "")
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            st.success(f"✅ Connected via IP!\n\n{version}")
        except Exception as e:
            st.error(f"❌ IP connection failed:\n\n{str(e)}")

# Fix suggestions
st.subheader("3. Cloud SQL Auth Proxy Check")

st.write("The missing socket file indicates the Cloud SQL Auth Proxy isn't working properly.")

cloudsql_dir = "/cloudsql"
if os.path.exists(cloudsql_dir):
    st.write("Checking Cloud SQL directories...")
    all_dirs = os.listdir(cloudsql_dir)
    
    if len(all_dirs) == 0:
        st.error("❌ The /cloudsql directory is empty - Cloud SQL Auth Proxy is not creating instance directories")
        st.info("Run this command to fix it:")
        st.code(f"gcloud run services update chat-rag --region=us-east4 --add-cloudsql-instances={conn_name}")
    else:
        st.write(f"Found directories: {all_dirs}")
        
        # Look for our specific instance
        expected_dir = conn_name.replace(":", ":")
        found = False
        
        for dir in all_dirs:
            if dir == expected_dir or expected_dir in dir:
                found = True
                st.success(f"✅ Found directory for this instance: {dir}")
                try:
                    socket_files = os.listdir(os.path.join(cloudsql_dir, dir))
                    st.write(f"Contents: {socket_files}")
                except:
                    st.error("❌ Could not read directory contents")
        
        if not found:
            st.error(f"❌ No directory found for instance {conn_name}")

# Run a process check to see if Cloud SQL Auth Proxy might be running
st.write("Checking for Cloud SQL Auth Proxy process...")
try:
    # Only works on Linux
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    output = result.stdout
    
    proxy_lines = [line for line in output.split('\n') if 'cloud_sql_proxy' in line or 'cloud-sql-proxy' in line]
    
    if proxy_lines:
        st.success("✅ Found Cloud SQL Auth Proxy process:")
        for line in proxy_lines:
            st.code(line)
    else:
        st.warning("⚠️ No Cloud SQL Auth Proxy process found. This is normal in Cloud Run.")
except Exception as e:
    st.info("Process check not available in this environment")

# Add specific next steps for socket connection issue
st.subheader("4. Direct Fix for Socket Connection")

st.write("Based on your results, here's a specific fix to try:")

if os.path.exists("/cloudsql") and len(os.listdir("/cloudsql")) == 0:
    st.error("⚠️ PROBLEM: Cloud SQL Auth Proxy is not creating instance directories")
    st.info("Run this command to fix it:")
    st.code(f"gcloud run services update chat-rag --region=us-east4 --add-cloudsql-instances={conn_name}")
elif socket_path and os.path.exists(socket_path) and '.s.PGSQL.5432' not in os.listdir(socket_path):
    st.error("⚠️ PROBLEM: PostgreSQL socket file is not being created")
    st.info("Restart the service by running:")
    st.code(f"gcloud run services update chat-rag --region=us-east4 --clear-env-vars=DUMMY --update-env-vars=PG_PASSWORD={os.environ.get('PG_PASSWORD', 'your_password_here')}")
else:
    st.write("Try manually configuring the service account:")
    
    st.code("""
# 1. Verify Cloud SQL Client role
gcloud projects add-iam-policy-binding chat-rag-v1 \\
  --member=serviceAccount:682048092511-compute@developer.gserviceaccount.com \\
  --role=roles/cloudsql.client \\
  --condition=None

# 2. Update Cloud Run service with correct configuration
gcloud run services update chat-rag \\
  --region=us-east4 \\
  --add-cloudsql-instances=chat-rag-v1:us-east4:chat-rag-db \\
  --update-env-vars=PG_PASSWORD=your_password_here
""")

st.subheader("5. Common Issues & Fixes")

issues = [
    {
        "problem": "Missing PostgreSQL socket file",
        "fix": "Ensure Cloud SQL instance is attached: `--add-cloudsql-instances=PROJECT:REGION:INSTANCE`"
    },
    {
        "problem": "No password set",
        "fix": "Set password: `--update-env-vars=PG_PASSWORD=your_password`"
    },
    {
        "problem": "Incorrect connection name",
        "fix": "Set correct name: `--update-env-vars=INSTANCE_CONNECTION_NAME=PROJECT:REGION:INSTANCE`"
    },
    {
        "problem": "Missing permissions",
        "fix": "Add Cloud SQL Client role: `gcloud projects add-iam-policy-binding PROJECT --member=serviceAccount:ACCOUNT --role=roles/cloudsql.client`"
    }
]

for issue in issues:
    with st.expander(f"📌 {issue['problem']}"):
        st.code(issue['fix']) 