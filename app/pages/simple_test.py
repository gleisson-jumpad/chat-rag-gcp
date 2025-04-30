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
    if st.button("Test Socket Connection"):
        try:
            conn = psycopg2.connect(
                host=socket_path,
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
st.subheader("3. Common Issues & Fixes")

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