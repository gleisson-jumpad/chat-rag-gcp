import streamlit as st
import os
import tempfile
import uuid
import time
from rag_utils import process_uploaded_file, is_supported_file, SUPPORTED_EXTENSIONS, get_existing_tables

st.set_page_config(page_title="Upload Debug", layout="wide")

st.title("🐞 Upload Debug Tool")
st.write("This tool helps debug file uploads and deletions")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.write(f"Session ID: {st.session_state.session_id}")

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Debug section for environment variables
with st.expander("Environment Variables"):
    st.code(f"""
OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set'}
DB_PUBLIC_IP: {os.getenv('DB_PUBLIC_IP', 'Not Set')}
PG_PORT: {os.getenv('PG_PORT', 'Not Set')}
PG_DB: {os.getenv('PG_DB', 'Not Set')}
PG_USER: {os.getenv('PG_USER', 'Not Set')}
PG_PASSWORD: {'Set' if os.getenv('PG_PASSWORD') else 'Not Set'}
    """)

# Simple direct upload
st.subheader("Test File Upload")

uploaded_file = st.file_uploader(
    "Choose a file to test upload",
    type=[ext[1:] for ext in SUPPORTED_EXTENSIONS]
)

if uploaded_file:
    st.write(f"Selected file: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    if st.button("Process File", key="debug_process"):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                # Set start time
                start_time = time.time()
                
                # Use basic logging to track progress
                st.write("1. Starting file processing...")
                
                # Process the file
                index = process_uploaded_file(uploaded_file, st.session_state.session_id)
                
                # Calculate process time
                process_time = time.time() - start_time
                
                # Add to session state
                if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                    st.session_state.uploaded_files.append({
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'table': f"vectors_{st.session_state.session_id}",
                        'processing_time': process_time
                    })
                
                st.success(f"✅ File processed successfully in {process_time:.2f} seconds!")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.write("Stack trace:")
                st.exception(e)

# Show list of uploaded files
st.subheader("Uploaded Files")

for idx, file_info in enumerate(st.session_state.uploaded_files):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f"{idx+1}. {file_info['name']} ({file_info.get('processing_time', 0):.2f}s)")
    with col2:
        if st.button("Delete", key=f"delete_{idx}"):
            try:
                # Direct database deletion
                from db_config import get_pg_connection
                with st.spinner(f"Deleting {file_info['name']}..."):
                    start_time = time.time()
                    
                    conn = get_pg_connection()
                    cursor = conn.cursor()
                    
                    # Delete document chunks from the table
                    table_name = file_info['table']
                    file_name = file_info['name']
                    
                    st.write(f"Deleting from table: {table_name}, file: {file_name}")
                    
                    # First check if the table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public'
                            AND table_name = %s
                        );
                    """, (table_name,))
                    
                    table_exists = cursor.fetchone()[0]
                    
                    if table_exists:
                        # Quote the table name to handle hyphens or special characters
                        cursor.execute(f"""
                            DELETE FROM "{table_name}"
                            WHERE metadata_->>'file_name' = %s
                        """, (file_name,))
                        
                        # Track deleted chunks
                        deleted_count = cursor.rowcount
                        
                        # Commit and close
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        # Calculate process time
                        delete_time = time.time() - start_time
                        
                        # Remove from session state
                        st.session_state.uploaded_files.pop(idx)
                        
                        st.success(f"✅ Deleted {deleted_count} chunks in {delete_time:.2f} seconds!")
                    else:
                        cursor.close()
                        conn.close()
                        
                        # Remove from session state anyway since table doesn't exist
                        st.session_state.uploaded_files.pop(idx)
                        
                        st.warning(f"⚠️ Table '{table_name}' doesn't exist, but file removed from list.")
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error deleting file: {str(e)}")
                st.write("Stack trace:")
                st.exception(e)

# Database Inspection Section
st.subheader("Database Tables")

if st.button("Refresh Tables"):
    with st.spinner("Querying database..."):
        try:
            vector_tables = get_existing_tables()
            
            if vector_tables:
                st.write(f"Found {len(vector_tables)} vector tables:")
                
                for table_name in vector_tables:
                    # Get table statistics
                    from db_config import get_pg_connection
                    
                    conn = get_pg_connection()
                    cursor = conn.cursor()
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    
                    # Get distinct file names
                    cursor.execute(f"""
                        SELECT DISTINCT metadata_->>'file_name' as file_name
                        FROM {table_name}
                        WHERE metadata_->>'file_name' IS NOT NULL
                    """)
                    files = [row[0] for row in cursor.fetchall()]
                    
                    cursor.close()
                    conn.close()
                    
                    with st.expander(f"{table_name} ({row_count} rows)"):
                        st.write("Files:")
                        for file in files:
                            st.write(f"- {file}")
            else:
                st.warning("No vector tables found in database.")
        except Exception as e:
            st.error(f"Error querying database: {str(e)}")
            st.exception(e) 