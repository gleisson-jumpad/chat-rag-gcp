#!/usr/bin/env python3
"""
Test direct table access to verify document content
"""
import os
from db_config import get_pg_connection

def list_document_content():
    """List document content directly from the database"""
    conn = get_pg_connection()
    cursor = conn.cursor()
    
    # First get a list of tables
    cursor.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' AND 
        (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
    """)
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(tables)} tables: {tables}")
    
    # Check each table for the contract document
    for table in tables:
        print(f"\n--- Table: {table} ---")
        
        # Check for document names
        cursor.execute(f"""
            SELECT DISTINCT metadata_->>'file_name' as file_name
            FROM {table}
            WHERE metadata_->>'file_name' IS NOT NULL
        """)
        files = [row[0] for row in cursor.fetchall()]
        print(f"Files in this table: {files}")
        
        # Look for contract document
        contract_files = [f for f in files if "coentro" in f.lower()]
        if contract_files:
            contract_file = contract_files[0]
            print(f"\nFound contract document: {contract_file}")
            
            # Get all text chunks for this document
            cursor.execute(f"""
                SELECT text 
                FROM {table}
                WHERE metadata_->>'file_name' = %s
                ORDER BY id
            """, (contract_file,))
            
            chunks = [row[0] for row in cursor.fetchall()]
            print(f"Document has {len(chunks)} chunks")
            
            print("\nContent samples:")
            for i, chunk in enumerate(chunks):
                print(f"\n--- Chunk {i+1} ---")
                print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    list_document_content() 