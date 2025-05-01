#!/usr/bin/env python3
"""
Direct PostgreSQL query for contract information
"""
import os
from openai import OpenAI
from db_config import get_pg_connection

def query_contract_info():
    # Connect to PostgreSQL
    conn = get_pg_connection()
    cursor = conn.cursor()
    
    # Find the table containing the contract document
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND 
        (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
    """)
    
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Found tables: {tables}")
    
    contract_info = []
    
    # Look in each table for the contract
    for table in tables:
        print(f"\nChecking table: {table}")
        
        # Check if this table has the contract
        cursor.execute(f"""
            SELECT DISTINCT metadata_->>'file_name' 
            FROM {table}
            WHERE metadata_->>'file_name' LIKE '%Coentro%' OR metadata_->>'file_name' LIKE '%Jumpad%'
        """)
        
        files = [row[0] for row in cursor.fetchall()]
        if files:
            print(f"Found contract files: {files}")
            
            # Get all the text content from this document
            for file in files:
                cursor.execute(f"""
                    SELECT text 
                    FROM {table}
                    WHERE metadata_->>'file_name' = %s
                    ORDER BY id
                """, (file,))
                
                chunks = [row[0] for row in cursor.fetchall()]
                print(f"Found {len(chunks)} chunks for file: {file}")
                
                # Combine all chunks
                document_text = "\n\n".join(chunks)
                contract_info.append({
                    "file": file,
                    "text": document_text
                })
    
    cursor.close()
    conn.close()
    
    # If we found the contract
    if contract_info:
        # Use OpenAI to answer the question
        client = OpenAI()
        
        for doc in contract_info:
            print(f"\nAnalyzing document: {doc['file']}")
            
            prompt = f"""
            Based on the following contract document, who signed the contract between Coentro and Jumpad?
            
            Document content:
            {doc['text']}
            
            Please provide just the names of the signatories with their roles.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            print("\nSignatories:")
            print(response.choices[0].message.content)
            return response.choices[0].message.content
    else:
        print("No contract documents found in the database.")
        return "No contract documents found in the database."

if __name__ == "__main__":
    query_contract_info() 