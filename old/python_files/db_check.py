#!/usr/bin/env python3
"""
Database check script to examine contract information directly
"""
import os
import sys
import logging

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from db_config import get_pg_connection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_check")

def check_contract_info():
    """Check contract information directly from tables"""
    conn = None
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Get available tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name LIKE 'data_vectors_%'
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(tables)} tables: {', '.join(tables)}")
        
        # Check each table for contract information
        for table in tables:
            logger.info(f"Checking table: {table}")
            
            # Check for contract value information
            cursor.execute(f"""
                SELECT text, metadata_->>'file_name' as file_name, metadata_->>'page_label' as page
                FROM {table}
                WHERE 
                    text ILIKE '%valor%' OR 
                    text ILIKE '%R$%' OR 
                    text ILIKE '%reais%' OR
                    text ILIKE '%pagamento%' OR
                    text ILIKE '%mensalidade%'
            """)
            
            results = cursor.fetchall()
            if results:
                print(f"\n========== CONTRACT INFO IN {table} ==========")
                print(f"Found {len(results)} chunks with payment/value information\n")
                
                for i, (text, filename, page) in enumerate(results):
                    print(f"CHUNK {i+1} (File: {filename}, Page: {page}):")
                    print(f"{text}\n")
            else:
                logger.info(f"No value information found in {table}")
        
        # Also check specifically for Coentro contract
        for table in tables:
            cursor.execute(f"""
                SELECT text, metadata_->>'file_name' as file_name, metadata_->>'page_label' as page
                FROM {table}
                WHERE 
                    metadata_->>'file_name' LIKE '%Coentro%' OR
                    metadata_->>'file_name' LIKE '%contrato%' OR
                    metadata_->>'file_name' LIKE '%Jumpad%'
            """)
            
            results = cursor.fetchall()
            if results:
                print(f"\n========== COENTRO CONTRACT IN {table} ==========")
                print(f"Found {len(results)} chunks from Coentro contract\n")
                
                for i, (text, filename, page) in enumerate(results):
                    print(f"CHUNK {i+1} (File: {filename}, Page: {page}):")
                    print(f"{text}\n")
    except Exception as e:
        logger.error(f"Error checking contract info: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_contract_info() 