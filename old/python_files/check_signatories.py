#!/usr/bin/env python3
"""
Script to directly check for signatories and dates in database tables
"""
import os
import sys
import logging
import psycopg2

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.db_config import get_pg_connection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("check_signatories")

def find_in_database(search_terms):
    """
    Directly search database tables for specific terms
    
    Args:
        search_terms: List of terms to search for
    """
    logger.info(f"Searching database for terms: {search_terms}")
    
    conn = None
    try:
        # Get database connection
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Get list of data vector tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND 
            (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(tables)} vector tables: {tables}")
        
        # Search each table for the terms
        for table_name in tables:
            logger.info(f"Searching table: {table_name}")
            
            # Construct search query
            search_conditions = []
            for term in search_terms:
                search_conditions.append(f"text ILIKE '%{term}%'")
            
            search_query = f"""
                SELECT id, text, metadata_->>'file_name' as filename 
                FROM {table_name}
                WHERE {' OR '.join(search_conditions)}
                LIMIT 20
            """
            
            # Execute the search
            cursor.execute(search_query)
            results = cursor.fetchall()
            
            # Display results
            if results:
                logger.info(f"Found {len(results)} matches in table {table_name}")
                print(f"\n===== RESULTS FROM TABLE: {table_name} =====")
                
                for i, (id, text, filename) in enumerate(results, 1):
                    print(f"\n--- Match {i} from file: {filename} ---")
                    print(text[:300] + '...' if len(text) > 300 else text)
            else:
                logger.info(f"No matches found in table {table_name}")
        
        cursor.close()
    except Exception as e:
        logger.error(f"Error searching database: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Search terms to look for
    search_terms = [
        "assinou", "assinado", "assina", "assinatura", 
        "signatário", "signatária", "representante",
        "representada", "CNPJ", "CPF", "Gleisson", 
        "Monique", "março", "data"
    ]
    
    find_in_database(search_terms) 