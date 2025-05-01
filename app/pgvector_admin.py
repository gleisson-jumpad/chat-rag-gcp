"""
PostgreSQL Vector Store Administration Utilities

This script provides utilities for managing and optimizing PostgreSQL vector tables
used by LlamaIndex for vector search.

Functions include:
- Listing vector tables
- Verifying table structures
- Creating and optimizing indices
- Gathering table statistics
- Vacuuming tables

Usage:
    python pgvector_admin.py list-tables  # List all vector tables
    python pgvector_admin.py optimize-table <table_name>  # Optimize a specific table
    python pgvector_admin.py verify-table <table_name>  # Verify table structure
    python pgvector_admin.py vacuum-tables  # Vacuum all vector tables
    python pgvector_admin.py create-indices  # Create HNSW indices on all tables
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Optional

from db_config import (
    get_pg_cursor, 
    verify_vector_table, 
    ensure_pgvector_extension,
    check_postgres_connection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pgvector_admin")

def list_vector_tables() -> List[str]:
    """List all vector tables in the database"""
    with get_pg_cursor() as cursor:
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND 
            (table_name LIKE 'vectors_%' OR table_name LIKE 'data_vectors_%')
            ORDER BY table_name
        """)
        
        tables = [table[0] for table in cursor.fetchall()]
        
        if not tables:
            logger.warning("No vector tables found in the database")
            return []
            
        logger.info(f"Found {len(tables)} vector tables:")
        for i, table in enumerate(tables, 1):
            logger.info(f"  {i}. {table}")
            
        return tables

def get_table_stats(table_name: str) -> Dict[str, Any]:
    """Get detailed statistics for a vector table"""
    stats = {}
    
    with get_pg_cursor() as cursor:
        # Basic row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        stats["row_count"] = cursor.fetchone()[0]
        
        # Document count (unique file names)
        cursor.execute(f"""
            SELECT COUNT(DISTINCT metadata_->>'file_name') 
            FROM {table_name}
            WHERE metadata_->>'file_name' IS NOT NULL
        """)
        stats["document_count"] = cursor.fetchone()[0]
        
        # Size information
        cursor.execute(f"""
            SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))
        """)
        stats["total_size"] = cursor.fetchone()[0]
        
        # Index information
        cursor.execute(f"""
            SELECT indexname, pg_size_pretty(pg_relation_size(indexname::text)) as size
            FROM pg_indexes 
            WHERE tablename = '{table_name}'
        """)
        indices = cursor.fetchall()
        stats["indices"] = [{"name": idx[0], "size": idx[1]} for idx in indices]
        
        # Sample documents
        cursor.execute(f"""
            SELECT DISTINCT metadata_->>'file_name' as file_name
            FROM {table_name}
            WHERE metadata_->>'file_name' IS NOT NULL
            LIMIT 5
        """)
        stats["sample_documents"] = [row[0] for row in cursor.fetchall() if row[0]]
        
        # Check for bloat
        cursor.execute(f"""
            SELECT 
                pg_size_pretty(pg_total_relation_size('{table_name}')),
                pg_size_pretty(pg_relation_size('{table_name}')),
                CASE 
                    WHEN pg_total_relation_size('{table_name}') > 0 
                    THEN ROUND(100.0 * pg_relation_size('{table_name}') / pg_total_relation_size('{table_name}'), 2)
                    ELSE 0
                END as table_percent
        """)
        size_info = cursor.fetchone()
        if size_info:
            stats["total_size"] = size_info[0]
            stats["table_size"] = size_info[1]
            stats["table_percent"] = float(size_info[2])
            stats["needs_vacuum"] = stats["table_percent"] < 70
        
        return stats

def optimize_table(table_name: str) -> Dict[str, Any]:
    """Optimize a vector table by creating indices and gathering statistics"""
    result = {
        "table_name": table_name,
        "actions": [],
        "success": False
    }
    
    # First verify the table
    table_info = verify_vector_table(table_name)
    if not table_info["exists"]:
        logger.error(f"Table {table_name} does not exist")
        result["error"] = "Table does not exist"
        return result
        
    if not table_info["vector_column"]:
        logger.error(f"Table {table_name} does not have an embedding column")
        result["error"] = "No embedding column found"
        return result
    
    # Create indices if needed
    indices_created = []
    
    with get_pg_cursor(commit=True) as cursor:
        # Check for and create HNSW index if needed
        has_hnsw = any(idx.get("type") == "hnsw" 
                      for idx in table_info.get("indices", []) 
                      if isinstance(idx, dict) and "type" in idx)
                      
        if not has_hnsw:
            try:
                logger.info(f"Creating HNSW index on table {table_name}")
                index_name = f"{table_name}_hnsw_idx"
                
                # Create HNSW index
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {table_name} 
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (
                        m = 16,
                        ef_construction = 64,
                        ef = 40
                    );
                """)
                
                indices_created.append({
                    "name": index_name,
                    "type": "hnsw"
                })
                
                result["actions"].append(f"Created HNSW index {index_name}")
            except Exception as e:
                logger.error(f"Failed to create HNSW index: {str(e)}")
                
                # Try to create regular index instead if HNSW fails
                try:
                    logger.info(f"Creating regular vector index on table {table_name}")
                    regular_index_name = f"{table_name}_vec_idx"
                    
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {regular_index_name} 
                        ON {table_name} 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    
                    indices_created.append({
                        "name": regular_index_name,
                        "type": "ivfflat"
                    })
                    
                    result["actions"].append(f"Created IVFFlat index {regular_index_name}")
                except Exception as e2:
                    logger.error(f"Failed to create regular vector index: {str(e2)}")
                    result["actions"].append(f"Failed to create vector indices: {str(e)}, {str(e2)}")
        
        # Create an index on metadata_->>'file_name' if it doesn't exist
        try:
            logger.info(f"Creating index on metadata_->>'file_name' for table {table_name}")
            file_index_name = f"{table_name}_file_idx"
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {file_index_name}
                ON {table_name} ((metadata_->>'file_name'));
            """)
            
            indices_created.append({
                "name": file_index_name,
                "type": "btree"
            })
            
            result["actions"].append(f"Created file name index {file_index_name}")
        except Exception as e:
            logger.error(f"Failed to create file name index: {str(e)}")
            result["actions"].append(f"Failed to create file name index: {str(e)}")
        
        # Vacuum and analyze the table
        try:
            logger.info(f"Vacuuming and analyzing table {table_name}")
            
            # Commit any pending transaction
            cursor.execute("COMMIT")
            
            # Need to close current transaction to run VACUUM
            cursor.execute(f"VACUUM ANALYZE {table_name}")
            
            result["actions"].append("Performed VACUUM ANALYZE")
        except Exception as e:
            logger.error(f"Failed to vacuum table: {str(e)}")
            result["actions"].append(f"Failed to vacuum table: {str(e)}")
    
    # Get updated statistics
    try:
        after_stats = get_table_stats(table_name)
        result["after_stats"] = after_stats
        result["success"] = True
        result["indices_created"] = indices_created
    except Exception as e:
        logger.error(f"Failed to get statistics after optimization: {str(e)}")
        result["error"] = f"Failed to get statistics: {str(e)}"
    
    return result

def vacuum_tables() -> Dict[str, Any]:
    """Vacuum all vector tables to optimize storage and performance"""
    result = {
        "vacuumed_tables": [],
        "failed_tables": [],
        "skipped_tables": []
    }
    
    tables = list_vector_tables()
    if not tables:
        return {"error": "No vector tables found"}
    
    for table_name in tables:
        try:
            # Check if table needs vacuum
            stats = get_table_stats(table_name)
            needs_vacuum = stats.get("needs_vacuum", False)
            
            if not needs_vacuum:
                logger.info(f"Table {table_name} does not need vacuuming, skipping")
                result["skipped_tables"].append(table_name)
                continue
            
            # Run vacuum analyze
            with get_pg_cursor() as cursor:
                logger.info(f"Vacuuming table {table_name}")
                
                # Commit any pending transaction
                cursor.execute("COMMIT")
                
                # Run vacuum
                cursor.execute(f"VACUUM ANALYZE {table_name}")
            
            result["vacuumed_tables"].append(table_name)
            logger.info(f"Successfully vacuumed table {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to vacuum table {table_name}: {str(e)}")
            result["failed_tables"].append({
                "table": table_name,
                "error": str(e)
            })
    
    return result

def create_indices() -> Dict[str, Any]:
    """Create HNSW indices on all vector tables"""
    result = {
        "optimized_tables": [],
        "failed_tables": []
    }
    
    tables = list_vector_tables()
    if not tables:
        return {"error": "No vector tables found"}
    
    for table_name in tables:
        try:
            logger.info(f"Optimizing table {table_name}")
            optimize_result = optimize_table(table_name)
            
            if optimize_result.get("success", False):
                result["optimized_tables"].append({
                    "table": table_name,
                    "actions": optimize_result.get("actions", [])
                })
            else:
                result["failed_tables"].append({
                    "table": table_name,
                    "error": optimize_result.get("error", "Unknown error")
                })
        except Exception as e:
            logger.error(f"Failed to optimize table {table_name}: {str(e)}")
            result["failed_tables"].append({
                "table": table_name,
                "error": str(e)
            })
    
    return result

def verify_all_tables() -> Dict[str, Any]:
    """Verify all vector tables in the database"""
    result = {
        "valid_tables": [],
        "invalid_tables": [],
        "total_count": 0
    }
    
    tables = list_vector_tables()
    result["total_count"] = len(tables)
    
    if not tables:
        return result
    
    for table_name in tables:
        try:
            logger.info(f"Verifying table {table_name}")
            table_info = verify_vector_table(table_name)
            
            if table_info["exists"] and table_info["vector_column"] and table_info["metadata_column"]:
                result["valid_tables"].append({
                    "table": table_name,
                    "row_count": table_info.get("row_count", 0),
                    "document_count": table_info.get("document_count", 0),
                    "has_index": table_info.get("has_index", False)
                })
            else:
                result["invalid_tables"].append({
                    "table": table_name,
                    "issues": [
                        "No vector column" if not table_info.get("vector_column") else None,
                        "No metadata column" if not table_info.get("metadata_column") else None,
                        "No indices" if not table_info.get("has_index") else None
                    ]
                })
        except Exception as e:
            logger.error(f"Failed to verify table {table_name}: {str(e)}")
            result["invalid_tables"].append({
                "table": table_name,
                "error": str(e)
            })
    
    return result

def main():
    """Main function to process command line arguments"""
    parser = argparse.ArgumentParser(description="PostgreSQL Vector Store Administration Utilities")
    
    # Define subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List tables command
    subparsers.add_parser('list-tables', help='List all vector tables')
    
    # Verify table command
    verify_parser = subparsers.add_parser('verify-table', help='Verify a table structure')
    verify_parser.add_argument('table_name', help='Name of the table to verify')
    
    # Verify all tables command
    subparsers.add_parser('verify-all', help='Verify all vector tables')
    
    # Optimize table command
    optimize_parser = subparsers.add_parser('optimize-table', help='Optimize a table')
    optimize_parser.add_argument('table_name', help='Name of the table to optimize')
    
    # Vacuum tables command
    subparsers.add_parser('vacuum-tables', help='Vacuum all vector tables')
    
    # Create indices command
    subparsers.add_parser('create-indices', help='Create HNSW indices on all tables')
    
    # Check database command
    subparsers.add_parser('check-db', help='Check database connection and pgvector extension')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check pgvector extension
    ensure_pgvector_extension()
    
    # Process commands
    if args.command == 'list-tables':
        list_vector_tables()
    
    elif args.command == 'verify-table':
        table_info = verify_vector_table(args.table_name)
        if table_info["exists"]:
            logger.info(f"Table {args.table_name} verification results:")
            for key, value in table_info.items():
                if key not in ['indices', 'sample_documents']:
                    logger.info(f"  {key}: {value}")
            
            if table_info.get("indices"):
                logger.info("  Indices:")
                for idx in table_info["indices"]:
                    logger.info(f"    - {idx.get('name')}: {idx.get('type', 'unknown')}")
            
            if table_info.get("sample_documents"):
                logger.info("  Sample documents:")
                for doc in table_info["sample_documents"]:
                    logger.info(f"    - {doc}")
        else:
            logger.error(f"Table {args.table_name} does not exist")
    
    elif args.command == 'verify-all':
        result = verify_all_tables()
        logger.info(f"Verified {result['total_count']} tables")
        logger.info(f"Valid tables: {len(result['valid_tables'])}")
        logger.info(f"Invalid tables: {len(result['invalid_tables'])}")
        
        if result["invalid_tables"]:
            logger.warning("Invalid tables found:")
            for table in result["invalid_tables"]:
                if "error" in table:
                    logger.warning(f"  {table['table']}: Error - {table['error']}")
                else:
                    issues = [i for i in table.get("issues", []) if i]
                    logger.warning(f"  {table['table']}: Issues - {', '.join(issues)}")
    
    elif args.command == 'optimize-table':
        result = optimize_table(args.table_name)
        if result.get("success", False):
            logger.info(f"Successfully optimized table {args.table_name}")
            for action in result.get("actions", []):
                logger.info(f"  - {action}")
        else:
            logger.error(f"Failed to optimize table {args.table_name}: {result.get('error', 'Unknown error')}")
    
    elif args.command == 'vacuum-tables':
        result = vacuum_tables()
        logger.info(f"Vacuumed {len(result.get('vacuumed_tables', []))} tables")
        logger.info(f"Skipped {len(result.get('skipped_tables', []))} tables")
        
        if result.get("failed_tables"):
            logger.warning(f"Failed to vacuum {len(result['failed_tables'])} tables")
            for table in result["failed_tables"]:
                logger.warning(f"  {table['table']}: {table['error']}")
    
    elif args.command == 'create-indices':
        result = create_indices()
        logger.info(f"Optimized {len(result.get('optimized_tables', []))} tables")
        
        if result.get("failed_tables"):
            logger.warning(f"Failed to optimize {len(result['failed_tables'])} tables")
            for table in result["failed_tables"]:
                logger.warning(f"  {table['table']}: {table['error']}")
    
    elif args.command == 'check-db':
        db_status = check_postgres_connection()
        logger.info("Database Connection Status:")
        
        if db_status.get("postgres_connection", False):
            logger.info(f"  Connection: SUCCESS")
            logger.info(f"  PostgreSQL version: {db_status.get('postgres_version', 'Unknown')}")
            
            if db_status.get("pgvector_installed", False):
                logger.info(f"  pgvector extension: INSTALLED (version {db_status.get('pgvector_version', 'Unknown')})")
                
                # Check if pgvector HNSW is enabled
                if db_status.get("pgvector_hnsw_enabled", False):
                    logger.info(f"  pgvector HNSW: ENABLED (ef_search: {db_status.get('pgvector_hnsw_ef_search', 'Unknown')})")
                else:
                    logger.warning(f"  pgvector HNSW: NOT ENABLED")
            else:
                logger.error(f"  pgvector extension: NOT INSTALLED")
                if db_status.get("pgvector_available", False):
                    logger.info(f"  pgvector is available but not installed. Run: {db_status.get('pgvector_installation_cmd', 'CREATE EXTENSION vector;')}")
                else:
                    logger.info(f"  pgvector is not available on this server. {db_status.get('pgvector_install_instructions', '')}")
            
            # Table information
            logger.info(f"  Vector tables: {db_status.get('vector_table_count', 0)}")
            
            if db_status.get("vector_tables"):
                logger.info("  Available tables:")
                for table in db_status.get("vector_tables", []):
                    table_stats = db_status.get("table_stats", {}).get(table, {})
                    if table_stats:
                        logger.info(f"    - {table}: {table_stats.get('row_count', '?')} rows, {table_stats.get('file_count', '?')} documents")
                    else:
                        logger.info(f"    - {table}")
        else:
            logger.error(f"  Connection: FAILED")
            logger.error(f"  Error: {db_status.get('error', 'Unknown error')}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 