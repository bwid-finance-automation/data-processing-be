#!/usr/bin/env python3
"""
Smart migration script for Alembic.

This script handles the case where tables already exist but Alembic's
version tracking is not set up, OR when tables exist but are missing
columns from migrations that were stamped but not actually run.

Usage:
    python scripts/migrate.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alembic.config import Config
from alembic import command


def get_database_url():
    """Get database URL from environment."""
    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url

    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "data_processing")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "password")

    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


async def get_connection(db_url: str):
    """Get asyncpg connection."""
    import asyncpg

    clean_url = db_url
    if clean_url.startswith("postgresql://"):
        clean_url = clean_url.replace("postgresql://", "")
    elif clean_url.startswith("postgres://"):
        clean_url = clean_url.replace("postgres://", "")

    if "?" in clean_url:
        clean_url, params = clean_url.split("?", 1)
    else:
        params = ""

    if "@" in clean_url:
        auth, rest = clean_url.split("@", 1)
        if ":" in auth:
            user, password = auth.split(":", 1)
        else:
            user, password = auth, ""
    else:
        user, password = "postgres", ""
        rest = clean_url

    if "/" in rest:
        host_port, database = rest.rsplit("/", 1)
    else:
        host_port, database = rest, "postgres"

    if ":" in host_port:
        host, port = host_port.split(":", 1)
        port = int(port)
    else:
        host, port = host_port, 5432

    ssl = "require" if "sslmode=require" in params or os.environ.get("DATABASE_URL") else None

    print(f"  Connecting to {host}:{port}/{database}...")

    return await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        ssl=ssl
    )


async def check_database_state(db_url: str):
    """Check database state using asyncpg directly."""
    conn = await get_connection(db_url)

    try:
        # Get all table names
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
        """)
        existing_tables = [t['tablename'] for t in tables]

        # Check alembic version
        alembic_version = None
        if 'alembic_version' in existing_tables:
            result = await conn.fetchrow("SELECT version_num FROM alembic_version")
            if result:
                alembic_version = result['version_num']

        return existing_tables, alembic_version

    finally:
        await conn.close()


async def get_table_columns(db_url: str, table_name: str):
    """Get columns for a specific table."""
    conn = await get_connection(db_url)

    try:
        columns = await conn.fetch("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1
        """, table_name)
        return [c['column_name'] for c in columns]
    finally:
        await conn.close()


async def check_index_exists(db_url: str, index_name: str):
    """Check if an index exists."""
    conn = await get_connection(db_url)

    try:
        result = await conn.fetchrow("""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'public' AND indexname = $1
        """, index_name)
        return result is not None
    finally:
        await conn.close()


async def execute_sql(db_url: str, sql: str):
    """Execute SQL statement."""
    conn = await get_connection(db_url)

    try:
        await conn.execute(sql)
        print(f"    Executed: {sql[:60]}...")
    finally:
        await conn.close()


async def fix_missing_schema(db_url: str, existing_tables: list):
    """Fix missing columns and tables from migrations."""
    fixes_applied = []

    print("\n→ Checking for missing schema elements...")

    # ============================================================
    # Migration: a1b2c3d4e5f6 - Add projects and project_cases
    # ============================================================

    # Check projects table
    if 'projects' not in existing_tables:
        print("  Adding missing table: projects")
        await execute_sql(db_url, """
            CREATE TABLE projects (
                id SERIAL PRIMARY KEY,
                uuid UUID NOT NULL UNIQUE,
                project_name VARCHAR(200) NOT NULL,
                description TEXT,
                is_protected BOOLEAN NOT NULL DEFAULT FALSE,
                password_hash VARCHAR(255),
                last_accessed_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
            )
        """)
        await execute_sql(db_url, "CREATE INDEX ix_projects_uuid ON projects (uuid)")
        await execute_sql(db_url, "CREATE INDEX ix_projects_project_name ON projects (project_name)")
        await execute_sql(db_url, "CREATE INDEX ix_projects_last_accessed_at ON projects (last_accessed_at)")
        fixes_applied.append("Created projects table")

    # Check project_cases table
    if 'project_cases' not in existing_tables:
        print("  Adding missing table: project_cases")
        await execute_sql(db_url, """
            CREATE TABLE project_cases (
                id SERIAL PRIMARY KEY,
                uuid UUID NOT NULL UNIQUE,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                case_type VARCHAR(50) NOT NULL,
                total_files INTEGER NOT NULL DEFAULT 0,
                last_processed_at TIMESTAMP,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                UNIQUE (project_id, case_type)
            )
        """)
        await execute_sql(db_url, "CREATE INDEX ix_project_cases_uuid ON project_cases (uuid)")
        await execute_sql(db_url, "CREATE INDEX ix_project_cases_project_id ON project_cases (project_id)")
        await execute_sql(db_url, "CREATE INDEX ix_project_cases_case_type ON project_cases (case_type)")
        fixes_applied.append("Created project_cases table")

    # Check case_id in bank_statements
    if 'bank_statements' in existing_tables:
        columns = await get_table_columns(db_url, 'bank_statements')

        if 'case_id' not in columns:
            print("  Adding missing column: bank_statements.case_id")
            await execute_sql(db_url, "ALTER TABLE bank_statements ADD COLUMN case_id INTEGER REFERENCES project_cases(id) ON DELETE SET NULL")
            await execute_sql(db_url, "CREATE INDEX ix_bank_statements_case_id ON bank_statements (case_id)")
            fixes_applied.append("Added bank_statements.case_id")

        if 'processed_at' not in columns:
            print("  Adding missing column: bank_statements.processed_at")
            await execute_sql(db_url, "ALTER TABLE bank_statements ADD COLUMN processed_at TIMESTAMP")
            fixes_applied.append("Added bank_statements.processed_at")

        # Migration: 34595774c5f5 - Add session_id to bank_statements
        if 'session_id' not in columns:
            print("  Adding missing column: bank_statements.session_id")
            await execute_sql(db_url, "ALTER TABLE bank_statements ADD COLUMN session_id VARCHAR(100)")
            if not await check_index_exists(db_url, 'ix_bank_statements_session_id'):
                await execute_sql(db_url, "CREATE INDEX ix_bank_statements_session_id ON bank_statements (session_id)")
            fixes_applied.append("Added bank_statements.session_id")

    # Check case_id, file_name, processed_at in contracts
    if 'contracts' in existing_tables:
        columns = await get_table_columns(db_url, 'contracts')

        if 'case_id' not in columns:
            print("  Adding missing column: contracts.case_id")
            await execute_sql(db_url, "ALTER TABLE contracts ADD COLUMN case_id INTEGER REFERENCES project_cases(id) ON DELETE SET NULL")
            await execute_sql(db_url, "CREATE INDEX ix_contracts_case_id ON contracts (case_id)")
            fixes_applied.append("Added contracts.case_id")

        if 'file_name' not in columns:
            print("  Adding missing column: contracts.file_name")
            await execute_sql(db_url, "ALTER TABLE contracts ADD COLUMN file_name VARCHAR(500)")
            fixes_applied.append("Added contracts.file_name")

        if 'processed_at' not in columns:
            print("  Adding missing column: contracts.processed_at")
            await execute_sql(db_url, "ALTER TABLE contracts ADD COLUMN processed_at TIMESTAMP")
            fixes_applied.append("Added contracts.processed_at")

    # Check case_id, file_name, processed_at in gla_projects
    if 'gla_projects' in existing_tables:
        columns = await get_table_columns(db_url, 'gla_projects')

        if 'case_id' not in columns:
            print("  Adding missing column: gla_projects.case_id")
            await execute_sql(db_url, "ALTER TABLE gla_projects ADD COLUMN case_id INTEGER REFERENCES project_cases(id) ON DELETE SET NULL")
            await execute_sql(db_url, "CREATE INDEX ix_gla_projects_case_id ON gla_projects (case_id)")
            fixes_applied.append("Added gla_projects.case_id")

        if 'file_name' not in columns:
            print("  Adding missing column: gla_projects.file_name")
            await execute_sql(db_url, "ALTER TABLE gla_projects ADD COLUMN file_name VARCHAR(500)")
            fixes_applied.append("Added gla_projects.file_name")

        if 'processed_at' not in columns:
            print("  Adding missing column: gla_projects.processed_at")
            await execute_sql(db_url, "ALTER TABLE gla_projects ADD COLUMN processed_at TIMESTAMP")
            fixes_applied.append("Added gla_projects.processed_at")

    # Check case_id in analysis_sessions
    if 'analysis_sessions' in existing_tables:
        columns = await get_table_columns(db_url, 'analysis_sessions')

        if 'case_id' not in columns:
            print("  Adding missing column: analysis_sessions.case_id")
            await execute_sql(db_url, "ALTER TABLE analysis_sessions ADD COLUMN case_id INTEGER REFERENCES project_cases(id) ON DELETE SET NULL")
            await execute_sql(db_url, "CREATE INDEX ix_analysis_sessions_case_id ON analysis_sessions (case_id)")
            fixes_applied.append("Added analysis_sessions.case_id")

    if fixes_applied:
        print(f"\nApplied {len(fixes_applied)} schema fixes:")
        for fix in fixes_applied:
            print(f"  - {fix}")
    else:
        print("  Schema is up to date")

    return fixes_applied


def determine_stamp_revision(existing_tables):
    """Determine which revision to stamp based on existing tables."""
    has_initial = "analysis_sessions" in existing_tables
    has_projects = "projects" in existing_tables and "project_cases" in existing_tables

    if has_projects:
        return "head"
    elif has_initial:
        return "52dd49119a74"

    return None


def main():
    print("=" * 60)
    print("Smart Migration Script")
    print("=" * 60)

    db_url = get_database_url()

    print(f"\nConnecting to database...")

    try:
        # Check current state using asyncpg
        existing_tables, current_version = asyncio.run(check_database_state(db_url))

        # Check for key tables
        initial_tables = ["analysis_sessions", "bank_statements", "contracts", "file_uploads"]
        tables_exist = {table: table in existing_tables for table in initial_tables}

        print(f"\nDatabase state:")
        print(f"  - Existing tables: {len(existing_tables)}")
        print(f"  - Key tables present: {tables_exist}")
        print(f"  - Alembic version: {current_version or 'Not set'}")

        # Fix missing schema elements FIRST (before alembic operations)
        if any(tables_exist.values()):
            asyncio.run(fix_missing_schema(db_url, existing_tables))

        # Re-check tables after schema fixes
        existing_tables, _ = asyncio.run(check_database_state(db_url))

        # Configure Alembic
        alembic_cfg = Config("alembic.ini")

        # Determine action
        if current_version:
            print(f"\nAlembic version is set to: {current_version}")
            print("  Running upgrade to head...")
            command.upgrade(alembic_cfg, "head")
            print("  Migrations complete!")

        elif any(tables_exist.values()):
            stamp_revision = determine_stamp_revision(existing_tables)

            if stamp_revision:
                print(f"\nTables exist but Alembic version not set!")
                print(f"  Stamping database at revision: {stamp_revision}")
                command.stamp(alembic_cfg, stamp_revision)
                print(f"  Stamped!")

                if stamp_revision != "head":
                    print(f"  Running remaining migrations to head...")
                    command.upgrade(alembic_cfg, "head")
                    print("  Migrations complete!")
            else:
                print("\nCould not determine appropriate stamp revision")
                sys.exit(1)

        else:
            print("\n→ Fresh database detected")
            print("  Running all migrations...")
            command.upgrade(alembic_cfg, "head")
            print("  Migrations complete!")

        # Verify final state
        _, final_version = asyncio.run(check_database_state(db_url))
        print(f"\nFinal Alembic version: {final_version}")
        print("=" * 60)

    except Exception as e:
        print(f"\nMigration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
