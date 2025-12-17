#!/usr/bin/env python3
"""
Smart migration script for Alembic.

This script handles the case where tables already exist but Alembic's
version tracking is not set up. It checks if tables exist and stamps
the database appropriately before running migrations.

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
    # Check for DATABASE_URL (Render/production)
    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        # Convert postgres:// to postgresql:// if needed
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url

    # Fallback to individual env vars (local development)
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "data_processing")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "password")

    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


async def check_database_state(db_url: str):
    """Check database state using asyncpg directly."""
    import asyncpg

    # Parse the URL for asyncpg
    # Remove postgresql:// prefix and any query params
    clean_url = db_url
    if clean_url.startswith("postgresql://"):
        clean_url = clean_url.replace("postgresql://", "")
    elif clean_url.startswith("postgres://"):
        clean_url = clean_url.replace("postgres://", "")

    # Handle query params (like ?sslmode=require)
    if "?" in clean_url:
        clean_url, params = clean_url.split("?", 1)
    else:
        params = ""

    # Parse user:pass@host:port/dbname
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

    # Determine SSL mode
    ssl = "require" if "sslmode=require" in params or os.environ.get("DATABASE_URL") else None

    print(f"  Connecting to {host}:{port}/{database}...")

    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        ssl=ssl
    )

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


def determine_stamp_revision(existing_tables):
    """Determine which revision to stamp based on existing tables."""
    has_initial = "analysis_sessions" in existing_tables
    has_projects = "projects" in existing_tables and "project_cases" in existing_tables

    if has_projects:
        # Has projects tables - stamp at head
        return "head"
    elif has_initial:
        # Has initial tables but not projects - stamp at initial
        return "52dd49119a74"

    return None


def main():
    print("=" * 60)
    print("Smart Migration Script")
    print("=" * 60)

    # Get database URL
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

        # Configure Alembic (it uses its own connection from env.py)
        alembic_cfg = Config("alembic.ini")

        # Determine action
        if current_version:
            print(f"\n✓ Alembic version is set to: {current_version}")
            print("  Running upgrade to head...")
            command.upgrade(alembic_cfg, "head")
            print("  ✓ Migrations complete!")

        elif any(tables_exist.values()):
            # Tables exist but no alembic version - need to stamp
            stamp_revision = determine_stamp_revision(existing_tables)

            if stamp_revision:
                print(f"\n⚠ Tables exist but Alembic version not set!")
                print(f"  Stamping database at revision: {stamp_revision}")
                command.stamp(alembic_cfg, stamp_revision)
                print(f"  ✓ Stamped!")

                if stamp_revision != "head":
                    print(f"  Running remaining migrations to head...")
                    command.upgrade(alembic_cfg, "head")
                    print("  ✓ Migrations complete!")
            else:
                print("\n✗ Could not determine appropriate stamp revision")
                sys.exit(1)

        else:
            # Fresh database - run all migrations
            print("\n→ Fresh database detected")
            print("  Running all migrations...")
            command.upgrade(alembic_cfg, "head")
            print("  ✓ Migrations complete!")

        # Verify final state
        _, final_version = asyncio.run(check_database_state(db_url))
        print(f"\n✓ Final Alembic version: {final_version}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
