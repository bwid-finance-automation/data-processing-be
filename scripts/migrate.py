#!/usr/bin/env python3
"""
Smart migration script for Alembic.

This script handles the case where tables already exist but Alembic's
version tracking is not set up. It checks if tables exist and stamps
the database appropriately before running migrations.

Usage:
    python scripts/migrate.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine, text, inspect


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


def check_tables_exist(engine):
    """Check if main tables already exist in the database."""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    # Check for key tables from initial migration
    initial_tables = ["analysis_sessions", "bank_statements", "contracts", "file_uploads"]

    tables_exist = {table: table in existing_tables for table in initial_tables}

    return tables_exist, existing_tables


def check_alembic_version(engine):
    """Check if alembic_version table exists and has a version."""
    inspector = inspect(engine)

    if "alembic_version" not in inspector.get_table_names():
        return None

    with engine.connect() as conn:
        result = conn.execute(text("SELECT version_num FROM alembic_version"))
        row = result.fetchone()
        return row[0] if row else None


def determine_stamp_revision(existing_tables):
    """Determine which revision to stamp based on existing tables."""
    # Check for tables from each migration
    has_initial = "analysis_sessions" in existing_tables
    has_projects = "projects" in existing_tables and "project_cases" in existing_tables

    # Check for columns added in later migrations
    # This is a simplified check - in production you might want to inspect columns too

    if has_projects:
        # Has projects tables, likely at or past a1b2c3d4e5f6
        # Could be at head (34595774c5f5)
        return "head"
    elif has_initial:
        # Has initial tables but not projects
        # Stamp at initial and let migrations add the rest
        return "52dd49119a74"

    return None


def main():
    print("=" * 60)
    print("Smart Migration Script")
    print("=" * 60)

    # Get database URL
    db_url = get_database_url()
    sync_url = db_url.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")

    print(f"\nConnecting to database...")

    # Create sync engine for inspection
    engine = create_engine(sync_url)

    try:
        # Check current state
        tables_exist, existing_tables = check_tables_exist(engine)
        current_version = check_alembic_version(engine)

        print(f"\nDatabase state:")
        print(f"  - Existing tables: {len(existing_tables)}")
        print(f"  - Key tables present: {tables_exist}")
        print(f"  - Alembic version: {current_version or 'Not set'}")

        # Configure Alembic
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", sync_url)

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
        final_version = check_alembic_version(engine)
        print(f"\n✓ Final Alembic version: {final_version}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
