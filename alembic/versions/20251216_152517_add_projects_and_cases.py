"""Add projects and project_cases tables, add case_id to related tables

Revision ID: a1b2c3d4e5f6
Revises: 4f20bae0351f
Create Date: 2025-12-16 15:25:17.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '4f20bae0351f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === 1. Create projects table ===
    op.create_table('projects',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', sa.UUID(), nullable=False),
        sa.Column('project_name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_protected', sa.Boolean(), nullable=False, default=False),
        sa.Column('password_hash', sa.String(length=255), nullable=True),
        sa.Column('last_accessed_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid')
    )
    op.create_index('ix_projects_uuid', 'projects', ['uuid'], unique=False)
    op.create_index('ix_projects_project_name', 'projects', ['project_name'], unique=False)
    op.create_index('ix_projects_last_accessed_at', 'projects', ['last_accessed_at'], unique=False)

    # === 2. Create project_cases table ===
    op.create_table('project_cases',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', sa.UUID(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('case_type', sa.String(length=50), nullable=False),
        sa.Column('total_files', sa.Integer(), nullable=False, default=0),
        sa.Column('last_processed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.UniqueConstraint('project_id', 'case_type', name='uq_project_case_type')
    )
    op.create_index('ix_project_cases_uuid', 'project_cases', ['uuid'], unique=False)
    op.create_index('ix_project_cases_project_id', 'project_cases', ['project_id'], unique=False)
    op.create_index('ix_project_cases_case_type', 'project_cases', ['case_type'], unique=False)

    # === 3. Add case_id and processed_at to bank_statements ===
    op.add_column('bank_statements', sa.Column('case_id', sa.Integer(), nullable=True))
    op.add_column('bank_statements', sa.Column('processed_at', sa.DateTime(), nullable=True))
    op.create_foreign_key(
        'fk_bank_statements_case_id',
        'bank_statements', 'project_cases',
        ['case_id'], ['id'],
        ondelete='SET NULL'
    )
    op.create_index('ix_bank_statements_case_id', 'bank_statements', ['case_id'], unique=False)

    # === 4. Add case_id, file_name, processed_at to contracts ===
    op.add_column('contracts', sa.Column('case_id', sa.Integer(), nullable=True))
    op.add_column('contracts', sa.Column('file_name', sa.String(length=500), nullable=True))
    op.add_column('contracts', sa.Column('processed_at', sa.DateTime(), nullable=True))
    op.create_foreign_key(
        'fk_contracts_case_id',
        'contracts', 'project_cases',
        ['case_id'], ['id'],
        ondelete='SET NULL'
    )
    op.create_index('ix_contracts_case_id', 'contracts', ['case_id'], unique=False)

    # === 5. Add case_id, file_name, processed_at to gla_projects ===
    op.add_column('gla_projects', sa.Column('case_id', sa.Integer(), nullable=True))
    op.add_column('gla_projects', sa.Column('file_name', sa.String(length=500), nullable=True))
    op.add_column('gla_projects', sa.Column('processed_at', sa.DateTime(), nullable=True))
    op.create_foreign_key(
        'fk_gla_projects_case_id',
        'gla_projects', 'project_cases',
        ['case_id'], ['id'],
        ondelete='SET NULL'
    )
    op.create_index('ix_gla_projects_case_id', 'gla_projects', ['case_id'], unique=False)

    # === 6. Add case_id to analysis_sessions ===
    op.add_column('analysis_sessions', sa.Column('case_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_analysis_sessions_case_id',
        'analysis_sessions', 'project_cases',
        ['case_id'], ['id'],
        ondelete='SET NULL'
    )
    op.create_index('ix_analysis_sessions_case_id', 'analysis_sessions', ['case_id'], unique=False)


def downgrade() -> None:
    # === 6. Remove case_id from analysis_sessions ===
    op.drop_index('ix_analysis_sessions_case_id', table_name='analysis_sessions')
    op.drop_constraint('fk_analysis_sessions_case_id', 'analysis_sessions', type_='foreignkey')
    op.drop_column('analysis_sessions', 'case_id')

    # === 5. Remove columns from gla_projects ===
    op.drop_index('ix_gla_projects_case_id', table_name='gla_projects')
    op.drop_constraint('fk_gla_projects_case_id', 'gla_projects', type_='foreignkey')
    op.drop_column('gla_projects', 'processed_at')
    op.drop_column('gla_projects', 'file_name')
    op.drop_column('gla_projects', 'case_id')

    # === 4. Remove columns from contracts ===
    op.drop_index('ix_contracts_case_id', table_name='contracts')
    op.drop_constraint('fk_contracts_case_id', 'contracts', type_='foreignkey')
    op.drop_column('contracts', 'processed_at')
    op.drop_column('contracts', 'file_name')
    op.drop_column('contracts', 'case_id')

    # === 3. Remove columns from bank_statements ===
    op.drop_index('ix_bank_statements_case_id', table_name='bank_statements')
    op.drop_constraint('fk_bank_statements_case_id', 'bank_statements', type_='foreignkey')
    op.drop_column('bank_statements', 'processed_at')
    op.drop_column('bank_statements', 'case_id')

    # === 2. Drop project_cases table ===
    op.drop_index('ix_project_cases_case_type', table_name='project_cases')
    op.drop_index('ix_project_cases_project_id', table_name='project_cases')
    op.drop_index('ix_project_cases_uuid', table_name='project_cases')
    op.drop_table('project_cases')

    # === 1. Drop projects table ===
    op.drop_index('ix_projects_last_accessed_at', table_name='projects')
    op.drop_index('ix_projects_project_name', table_name='projects')
    op.drop_index('ix_projects_uuid', table_name='projects')
    op.drop_table('projects')
