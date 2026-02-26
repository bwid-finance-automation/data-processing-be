"""Remove project feature, replace with user_id tracking

- Add user_id (FK to users.id) to bank_statements, contracts, gla_projects
- Change analysis_sessions.user_id from String(100) to Integer FK to users.id
- Drop case_id from bank_statements, contracts, gla_projects, analysis_sessions
- Drop project_id and case_id from ai_usage_logs
- Drop project_cases and projects tables

Revision ID: remove_projects_user_id
Revises: add_user_id_cash_report
Create Date: 2026-02-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'remove_projects_user_id'
down_revision: Union[str, None] = 'add_user_id_cash_report'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === 1. Add user_id to bank_statements ===
    op.add_column('bank_statements', sa.Column('user_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_bank_statements_user_id', 'bank_statements', 'users',
        ['user_id'], ['id'], ondelete='SET NULL'
    )
    op.create_index('ix_bank_statements_user_id', 'bank_statements', ['user_id'], unique=False)

    # Drop case_id from bank_statements
    op.drop_index('ix_bank_statements_case_id', table_name='bank_statements')
    op.drop_constraint('fk_bank_statements_case_id', 'bank_statements', type_='foreignkey')
    op.drop_column('bank_statements', 'case_id')

    # === 2. Add user_id to contracts ===
    op.add_column('contracts', sa.Column('user_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_contracts_user_id', 'contracts', 'users',
        ['user_id'], ['id'], ondelete='SET NULL'
    )
    op.create_index('ix_contracts_user_id', 'contracts', ['user_id'], unique=False)

    # Drop case_id from contracts
    op.drop_index('ix_contracts_case_id', table_name='contracts')
    op.drop_constraint('fk_contracts_case_id', 'contracts', type_='foreignkey')
    op.drop_column('contracts', 'case_id')

    # === 3. Add user_id to gla_projects ===
    op.add_column('gla_projects', sa.Column('user_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_gla_projects_user_id', 'gla_projects', 'users',
        ['user_id'], ['id'], ondelete='SET NULL'
    )
    op.create_index('ix_gla_projects_user_id', 'gla_projects', ['user_id'], unique=False)

    # Drop case_id from gla_projects
    op.drop_index('ix_gla_projects_case_id', table_name='gla_projects')
    op.drop_constraint('fk_gla_projects_case_id', 'gla_projects', type_='foreignkey')
    op.drop_column('gla_projects', 'case_id')

    # === 4. Change analysis_sessions.user_id from String to Integer FK ===
    # Drop old user_id (String) index and column
    op.drop_index('ix_analysis_sessions_user_id', table_name='analysis_sessions')
    op.drop_column('analysis_sessions', 'user_id')
    # Add new user_id as Integer FK
    op.add_column('analysis_sessions', sa.Column('user_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_analysis_sessions_user_id', 'analysis_sessions', 'users',
        ['user_id'], ['id'], ondelete='SET NULL'
    )
    op.create_index('ix_analysis_sessions_user_id', 'analysis_sessions', ['user_id'], unique=False)

    # Drop case_id from analysis_sessions
    op.drop_index('ix_analysis_sessions_case_id', table_name='analysis_sessions')
    op.drop_constraint('fk_analysis_sessions_case_id', 'analysis_sessions', type_='foreignkey')
    op.drop_column('analysis_sessions', 'case_id')

    # === 5. Drop project_id and case_id from ai_usage_logs ===
    # Original FKs were created via sa.ForeignKeyConstraint (unnamed) → PostgreSQL auto-names them
    op.drop_index('ix_ai_usage_logs_project_id', table_name='ai_usage_logs')
    op.drop_constraint('ai_usage_logs_project_id_fkey', 'ai_usage_logs', type_='foreignkey')
    op.drop_column('ai_usage_logs', 'project_id')

    op.drop_index('ix_ai_usage_logs_case_id', table_name='ai_usage_logs')
    op.drop_constraint('ai_usage_logs_case_id_fkey', 'ai_usage_logs', type_='foreignkey')
    op.drop_column('ai_usage_logs', 'case_id')

    # === 6. Drop project_cases table (depends on projects) ===
    op.drop_index('ix_project_cases_case_type', table_name='project_cases')
    op.drop_index('ix_project_cases_project_id', table_name='project_cases')
    op.drop_index('ix_project_cases_uuid', table_name='project_cases')
    op.drop_table('project_cases')

    # === 7. Drop projects table ===
    op.drop_index('ix_projects_last_accessed_at', table_name='projects')
    op.drop_index('ix_projects_project_name', table_name='projects')
    op.drop_index('ix_projects_uuid', table_name='projects')
    op.drop_table('projects')


def downgrade() -> None:
    # === 7. Recreate projects table ===
    op.create_table('projects',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', sa.UUID(), nullable=False),
        sa.Column('project_name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_protected', sa.Boolean(), nullable=False, server_default=sa.text('false')),
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

    # === 6. Recreate project_cases table ===
    op.create_table('project_cases',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', sa.UUID(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('case_type', sa.String(length=50), nullable=False),
        sa.Column('total_files', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('last_processed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'case_type', name='uq_project_case_type'),
        sa.UniqueConstraint('uuid')
    )
    op.create_index('ix_project_cases_uuid', 'project_cases', ['uuid'], unique=False)
    op.create_index('ix_project_cases_project_id', 'project_cases', ['project_id'], unique=False)
    op.create_index('ix_project_cases_case_type', 'project_cases', ['case_type'], unique=False)

    # === 5. Re-add project_id and case_id to ai_usage_logs ===
    op.add_column('ai_usage_logs', sa.Column('case_id', sa.Integer(), nullable=True))
    op.create_foreign_key('ai_usage_logs_case_id_fkey', 'ai_usage_logs', 'project_cases', ['case_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_ai_usage_logs_case_id', 'ai_usage_logs', ['case_id'], unique=False)

    op.add_column('ai_usage_logs', sa.Column('project_id', sa.Integer(), nullable=True))
    op.create_foreign_key('ai_usage_logs_project_id_fkey', 'ai_usage_logs', 'projects', ['project_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_ai_usage_logs_project_id', 'ai_usage_logs', ['project_id'], unique=False)

    # === 4. Restore analysis_sessions ===
    # Re-add case_id
    op.add_column('analysis_sessions', sa.Column('case_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_analysis_sessions_case_id', 'analysis_sessions', 'project_cases', ['case_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_analysis_sessions_case_id', 'analysis_sessions', ['case_id'], unique=False)

    # Revert user_id from Integer to String(100)
    op.drop_index('ix_analysis_sessions_user_id', table_name='analysis_sessions')
    op.drop_constraint('fk_analysis_sessions_user_id', 'analysis_sessions', type_='foreignkey')
    op.drop_column('analysis_sessions', 'user_id')
    op.add_column('analysis_sessions', sa.Column('user_id', sa.String(length=100), nullable=True))
    op.create_index('ix_analysis_sessions_user_id', 'analysis_sessions', ['user_id'], unique=False)

    # === 3. Restore case_id to gla_projects, drop user_id ===
    op.drop_index('ix_gla_projects_user_id', table_name='gla_projects')
    op.drop_constraint('fk_gla_projects_user_id', 'gla_projects', type_='foreignkey')
    op.drop_column('gla_projects', 'user_id')
    op.add_column('gla_projects', sa.Column('case_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_gla_projects_case_id', 'gla_projects', 'project_cases', ['case_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_gla_projects_case_id', 'gla_projects', ['case_id'], unique=False)

    # === 2. Restore case_id to contracts, drop user_id ===
    op.drop_index('ix_contracts_user_id', table_name='contracts')
    op.drop_constraint('fk_contracts_user_id', 'contracts', type_='foreignkey')
    op.drop_column('contracts', 'user_id')
    op.add_column('contracts', sa.Column('case_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_contracts_case_id', 'contracts', 'project_cases', ['case_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_contracts_case_id', 'contracts', ['case_id'], unique=False)

    # === 1. Restore case_id to bank_statements, drop user_id ===
    op.drop_index('ix_bank_statements_user_id', table_name='bank_statements')
    op.drop_constraint('fk_bank_statements_user_id', 'bank_statements', type_='foreignkey')
    op.drop_column('bank_statements', 'user_id')
    op.add_column('bank_statements', sa.Column('case_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_bank_statements_case_id', 'bank_statements', 'project_cases', ['case_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_bank_statements_case_id', 'bank_statements', ['case_id'], unique=False)
