"""Add cash_report_sessions and cash_report_uploaded_files tables.

Revision ID: add_cash_report_sessions
Revises: dce038d69129
Create Date: 2026-02-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_cash_report_sessions'
down_revision: Union[str, None] = 'dce038d69129'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum type for session status
    op.execute("CREATE TYPE cashreportsessionstatus AS ENUM ('active', 'processing', 'completed', 'archived')")

    # Create cash_report_sessions table
    op.create_table(
        'cash_report_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.String(length=100), nullable=False),
        sa.Column('status', postgresql.ENUM('active', 'processing', 'completed', 'archived', name='cashreportsessionstatus', create_type=False), nullable=False),
        sa.Column('period_name', sa.String(length=100), nullable=True),
        sa.Column('opening_date', sa.Date(), nullable=True),
        sa.Column('ending_date', sa.Date(), nullable=True),
        sa.Column('fx_rate', sa.Numeric(precision=20, scale=4), nullable=True),
        sa.Column('working_file_path', sa.String(length=1000), nullable=True),
        sa.Column('total_transactions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_files_uploaded', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.UniqueConstraint('session_id'),
    )

    # Create indexes for cash_report_sessions
    op.create_index('ix_cash_report_sessions_session_id', 'cash_report_sessions', ['session_id'])
    op.create_index('ix_cash_report_sessions_status', 'cash_report_sessions', ['status'])
    op.create_index('ix_cash_report_sessions_opening_date', 'cash_report_sessions', ['opening_date'])
    op.create_index('ix_cash_report_sessions_ending_date', 'cash_report_sessions', ['ending_date'])

    # Create cash_report_uploaded_files table
    op.create_table(
        'cash_report_uploaded_files',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('original_filename', sa.String(length=500), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('transactions_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('transactions_added', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('transactions_skipped', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['cash_report_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create index for uploaded files
    op.create_index('ix_cash_report_uploaded_files_session_id', 'cash_report_uploaded_files', ['session_id'])


def downgrade() -> None:
    # Drop cash_report_uploaded_files table
    op.drop_index('ix_cash_report_uploaded_files_session_id', table_name='cash_report_uploaded_files')
    op.drop_table('cash_report_uploaded_files')

    # Drop cash_report_sessions table
    op.drop_index('ix_cash_report_sessions_ending_date', table_name='cash_report_sessions')
    op.drop_index('ix_cash_report_sessions_opening_date', table_name='cash_report_sessions')
    op.drop_index('ix_cash_report_sessions_status', table_name='cash_report_sessions')
    op.drop_index('ix_cash_report_sessions_session_id', table_name='cash_report_sessions')
    op.drop_table('cash_report_sessions')

    # Drop enum type
    op.execute("DROP TYPE cashreportsessionstatus")
