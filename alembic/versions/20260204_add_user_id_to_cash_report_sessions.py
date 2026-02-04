"""Add user_id to cash_report_sessions for per-user session isolation.

Revision ID: add_user_id_cash_report
Revises: add_cash_report_sessions
Create Date: 2026-02-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_user_id_cash_report'
down_revision: Union[str, None] = 'add_cash_report_sessions'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'cash_report_sessions',
        sa.Column('user_id', sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        'fk_cash_report_sessions_user_id',
        'cash_report_sessions',
        'users',
        ['user_id'],
        ['id'],
        ondelete='CASCADE',
    )
    op.create_index(
        'ix_cash_report_sessions_user_status',
        'cash_report_sessions',
        ['user_id', 'status'],
    )


def downgrade() -> None:
    op.drop_index('ix_cash_report_sessions_user_status', table_name='cash_report_sessions')
    op.drop_constraint('fk_cash_report_sessions_user_id', 'cash_report_sessions', type_='foreignkey')
    op.drop_column('cash_report_sessions', 'user_id')
