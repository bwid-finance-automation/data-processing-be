"""add file_type to cash_report_uploaded_files

Revision ID: 9292ea923816
Revises: remove_projects_user_id
Create Date: 2026-03-04 16:44:44.953139

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9292ea923816'
down_revision: Union[str, None] = 'remove_projects_user_id'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('cash_report_uploaded_files', sa.Column('file_type', sa.String(length=50), server_default='bank_statement', nullable=False))


def downgrade() -> None:
    op.drop_column('cash_report_uploaded_files', 'file_type')
