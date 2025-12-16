"""drop upload_timestamp from file_uploads

Revision ID: 4f20bae0351f
Revises: 52dd49119a74
Create Date: 2025-12-16 14:39:24.388943

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4f20bae0351f'
down_revision: Union[str, None] = '52dd49119a74'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop index first
    op.drop_index('ix_file_uploads_upload_timestamp', table_name='file_uploads')
    # Drop column
    op.drop_column('file_uploads', 'upload_timestamp')
    # Add new index on created_at
    op.create_index('ix_file_uploads_created_at', 'file_uploads', ['created_at'], unique=False)


def downgrade() -> None:
    # Drop new index
    op.drop_index('ix_file_uploads_created_at', table_name='file_uploads')
    # Add column back
    op.add_column('file_uploads', sa.Column('upload_timestamp', sa.DateTime(), nullable=False, server_default=sa.text('now()')))
    # Recreate old index
    op.create_index('ix_file_uploads_upload_timestamp', 'file_uploads', ['upload_timestamp'], unique=False)
