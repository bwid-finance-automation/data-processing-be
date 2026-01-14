"""Add username and password_hash for local auth, make google_id nullable.

Revision ID: 45678abcdef0
Revises: 32564f62edd7
Create Date: 2026-01-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '45678abcdef0'
down_revision: Union[str, None] = '32564f62edd7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add username column
    op.add_column(
        'users',
        sa.Column('username', sa.String(length=100), nullable=True)
    )

    # Add password_hash column
    op.add_column(
        'users',
        sa.Column('password_hash', sa.String(length=255), nullable=True)
    )

    # Make google_id nullable (for local accounts)
    op.alter_column(
        'users',
        'google_id',
        existing_type=sa.String(length=100),
        nullable=True
    )

    # Add unique constraint for username
    op.create_unique_constraint('uq_users_username', 'users', ['username'])

    # Add index for username
    op.create_index('ix_users_username', 'users', ['username'])


def downgrade() -> None:
    # Drop index and constraint
    op.drop_index('ix_users_username', table_name='users')
    op.drop_constraint('uq_users_username', 'users', type_='unique')

    # Make google_id non-nullable again
    op.alter_column(
        'users',
        'google_id',
        existing_type=sa.String(length=100),
        nullable=False
    )

    # Drop columns
    op.drop_column('users', 'password_hash')
    op.drop_column('users', 'username')
