#!/usr/bin/env python3
"""Script to create or update an admin account with local login credentials."""

import asyncio
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

from app.infrastructure.database.connection import get_db_context, init_db, close_db
from app.infrastructure.database.models.user import UserModel
from app.application.auth.auth_service import hash_password
from sqlalchemy import select


async def create_admin(
    username: str,
    password: str,
    email: str,
    full_name: str = "System Admin",
):
    """Create or update an admin user with local credentials."""
    await init_db()

    async with get_db_context() as db:
        try:
            # Check if user exists by username or email
            result = await db.execute(
                select(UserModel).where(
                    (UserModel.username == username) | (UserModel.email == email)
                )
            )
            existing_user = result.scalar_one_or_none()

            if existing_user:
                # Update existing user to admin with password
                existing_user.username = username
                existing_user.password_hash = hash_password(password)
                existing_user.role = "admin"
                existing_user.is_active = True
                existing_user.full_name = full_name

                print(f"Updated existing user '{email}' to admin with local login")
                print(f"   Username: {username}")
            else:
                # Create new admin user
                admin = UserModel(
                    username=username,
                    password_hash=hash_password(password),
                    email=email,
                    email_verified=True,
                    full_name=full_name,
                    role="admin",
                    is_active=True,
                )
                db.add(admin)
                print(f"Created new admin user")
                print(f"   Username: {username}")
                print(f"   Email: {email}")

            await db.commit()

            print(f"\nLogin credentials:")
            print(f"   Username: {username}")
            print(f"   Password: {password}")

        except Exception as e:
            print(f"Error: {e}")
            raise

    await close_db()


def main():
    print("=" * 50)
    print("Admin Account Setup")
    print("=" * 50)

    # Get input from user or use defaults
    if len(sys.argv) >= 3:
        username = sys.argv[1]
        password = sys.argv[2]
        email = sys.argv[3] if len(sys.argv) > 3 else f"{username}@admin.local"
        full_name = sys.argv[4] if len(sys.argv) > 4 else "System Admin"
    else:
        print("\nEnter admin account details (or press Enter for defaults):")
        username = input("Username [admin]: ").strip() or "admin"
        password = input("Password [Admin@123]: ").strip() or "Admin@123"
        email = input(f"Email [{username}@admin.local]: ").strip() or f"{username}@admin.local"
        full_name = input("Full Name [System Admin]: ").strip() or "System Admin"

    print(f"\nCreating admin with:")
    print(f"  Username: {username}")
    print(f"  Email: {email}")
    print(f"  Full Name: {full_name}")

    asyncio.run(create_admin(username, password, email, full_name))


if __name__ == "__main__":
    main()
