"""
Database seeding functions.
Automatically creates default data on application startup.
"""

import os
import logging
from sqlalchemy import select
from app.infrastructure.database.connection import get_db_context
from app.infrastructure.database.models.user import UserModel
from app.infrastructure.database.models.system_settings import SystemSettingsModel
from app.application.auth.auth_service import hash_password

logger = logging.getLogger(__name__)


async def seed_admin_user():
    """
    Seed default admin user if not exists.
    Reads credentials from environment variables:
    - ADMIN_USERNAME (default: admin)
    - ADMIN_PASSWORD (default: Admin@123)
    - ADMIN_EMAIL (default: admin@bwid.local)
    - ADMIN_FULL_NAME (default: System Admin)
    """
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "Admin@123")
    email = os.getenv("ADMIN_EMAIL", "admin@bwid.local")
    full_name = os.getenv("ADMIN_FULL_NAME", "System Admin")

    async with get_db_context() as db:
        try:
            # Check if admin user already exists
            result = await db.execute(
                select(UserModel).where(
                    (UserModel.username == username) | (UserModel.email == email)
                )
            )
            existing_user = result.scalar_one_or_none()

            if existing_user:
                logger.info(f"Admin user '{username}' already exists, skipping seed")
                return False

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
            await db.commit()

            logger.info(f"Created default admin user: {username} ({email})")
            return True

        except Exception as e:
            logger.error(f"Failed to seed admin user: {e}")
            return False


async def seed_feature_toggles():
    """Seed default feature toggles settings if not exists."""
    default_features = {
        "bankStatementOcr": {
            "enabled": True,
            "disabledMessage": "Bank Statement OCR is temporarily unavailable. Please contact Admin for assistance.",
        },
        "contractOcr": {
            "enabled": True,
            "disabledMessage": "Contract OCR is temporarily unavailable. Please contact Admin for assistance.",
        },
        "varianceAnalysis": {
            "enabled": True,
            "disabledMessage": "Variance Analysis is temporarily unavailable. Please contact Admin for assistance.",
        },
        "glaAnalysis": {
            "enabled": True,
            "disabledMessage": "GLA Analysis is temporarily unavailable. Please contact Admin for assistance.",
        },
        "excelComparison": {
            "enabled": True,
            "disabledMessage": "Excel Comparison is temporarily unavailable. Please contact Admin for assistance.",
        },
        "utilityBilling": {
            "enabled": True,
            "disabledMessage": "Utility Billing is temporarily unavailable. Please contact Admin for assistance.",
        },
        "cashReport": {
            "enabled": True,
            "disabledMessage": "Cash Report is temporarily unavailable. Please contact Admin for assistance.",
        },
        "ntmEbitdaAnalysis": {
            "enabled": True,
            "disabledMessage": "NTM EBITDA Analysis is temporarily unavailable. Please contact Admin for assistance.",
        },
    }

    async with get_db_context() as db:
        try:
            result = await db.execute(
                select(SystemSettingsModel).where(SystemSettingsModel.key == "feature_toggles")
            )
            existing_setting = result.scalar_one_or_none()

            if existing_setting:
                logger.info("Feature toggles already exist, skipping seed")
                return False

            setting = SystemSettingsModel(
                key="feature_toggles",
                value=default_features,
                description="Feature toggles for enabling/disabling application features",
            )
            db.add(setting)
            await db.commit()

            logger.info("Created default feature toggles setting")
            return True
        except Exception as e:
            logger.error(f"Failed to seed feature toggles: {e}")
            return False


async def run_all_seeds():
    """Run all database seeds."""
    logger.info("Running database seeds...")

    # Seed admin user
    await seed_admin_user()
    await seed_feature_toggles()

    logger.info("Database seeding complete")
