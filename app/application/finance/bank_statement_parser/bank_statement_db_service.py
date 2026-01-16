"""Service for saving bank statement data to database."""

import math
import os
import shutil
import aiofiles
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, and_

from app.infrastructure.database.models.bank_statement import (
    BankStatementModel,
    BankTransactionModel,
    BankBalanceModel,
)
from app.infrastructure.database.models.file_upload import FileUploadModel
from app.infrastructure.database.models.project import ProjectModel, ProjectCaseModel
from app.domain.finance.bank_statement_parser.models.bank_statement import (
    BankStatement,
    BankTransaction,
    BankBalance,
)
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Base upload directory
UPLOAD_BASE_DIR = Path("uploads/bank_statements")

# File retention period in days
FILE_RETENTION_DAYS = 7


class BankStatementDbService:
    """Service for persisting bank statement data to database."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def save_file_upload(
        self,
        filename: str,
        file_size: int,
        file_content: Optional[bytes] = None,
        file_type: str = "bank_statement",
        session_id: Optional[str] = None,
        content_type: Optional[str] = None,
        processing_status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FileUploadModel:
        """
        Save file upload record to database and optionally save file to disk.

        Args:
            filename: Original filename
            file_size: Size of file in bytes
            file_content: File content bytes (if provided, file will be saved to disk)
            file_type: Type of file (bank_statement, contract, etc.)
            session_id: Optional session ID
            content_type: MIME type
            processing_status: Status (pending, processing, completed, failed)
            metadata: Additional metadata

        Returns:
            Created FileUploadModel
        """
        # Extract file extension
        file_extension = filename.split(".")[-1].lower() if "." in filename else ""

        # Generate file path
        file_path = None
        if file_content and session_id:
            file_path = await self._save_file_to_disk(filename, file_content, session_id)

        file_upload = FileUploadModel(
            filename=filename,
            original_filename=filename,
            file_path=file_path or "",
            file_extension=file_extension,
            content_type=content_type or self._get_content_type(file_extension),
            file_size=file_size,
            file_type=file_type,
            processing_status=processing_status,
            session_id=session_id,
            validation_status="valid",
            security_scan_passed=True,
            metadata_json=metadata,
            # created_at is auto-set by TimestampMixin
        )

        self.db.add(file_upload)
        await self.db.flush()

        logger.info(f"Saved file upload record: {filename} (ID: {file_upload.id}, Path: {file_path})")
        return file_upload

    async def _save_file_to_disk(
        self,
        filename: str,
        content: bytes,
        session_id: str,
    ) -> str:
        """
        Save file content to disk.

        Args:
            filename: Original filename
            content: File content bytes
            session_id: Session ID for organizing files

        Returns:
            Relative file path
        """
        # Create directory structure: uploads/bank_statements/{session_id}/
        upload_dir = UPLOAD_BASE_DIR / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        file_path = upload_dir / safe_filename

        # Write file asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)

        logger.info(f"Saved file to disk: {file_path}")
        return str(file_path)

    async def save_excel_output(
        self,
        session_id: str,
        excel_bytes: bytes,
    ) -> str:
        """
        Save generated Excel output file to disk.

        Args:
            session_id: Session ID for organizing files
            excel_bytes: Excel file content bytes

        Returns:
            File path where Excel was saved
        """
        # Create directory structure: uploads/bank_statements/{session_id}/
        upload_dir = UPLOAD_BASE_DIR / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Use fixed filename for Excel output
        filename = f"output_{session_id[:8]}.xlsx"
        file_path = upload_dir / filename

        # Write file asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(excel_bytes)

        logger.info(f"Saved Excel output to disk: {file_path}")
        return str(file_path)

    async def get_excel_output(self, session_id: str) -> Optional[bytes]:
        """
        Get Excel output file content from disk.

        Args:
            session_id: Session ID

        Returns:
            Excel file bytes or None if not found
        """
        # Check for output file
        upload_dir = UPLOAD_BASE_DIR / session_id
        filename = f"output_{session_id[:8]}.xlsx"
        file_path = upload_dir / filename

        if not file_path.exists():
            logger.warning(f"Excel output not found: {file_path}")
            return None

        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()

        return content

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks."""
        # Remove path separators and null bytes
        safe_name = filename.replace('/', '_').replace('\\', '_').replace('\x00', '')
        # Keep only the last component if there's a path
        safe_name = os.path.basename(safe_name)
        return safe_name if safe_name else "unnamed_file"

    async def get_file_by_id(self, file_id: int) -> Optional[FileUploadModel]:
        """Get file upload record by ID."""
        result = await self.db.execute(
            select(FileUploadModel).where(FileUploadModel.id == file_id)
        )
        return result.scalar_one_or_none()

    async def get_files_by_session(self, session_id: str) -> List[FileUploadModel]:
        """Get all file uploads for a session."""
        result = await self.db.execute(
            select(FileUploadModel)
            .where(FileUploadModel.session_id == session_id)
            .order_by(FileUploadModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_statements_by_session(self, session_id: str) -> List[BankStatementModel]:
        """Get all bank statements for a session with transactions and balances loaded."""
        from sqlalchemy.orm import selectinload
        result = await self.db.execute(
            select(BankStatementModel)
            .options(
                selectinload(BankStatementModel.transactions),
                selectinload(BankStatementModel.balances),
            )
            .where(BankStatementModel.session_id == session_id)
            .order_by(BankStatementModel.id.asc())  # Maintain upload order
        )
        return list(result.scalars().all())

    async def get_file_content(self, file_id: int) -> Optional[Tuple[bytes, str, str]]:
        """
        Get file content by ID.

        Returns:
            Tuple of (content, filename, content_type) or None if not found
        """
        file_record = await self.get_file_by_id(file_id)
        if not file_record or not file_record.file_path:
            return None

        file_path = Path(file_record.file_path)
        if not file_path.exists():
            logger.warning(f"File not found on disk: {file_path}")
            return None

        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()

        return content, file_record.original_filename, file_record.content_type

    async def get_or_create_case(
        self,
        project_uuid: UUID,
    ) -> Optional[ProjectCaseModel]:
        """
        Get or create bank_statement case for a project.

        Args:
            project_uuid: Project UUID

        Returns:
            ProjectCaseModel or None if project not found
        """
        # Find project
        result = await self.db.execute(
            select(ProjectModel).where(ProjectModel.uuid == project_uuid)
        )
        project = result.scalar_one_or_none()
        if not project:
            logger.warning(f"Project not found: {project_uuid}")
            return None

        # Find or create case
        result = await self.db.execute(
            select(ProjectCaseModel).where(
                ProjectCaseModel.project_id == project.id,
                ProjectCaseModel.case_type == "bank_statement",
            )
        )
        case = result.scalar_one_or_none()

        if not case:
            case = ProjectCaseModel(
                project_id=project.id,
                case_type="bank_statement",
                total_files=0,
            )
            self.db.add(case)
            await self.db.flush()
            logger.info(f"Created bank_statement case for project: {project_uuid}")

        return case

    async def _update_case_file_count(self, case_id: int, count: int = 1) -> None:
        """Update file count for a case (increment by count)."""
        result = await self.db.execute(
            select(ProjectCaseModel).where(ProjectCaseModel.id == case_id)
        )
        case = result.scalar_one_or_none()
        if case:
            case.total_files += count
            case.last_processed_at = datetime.utcnow()

    async def save_bank_statement(
        self,
        statement: BankStatement,
        session_id: Optional[str] = None,
        case_id: Optional[int] = None,
    ) -> BankStatementModel:
        """
        Save a bank statement with its transactions and balances to database.

        Args:
            statement: Domain BankStatement object
            session_id: Optional session ID for tracking
            case_id: Optional project case ID for linking to project

        Returns:
            Created BankStatementModel with related records
        """
        # Create bank statement record
        db_statement = BankStatementModel(
            bank_name=statement.bank_name,
            file_name=statement.file_name,
            uploaded_at=datetime.utcnow(),
            processed_at=datetime.utcnow(),
            case_id=case_id,
            session_id=session_id,  # Store session_id directly for grouping
            metadata_json={
                "transaction_count": len(statement.transactions),
                "has_balance": statement.balance is not None,
            },
        )

        self.db.add(db_statement)
        await self.db.flush()  # Get the ID

        # Save transactions
        for tx in statement.transactions:
            await self._save_transaction(db_statement.id, tx)

        # Save balance if exists
        if statement.balance:
            await self._save_balance(db_statement.id, statement.balance)

        return db_statement

    async def save_statements_batch(
        self,
        statements: List[BankStatement],
        session_id: Optional[str] = None,
        project_uuid: Optional[UUID] = None,
    ) -> List[BankStatementModel]:
        """
        Save multiple bank statements in a batch.

        Args:
            statements: List of domain BankStatement objects
            session_id: Optional session ID for tracking
            project_uuid: Optional project UUID for linking to project

        Returns:
            List of created BankStatementModel objects
        """
        saved_statements = []

        # Get or create case if project_uuid is provided
        case_id = None
        if project_uuid:
            case = await self.get_or_create_case(project_uuid)
            if case:
                case_id = case.id

        for statement in statements:
            try:
                db_statement = await self.save_bank_statement(statement, session_id, case_id)
                saved_statements.append(db_statement)
            except Exception as e:
                logger.error(f"Failed to save statement {statement.file_name}: {e}")
                # Continue with other statements
                continue

        # Update case file count - increment by actual number of saved statements
        if case_id and saved_statements:
            await self._update_case_file_count(case_id, len(saved_statements))

        logger.info(f"Batch saved {len(saved_statements)} of {len(statements)} statements (case_id: {case_id})")
        return saved_statements

    async def _save_transaction(
        self,
        statement_id: int,
        tx: BankTransaction,
    ) -> BankTransactionModel:
        """Save a single transaction."""
        # Handle NaN values
        debit = tx.debit
        credit = tx.credit
        if isinstance(debit, float) and math.isnan(debit):
            debit = None
        if isinstance(credit, float) and math.isnan(credit):
            credit = None

        db_transaction = BankTransactionModel(
            statement_id=statement_id,
            bank_name=tx.bank_name,
            acc_no=tx.acc_no or "",
            transaction_date=tx.date,
            description=tx.description or "",
            debit=Decimal(str(debit)) if debit is not None else None,
            credit=Decimal(str(credit)) if credit is not None else None,
            currency=tx.currency or "VND",
            transaction_id=tx.transaction_id or "",
            beneficiary_bank=tx.beneficiary_bank or "",
            beneficiary_acc_no=tx.beneficiary_acc_no or "",
            beneficiary_acc_name=tx.beneficiary_acc_name or "",
        )

        self.db.add(db_transaction)
        return db_transaction

    async def _save_balance(
        self,
        statement_id: int,
        balance: BankBalance,
    ) -> BankBalanceModel:
        """Save balance record."""
        # Handle NaN values
        opening = balance.opening_balance
        closing = balance.closing_balance
        if isinstance(opening, float) and math.isnan(opening):
            opening = 0.0
        if isinstance(closing, float) and math.isnan(closing):
            closing = 0.0

        db_balance = BankBalanceModel(
            statement_id=statement_id,
            bank_name=balance.bank_name,
            acc_no=balance.acc_no or "",
            currency=balance.currency or "VND",
            opening_balance=Decimal(str(opening)),
            closing_balance=Decimal(str(closing)),
        )

        self.db.add(db_balance)
        return db_balance

    def _get_content_type(self, extension: str) -> str:
        """Get content type from file extension."""
        content_types = {
            "pdf": "application/pdf",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls": "application/vnd.ms-excel",
            "csv": "text/csv",
        }
        return content_types.get(extension, "application/octet-stream")

    async def cleanup_old_files(self, retention_days: int = FILE_RETENTION_DAYS) -> Dict[str, Any]:
        """
        Clean up files older than retention period.

        Args:
            retention_days: Number of days to keep files (default: 30)

        Returns:
            Cleanup statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        stats = {
            "files_deleted": 0,
            "files_failed": 0,
            "disk_space_freed": 0,
            "db_records_updated": 0,
            "sessions_cleaned": set(),
        }

        try:
            # Find old file records
            result = await self.db.execute(
                select(FileUploadModel).where(
                    and_(
                        FileUploadModel.created_at < cutoff_date,
                        FileUploadModel.file_path != "",
                        FileUploadModel.file_path.isnot(None),
                    )
                )
            )
            old_files = list(result.scalars().all())

            for file_record in old_files:
                try:
                    file_path = Path(file_record.file_path)

                    # Delete file from disk
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        stats["disk_space_freed"] += file_size
                        stats["files_deleted"] += 1
                        logger.info(f"Deleted old file: {file_path}")

                    # Update database record (clear file_path but keep metadata)
                    file_record.file_path = ""
                    stats["db_records_updated"] += 1

                    if file_record.session_id:
                        stats["sessions_cleaned"].add(file_record.session_id)

                except Exception as e:
                    logger.error(f"Failed to delete file {file_record.file_path}: {e}")
                    stats["files_failed"] += 1

            # Clean up empty session directories
            await self._cleanup_empty_directories()

            await self.db.commit()

            stats["sessions_cleaned"] = len(stats["sessions_cleaned"])
            logger.info(
                f"Cleanup completed: {stats['files_deleted']} files deleted, "
                f"{stats['disk_space_freed'] / 1024 / 1024:.2f} MB freed"
            )

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            await self.db.rollback()
            raise

        return stats

    async def _cleanup_empty_directories(self) -> None:
        """Remove empty session directories."""
        if not UPLOAD_BASE_DIR.exists():
            return

        for session_dir in UPLOAD_BASE_DIR.iterdir():
            if session_dir.is_dir():
                try:
                    # Check if directory is empty
                    if not any(session_dir.iterdir()):
                        session_dir.rmdir()
                        logger.info(f"Removed empty directory: {session_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {session_dir}: {e}")

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "files_with_content": 0,
            "files_expired": 0,
        }

        cutoff_date = datetime.utcnow() - timedelta(days=FILE_RETENTION_DAYS)

        # Count all files
        result = await self.db.execute(
            select(FileUploadModel).where(FileUploadModel.file_type == "bank_statement")
        )
        all_files = list(result.scalars().all())

        for f in all_files:
            stats["total_files"] += 1
            stats["total_size_bytes"] += f.file_size or 0

            if f.file_path:
                stats["files_with_content"] += 1

            if f.created_at and f.created_at < cutoff_date:
                stats["files_expired"] += 1

        stats["total_size_mb"] = round(stats["total_size_bytes"] / 1024 / 1024, 2)

        return stats
