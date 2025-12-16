"""File upload repository implementation."""

from datetime import datetime, timedelta
from typing import Optional, List

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.repositories.base import BaseRepository
from app.infrastructure.database.models.file_upload import FileUploadModel


class FileUploadRepository(BaseRepository[FileUploadModel]):
    """Repository for file upload operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(FileUploadModel, session)

    async def get_by_filename(
        self,
        filename: str,
    ) -> Optional[FileUploadModel]:
        """Get file upload by filename."""
        result = await self.session.execute(
            select(FileUploadModel)
            .where(
                and_(
                    FileUploadModel.filename == filename,
                    FileUploadModel.is_deleted == False,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_by_session(
        self,
        session_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[FileUploadModel]:
        """Get files by session ID."""
        result = await self.session.execute(
            select(FileUploadModel)
            .where(
                and_(
                    FileUploadModel.session_id == session_id,
                    FileUploadModel.is_deleted == False,
                )
            )
            .order_by(FileUploadModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_file_type(
        self,
        file_type: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[FileUploadModel]:
        """Get files by file type."""
        result = await self.session.execute(
            select(FileUploadModel)
            .where(
                and_(
                    FileUploadModel.file_type == file_type,
                    FileUploadModel.is_deleted == False,
                )
            )
            .order_by(FileUploadModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_processing_status(
        self,
        status: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[FileUploadModel]:
        """Get files by processing status."""
        result = await self.session.execute(
            select(FileUploadModel)
            .where(
                and_(
                    FileUploadModel.processing_status == status,
                    FileUploadModel.is_deleted == False,
                )
            )
            .order_by(FileUploadModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_pending_files(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[FileUploadModel]:
        """Get files pending processing."""
        return await self.get_by_processing_status("pending", skip, limit)

    async def get_recent_uploads(
        self,
        hours: int = 24,
        skip: int = 0,
        limit: int = 100,
    ) -> List[FileUploadModel]:
        """Get files uploaded in the last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(FileUploadModel)
            .where(
                and_(
                    FileUploadModel.created_at >= since,
                    FileUploadModel.is_deleted == False,
                )
            )
            .order_by(FileUploadModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_processing_status(
        self,
        id: int,
        status: str,
    ) -> Optional[FileUploadModel]:
        """Update file processing status."""
        return await self.update(id, processing_status=status)

    async def update_validation_status(
        self,
        id: int,
        status: str,
        security_scan_passed: bool = True,
    ) -> Optional[FileUploadModel]:
        """Update file validation status."""
        return await self.update(
            id,
            validation_status=status,
            security_scan_passed=security_scan_passed,
        )

    async def get_total_size_by_session(
        self,
        session_id: str,
    ) -> int:
        """Get total file size for a session."""
        from sqlalchemy import func

        result = await self.session.execute(
            select(func.coalesce(func.sum(FileUploadModel.file_size), 0))
            .where(
                and_(
                    FileUploadModel.session_id == session_id,
                    FileUploadModel.is_deleted == False,
                )
            )
        )
        return result.scalar_one()

    async def count_by_session(
        self,
        session_id: str,
    ) -> int:
        """Count files for a session."""
        from sqlalchemy import func

        result = await self.session.execute(
            select(func.count())
            .select_from(FileUploadModel)
            .where(
                and_(
                    FileUploadModel.session_id == session_id,
                    FileUploadModel.is_deleted == False,
                )
            )
        )
        return result.scalar_one()

    async def cleanup_old_files(
        self,
        days: int = 30,
    ) -> int:
        """Soft delete files older than N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        result = await self.session.execute(
            select(FileUploadModel)
            .where(
                and_(
                    FileUploadModel.created_at < cutoff,
                    FileUploadModel.is_deleted == False,
                )
            )
        )
        files = result.scalars().all()

        count = 0
        for file in files:
            file.soft_delete()
            count += 1

        await self.session.flush()
        return count
