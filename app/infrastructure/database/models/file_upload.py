"""File upload tracking database model."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Integer, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base, TimestampMixin, SoftDeleteMixin


class FileUploadModel(Base, TimestampMixin, SoftDeleteMixin):
    """Model for tracking uploaded files."""

    __tablename__ = "file_uploads"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        nullable=False,
    )

    # File info
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(20), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)

    # Processing info
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)  # bank_statement, contract, gla, etc.
    processing_status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Validation
    validation_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    security_scan_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    estimated_sheets: Mapped[int] = mapped_column(Integer, default=0)

    # Metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    upload_timestamp: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_file_uploads_filename", "filename"),
        Index("ix_file_uploads_file_type", "file_type"),
        Index("ix_file_uploads_session_id", "session_id"),
        Index("ix_file_uploads_processing_status", "processing_status"),
        Index("ix_file_uploads_upload_timestamp", "upload_timestamp"),
        Index("ix_file_uploads_is_deleted", "is_deleted"),
    )

    def __repr__(self) -> str:
        return f"<FileUpload(id={self.id}, filename={self.filename}, type={self.file_type})>"
