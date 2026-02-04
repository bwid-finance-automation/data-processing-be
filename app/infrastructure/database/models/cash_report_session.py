"""Cash Report Session database models."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List
import uuid as uuid_lib

from sqlalchemy import String, Text, Date, Numeric, ForeignKey, Index, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.infrastructure.database.base import Base, TimestampMixin


class CashReportSessionStatus(enum.Enum):
    """Status of a cash report session."""
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class CashReportSessionModel(Base, TimestampMixin):
    """Model for cash report automation sessions."""

    __tablename__ = "cash_report_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid_lib.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid_lib.uuid4,
        unique=True,
        nullable=False,
    )

    # Owner
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Session configuration
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    status: Mapped[CashReportSessionStatus] = mapped_column(
        Enum(
            CashReportSessionStatus,
            values_callable=lambda obj: [e.value for e in obj],
            name="cashreportsessionstatus",
            create_type=False,  # Already created in migration
        ),
        default=CashReportSessionStatus.ACTIVE,
        nullable=False,
    )

    # Report period configuration
    period_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    opening_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    ending_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    fx_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 4), nullable=True)

    # File paths
    working_file_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Statistics
    total_transactions: Mapped[int] = mapped_column(default=0, nullable=False)
    total_files_uploaded: Mapped[int] = mapped_column(default=0, nullable=False)

    # Metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    uploaded_files: Mapped[List["CashReportUploadedFileModel"]] = relationship(
        "CashReportUploadedFileModel",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="CashReportUploadedFileModel.id",
    )

    __table_args__ = (
        Index("ix_cash_report_sessions_status", "status"),
        Index("ix_cash_report_sessions_opening_date", "opening_date"),
        Index("ix_cash_report_sessions_ending_date", "ending_date"),
        Index("ix_cash_report_sessions_user_status", "user_id", "status"),
    )

    def __repr__(self) -> str:
        return f"<CashReportSession(id={self.id}, session_id={self.session_id}, status={self.status})>"


class CashReportUploadedFileModel(Base, TimestampMixin):
    """Model for tracking uploaded bank statement files in a cash report session."""

    __tablename__ = "cash_report_uploaded_files"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("cash_report_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # File info
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int] = mapped_column(default=0, nullable=False)

    # Processing results
    transactions_count: Mapped[int] = mapped_column(default=0, nullable=False)
    transactions_added: Mapped[int] = mapped_column(default=0, nullable=False)
    transactions_skipped: Mapped[int] = mapped_column(default=0, nullable=False)

    # Processing status
    processed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    session: Mapped["CashReportSessionModel"] = relationship(
        "CashReportSessionModel",
        back_populates="uploaded_files",
    )

    __table_args__ = (
        Index("ix_cash_report_uploaded_files_session_id", "session_id"),
    )

    def __repr__(self) -> str:
        return f"<CashReportUploadedFile(id={self.id}, filename={self.original_filename})>"
