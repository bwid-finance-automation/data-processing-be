"""Analysis session related database models."""

from datetime import datetime
from typing import Optional, List
import uuid
import enum

from sqlalchemy import String, Text, Integer, Float, ForeignKey, Index, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base, TimestampMixin


class SessionStatus(str, enum.Enum):
    """Session status enumeration."""
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    PROCESSING = "PROCESSING"
    ANALYZING = "ANALYZING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class AnalysisType(str, enum.Enum):
    """Analysis type enumeration."""
    REVENUE_VARIANCE = "REVENUE_VARIANCE"
    COMPREHENSIVE = "COMPREHENSIVE"
    AI_POWERED = "AI_POWERED"
    CUSTOM = "CUSTOM"


class RiskLevel(str, enum.Enum):
    """Risk level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnalysisSessionModel(Base, TimestampMixin):
    """Model for analysis session tracking."""

    __tablename__ = "analysis_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=SessionStatus.CREATED.value,
    )
    analysis_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    # Progress
    files_count: Mapped[int] = mapped_column(Integer, default=0)
    progress_percentage: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    current_step_number: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    timeout_minutes: Mapped[int] = mapped_column(Integer, default=60)

    # Configuration and metadata
    configuration: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    processing_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    warnings: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    errors: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # Relationships
    results: Mapped[List["AnalysisResultModel"]] = relationship(
        "AnalysisResultModel",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_analysis_sessions_session_id", "session_id"),
        Index("ix_analysis_sessions_status", "status"),
        Index("ix_analysis_sessions_user_id", "user_id"),
        Index("ix_analysis_sessions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisSession(id={self.id}, session_id={self.session_id}, status={self.status})>"


class AnalysisResultModel(Base, TimestampMixin):
    """Model for analysis results."""

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        nullable=False,
    )
    session_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Result metadata
    analysis_type: Mapped[str] = mapped_column(String(30), nullable=False)
    subsidiary_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    filename: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Timing
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Results (stored as JSON for flexibility)
    summary: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    key_insights: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    detailed_results: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    configuration_used: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Quality scores
    data_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_assessment: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # File references
    result_files: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    debug_files: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # Issues
    warnings: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    errors: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # Relationship
    session: Mapped["AnalysisSessionModel"] = relationship(
        "AnalysisSessionModel",
        back_populates="results",
    )

    __table_args__ = (
        Index("ix_analysis_results_session_id", "session_id"),
        Index("ix_analysis_results_analysis_type", "analysis_type"),
        Index("ix_analysis_results_subsidiary_name", "subsidiary_name"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisResult(id={self.id}, type={self.analysis_type}, subsidiary={self.subsidiary_name})>"
