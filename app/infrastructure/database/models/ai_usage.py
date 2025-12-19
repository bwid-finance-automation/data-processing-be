"""AI Usage tracking database models."""

from datetime import datetime
from typing import Optional, TYPE_CHECKING
import uuid as uuid_lib
import enum

from sqlalchemy import String, Text, Integer, Float, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.infrastructure.database.models.project import ProjectModel, ProjectCaseModel


class AIProvider(str, enum.Enum):
    """AI Provider enumeration."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OTHER = "other"


class AITaskType(str, enum.Enum):
    """AI Task type enumeration."""
    OCR = "ocr"
    PARSING = "parsing"
    ANALYSIS = "analysis"
    CHAT = "chat"
    OTHER = "other"


class AIUsageModel(Base, TimestampMixin):
    """Model for tracking AI API usage."""

    __tablename__ = "ai_usage_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid_lib.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid_lib.uuid4,
        unique=True,
        nullable=False,
    )

    # Link to project (optional)
    project_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Link to project case (optional)
    case_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("project_cases.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Session ID for grouping related requests
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # AI Provider info
    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="gemini")
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Task info
    task_type: Mapped[str] = mapped_column(String(50), nullable=False, default="ocr")
    task_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # File info (for file-based tasks)
    file_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Token usage
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Processing time (in milliseconds)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Cost estimation (in USD)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Status
    success: Mapped[bool] = mapped_column(default=True, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Additional metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Timestamp when the AI request was made
    requested_at: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)

    # Relationships
    project: Mapped[Optional["ProjectModel"]] = relationship(
        "ProjectModel",
        foreign_keys=[project_id],
    )
    case: Mapped[Optional["ProjectCaseModel"]] = relationship(
        "ProjectCaseModel",
        foreign_keys=[case_id],
    )

    __table_args__ = (
        Index("ix_ai_usage_logs_project_id", "project_id"),
        Index("ix_ai_usage_logs_case_id", "case_id"),
        Index("ix_ai_usage_logs_session_id", "session_id"),
        Index("ix_ai_usage_logs_provider", "provider"),
        Index("ix_ai_usage_logs_task_type", "task_type"),
        Index("ix_ai_usage_logs_requested_at", "requested_at"),
        Index("ix_ai_usage_logs_success", "success"),
    )

    def __repr__(self) -> str:
        return f"<AIUsage(id={self.id}, provider={self.provider}, model={self.model_name}, tokens={self.total_tokens})>"

    @property
    def processing_time_seconds(self) -> float:
        """Return processing time in seconds."""
        return self.processing_time_ms / 1000.0
