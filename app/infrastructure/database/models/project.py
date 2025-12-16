"""Project and ProjectCase database models."""

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
import uuid as uuid_lib
import enum

from sqlalchemy import String, Text, Boolean, Integer, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.infrastructure.database.models.bank_statement import BankStatementModel
    from app.infrastructure.database.models.contract import ContractModel
    from app.infrastructure.database.models.gla import GLAProjectModel
    from app.infrastructure.database.models.analysis_session import AnalysisSessionModel


class CaseType(str, enum.Enum):
    """Case type enumeration."""
    BANK_STATEMENT = "bank_statement"
    CONTRACT = "contract"
    GLA = "gla"
    VARIANCE = "variance"


class ProjectModel(Base, TimestampMixin):
    """Model for project - represents a user/workspace."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid_lib.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid_lib.uuid4,
        unique=True,
        nullable=False,
    )

    # Project info
    project_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Password protection (optional)
    is_protected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Tracking
    last_accessed_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationships
    cases: Mapped[List["ProjectCaseModel"]] = relationship(
        "ProjectCaseModel",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_projects_uuid", "uuid"),
        Index("ix_projects_project_name", "project_name"),
        Index("ix_projects_last_accessed_at", "last_accessed_at"),
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.project_name}, protected={self.is_protected})>"


class ProjectCaseModel(Base, TimestampMixin):
    """Model for project case - auto created when processing files."""

    __tablename__ = "project_cases"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid_lib.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid_lib.uuid4,
        unique=True,
        nullable=False,
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Case info
    case_type: Mapped[str] = mapped_column(String(50), nullable=False)
    total_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_processed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationships
    project: Mapped["ProjectModel"] = relationship(
        "ProjectModel",
        back_populates="cases",
    )

    # Reverse relationships to data models
    bank_statements: Mapped[List["BankStatementModel"]] = relationship(
        "BankStatementModel",
        back_populates="case",
        cascade="all, delete-orphan",
    )
    contracts: Mapped[List["ContractModel"]] = relationship(
        "ContractModel",
        back_populates="case",
        cascade="all, delete-orphan",
    )
    gla_projects: Mapped[List["GLAProjectModel"]] = relationship(
        "GLAProjectModel",
        back_populates="case",
        cascade="all, delete-orphan",
    )
    analysis_sessions: Mapped[List["AnalysisSessionModel"]] = relationship(
        "AnalysisSessionModel",
        back_populates="case",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_project_cases_uuid", "uuid"),
        Index("ix_project_cases_project_id", "project_id"),
        Index("ix_project_cases_case_type", "case_type"),
        UniqueConstraint("project_id", "case_type", name="uq_project_case_type"),
    )

    def __repr__(self) -> str:
        return f"<ProjectCase(id={self.id}, project_id={self.project_id}, type={self.case_type}, files={self.total_files})>"
