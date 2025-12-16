"""GLA (Gross Leasable Area) related database models."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List, TYPE_CHECKING
import uuid as uuid_lib

from sqlalchemy import String, Date, Numeric, ForeignKey, Index, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.infrastructure.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.infrastructure.database.models.project import ProjectCaseModel


class ProductType(str, enum.Enum):
    """Product type enumeration."""
    RBF = "RBF"
    RBW = "RBW"
    BTSW = "BTSW"
    OFFICE = "OFFICE"
    SERVICE = "SERVICE"


class Region(str, enum.Enum):
    """Region enumeration."""
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    CENTRAL = "CENTRAL"


class LeaseStatus(str, enum.Enum):
    """Lease status enumeration."""
    HANDED_OVER = "HANDED_OVER"
    OPEN = "OPEN"
    TERMINATED = "TERMINATED"
    VOIDED = "VOIDED"
    NONE = "NONE"


class GLAProjectModel(Base, TimestampMixin):
    """Model for GLA project summary."""

    __tablename__ = "gla_projects"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid_lib.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid_lib.uuid4,
        unique=True,
        nullable=False,
    )

    # Link to project case (nullable for backward compatibility)
    case_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("project_cases.id", ondelete="SET NULL"),
        nullable=True,
    )

    # File info
    file_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    processed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True, default=datetime.utcnow)

    project_code: Mapped[str] = mapped_column(String(50), nullable=False)
    project_name: Mapped[str] = mapped_column(String(200), nullable=False)
    product_type: Mapped[str] = mapped_column(String(20), nullable=False)
    region: Mapped[str] = mapped_column(String(20), nullable=False)
    total_gla_sqm: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False, default=0)

    # Period tracking
    period_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    period_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Relationships
    case: Mapped[Optional["ProjectCaseModel"]] = relationship(
        "ProjectCaseModel",
        back_populates="gla_projects",
    )
    records: Mapped[List["GLARecordModel"]] = relationship(
        "GLARecordModel",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    tenants: Mapped[List["GLATenantModel"]] = relationship(
        "GLATenantModel",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_gla_projects_case_id", "case_id"),
        Index("ix_gla_projects_project_code", "project_code"),
        Index("ix_gla_projects_product_type", "product_type"),
        Index("ix_gla_projects_region", "region"),
        Index("ix_gla_projects_period_date", "period_date"),
    )

    def __repr__(self) -> str:
        return f"<GLAProject(id={self.id}, code={self.project_code}, name={self.project_name})>"


class GLARecordModel(Base, TimestampMixin):
    """Model for individual GLA (Unit for Lease) records."""

    __tablename__ = "gla_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("gla_projects.id", ondelete="SET NULL"),
        nullable=True,
    )

    # UFL fields
    ufl_id: Mapped[int] = mapped_column(nullable=False)
    project_code: Mapped[str] = mapped_column(String(50), nullable=False)
    project_name: Mapped[str] = mapped_column(String(200), nullable=False)
    product_type: Mapped[str] = mapped_column(String(20), nullable=False)
    region: Mapped[str] = mapped_column(String(20), nullable=False)
    gla_for_lease: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    tenant: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Dates
    start_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    end_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Period tracking
    period_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Relationship
    project: Mapped[Optional["GLAProjectModel"]] = relationship(
        "GLAProjectModel",
        back_populates="records",
    )

    __table_args__ = (
        Index("ix_gla_records_project_id", "project_id"),
        Index("ix_gla_records_ufl_id", "ufl_id"),
        Index("ix_gla_records_project_code", "project_code"),
        Index("ix_gla_records_status", "status"),
        Index("ix_gla_records_tenant", "tenant"),
        Index("ix_gla_records_period_date", "period_date"),
    )

    def __repr__(self) -> str:
        return f"<GLARecord(id={self.id}, ufl_id={self.ufl_id}, gla={self.gla_for_lease})>"


class GLATenantModel(Base, TimestampMixin):
    """Model for tenant GLA information."""

    __tablename__ = "gla_tenants"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("gla_projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    tenant_name: Mapped[str] = mapped_column(String(500), nullable=False)
    gla_sqm: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="")  # Open or Handed Over

    # Relationship
    project: Mapped["GLAProjectModel"] = relationship(
        "GLAProjectModel",
        back_populates="tenants",
    )

    __table_args__ = (
        Index("ix_gla_tenants_project_id", "project_id"),
        Index("ix_gla_tenants_tenant_name", "tenant_name"),
    )

    def __repr__(self) -> str:
        return f"<GLATenant(id={self.id}, name={self.tenant_name}, gla={self.gla_sqm})>"
