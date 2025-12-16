"""Contract related database models."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List, TYPE_CHECKING
import uuid as uuid_lib

from sqlalchemy import String, Text, Date, Numeric, ForeignKey, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.infrastructure.database.models.project import ProjectCaseModel


class ContractModel(Base, TimestampMixin):
    """Model for contract information extracted via OCR."""

    __tablename__ = "contracts"

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

    # Contract basic info
    contract_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    contract_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    contract_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    effective_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    expiration_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    contract_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Financial info
    contract_value: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    deposit_amount: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    payment_terms: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Tenant/Customer info
    tenant: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    customer_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Lease-specific fields
    unit_for_lease: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    gla_for_lease: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    gfa: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    months: Mapped[Optional[int]] = mapped_column(nullable=True)
    handover_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Service charges
    service_charge_rate: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    service_charge_applies_to: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Legal info
    governing_law: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    termination_clauses: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    signatures_present: Mapped[bool] = mapped_column(Boolean, default=False)

    # JSON fields for flexible data
    key_obligations: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    special_conditions: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # OCR metadata
    source_file: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    processing_time: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Relationships
    case: Mapped[Optional["ProjectCaseModel"]] = relationship(
        "ProjectCaseModel",
        back_populates="contracts",
    )
    parties: Mapped[List["ContractPartyModel"]] = relationship(
        "ContractPartyModel",
        back_populates="contract",
        cascade="all, delete-orphan",
    )
    rate_periods: Mapped[List["ContractRatePeriodModel"]] = relationship(
        "ContractRatePeriodModel",
        back_populates="contract",
        cascade="all, delete-orphan",
    )
    units: Mapped[List["ContractUnitModel"]] = relationship(
        "ContractUnitModel",
        back_populates="contract",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_contracts_case_id", "case_id"),
        Index("ix_contracts_contract_number", "contract_number"),
        Index("ix_contracts_tenant", "tenant"),
        Index("ix_contracts_customer_name", "customer_name"),
        Index("ix_contracts_effective_date", "effective_date"),
    )

    def __repr__(self) -> str:
        return f"<Contract(id={self.id}, number={self.contract_number}, tenant={self.tenant})>"


class ContractPartyModel(Base, TimestampMixin):
    """Model for parties involved in a contract."""

    __tablename__ = "contract_parties"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    contract_id: Mapped[int] = mapped_column(
        ForeignKey("contracts.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(500), nullable=False)
    role: Mapped[str] = mapped_column(String(100), nullable=False)  # Buyer, Seller, etc.
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationship
    contract: Mapped["ContractModel"] = relationship(
        "ContractModel",
        back_populates="parties",
    )

    __table_args__ = (
        Index("ix_contract_parties_contract_id", "contract_id"),
        Index("ix_contract_parties_role", "role"),
    )

    def __repr__(self) -> str:
        return f"<ContractParty(id={self.id}, name={self.name}, role={self.role})>"


class ContractRatePeriodModel(Base, TimestampMixin):
    """Model for rental rate periods in a contract."""

    __tablename__ = "contract_rate_periods"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    contract_id: Mapped[int] = mapped_column(
        ForeignKey("contracts.id", ondelete="CASCADE"),
        nullable=False,
    )

    start_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    end_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    monthly_rate_per_sqm: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    total_monthly_rate: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    num_months: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Free of charge period
    foc_from: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    foc_to: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    foc_num_months: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    service_charge_rate_per_sqm: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationship
    contract: Mapped["ContractModel"] = relationship(
        "ContractModel",
        back_populates="rate_periods",
    )

    __table_args__ = (
        Index("ix_contract_rate_periods_contract_id", "contract_id"),
    )

    def __repr__(self) -> str:
        return f"<ContractRatePeriod(id={self.id}, start={self.start_date}, end={self.end_date})>"


class ContractUnitModel(Base, TimestampMixin):
    """Model for unit breakdown in a contract."""

    __tablename__ = "contract_units"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    contract_id: Mapped[int] = mapped_column(
        ForeignKey("contracts.id", ondelete="CASCADE"),
        nullable=False,
    )

    unit: Mapped[str] = mapped_column(String(100), nullable=False)
    gfa: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False, default=0)
    customer_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    customer_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    tax_rate: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Relationship
    contract: Mapped["ContractModel"] = relationship(
        "ContractModel",
        back_populates="units",
    )

    __table_args__ = (
        Index("ix_contract_units_contract_id", "contract_id"),
        Index("ix_contract_units_unit", "unit"),
    )

    def __repr__(self) -> str:
        return f"<ContractUnit(id={self.id}, unit={self.unit}, gfa={self.gfa})>"
