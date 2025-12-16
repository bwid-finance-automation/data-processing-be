"""Bank statement related database models."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List
import uuid

from sqlalchemy import String, Text, Date, Numeric, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base, TimestampMixin


class BankStatementModel(Base, TimestampMixin):
    """Model for bank statement metadata."""

    __tablename__ = "bank_statements"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        nullable=False,
    )
    bank_name: Mapped[str] = mapped_column(String(100), nullable=False)
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)

    # Metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    transactions: Mapped[List["BankTransactionModel"]] = relationship(
        "BankTransactionModel",
        back_populates="statement",
        cascade="all, delete-orphan",
    )
    balances: Mapped[List["BankBalanceModel"]] = relationship(
        "BankBalanceModel",
        back_populates="statement",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_bank_statements_bank_name", "bank_name"),
        Index("ix_bank_statements_uploaded_at", "uploaded_at"),
    )

    def __repr__(self) -> str:
        return f"<BankStatement(id={self.id}, bank={self.bank_name}, file={self.file_name})>"


class BankTransactionModel(Base, TimestampMixin):
    """Model for individual bank transactions."""

    __tablename__ = "bank_transactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    statement_id: Mapped[int] = mapped_column(
        ForeignKey("bank_statements.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Transaction fields
    bank_name: Mapped[str] = mapped_column(String(100), nullable=False)
    acc_no: Mapped[str] = mapped_column(String(50), nullable=False)
    transaction_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    debit: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    credit: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="VND")
    transaction_id: Mapped[str] = mapped_column(String(100), nullable=False, default="")

    # Beneficiary info
    beneficiary_bank: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    beneficiary_acc_no: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    beneficiary_acc_name: Mapped[str] = mapped_column(String(500), nullable=False, default="")

    # Relationships
    statement: Mapped["BankStatementModel"] = relationship(
        "BankStatementModel",
        back_populates="transactions",
    )

    __table_args__ = (
        Index("ix_bank_transactions_statement_id", "statement_id"),
        Index("ix_bank_transactions_acc_no", "acc_no"),
        Index("ix_bank_transactions_date", "transaction_date"),
        Index("ix_bank_transactions_bank_name", "bank_name"),
    )

    def __repr__(self) -> str:
        return f"<BankTransaction(id={self.id}, acc={self.acc_no}, debit={self.debit}, credit={self.credit})>"


class BankBalanceModel(Base, TimestampMixin):
    """Model for bank account balance snapshots."""

    __tablename__ = "bank_balances"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    statement_id: Mapped[int] = mapped_column(
        ForeignKey("bank_statements.id", ondelete="CASCADE"),
        nullable=False,
    )

    bank_name: Mapped[str] = mapped_column(String(100), nullable=False)
    acc_no: Mapped[str] = mapped_column(String(50), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="VND")
    opening_balance: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False, default=0)
    closing_balance: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False, default=0)

    # Relationships
    statement: Mapped["BankStatementModel"] = relationship(
        "BankStatementModel",
        back_populates="balances",
    )

    __table_args__ = (
        Index("ix_bank_balances_statement_id", "statement_id"),
        Index("ix_bank_balances_acc_no", "acc_no"),
    )

    def __repr__(self) -> str:
        return f"<BankBalance(id={self.id}, acc={self.acc_no}, opening={self.opening_balance}, closing={self.closing_balance})>"
