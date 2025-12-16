"""Bank statement repository implementation."""

from datetime import date
from typing import Optional, List
from decimal import Decimal

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure.persistence.repositories.base import BaseRepository
from app.infrastructure.database.models.bank_statement import (
    BankStatementModel,
    BankTransactionModel,
    BankBalanceModel,
)


class BankStatementRepository(BaseRepository[BankStatementModel]):
    """Repository for bank statement operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(BankStatementModel, session)

    async def get_with_relations(self, id: int) -> Optional[BankStatementModel]:
        """Get statement with transactions and balances loaded."""
        result = await self.session.execute(
            select(BankStatementModel)
            .options(
                selectinload(BankStatementModel.transactions),
                selectinload(BankStatementModel.balances),
            )
            .where(BankStatementModel.id == id)
        )
        return result.scalar_one_or_none()

    async def get_by_bank_name(
        self,
        bank_name: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[BankStatementModel]:
        """Get statements by bank name."""
        result = await self.session.execute(
            select(BankStatementModel)
            .where(BankStatementModel.bank_name == bank_name)
            .order_by(BankStatementModel.uploaded_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_file_name(self, file_name: str) -> Optional[BankStatementModel]:
        """Get statement by file name."""
        result = await self.session.execute(
            select(BankStatementModel)
            .where(BankStatementModel.file_name == file_name)
        )
        return result.scalar_one_or_none()


class BankTransactionRepository(BaseRepository[BankTransactionModel]):
    """Repository for bank transaction operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(BankTransactionModel, session)

    async def get_by_statement(
        self,
        statement_id: int,
        skip: int = 0,
        limit: int = 1000,
    ) -> List[BankTransactionModel]:
        """Get transactions for a statement."""
        result = await self.session.execute(
            select(BankTransactionModel)
            .where(BankTransactionModel.statement_id == statement_id)
            .order_by(BankTransactionModel.transaction_date.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_account(
        self,
        acc_no: str,
        skip: int = 0,
        limit: int = 1000,
    ) -> List[BankTransactionModel]:
        """Get transactions by account number."""
        result = await self.session.execute(
            select(BankTransactionModel)
            .where(BankTransactionModel.acc_no == acc_no)
            .order_by(BankTransactionModel.transaction_date.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_date_range(
        self,
        start_date: date,
        end_date: date,
        acc_no: Optional[str] = None,
        skip: int = 0,
        limit: int = 1000,
    ) -> List[BankTransactionModel]:
        """Get transactions within a date range."""
        conditions = [
            BankTransactionModel.transaction_date >= start_date,
            BankTransactionModel.transaction_date <= end_date,
        ]

        if acc_no:
            conditions.append(BankTransactionModel.acc_no == acc_no)

        result = await self.session.execute(
            select(BankTransactionModel)
            .where(and_(*conditions))
            .order_by(BankTransactionModel.transaction_date.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_total_debit(
        self,
        statement_id: int,
    ) -> Decimal:
        """Get total debit amount for a statement."""
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.coalesce(func.sum(BankTransactionModel.debit), 0))
            .where(BankTransactionModel.statement_id == statement_id)
        )
        return result.scalar_one()

    async def get_total_credit(
        self,
        statement_id: int,
    ) -> Decimal:
        """Get total credit amount for a statement."""
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.coalesce(func.sum(BankTransactionModel.credit), 0))
            .where(BankTransactionModel.statement_id == statement_id)
        )
        return result.scalar_one()


class BankBalanceRepository(BaseRepository[BankBalanceModel]):
    """Repository for bank balance operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(BankBalanceModel, session)

    async def get_by_statement(
        self,
        statement_id: int,
    ) -> List[BankBalanceModel]:
        """Get balances for a statement."""
        result = await self.session.execute(
            select(BankBalanceModel)
            .where(BankBalanceModel.statement_id == statement_id)
        )
        return list(result.scalars().all())

    async def get_by_account(
        self,
        acc_no: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[BankBalanceModel]:
        """Get balance history for an account."""
        result = await self.session.execute(
            select(BankBalanceModel)
            .where(BankBalanceModel.acc_no == acc_no)
            .order_by(BankBalanceModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
