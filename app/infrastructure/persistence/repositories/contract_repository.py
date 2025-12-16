"""Contract repository implementation."""

from datetime import date
from typing import Optional, List

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure.persistence.repositories.base import BaseRepository
from app.infrastructure.database.models.contract import (
    ContractModel,
    ContractPartyModel,
    ContractRatePeriodModel,
    ContractUnitModel,
)


class ContractRepository(BaseRepository[ContractModel]):
    """Repository for contract operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(ContractModel, session)

    async def get_with_relations(self, id: int) -> Optional[ContractModel]:
        """Get contract with all related data loaded."""
        result = await self.session.execute(
            select(ContractModel)
            .options(
                selectinload(ContractModel.parties),
                selectinload(ContractModel.rate_periods),
                selectinload(ContractModel.units),
            )
            .where(ContractModel.id == id)
        )
        return result.scalar_one_or_none()

    async def get_by_contract_number(
        self,
        contract_number: str,
    ) -> Optional[ContractModel]:
        """Get contract by contract number."""
        result = await self.session.execute(
            select(ContractModel)
            .where(ContractModel.contract_number == contract_number)
        )
        return result.scalar_one_or_none()

    async def get_by_tenant(
        self,
        tenant: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ContractModel]:
        """Get contracts by tenant name (partial match)."""
        result = await self.session.execute(
            select(ContractModel)
            .where(
                or_(
                    ContractModel.tenant.ilike(f"%{tenant}%"),
                    ContractModel.customer_name.ilike(f"%{tenant}%"),
                )
            )
            .order_by(ContractModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_date_range(
        self,
        start_date: date,
        end_date: date,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ContractModel]:
        """Get contracts within effective date range."""
        result = await self.session.execute(
            select(ContractModel)
            .where(
                and_(
                    ContractModel.effective_date >= start_date,
                    ContractModel.effective_date <= end_date,
                )
            )
            .order_by(ContractModel.effective_date.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_active_contracts(
        self,
        as_of_date: Optional[date] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ContractModel]:
        """Get contracts that are currently active."""
        if as_of_date is None:
            as_of_date = date.today()

        result = await self.session.execute(
            select(ContractModel)
            .where(
                and_(
                    ContractModel.effective_date <= as_of_date,
                    or_(
                        ContractModel.expiration_date.is_(None),
                        ContractModel.expiration_date >= as_of_date,
                    ),
                )
            )
            .order_by(ContractModel.effective_date.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def search(
        self,
        query: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ContractModel]:
        """Search contracts by multiple fields."""
        search_term = f"%{query}%"
        result = await self.session.execute(
            select(ContractModel)
            .where(
                or_(
                    ContractModel.contract_title.ilike(search_term),
                    ContractModel.contract_number.ilike(search_term),
                    ContractModel.tenant.ilike(search_term),
                    ContractModel.customer_name.ilike(search_term),
                    ContractModel.unit_for_lease.ilike(search_term),
                )
            )
            .order_by(ContractModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


class ContractPartyRepository(BaseRepository[ContractPartyModel]):
    """Repository for contract party operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(ContractPartyModel, session)

    async def get_by_contract(
        self,
        contract_id: int,
    ) -> List[ContractPartyModel]:
        """Get all parties for a contract."""
        result = await self.session.execute(
            select(ContractPartyModel)
            .where(ContractPartyModel.contract_id == contract_id)
        )
        return list(result.scalars().all())

    async def get_by_role(
        self,
        role: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ContractPartyModel]:
        """Get parties by role."""
        result = await self.session.execute(
            select(ContractPartyModel)
            .where(ContractPartyModel.role == role)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


class ContractRatePeriodRepository(BaseRepository[ContractRatePeriodModel]):
    """Repository for contract rate period operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(ContractRatePeriodModel, session)

    async def get_by_contract(
        self,
        contract_id: int,
    ) -> List[ContractRatePeriodModel]:
        """Get all rate periods for a contract."""
        result = await self.session.execute(
            select(ContractRatePeriodModel)
            .where(ContractRatePeriodModel.contract_id == contract_id)
        )
        return list(result.scalars().all())


class ContractUnitRepository(BaseRepository[ContractUnitModel]):
    """Repository for contract unit operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(ContractUnitModel, session)

    async def get_by_contract(
        self,
        contract_id: int,
    ) -> List[ContractUnitModel]:
        """Get all units for a contract."""
        result = await self.session.execute(
            select(ContractUnitModel)
            .where(ContractUnitModel.contract_id == contract_id)
        )
        return list(result.scalars().all())

    async def get_by_unit_name(
        self,
        unit: str,
    ) -> List[ContractUnitModel]:
        """Get units by unit name."""
        result = await self.session.execute(
            select(ContractUnitModel)
            .where(ContractUnitModel.unit.ilike(f"%{unit}%"))
        )
        return list(result.scalars().all())
