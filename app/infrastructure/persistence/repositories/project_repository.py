"""Repository for Project and ProjectCase operations."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure.database.models.project import ProjectModel, ProjectCaseModel
from app.infrastructure.database.models.bank_statement import BankStatementModel
from app.infrastructure.database.models.contract import ContractModel
from app.infrastructure.database.models.gla import GLAProjectModel
from app.infrastructure.database.models.analysis_session import AnalysisSessionModel
from app.infrastructure.persistence.repositories.base import BaseRepository


class ProjectRepository(BaseRepository[ProjectModel]):
    """Repository for Project operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(ProjectModel, session)

    async def get_by_uuid(self, uuid: UUID) -> Optional[ProjectModel]:
        """Get project by UUID."""
        result = await self.session.execute(
            select(ProjectModel).where(ProjectModel.uuid == uuid)
        )
        return result.scalar_one_or_none()

    async def get_with_cases(self, uuid: UUID) -> Optional[ProjectModel]:
        """Get project with all cases loaded."""
        result = await self.session.execute(
            select(ProjectModel)
            .options(selectinload(ProjectModel.cases))
            .where(ProjectModel.uuid == uuid)
        )
        return result.scalar_one_or_none()

    async def get_all_with_cases(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
    ) -> List[ProjectModel]:
        """Get all projects with cases, optionally filtered by search term."""
        query = (
            select(ProjectModel)
            .options(selectinload(ProjectModel.cases))
            .order_by(ProjectModel.last_accessed_at.desc())
        )

        if search:
            query = query.where(
                ProjectModel.project_name.ilike(f"%{search}%")
            )

        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_projects(self, search: Optional[str] = None) -> int:
        """Count total projects, optionally filtered by search."""
        query = select(func.count()).select_from(ProjectModel)
        if search:
            query = query.where(ProjectModel.project_name.ilike(f"%{search}%"))
        result = await self.session.execute(query)
        return result.scalar_one()

    async def update_last_accessed(self, project_id: int) -> None:
        """Update last_accessed_at timestamp."""
        await self.update(project_id, last_accessed_at=datetime.utcnow())

    async def search_by_name(self, name: str, limit: int = 10) -> List[ProjectModel]:
        """Search projects by name (partial match)."""
        result = await self.session.execute(
            select(ProjectModel)
            .where(ProjectModel.project_name.ilike(f"%{name}%"))
            .order_by(ProjectModel.last_accessed_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class ProjectCaseRepository(BaseRepository[ProjectCaseModel]):
    """Repository for ProjectCase operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(ProjectCaseModel, session)

    async def get_by_uuid(self, uuid: UUID) -> Optional[ProjectCaseModel]:
        """Get case by UUID."""
        result = await self.session.execute(
            select(ProjectCaseModel).where(ProjectCaseModel.uuid == uuid)
        )
        return result.scalar_one_or_none()

    async def get_by_project_and_type(
        self,
        project_id: int,
        case_type: str,
    ) -> Optional[ProjectCaseModel]:
        """Get case by project ID and case type."""
        result = await self.session.execute(
            select(ProjectCaseModel).where(
                ProjectCaseModel.project_id == project_id,
                ProjectCaseModel.case_type == case_type,
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create(
        self,
        project_id: int,
        case_type: str,
    ) -> ProjectCaseModel:
        """Get existing case or create new one."""
        case = await self.get_by_project_and_type(project_id, case_type)
        if case:
            return case

        # Create new case
        new_case = ProjectCaseModel(
            project_id=project_id,
            case_type=case_type,
            total_files=0,
        )
        return await self.create(new_case)

    async def increment_file_count(self, case_id: int) -> None:
        """Increment total_files and update last_processed_at."""
        case = await self.get(case_id)
        if case:
            await self.update(
                case_id,
                total_files=case.total_files + 1,
                last_processed_at=datetime.utcnow(),
            )

    async def get_cases_by_project(self, project_id: int) -> List[ProjectCaseModel]:
        """Get all cases for a project."""
        result = await self.session.execute(
            select(ProjectCaseModel)
            .where(ProjectCaseModel.project_id == project_id)
            .order_by(ProjectCaseModel.last_processed_at.desc().nullslast())
        )
        return list(result.scalars().all())

    async def get_case_with_bank_statements(
        self,
        case_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[Optional[ProjectCaseModel], List[BankStatementModel]]:
        """Get case with its bank statements."""
        # Get case
        case = await self.get(case_id)
        if not case:
            return None, []

        # Get bank statements
        result = await self.session.execute(
            select(BankStatementModel)
            .where(BankStatementModel.case_id == case_id)
            .order_by(BankStatementModel.processed_at.desc().nullslast())
            .offset(skip)
            .limit(limit)
        )
        statements = list(result.scalars().all())

        return case, statements

    async def get_bank_statements_by_case(
        self,
        case_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> List[BankStatementModel]:
        """Get bank statements for a case."""
        result = await self.session.execute(
            select(BankStatementModel)
            .options(
                selectinload(BankStatementModel.transactions),
                selectinload(BankStatementModel.balances),
            )
            .where(BankStatementModel.case_id == case_id)
            .order_by(BankStatementModel.processed_at.desc().nullslast())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_bank_statements_by_case(self, case_id: int) -> int:
        """Count bank statements in a case."""
        result = await self.session.execute(
            select(func.count())
            .select_from(BankStatementModel)
            .where(BankStatementModel.case_id == case_id)
        )
        return result.scalar_one()
