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

    async def get_parse_sessions_by_case(
        self,
        case_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> List[dict]:
        """
        Get parse sessions grouped by session_id for a case.
        Returns list of sessions with aggregated info.
        Files within each session are ordered by upload order (id asc).
        """
        # Get all statements for this case, ordered by id (upload order)
        result = await self.session.execute(
            select(BankStatementModel)
            .options(
                selectinload(BankStatementModel.transactions),
            )
            .where(BankStatementModel.case_id == case_id)
            .order_by(BankStatementModel.id.asc())  # Keep upload order
        )
        statements = list(result.scalars().all())

        # Group by session_id
        sessions_dict = {}
        for stmt in statements:
            session_id = stmt.session_id or str(stmt.uuid)  # Fallback to uuid if no session_id
            if session_id not in sessions_dict:
                sessions_dict[session_id] = {
                    "session_id": session_id,
                    "processed_at": stmt.processed_at,
                    "files": [],
                    "total_transactions": 0,
                    "banks": set(),
                }
            sessions_dict[session_id]["files"].append({
                "uuid": str(stmt.uuid),
                "file_name": stmt.file_name,
                "bank_name": stmt.bank_name,
                "transaction_count": len(stmt.transactions),
            })
            sessions_dict[session_id]["total_transactions"] += len(stmt.transactions)
            sessions_dict[session_id]["banks"].add(stmt.bank_name)
            # Update processed_at to latest
            if stmt.processed_at and (
                sessions_dict[session_id]["processed_at"] is None or
                stmt.processed_at > sessions_dict[session_id]["processed_at"]
            ):
                sessions_dict[session_id]["processed_at"] = stmt.processed_at

        # Convert to list and sort sessions by processed_at (newest first)
        sessions = list(sessions_dict.values())
        for s in sessions:
            s["banks"] = list(s["banks"])
            s["file_count"] = len(s["files"])
        sessions.sort(key=lambda x: x["processed_at"] or datetime.min, reverse=True)

        # Apply pagination
        return sessions[skip:skip + limit]

    async def count_parse_sessions_by_case(self, case_id: int) -> int:
        """Count distinct parse sessions in a case."""
        result = await self.session.execute(
            select(func.count(func.distinct(BankStatementModel.session_id)))
            .where(BankStatementModel.case_id == case_id)
        )
        return result.scalar_one() or 0

    # ============== Contract Case Operations ==============

    async def get_contracts_by_case(
        self,
        case_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> List[ContractModel]:
        """Get contracts for a case."""
        from sqlalchemy.orm import selectinload
        result = await self.session.execute(
            select(ContractModel)
            .options(
                selectinload(ContractModel.parties),
                selectinload(ContractModel.rate_periods),
                selectinload(ContractModel.units),
            )
            .where(ContractModel.case_id == case_id)
            .order_by(ContractModel.processed_at.desc().nullslast())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_contracts_by_case(self, case_id: int) -> int:
        """Count contracts in a case."""
        result = await self.session.execute(
            select(func.count())
            .select_from(ContractModel)
            .where(ContractModel.case_id == case_id)
        )
        return result.scalar_one()

    async def get_contract_sessions_by_case(
        self,
        case_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> List[dict]:
        """
        Get contract processing sessions grouped by session_id for a case.
        Returns list of sessions with aggregated info.
        """
        from sqlalchemy.orm import selectinload
        # Get all contracts for this case, ordered by id (upload order)
        result = await self.session.execute(
            select(ContractModel)
            .options(
                selectinload(ContractModel.parties),
                selectinload(ContractModel.rate_periods),
                selectinload(ContractModel.units),
            )
            .where(ContractModel.case_id == case_id)
            .order_by(ContractModel.id.asc())
        )
        contracts = list(result.scalars().all())

        # Group by session_id (using source_file as session marker for now)
        sessions_dict = {}
        for contract in contracts:
            # Use processed_at timestamp as session identifier
            session_key = contract.processed_at.strftime("%Y%m%d%H%M%S") if contract.processed_at else str(contract.uuid)
            if session_key not in sessions_dict:
                sessions_dict[session_key] = {
                    "session_id": session_key,
                    "processed_at": contract.processed_at,
                    "files": [],
                    "total_contracts": 0,
                    "tenants": set(),
                }
            sessions_dict[session_key]["files"].append({
                "uuid": str(contract.uuid),
                "file_name": contract.file_name or contract.source_file,
                "contract_number": contract.contract_number,
                "tenant": contract.tenant or contract.customer_name,
                "unit_for_lease": contract.unit_for_lease,
                "contract_title": contract.contract_title,
            })
            sessions_dict[session_key]["total_contracts"] += 1
            if contract.tenant:
                sessions_dict[session_key]["tenants"].add(contract.tenant)
            elif contract.customer_name:
                sessions_dict[session_key]["tenants"].add(contract.customer_name)

        # Convert to list and sort sessions by processed_at (newest first)
        sessions = list(sessions_dict.values())
        for s in sessions:
            s["tenants"] = list(s["tenants"])
            s["file_count"] = len(s["files"])
        sessions.sort(key=lambda x: x["processed_at"] or datetime.min, reverse=True)

        # Apply pagination
        return sessions[skip:skip + limit]

    async def count_contract_sessions_by_case(self, case_id: int) -> int:
        """Count distinct contract processing sessions in a case."""
        # Count unique processed_at timestamps (approximate session count)
        result = await self.session.execute(
            select(func.count(func.distinct(ContractModel.processed_at)))
            .where(ContractModel.case_id == case_id)
        )
        return result.scalar_one() or 0

    # ============== GLA Case Operations ==============

    async def get_gla_projects_by_case(
        self,
        case_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> List[dict]:
        """Get GLA projects for a case."""
        result = await self.session.execute(
            select(GLAProjectModel)
            .where(GLAProjectModel.case_id == case_id)
            .order_by(GLAProjectModel.processed_at.desc().nullslast())
            .offset(skip)
            .limit(limit)
        )
        projects = list(result.scalars().all())

        return [
            {
                "uuid": str(p.uuid),
                "file_name": p.file_name,
                "processed_at": p.processed_at,
                "project_code": p.project_code,
                "project_name": p.project_name,
                "product_type": p.product_type,
                "region": p.region,
                "total_gla_sqm": float(p.total_gla_sqm) if p.total_gla_sqm else 0,
                "period_label": p.period_label,
            }
            for p in projects
        ]

    async def count_gla_projects_by_case(self, case_id: int) -> int:
        """Count GLA projects in a case."""
        result = await self.session.execute(
            select(func.count())
            .select_from(GLAProjectModel)
            .where(GLAProjectModel.case_id == case_id)
        )
        return result.scalar_one() or 0

    # ============== Analysis Session Operations (Variance, Utility Billing, Excel Comparison) ==============

    async def get_analysis_sessions_by_case(
        self,
        case_id: int,
        skip: int = 0,
        limit: int = 50,
        analysis_type: str = None,
    ) -> List[dict]:
        """Get analysis sessions for a case, optionally filtered by type."""
        query = (
            select(AnalysisSessionModel)
            .where(AnalysisSessionModel.case_id == case_id)
        )

        if analysis_type:
            query = query.where(AnalysisSessionModel.analysis_type == analysis_type)

        query = query.order_by(AnalysisSessionModel.created_at.desc()).offset(skip).limit(limit)

        result = await self.session.execute(query)
        sessions = list(result.scalars().all())

        return [
            {
                "session_id": s.session_id,
                "analysis_type": s.analysis_type,
                "status": s.status,
                "files_count": s.files_count,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "processing_details": s.processing_details,
                "created_at": s.created_at,
            }
            for s in sessions
        ]

    async def count_analysis_sessions_by_case(
        self,
        case_id: int,
        analysis_type: str = None,
    ) -> int:
        """Count analysis sessions in a case, optionally filtered by type."""
        query = (
            select(func.count())
            .select_from(AnalysisSessionModel)
            .where(AnalysisSessionModel.case_id == case_id)
        )

        if analysis_type:
            query = query.where(AnalysisSessionModel.analysis_type == analysis_type)

        result = await self.session.execute(query)
        return result.scalar_one() or 0
