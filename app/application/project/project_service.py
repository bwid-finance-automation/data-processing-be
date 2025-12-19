"""Service for Project and ProjectCase operations."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from passlib.hash import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models.project import ProjectModel, ProjectCaseModel
from app.infrastructure.database.models.bank_statement import BankStatementModel
from app.infrastructure.database.models.contract import ContractModel
from app.infrastructure.persistence.repositories.project_repository import (
    ProjectRepository,
    ProjectCaseRepository,
)
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class ProjectService:
    """Service for managing projects and cases."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.project_repo = ProjectRepository(db)
        self.case_repo = ProjectCaseRepository(db)

    # ============== Password Utilities ==============

    @staticmethod
    def _truncate_password(password: str) -> bytes:
        """Truncate password to 72 bytes (bcrypt limitation)."""
        return password.encode("utf-8")[:72]

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        truncated = ProjectService._truncate_password(password)
        return bcrypt.hash(truncated)

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against bcrypt hash."""
        try:
            truncated = ProjectService._truncate_password(password)
            return bcrypt.verify(truncated, password_hash)
        except ValueError:
            # Fallback for legacy SHA256 hashes (backward compatibility)
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest() == password_hash

    # ============== Project Operations ==============

    async def create_project(
        self,
        project_name: str,
        description: Optional[str] = None,
        password: Optional[str] = None,
    ) -> ProjectModel:
        """Create a new project."""
        password_hash = self.hash_password(password) if password else None

        logger.info(f"Creating project: {project_name}, protected={password is not None}")
        if password:
            logger.info(f"Password provided, hash: {password_hash[:20]}...")

        project = ProjectModel(
            project_name=project_name,
            description=description,
            is_protected=password is not None,
            password_hash=password_hash,
            last_accessed_at=datetime.utcnow(),
        )

        created = await self.project_repo.create(project)
        await self.db.commit()

        logger.info(f"Created project: {created.project_name} (uuid={created.uuid})")
        return created

    async def get_project(self, uuid: UUID) -> Optional[ProjectModel]:
        """Get project by UUID."""
        return await self.project_repo.get_by_uuid(uuid)

    async def get_project_with_cases(self, uuid: UUID) -> Optional[ProjectModel]:
        """Get project with all cases."""
        project = await self.project_repo.get_with_cases(uuid)
        if project:
            # Update last accessed
            await self.project_repo.update_last_accessed(project.id)
            await self.db.commit()
        return project

    async def list_projects(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
    ) -> tuple[List[ProjectModel], int]:
        """List projects with pagination and optional search."""
        projects = await self.project_repo.get_all_with_cases(skip, limit, search)
        total = await self.project_repo.count_projects(search)
        return projects, total

    async def update_project(
        self,
        uuid: UUID,
        project_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[ProjectModel]:
        """Update project details."""
        project = await self.project_repo.get_by_uuid(uuid)
        if not project:
            return None

        update_data = {}
        if project_name is not None:
            update_data["project_name"] = project_name
        if description is not None:
            update_data["description"] = description

        if update_data:
            await self.project_repo.update(project.id, **update_data)
            await self.db.commit()
            logger.info(f"Updated project: {uuid}")

        return await self.project_repo.get_by_uuid(uuid)

    async def delete_project(self, uuid: UUID) -> bool:
        """Delete a project and all its cases."""
        project = await self.project_repo.get_by_uuid(uuid)
        if not project:
            return False

        deleted = await self.project_repo.delete(project.id)
        await self.db.commit()

        if deleted:
            logger.info(f"Deleted project: {uuid}")
        return deleted

    async def set_password(
        self,
        uuid: UUID,
        password: Optional[str],
    ) -> Optional[ProjectModel]:
        """Set or remove project password."""
        project = await self.project_repo.get_by_uuid(uuid)
        if not project:
            return None

        if password:
            await self.project_repo.update(
                project.id,
                is_protected=True,
                password_hash=self.hash_password(password),
            )
        else:
            await self.project_repo.update(
                project.id,
                is_protected=False,
                password_hash=None,
            )

        await self.db.commit()
        logger.info(f"Updated password for project: {uuid}")
        return await self.project_repo.get_by_uuid(uuid)

    async def verify_project_password(self, uuid: UUID, password: str) -> bool:
        """Verify project password."""
        project = await self.project_repo.get_by_uuid(uuid)
        if not project:
            return False

        if not project.is_protected:
            return True

        return ProjectService.verify_password(password, project.password_hash)

    async def search_projects(self, name: str, limit: int = 10) -> List[ProjectModel]:
        """Search projects by name."""
        return await self.project_repo.search_by_name(name, limit)

    # ============== Case Operations ==============

    async def get_or_create_case(
        self,
        project_uuid: UUID,
        case_type: str,
    ) -> Optional[ProjectCaseModel]:
        """Get or create a case for a project."""
        project = await self.project_repo.get_by_uuid(project_uuid)
        if not project:
            logger.warning(f"Project not found: {project_uuid}")
            return None

        case = await self.case_repo.get_or_create(project.id, case_type)
        await self.db.commit()

        return case

    async def get_case(self, uuid: UUID) -> Optional[ProjectCaseModel]:
        """Get case by UUID."""
        return await self.case_repo.get_by_uuid(uuid)

    async def get_cases_by_project(self, project_uuid: UUID) -> List[ProjectCaseModel]:
        """Get all cases for a project."""
        project = await self.project_repo.get_by_uuid(project_uuid)
        if not project:
            return []
        return await self.case_repo.get_cases_by_project(project.id)

    async def increment_case_file_count(self, case_id: int) -> None:
        """Increment file count for a case."""
        await self.case_repo.increment_file_count(case_id)
        await self.db.commit()

    # ============== Bank Statement Case Operations ==============

    async def get_bank_statement_history(
        self,
        case_uuid: UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[Optional[ProjectCaseModel], List[BankStatementModel], int]:
        """Get bank statement history for a case."""
        case = await self.case_repo.get_by_uuid(case_uuid)
        if not case:
            return None, [], 0

        statements = await self.case_repo.get_bank_statements_by_case(
            case.id, skip, limit
        )
        total = await self.case_repo.count_bank_statements_by_case(case.id)

        return case, statements, total

    async def get_bank_statement_history_by_project(
        self,
        project_uuid: UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[Optional[ProjectCaseModel], List[BankStatementModel], int]:
        """Get bank statement history for a project (via its bank_statement case)."""
        project = await self.project_repo.get_by_uuid(project_uuid)
        if not project:
            return None, [], 0

        case = await self.case_repo.get_by_project_and_type(project.id, "bank_statement")
        if not case:
            return None, [], 0

        statements = await self.case_repo.get_bank_statements_by_case(
            case.id, skip, limit
        )
        total = await self.case_repo.count_bank_statements_by_case(case.id)

        return case, statements, total

    async def get_bank_statement_sessions_by_project(
        self,
        project_uuid: UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[Optional[ProjectCaseModel], List[dict], int]:
        """Get bank statement parse sessions grouped by session_id."""
        project = await self.project_repo.get_by_uuid(project_uuid)
        if not project:
            return None, [], 0

        case = await self.case_repo.get_by_project_and_type(project.id, "bank_statement")
        if not case:
            return None, [], 0

        sessions = await self.case_repo.get_parse_sessions_by_case(
            case.id, skip, limit
        )
        total = await self.case_repo.count_parse_sessions_by_case(case.id)

        return case, sessions, total

    # ============== Contract Case Operations ==============

    async def get_contract_sessions_by_project(
        self,
        project_uuid: UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[Optional[ProjectCaseModel], List[dict], int]:
        """Get contract processing sessions grouped by session for a project."""
        project = await self.project_repo.get_by_uuid(project_uuid)
        if not project:
            return None, [], 0

        case = await self.case_repo.get_by_project_and_type(project.id, "contract")
        if not case:
            return None, [], 0

        sessions = await self.case_repo.get_contract_sessions_by_case(
            case.id, skip, limit
        )
        total = await self.case_repo.count_contract_sessions_by_case(case.id)

        return case, sessions, total
