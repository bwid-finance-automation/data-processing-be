"""Analysis session repository implementation."""

from datetime import datetime, timedelta
from typing import Optional, List

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure.persistence.repositories.base import BaseRepository
from app.infrastructure.database.models.analysis_session import (
    AnalysisSessionModel,
    AnalysisResultModel,
    SessionStatus,
)


class AnalysisSessionRepository(BaseRepository[AnalysisSessionModel]):
    """Repository for analysis session operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(AnalysisSessionModel, session)

    async def get_by_session_id(
        self,
        session_id: str,
    ) -> Optional[AnalysisSessionModel]:
        """Get session by session_id string."""
        result = await self.session.execute(
            select(AnalysisSessionModel)
            .where(AnalysisSessionModel.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_with_results(
        self,
        session_id: str,
    ) -> Optional[AnalysisSessionModel]:
        """Get session with results loaded."""
        result = await self.session.execute(
            select(AnalysisSessionModel)
            .options(selectinload(AnalysisSessionModel.results))
            .where(AnalysisSessionModel.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_by_status(
        self,
        status: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AnalysisSessionModel]:
        """Get sessions by status."""
        result = await self.session.execute(
            select(AnalysisSessionModel)
            .where(AnalysisSessionModel.status == status)
            .order_by(AnalysisSessionModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AnalysisSessionModel]:
        """Get sessions by user ID."""
        result = await self.session.execute(
            select(AnalysisSessionModel)
            .where(AnalysisSessionModel.user_id == user_id)
            .order_by(AnalysisSessionModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_active_sessions(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AnalysisSessionModel]:
        """Get sessions that are currently active (not completed/failed)."""
        active_statuses = [
            SessionStatus.CREATED.value,
            SessionStatus.INITIALIZING.value,
            SessionStatus.PROCESSING.value,
            SessionStatus.ANALYZING.value,
            SessionStatus.COMPLETING.value,
        ]

        result = await self.session.execute(
            select(AnalysisSessionModel)
            .where(AnalysisSessionModel.status.in_(active_statuses))
            .order_by(AnalysisSessionModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_expired_sessions(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AnalysisSessionModel]:
        """Get sessions that have exceeded their timeout."""
        active_statuses = [
            SessionStatus.CREATED.value,
            SessionStatus.INITIALIZING.value,
            SessionStatus.PROCESSING.value,
            SessionStatus.ANALYZING.value,
        ]

        result = await self.session.execute(
            select(AnalysisSessionModel)
            .where(
                and_(
                    AnalysisSessionModel.status.in_(active_statuses),
                    AnalysisSessionModel.created_at < datetime.utcnow() - timedelta(minutes=60),
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_progress(
        self,
        session_id: str,
        progress_percentage: int,
        current_step: Optional[str] = None,
        current_step_number: Optional[int] = None,
    ) -> Optional[AnalysisSessionModel]:
        """Update session progress."""
        session = await self.get_by_session_id(session_id)
        if session:
            session.progress_percentage = progress_percentage
            if current_step:
                session.current_step = current_step
            if current_step_number is not None:
                session.current_step_number = current_step_number
            await self.session.flush()
            await self.session.refresh(session)
        return session

    async def update_status(
        self,
        session_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> Optional[AnalysisSessionModel]:
        """Update session status."""
        session = await self.get_by_session_id(session_id)
        if session:
            session.status = status
            if error_message:
                session.error_message = error_message
            if status == SessionStatus.COMPLETED.value:
                session.completed_at = datetime.utcnow()
                session.progress_percentage = 100
            await self.session.flush()
            await self.session.refresh(session)
        return session

    async def count_active_sessions(self) -> int:
        """Count currently active sessions."""
        from sqlalchemy import func

        active_statuses = [
            SessionStatus.CREATED.value,
            SessionStatus.INITIALIZING.value,
            SessionStatus.PROCESSING.value,
            SessionStatus.ANALYZING.value,
            SessionStatus.COMPLETING.value,
        ]

        result = await self.session.execute(
            select(func.count())
            .select_from(AnalysisSessionModel)
            .where(AnalysisSessionModel.status.in_(active_statuses))
        )
        return result.scalar_one()


class AnalysisResultRepository(BaseRepository[AnalysisResultModel]):
    """Repository for analysis result operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(AnalysisResultModel, session)

    async def get_by_session(
        self,
        session_id: int,
    ) -> List[AnalysisResultModel]:
        """Get all results for a session."""
        result = await self.session.execute(
            select(AnalysisResultModel)
            .where(AnalysisResultModel.session_id == session_id)
            .order_by(AnalysisResultModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_analysis_type(
        self,
        analysis_type: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AnalysisResultModel]:
        """Get results by analysis type."""
        result = await self.session.execute(
            select(AnalysisResultModel)
            .where(AnalysisResultModel.analysis_type == analysis_type)
            .order_by(AnalysisResultModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_subsidiary(
        self,
        subsidiary_name: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AnalysisResultModel]:
        """Get results by subsidiary name."""
        result = await self.session.execute(
            select(AnalysisResultModel)
            .where(AnalysisResultModel.subsidiary_name.ilike(f"%{subsidiary_name}%"))
            .order_by(AnalysisResultModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_high_risk_results(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AnalysisResultModel]:
        """Get results with high or critical risk."""
        result = await self.session.execute(
            select(AnalysisResultModel)
            .where(AnalysisResultModel.risk_assessment.in_(["HIGH", "CRITICAL"]))
            .order_by(AnalysisResultModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
