"""AI Usage repository for database operations."""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy import select, func, and_, or_, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.repositories.base import BaseRepository
from app.infrastructure.database.models.ai_usage import AIUsageModel


class AIUsageRepository(BaseRepository[AIUsageModel]):
    """Repository for AI usage log operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(AIUsageModel, session)

    async def get_by_session_id(self, session_id: str) -> List[AIUsageModel]:
        """Get all usage logs for a session."""
        result = await self.session.execute(
            select(AIUsageModel)
            .where(AIUsageModel.session_id == session_id)
            .order_by(AIUsageModel.requested_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_project(
        self,
        project_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AIUsageModel]:
        """Get all usage logs for a project."""
        result = await self.session.execute(
            select(AIUsageModel)
            .where(AIUsageModel.project_id == project_id)
            .order_by(AIUsageModel.requested_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_case(
        self,
        case_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AIUsageModel]:
        """Get all usage logs for a case."""
        result = await self.session.execute(
            select(AIUsageModel)
            .where(AIUsageModel.case_id == case_id)
            .order_by(AIUsageModel.requested_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_filtered(
        self,
        project_id: Optional[int] = None,
        case_id: Optional[int] = None,
        provider: Optional[str] = None,
        task_type: Optional[str] = None,
        success: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AIUsageModel]:
        """Get usage logs with filters."""
        query = select(AIUsageModel)
        conditions = []

        if project_id is not None:
            conditions.append(AIUsageModel.project_id == project_id)
        if case_id is not None:
            conditions.append(AIUsageModel.case_id == case_id)
        if provider is not None:
            conditions.append(AIUsageModel.provider == provider)
        if task_type is not None:
            conditions.append(AIUsageModel.task_type == task_type)
        if success is not None:
            conditions.append(AIUsageModel.success == success)
        if start_date is not None:
            conditions.append(AIUsageModel.requested_at >= start_date)
        if end_date is not None:
            conditions.append(AIUsageModel.requested_at <= end_date)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(AIUsageModel.requested_at.desc())
        query = query.offset(skip).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_filtered(
        self,
        project_id: Optional[int] = None,
        case_id: Optional[int] = None,
        provider: Optional[str] = None,
        task_type: Optional[str] = None,
        success: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Count usage logs with filters."""
        query = select(func.count()).select_from(AIUsageModel)
        conditions = []

        if project_id is not None:
            conditions.append(AIUsageModel.project_id == project_id)
        if case_id is not None:
            conditions.append(AIUsageModel.case_id == case_id)
        if provider is not None:
            conditions.append(AIUsageModel.provider == provider)
        if task_type is not None:
            conditions.append(AIUsageModel.task_type == task_type)
        if success is not None:
            conditions.append(AIUsageModel.success == success)
        if start_date is not None:
            conditions.append(AIUsageModel.requested_at >= start_date)
        if end_date is not None:
            conditions.append(AIUsageModel.requested_at <= end_date)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(query)
        return result.scalar_one()

    async def get_aggregated_stats(
        self,
        project_id: Optional[int] = None,
        case_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get aggregated statistics for usage logs."""
        conditions = []

        if project_id is not None:
            conditions.append(AIUsageModel.project_id == project_id)
        if case_id is not None:
            conditions.append(AIUsageModel.case_id == case_id)
        if start_date is not None:
            conditions.append(AIUsageModel.requested_at >= start_date)
        if end_date is not None:
            conditions.append(AIUsageModel.requested_at <= end_date)

        where_clause = and_(*conditions) if conditions else True

        # Main aggregation query
        query = select(
            func.count().label("total_requests"),
            func.sum(AIUsageModel.input_tokens).label("total_input_tokens"),
            func.sum(AIUsageModel.output_tokens).label("total_output_tokens"),
            func.sum(AIUsageModel.total_tokens).label("total_tokens"),
            func.sum(AIUsageModel.processing_time_ms).label("total_processing_time_ms"),
            func.sum(AIUsageModel.estimated_cost_usd).label("total_cost_usd"),
            func.sum(AIUsageModel.file_count).label("total_files_processed"),
            func.sum(case((AIUsageModel.success == True, 1), else_=0)).label("successful_requests"),
            func.sum(case((AIUsageModel.success == False, 1), else_=0)).label("failed_requests"),
        ).where(where_clause)

        result = await self.session.execute(query)
        row = result.one()

        return {
            "total_requests": row.total_requests or 0,
            "total_input_tokens": row.total_input_tokens or 0,
            "total_output_tokens": row.total_output_tokens or 0,
            "total_tokens": row.total_tokens or 0,
            "total_processing_time_ms": row.total_processing_time_ms or 0,
            "total_processing_time_seconds": (row.total_processing_time_ms or 0) / 1000,
            "total_cost_usd": row.total_cost_usd or 0,
            "total_files_processed": row.total_files_processed or 0,
            "successful_requests": row.successful_requests or 0,
            "failed_requests": row.failed_requests or 0,
        }

    async def get_usage_by_provider(
        self,
        project_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get usage aggregated by provider."""
        conditions = []

        if project_id is not None:
            conditions.append(AIUsageModel.project_id == project_id)
        if start_date is not None:
            conditions.append(AIUsageModel.requested_at >= start_date)
        if end_date is not None:
            conditions.append(AIUsageModel.requested_at <= end_date)

        where_clause = and_(*conditions) if conditions else True

        query = select(
            AIUsageModel.provider,
            func.count().label("request_count"),
            func.sum(AIUsageModel.total_tokens).label("total_tokens"),
            func.sum(AIUsageModel.estimated_cost_usd).label("total_cost_usd"),
        ).where(where_clause).group_by(AIUsageModel.provider)

        result = await self.session.execute(query)
        return [
            {
                "provider": row.provider,
                "request_count": row.request_count,
                "total_tokens": row.total_tokens or 0,
                "total_cost_usd": row.total_cost_usd or 0,
            }
            for row in result.all()
        ]

    async def get_usage_by_task_type(
        self,
        project_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get usage aggregated by task type."""
        conditions = []

        if project_id is not None:
            conditions.append(AIUsageModel.project_id == project_id)
        if start_date is not None:
            conditions.append(AIUsageModel.requested_at >= start_date)
        if end_date is not None:
            conditions.append(AIUsageModel.requested_at <= end_date)

        where_clause = and_(*conditions) if conditions else True

        query = select(
            AIUsageModel.task_type,
            func.count().label("request_count"),
            func.sum(AIUsageModel.total_tokens).label("total_tokens"),
            func.sum(AIUsageModel.estimated_cost_usd).label("total_cost_usd"),
        ).where(where_clause).group_by(AIUsageModel.task_type)

        result = await self.session.execute(query)
        return [
            {
                "task_type": row.task_type,
                "request_count": row.request_count,
                "total_tokens": row.total_tokens or 0,
                "total_cost_usd": row.total_cost_usd or 0,
            }
            for row in result.all()
        ]

    async def get_daily_usage(
        self,
        project_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get daily usage statistics."""
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=days)
        if end_date is None:
            end_date = datetime.utcnow()

        conditions = [
            AIUsageModel.requested_at >= start_date,
            AIUsageModel.requested_at <= end_date,
        ]

        if project_id is not None:
            conditions.append(AIUsageModel.project_id == project_id)

        where_clause = and_(*conditions)

        query = select(
            func.date(AIUsageModel.requested_at).label("date"),
            func.count().label("request_count"),
            func.sum(AIUsageModel.total_tokens).label("total_tokens"),
            func.sum(AIUsageModel.estimated_cost_usd).label("total_cost_usd"),
        ).where(where_clause).group_by(
            func.date(AIUsageModel.requested_at)
        ).order_by(
            func.date(AIUsageModel.requested_at)
        )

        result = await self.session.execute(query)
        return [
            {
                "date": str(row.date),
                "request_count": row.request_count,
                "total_tokens": row.total_tokens or 0,
                "total_cost_usd": row.total_cost_usd or 0,
            }
            for row in result.all()
        ]
