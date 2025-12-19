"""API router for AI Usage tracking operations."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.dependencies import get_ai_usage_repository, get_db
from app.infrastructure.persistence.repositories import AIUsageRepository
from app.infrastructure.database.models.ai_usage import AIUsageModel
from app.presentation.schemas.ai_usage_schemas import (
    AIUsageLogCreate,
    AIUsageLogResponse,
    AIUsageListResponse,
    AIUsageStatsResponse,
    AIUsageByProviderResponse,
    AIUsageByTaskTypeResponse,
    AIUsageDailyResponse,
    AIUsageDashboardResponse,
)

router = APIRouter(prefix="/ai-usage", tags=["AI Usage"])


def _model_to_response(model: AIUsageModel) -> AIUsageLogResponse:
    """Convert model to response schema."""
    return AIUsageLogResponse(
        id=model.id,
        uuid=str(model.uuid),
        project_id=model.project_id,
        case_id=model.case_id,
        session_id=model.session_id,
        provider=model.provider,
        model_name=model.model_name,
        task_type=model.task_type,
        task_description=model.task_description,
        file_name=model.file_name,
        file_count=model.file_count,
        input_tokens=model.input_tokens,
        output_tokens=model.output_tokens,
        total_tokens=model.total_tokens,
        processing_time_ms=model.processing_time_ms,
        processing_time_seconds=model.processing_time_seconds,
        estimated_cost_usd=model.estimated_cost_usd,
        success=model.success,
        error_message=model.error_message,
        requested_at=model.requested_at,
        created_at=model.created_at,
    )


@router.get("/dashboard", response_model=AIUsageDashboardResponse)
async def get_dashboard(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    days: int = Query(30, ge=1, le=365, description="Number of days for daily stats"),
    repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """
    Get AI usage dashboard with aggregated statistics.

    Returns:
    - Overall stats (total tokens, cost, requests, etc.)
    - Usage breakdown by provider
    - Usage breakdown by task type
    - Daily usage for the specified period
    - Recent usage logs
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    end_date = datetime.utcnow()

    # Get aggregated stats
    stats_data = await repo.get_aggregated_stats(
        project_id=project_id,
        start_date=start_date,
        end_date=end_date,
    )

    # Calculate success rate
    total = stats_data["total_requests"]
    success_rate = (stats_data["successful_requests"] / total * 100) if total > 0 else 0.0

    stats = AIUsageStatsResponse(
        **stats_data,
        success_rate=round(success_rate, 2),
    )

    # Get usage by provider
    by_provider_data = await repo.get_usage_by_provider(
        project_id=project_id,
        start_date=start_date,
        end_date=end_date,
    )
    by_provider = [AIUsageByProviderResponse(**p) for p in by_provider_data]

    # Get usage by task type
    by_task_type_data = await repo.get_usage_by_task_type(
        project_id=project_id,
        start_date=start_date,
        end_date=end_date,
    )
    by_task_type = [AIUsageByTaskTypeResponse(**t) for t in by_task_type_data]

    # Get daily usage
    daily_data = await repo.get_daily_usage(
        project_id=project_id,
        start_date=start_date,
        end_date=end_date,
    )
    daily_usage = [AIUsageDailyResponse(**d) for d in daily_data]

    # Get recent logs
    recent = await repo.get_filtered(
        project_id=project_id,
        skip=0,
        limit=10,
    )
    recent_logs = [_model_to_response(r) for r in recent]

    return AIUsageDashboardResponse(
        stats=stats,
        by_provider=by_provider,
        by_task_type=by_task_type,
        daily_usage=daily_usage,
        recent_logs=recent_logs,
    )


@router.get("/stats", response_model=AIUsageStatsResponse)
async def get_stats(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    case_id: Optional[int] = Query(None, description="Filter by case ID"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """Get aggregated AI usage statistics."""
    stats_data = await repo.get_aggregated_stats(
        project_id=project_id,
        case_id=case_id,
        start_date=start_date,
        end_date=end_date,
    )

    total = stats_data["total_requests"]
    success_rate = (stats_data["successful_requests"] / total * 100) if total > 0 else 0.0

    return AIUsageStatsResponse(
        **stats_data,
        success_rate=round(success_rate, 2),
    )


@router.get("/logs", response_model=AIUsageListResponse)
async def get_logs(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    case_id: Optional[int] = Query(None, description="Filter by case ID"),
    provider: Optional[str] = Query(None, description="Filter by provider (gemini, openai, etc.)"),
    task_type: Optional[str] = Query(None, description="Filter by task type (ocr, parsing, etc.)"),
    success: Optional[bool] = Query(None, description="Filter by success status"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of records"),
    repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """
    Get AI usage logs with filtering and pagination.
    """
    logs = await repo.get_filtered(
        project_id=project_id,
        case_id=case_id,
        provider=provider,
        task_type=task_type,
        success=success,
        start_date=start_date,
        end_date=end_date,
        skip=skip,
        limit=limit,
    )

    total = await repo.count_filtered(
        project_id=project_id,
        case_id=case_id,
        provider=provider,
        task_type=task_type,
        success=success,
        start_date=start_date,
        end_date=end_date,
    )

    return AIUsageListResponse(
        items=[_model_to_response(log) for log in logs],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/logs/{log_id}", response_model=AIUsageLogResponse)
async def get_log(
    log_id: int,
    repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """Get a single AI usage log by ID."""
    log = await repo.get(log_id)
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AI usage log not found: {log_id}",
        )
    return _model_to_response(log)


@router.get("/by-provider")
async def get_usage_by_provider(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """Get AI usage aggregated by provider."""
    data = await repo.get_usage_by_provider(
        project_id=project_id,
        start_date=start_date,
        end_date=end_date,
    )
    return [AIUsageByProviderResponse(**p) for p in data]


@router.get("/by-task-type")
async def get_usage_by_task_type(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """Get AI usage aggregated by task type."""
    data = await repo.get_usage_by_task_type(
        project_id=project_id,
        start_date=start_date,
        end_date=end_date,
    )
    return [AIUsageByTaskTypeResponse(**t) for t in data]


@router.get("/daily")
async def get_daily_usage(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """Get daily AI usage statistics."""
    data = await repo.get_daily_usage(
        project_id=project_id,
        days=days,
    )
    return [AIUsageDailyResponse(**d) for d in data]


@router.post("/log", response_model=AIUsageLogResponse, status_code=status.HTTP_201_CREATED)
async def create_log(
    request: AIUsageLogCreate,
    repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """
    Create a new AI usage log entry.

    This endpoint is typically called internally after AI operations,
    but can also be used for manual logging.
    """
    log = AIUsageModel(
        project_id=request.project_id,
        case_id=request.case_id,
        session_id=request.session_id,
        provider=request.provider,
        model_name=request.model_name,
        task_type=request.task_type,
        task_description=request.task_description,
        file_name=request.file_name,
        file_count=request.file_count,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        total_tokens=request.total_tokens,
        processing_time_ms=request.processing_time_ms,
        estimated_cost_usd=request.estimated_cost_usd,
        success=request.success,
        error_message=request.error_message,
        metadata_json=request.metadata_json,
        requested_at=datetime.utcnow(),
    )

    created = await repo.create(log)
    return _model_to_response(created)
