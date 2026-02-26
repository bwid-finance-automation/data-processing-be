"""API router for file processing history by user."""

from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, delete

from app.core.dependencies import get_db, get_current_user, get_current_user_optional
from app.infrastructure.database.models.user import UserModel
from app.infrastructure.database.models.bank_statement import BankStatementModel
from app.infrastructure.database.models.contract import ContractModel
from app.infrastructure.database.models.gla import GLAProjectModel
from app.infrastructure.database.models.analysis_session import AnalysisSessionModel
from app.presentation.schemas.history_schemas import (
    HistorySummaryResponse,
    ModuleSummary,
    BankStatementHistoryResponse,
    BankStatementSessionItem,
    ContractHistoryResponse,
    ContractSessionItem,
    GLAHistoryResponse,
    GLASessionItem,
    AnalysisHistoryResponse,
    AnalysisSessionItem,
)
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/all", response_model=HistorySummaryResponse)
async def get_history_summary(
    db: AsyncSession = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Get summary of all processing history for the current user."""
    user_id = current_user.id if current_user else None
    modules = []
    total_sessions = 0
    total_files = 0

    # Bank statements
    bs_query = select(
        func.count(func.distinct(BankStatementModel.session_id)),
        func.count(BankStatementModel.id),
        func.max(BankStatementModel.processed_at),
    )
    if user_id is not None:
        bs_query = bs_query.where(BankStatementModel.user_id == user_id)
    else:
        bs_query = bs_query.where(BankStatementModel.user_id.is_(None))
    bs_result = (await db.execute(bs_query)).one()
    if bs_result[1] > 0:
        modules.append(ModuleSummary(
            module="bank_statement",
            total_sessions=bs_result[0] or 0,
            total_files=bs_result[1] or 0,
            last_processed_at=bs_result[2],
        ))
        total_sessions += bs_result[0] or 0
        total_files += bs_result[1] or 0

    # Contracts
    ct_query = select(
        func.count(ContractModel.id),
        func.max(ContractModel.processed_at),
    )
    if user_id is not None:
        ct_query = ct_query.where(ContractModel.user_id == user_id)
    else:
        ct_query = ct_query.where(ContractModel.user_id.is_(None))
    ct_result = (await db.execute(ct_query)).one()
    if ct_result[0] > 0:
        modules.append(ModuleSummary(
            module="contract",
            total_sessions=ct_result[0] or 0,
            total_files=ct_result[0] or 0,
            last_processed_at=ct_result[1],
        ))
        total_sessions += ct_result[0] or 0
        total_files += ct_result[0] or 0

    # GLA
    gla_query = select(
        func.count(GLAProjectModel.id),
        func.max(GLAProjectModel.processed_at),
    )
    if user_id is not None:
        gla_query = gla_query.where(GLAProjectModel.user_id == user_id)
    else:
        gla_query = gla_query.where(GLAProjectModel.user_id.is_(None))
    gla_result = (await db.execute(gla_query)).one()
    if gla_result[0] > 0:
        modules.append(ModuleSummary(
            module="gla",
            total_sessions=gla_result[0] or 0,
            total_files=gla_result[0] or 0,
            last_processed_at=gla_result[1],
        ))
        total_sessions += gla_result[0] or 0
        total_files += gla_result[0] or 0

    # Analysis sessions (variance, utility billing, excel comparison)
    for analysis_type, module_name in [
        ("PYTHON_VARIANCE", "variance"),
        ("AI_POWERED", "variance"),
        ("REVENUE_VARIANCE", "variance"),
        ("UTILITY_BILLING", "utility_billing"),
        ("EXCEL_COMPARISON", "excel_comparison"),
    ]:
        as_query = select(
            func.count(AnalysisSessionModel.id),
            func.sum(AnalysisSessionModel.files_count),
            func.max(AnalysisSessionModel.completed_at),
        ).where(AnalysisSessionModel.analysis_type == analysis_type)
        if user_id is not None:
            as_query = as_query.where(AnalysisSessionModel.user_id == user_id)
        else:
            as_query = as_query.where(AnalysisSessionModel.user_id.is_(None))
        as_result = (await db.execute(as_query)).one()
        if as_result[0] > 0:
            # Check if this module already exists (variance has multiple types)
            existing = next((m for m in modules if m.module == module_name), None)
            if existing:
                existing.total_sessions += as_result[0] or 0
                existing.total_files += as_result[1] or 0
                if as_result[2] and (not existing.last_processed_at or as_result[2] > existing.last_processed_at):
                    existing.last_processed_at = as_result[2]
            else:
                modules.append(ModuleSummary(
                    module=module_name,
                    total_sessions=as_result[0] or 0,
                    total_files=as_result[1] or 0,
                    last_processed_at=as_result[2],
                ))
            total_sessions += as_result[0] or 0
            total_files += as_result[1] or 0

    return HistorySummaryResponse(
        modules=modules,
        total_sessions=total_sessions,
        total_files=total_files,
    )


@router.get("/bank-statements", response_model=BankStatementHistoryResponse)
async def get_bank_statement_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Get bank statement parse sessions for the current user. Requires authentication."""
    user_id = current_user.id

    # Get distinct sessions
    session_query = (
        select(
            BankStatementModel.session_id,
            func.count(BankStatementModel.id).label("file_count"),
            func.max(BankStatementModel.processed_at).label("processed_at"),
        )
        .where(BankStatementModel.user_id == user_id)
        .where(BankStatementModel.session_id.isnot(None))
        .group_by(BankStatementModel.session_id)
        .order_by(desc("processed_at"))
        .offset(skip)
        .limit(limit)
    )
    sessions_result = (await db.execute(session_query)).all()

    # Count total sessions
    count_query = (
        select(func.count(func.distinct(BankStatementModel.session_id)))
        .where(BankStatementModel.user_id == user_id)
        .where(BankStatementModel.session_id.isnot(None))
    )
    total = (await db.execute(count_query)).scalar() or 0

    # Build session items with details
    items = []
    for row in sessions_result:
        sid = row.session_id
        # Get details for this session
        detail_query = select(BankStatementModel).where(
            BankStatementModel.session_id == sid,
            BankStatementModel.user_id == user_id,
        )
        details = (await db.execute(detail_query)).scalars().all()

        banks = list(set(d.bank_name for d in details))
        files = [d.file_name for d in details]
        total_tx = sum(
            (d.metadata_json or {}).get("transaction_count", 0) for d in details
        )

        items.append(BankStatementSessionItem(
            session_id=sid,
            file_count=row.file_count,
            total_transactions=total_tx,
            banks=banks,
            files=files,
            processed_at=row.processed_at,
            download_url=f"/api/finance/bank-statements/download-history/{sid}",
        ))

    return BankStatementHistoryResponse(
        sessions=items, total=total, skip=skip, limit=limit
    )


@router.get("/contracts", response_model=ContractHistoryResponse)
async def get_contract_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Get contract OCR history for the current user."""
    user_id = current_user.id if current_user else None

    base = select(ContractModel)
    count_base = select(func.count(ContractModel.id))
    if user_id is not None:
        base = base.where(ContractModel.user_id == user_id)
        count_base = count_base.where(ContractModel.user_id == user_id)
    else:
        base = base.where(ContractModel.user_id.is_(None))
        count_base = count_base.where(ContractModel.user_id.is_(None))

    total = (await db.execute(count_base)).scalar() or 0
    contracts = (
        await db.execute(
            base.order_by(desc(ContractModel.processed_at)).offset(skip).limit(limit)
        )
    ).scalars().all()

    items = [
        ContractSessionItem(
            file_name=c.file_name,
            contract_number=c.contract_number,
            contract_title=c.contract_title,
            tenant=c.tenant,
            processed_at=c.processed_at,
        )
        for c in contracts
    ]

    return ContractHistoryResponse(
        contracts=items, total=total, skip=skip, limit=limit
    )


@router.get("/gla", response_model=GLAHistoryResponse)
async def get_gla_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Get GLA analysis history for the current user."""
    user_id = current_user.id if current_user else None

    base = select(GLAProjectModel)
    count_base = select(func.count(GLAProjectModel.id))
    if user_id is not None:
        base = base.where(GLAProjectModel.user_id == user_id)
        count_base = count_base.where(GLAProjectModel.user_id == user_id)
    else:
        base = base.where(GLAProjectModel.user_id.is_(None))
        count_base = count_base.where(GLAProjectModel.user_id.is_(None))

    total = (await db.execute(count_base)).scalar() or 0
    gla_items = (
        await db.execute(
            base.order_by(desc(GLAProjectModel.processed_at)).offset(skip).limit(limit)
        )
    ).scalars().all()

    items = [
        GLASessionItem(
            file_name=g.file_name,
            project_code=g.project_code,
            project_name=g.project_name,
            product_type=g.product_type,
            region=g.region,
            period_label=g.period_label,
            processed_at=g.processed_at,
        )
        for g in gla_items
    ]

    return GLAHistoryResponse(
        sessions=items, total=total, skip=skip, limit=limit
    )


@router.get("/variance", response_model=AnalysisHistoryResponse)
async def get_variance_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Get variance analysis history for the current user."""
    return await _get_analysis_history(
        db, current_user,
        ["PYTHON_VARIANCE", "AI_POWERED", "REVENUE_VARIANCE"],
        skip, limit,
    )


@router.get("/utility-billing", response_model=AnalysisHistoryResponse)
async def get_utility_billing_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Get utility billing history for the current user."""
    return await _get_analysis_history(
        db, current_user, ["UTILITY_BILLING"], skip, limit
    )


@router.get("/excel-comparison", response_model=AnalysisHistoryResponse)
async def get_excel_comparison_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Get excel comparison history for the current user."""
    return await _get_analysis_history(
        db, current_user, ["EXCEL_COMPARISON"], skip, limit
    )


async def _get_analysis_history(
    db: AsyncSession,
    current_user: Optional[UserModel],
    analysis_types: list,
    skip: int,
    limit: int,
) -> AnalysisHistoryResponse:
    """Common helper for analysis-type history queries."""
    user_id = current_user.id if current_user else None

    base = select(AnalysisSessionModel).where(
        AnalysisSessionModel.analysis_type.in_(analysis_types)
    )
    count_base = select(func.count(AnalysisSessionModel.id)).where(
        AnalysisSessionModel.analysis_type.in_(analysis_types)
    )

    if user_id is not None:
        base = base.where(AnalysisSessionModel.user_id == user_id)
        count_base = count_base.where(AnalysisSessionModel.user_id == user_id)
    else:
        base = base.where(AnalysisSessionModel.user_id.is_(None))
        count_base = count_base.where(AnalysisSessionModel.user_id.is_(None))

    total = (await db.execute(count_base)).scalar() or 0
    sessions = (
        await db.execute(
            base.order_by(desc(AnalysisSessionModel.completed_at))
            .offset(skip)
            .limit(limit)
        )
    ).scalars().all()

    items = []
    for s in sessions:
        download_url = None
        details = s.processing_details or {}

        if s.analysis_type == "EXCEL_COMPARISON" and details.get("output_file"):
            download_url = f"/api/finance/fpa/download/{details['output_file']}"
        elif s.analysis_type in ("AI_POWERED", "PYTHON_VARIANCE", "REVENUE_VARIANCE"):
            download_url = f"/api/finance/process/download/{s.session_id}"

        items.append(AnalysisSessionItem(
            session_id=s.session_id,
            analysis_type=s.analysis_type,
            status=s.status,
            files_count=s.files_count,
            processing_details=s.processing_details,
            started_at=s.started_at,
            completed_at=s.completed_at,
            download_url=download_url,
        ))

    return AnalysisHistoryResponse(
        sessions=items, total=total, skip=skip, limit=limit
    )


RETENTION_DAYS = 14


async def cleanup_old_history(db: AsyncSession) -> dict:
    """Delete history records older than RETENTION_DAYS. Returns counts of deleted records."""
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    counts = {}

    # Bank statements
    result = await db.execute(
        delete(BankStatementModel).where(BankStatementModel.processed_at < cutoff)
    )
    counts["bank_statements"] = result.rowcount

    # Contracts
    result = await db.execute(
        delete(ContractModel).where(ContractModel.processed_at < cutoff)
    )
    counts["contracts"] = result.rowcount

    # GLA projects (cascade deletes records and tenants)
    result = await db.execute(
        delete(GLAProjectModel).where(GLAProjectModel.processed_at < cutoff)
    )
    counts["gla_projects"] = result.rowcount

    # Analysis sessions (cascade deletes results)
    result = await db.execute(
        delete(AnalysisSessionModel).where(AnalysisSessionModel.completed_at < cutoff)
    )
    counts["analysis_sessions"] = result.rowcount

    await db.commit()
    return counts


@router.delete("/cleanup", summary="Cleanup history older than 14 days")
async def run_cleanup(
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Manually trigger cleanup of history records older than 14 days."""
    counts = await cleanup_old_history(db)
    total = sum(counts.values())
    logger.info(f"History cleanup: deleted {total} records ({counts})")
    return {"deleted": counts, "total": total, "retention_days": RETENTION_DAYS}
