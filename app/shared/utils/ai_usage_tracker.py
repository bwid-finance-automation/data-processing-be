"""
Shared AI usage tracking utility.
Provides a centralized helper for logging AI API usage (tokens, cost, provider)
across all services that call LLM/AI APIs.
"""
from datetime import datetime
from typing import Optional

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Pricing table: model_prefix -> (input_cost_per_token, output_cost_per_token)
AI_PRICING = {
    # Google Gemini
    "gemini-2.0-flash": (0.0000001, 0.0000004),
    "gemini-1.5-flash": (0.000000075, 0.0000003),
    "gemini-1.5-pro": (0.00000125, 0.000005),
    # OpenAI
    "gpt-4o-mini": (0.00000015, 0.0000006),
    "gpt-4o": (0.0000025, 0.00001),
    "gpt-5-mini": (0.000002, 0.000008),
    "gpt-5": (0.000002, 0.000008),
    # Anthropic Claude
    "claude-opus-4-5": (0.000015, 0.000075),
    "claude-opus-4": (0.000015, 0.000075),
    "claude-sonnet-4": (0.000003, 0.000015),
    "claude-3-5-sonnet": (0.000003, 0.000015),
    "claude-3-5-haiku": (0.0000008, 0.000004),
}


def detect_provider(model_name: str) -> str:
    """Detect AI provider from model name."""
    model = model_name.lower()
    if "gemini" in model:
        return "gemini"
    if "gpt" in model or "o1" in model or "o3" in model:
        return "openai"
    if "claude" in model:
        return "anthropic"
    return "other"


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD based on model pricing."""
    model = model_name.lower()
    # Find best matching pricing entry (longest prefix match)
    best_match = None
    best_len = 0
    for prefix, pricing in AI_PRICING.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best_match = pricing
            best_len = len(prefix)

    if not best_match:
        return 0.0

    input_cost, output_cost = best_match
    return (input_tokens * input_cost) + (output_tokens * output_cost)


async def log_ai_usage(
    ai_usage_repo,
    *,
    provider: str,
    model_name: str,
    task_type: str,
    input_tokens: int,
    output_tokens: int,
    processing_time_ms: float = 0.0,
    task_description: str = "",
    file_name: str = "",
    file_count: int = 1,
    session_id: Optional[str] = None,
    project_id: Optional[int] = None,
    case_id: Optional[int] = None,
    user_id: Optional[int] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Log AI API usage to the database.

    This function is safe to call from any endpoint - it wraps everything
    in try/except so tracking failures never break the main request.
    """
    try:
        from app.infrastructure.database.models.ai_usage import AIUsageModel

        total_tokens = input_tokens + output_tokens
        cost = estimate_cost(model_name, input_tokens, output_tokens)

        ai_usage_log = AIUsageModel(
            project_id=project_id,
            case_id=case_id,
            user_id=user_id,
            session_id=session_id,
            provider=provider or detect_provider(model_name),
            model_name=model_name,
            task_type=task_type,
            task_description=task_description,
            file_name=file_name,
            file_count=file_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            processing_time_ms=processing_time_ms,
            estimated_cost_usd=cost,
            success=success,
            error_message=error_message,
            metadata_json=metadata or {},
            requested_at=datetime.utcnow(),
        )

        await ai_usage_repo.create(ai_usage_log)
        await ai_usage_repo.session.commit()

        logger.info(
            f"AI usage logged: {provider}/{model_name} | "
            f"{task_type} | {input_tokens}+{output_tokens}={total_tokens} tokens | "
            f"${cost:.6f}"
        )
    except Exception as e:
        logger.error(f"Failed to log AI usage: {e}")
        try:
            await ai_usage_repo.session.rollback()
        except Exception:
            pass
