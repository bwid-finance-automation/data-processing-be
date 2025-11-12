# app/utils/helpers.py
"""Utility helper functions."""

from typing import Optional, List, Dict, Any

def split_list_string(s: Optional[str]) -> Optional[List[str]]:
    """Split comma or pipe separated string into list."""
    if s is None:
        return None
    raw = s.replace("|", ",")
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return vals or None

def build_config_overrides(
    materiality_vnd: Optional[float] = None,
    recurring_pct_threshold: Optional[float] = None,
    revenue_opex_pct_threshold: Optional[float] = None,
    bs_pct_threshold: Optional[float] = None,
    recurring_code_prefixes: Optional[str] = None,
    min_trend_periods: Optional[int] = None,
    gm_drop_threshold_pct: Optional[float] = None,
    dep_pct_only_prefixes: Optional[str] = None,
    customer_column_hints: Optional[str] = None,
) -> Dict[str, Any]:
    """Build configuration overrides from optional parameters."""
    overrides = {}

    if materiality_vnd is not None:
        overrides["materiality_vnd"] = float(materiality_vnd)
    if recurring_pct_threshold is not None:
        overrides["recurring_pct_threshold"] = float(recurring_pct_threshold)
    if revenue_opex_pct_threshold is not None:
        overrides["revenue_opex_pct_threshold"] = float(revenue_opex_pct_threshold)
    if bs_pct_threshold is not None:
        overrides["bs_pct_threshold"] = float(bs_pct_threshold)
    if min_trend_periods is not None:
        overrides["min_trend_periods"] = int(min_trend_periods)
    if gm_drop_threshold_pct is not None:
        overrides["gm_drop_threshold_pct"] = float(gm_drop_threshold_pct)

    rc = split_list_string(recurring_code_prefixes)
    if rc is not None:
        overrides["recurring_code_prefixes"] = rc

    dep = split_list_string(dep_pct_only_prefixes)
    if dep is not None:
        overrides["dep_pct_only_prefixes"] = dep

    cust = split_list_string(customer_column_hints)
    if cust is not None:
        overrides["customer_column_hints"] = cust

    return overrides