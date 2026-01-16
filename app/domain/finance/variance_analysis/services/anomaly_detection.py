# app/anomaly_detection.py
"""Anomaly detection functions for financial data analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List

from app.infrastructure.persistence.excel.data_utils import (
    normalize_period_label, month_key, match_codes, _months,
    _anom_record, _is_511, _is_632, _is_641, _is_642, _is_635,
    _is_515, _is_217, _is_214, _pct_change, _series_hist_pct_of_rev,
    find_customer_column, compute_mom_with_trends, classify_pl_account,
    get_threshold_cause, ACCT_THRESH
)
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Import AI analyzer for AI mode
try:
    from .llm_analyzer import LLMFinancialAnalyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI analyzer not available - AI mode will be disabled")

def build_corr_anoms(
    sub: str,
    combined: pd.DataFrame,
    corr_rules: pd.DataFrame,
    periods: list[str],
    materiality: int,
) -> list[dict]:
    """Correlation-rule anomalies; flags when left/right deltas break expected direction."""
    items: list[dict] = []

    cols = {c.lower(): c for c in corr_rules.columns}
    def pick(opts: list[str]) -> str | None:
        for n in opts:
            if n in cols:
                return cols[n]
        return None

    left_col  = pick(["left_patterns","left_pattern","left_patter","left"])
    right_col = pick(["right_patterns","right_pattern","right_patter","right"])
    cause_col = pick(["cause_message","cause","message","notes"])
    type_col  = pick(["relation_type","type","direction"])

    for _, rule in corr_rules.iterrows():
        lp = str(rule[left_col]) if left_col else ""
        rp = str(rule[right_col]) if right_col else ""
        cause = str(rule[cause_col]) if cause_col else "Correlation mismatch"
        rel = str(rule.get(type_col, "directional")).strip().lower() if type_col else "directional"
        inverse = rel in ("inverse","opposite","neg","negative")
        if not lp or not rp:
            continue

        for per in periods:
            mom = combined[combined["Norm_Period"] == per]
            if mom.empty:
                continue

            l = mom[match_codes(mom["Account Code"], lp)]["Delta"].sum()
            r = mom[match_codes(mom["Account Code"], rp)]["Delta"].sum()

            ok = ((l > 0 and r < 0) or (l < 0 and r > 0)) if inverse else ((l > 0 and r > 0) or (l < 0 and r < 0))
            if abs(l) >= materiality and not ok:
                items.append({
                    "Subsidiary": sub,
                    "Account": f"{lp} ↔ {rp}",
                    "Period": per,
                    "Pct Change": "",
                    "Abs Change (VND)": int(l),
                    "Trigger(s)": "Correlation break",
                    "Suggested likely cause": cause,
                    "Status": "Needs Review",
                    "Notes": f"Left Δ={int(l):,}, Right Δ={int(r):,}, relation={'inverse' if inverse else 'directional'}",
                })
    return items

def build_gross_margin_anoms(
    sub: str,
    pl_data: pd.DataFrame,
    pl_cols: list[str],
    CONFIG: dict,
) -> list[dict]:
    """Gross margin % = (Revenue(511*) - COGS(632*)) / Revenue(511*).
    Flag when current GM% drops vs prior by >= gm_drop_threshold_pct (absolute).
    """
    items: list[dict] = []
    if pl_data is None or pl_data.empty or len(pl_cols) < 2:
        return items

    # Sum revenue & COGS by month across all 511* / 632* rows
    is_rev = pl_data["Account Code"].astype(str).str.startswith("511")
    is_cogs = pl_data["Account Code"].astype(str).str.startswith("632")

    rev = pl_data.loc[is_rev, pl_cols].sum(numeric_only=True)
    cogs = pl_data.loc[is_cogs, pl_cols].sum(numeric_only=True)

    # Walk month to month
    for i in range(1, len(pl_cols)):
        prev, cur = pl_cols[i-1], pl_cols[i]
        rev_prev, rev_cur = float(rev.get(prev, 0.0) or 0.0), float(rev.get(cur, 0.0) or 0.0)
        cogs_prev, cogs_cur = float(cogs.get(prev, 0.0) or 0.0), float(cogs.get(cur, 0.0) or 0.0)

        if rev_prev == 0 or rev_cur == 0:
            continue  # cannot compute a meaningful margin

        gm_prev = (rev_prev - cogs_prev) / rev_prev
        gm_cur  = (rev_cur  - cogs_cur)  / rev_cur
        drop = gm_cur - gm_prev  # negative when margin worsens

        if drop <= -abs(CONFIG.get("gm_drop_threshold_pct", 0.01)):
            items.append({
                "Subsidiary": sub,
                "Account": "Gross Margin (511-632)",
                "Period": normalize_period_label(cur),
                "Pct Change": round((gm_cur - gm_prev) * 100, 2),
                "Abs Change (VND)": "",
                "Trigger(s)": f"Gross margin drop ≥ {int(CONFIG.get('gm_drop_threshold_pct', 0.01)*100)}%",
                "Suggested likely cause": "COGS moved vs revenue; check pricing, mix, or timing.",
                "Status": "Needs Review",
                "Notes": f"GM {normalize_period_label(prev)}={gm_prev:.2%} → {normalize_period_label(cur)}={gm_cur:.2%}",
            })

    return items

def build_revenue_by_customer_anoms(
    sub: str,
    pl_data: pd.DataFrame,
    pl_cols: list[str],
    CONFIG: dict,
) -> list[dict]:
    """If pl_data includes a customer column, compute MoM by customer for 511* rows."""
    items: list[dict] = []
    if pl_data is None or pl_data.empty or len(pl_cols) < 2:
        return items

    cust_col = find_customer_column(pl_data, CONFIG)
    if not cust_col or cust_col not in pl_data.columns:
        return items  # nothing to do

    # Filter revenue rows
    rev_df = pl_data[pl_data["Account Code"].astype(str).str.startswith("511")].copy()
    if rev_df.empty:
        return items

    # Group by customer per month; then compute MoM % and abs deltas
    g = rev_df.groupby(cust_col)[pl_cols].sum(numeric_only=True)
    customers = g.index.tolist()

    for cust in customers:
        series = g.loc[cust]
        for i in range(1, len(pl_cols)):
            prev, cur = pl_cols[i-1], pl_cols[i]
            prev_v = float(series.get(prev, 0.0) or 0.0)
            cur_v  = float(series.get(cur, 0.0) or 0.0)
            delta  = cur_v - prev_v
            pct    = (delta / prev_v) if prev_v != 0 else np.nan

            # Use the same Revenue/OPEX rule: % > threshold OR abs ≥ materiality
            cond_pct = (pd.notna(pct) and abs(pct) > CONFIG["revenue_opex_pct_threshold"])
            cond_abs = (abs(delta) >= CONFIG["materiality_vnd"])
            if cond_pct or cond_abs:
                items.append({
                    "Subsidiary": sub,
                    "Account": f"Revenue 511* — Customer: {cust}",
                    "Period": normalize_period_label(cur),
                    "Pct Change": round(pct*100, 2) if pd.notna(pct) else "",
                    "Abs Change (VND)": int(delta),
                    "Trigger(s)": "Revenue by customer variance",
                    "Suggested likely cause": "Customer-level shift; check orders, churn, or timing.",
                    "Status": "Needs Review",
                    "Notes": f"{normalize_period_label(prev)}={int(prev_v):,} → {normalize_period_label(cur)}={int(cur_v):,}",
                })

    return items

# -----------------------------------------------------------------------------
# AI Analysis Integration
# -----------------------------------------------------------------------------

def build_anoms_ai_mode(
    sub: str,
    excel_bytes: bytes,
    filename: str,
    CONFIG: dict,
    progress_callback=None,
    initial_progress=0,
) -> pd.DataFrame:
    """AI-only anomaly detection using complete raw Excel file."""
    anomalies: list[dict] = []

    if not AI_AVAILABLE:
        logger.error(f"AI analysis requested but AI analyzer not available for '{sub}'")
        return pd.DataFrame([{
            "Subsidiary": sub,
            "Account": "AI_NOT_AVAILABLE",
            "Period": "N/A",
            "Pct Change": 0,
            "Abs Change (VND)": 0,
            "Trigger(s)": "AI Analysis Not Available",
            "Suggested likely cause": "AI analyzer module not found - install required dependencies",
            "Status": "Error",
            "Notes": "Check if llm_analyzer.py is available and dependencies are installed",
        }])

    logger.info(f"Starting AI analysis for '{sub}'...")
    try:
        llm_analyzer = LLMFinancialAnalyzer(CONFIG.get("llm_model", "gpt-4o"), progress_callback=progress_callback, initial_progress=initial_progress)
        logger.info(f"AI analyzer initialized with model: {CONFIG.get('llm_model', 'gpt-4o')}")

        logger.info("Running AI analysis on complete raw Excel file...")
        llm_anomalies = llm_analyzer.analyze_raw_excel_file(excel_bytes, filename, sub, CONFIG)
        logger.info(f"AI analysis completed, processing {len(llm_anomalies)} results")

        logger.info(f"Converting {len(llm_anomalies)} AI results to report format...")
        for idx, anom in enumerate(llm_anomalies, 1):
            logger.debug(f"   • Processing anomaly {idx}: Account {anom.get('account_code', 'Unknown')}")

            # Skip anomalies where both change_amount and change_percent are 0
            change_amount = anom.get("change_amount", 0)
            change_percent = anom.get("change_percent", 0)

            if change_amount == 0 and change_percent == 0:
                logger.debug(f"   • Skipping anomaly {idx}: No change detected (both amount and percent are 0)")
                continue

            # Map severity to emoji for Excel coloring
            severity = anom.get("severity", "Low")
            if severity == "Critical":
                priority_emoji = "🔴 Critical"
            elif severity == "High":
                priority_emoji = "🔴 High"
            elif severity == "Medium":
                priority_emoji = "🟡 Review"
            else:  # Low
                priority_emoji = "🟢 Info"

            anomalies.append({
                "Subsidiary": anom["subsidiary"],
                "Account": anom["account_code"],
                "Period": anom.get("period", "Current"),
                "Pct Change": change_percent,
                "Abs Change (VND)": int(change_amount),
                "Trigger(s)": anom["rule_name"],
                "Suggested likely cause": anom["details"],
                "Status": "AI Analysis",
                "Priority": priority_emoji,
                "Notes": anom["details"],
            })
        logger.info("Successfully converted all AI results to report format")

        logger.info(f"AI anomaly detection completed for '{sub}' - returning {len(anomalies)} records")

        # Create DataFrame and filter out zero changes
        result_df = pd.DataFrame(anomalies)

        if not result_df.empty:
            def is_zero_or_empty(val):
                if pd.isna(val) or val == "" or val == 0 or val == "0":
                    return True
                try:
                    return float(val) == 0
                except (ValueError, TypeError):
                    return False

            # Keep only rows where at least one of the change fields is non-zero
            mask = ~(result_df["Pct Change"].apply(is_zero_or_empty) & result_df["Abs Change (VND)"].apply(is_zero_or_empty))
            result_df = result_df[mask].reset_index(drop=True)
            logger.info(f"After filtering zero changes: {len(result_df)} records remaining")

        return result_df

    except Exception as e:
        logger.error(f"AI analysis failed for '{sub}': {e}", exc_info=True)
        error_record = pd.DataFrame([{
            "Subsidiary": sub,
            "Account": "AI_ERROR",
            "Period": "N/A",
            "Pct Change": 0,
            "Abs Change (VND)": 0,
            "Trigger(s)": "AI Analysis Failed",
            "Suggested likely cause": f"AI model error: {str(e)[:100]}...",
            "Status": "Error",
            "Notes": "Check if OpenAI is running and model is available",
        }])
        logger.warning("Returning error record to continue processing other files")
        return error_record

# -----------------------------------------------------------------------------
# Python Analysis Mode (Traditional Rule-Based)
# -----------------------------------------------------------------------------

def build_anoms_python_mode(
    sub: str,
    bs_data: pd.DataFrame, bs_cols: list[str],
    pl_data: pd.DataFrame, pl_cols: list[str],
    corr_rules: pd.DataFrame, season_rules: pd.DataFrame,
    CONFIG: dict,
) -> pd.DataFrame:
    """Apply all anomaly rules to one subsidiary's cleaned BS/PL frames."""
    anomalies: list[dict] = []
    materiality = CONFIG["materiality_vnd"]

    bs_mom = compute_mom_with_trends(bs_data, bs_cols, CONFIG)
    pl_mom = compute_mom_with_trends(pl_data, pl_cols, CONFIG)

    # Balance Sheet rule
    for _, row in bs_mom.iterrows():
        abs_delta = abs(row["Delta"])
        pct_change = row["Pct Change"]

        # Skip if both change amount and percent are 0
        if abs_delta == 0 and (pd.isna(pct_change) or pct_change == 0):
            continue

        if (abs_delta >= materiality and pd.notna(pct_change) and abs(pct_change) > CONFIG["bs_pct_threshold"]):
            anomalies.append({
                "Subsidiary": sub,
                "Account": f'{row["Account Code"]}-{row["Account Name"]}',
                "Period": row["Period"],
                "Pct Change": round(pct_change * 100, 2),
                "Abs Change (VND)": int(row["Delta"]),
                "Trigger(s)": "BS >5% & ≥1B",
                "Suggested likely cause": get_threshold_cause("BS", row["Account Code"], CONFIG),
                "Status": "Needs Review",
                "Notes": "",
            })

    # P/L rules split: Recurring vs Revenue/OPEX (+ special %-only for dep accounts)
    dep_prefixes = [str(p) for p in CONFIG.get("dep_pct_only_prefixes", [])]

    def is_dep_pct_only(code: object) -> bool:
        cs = str(code)
        return any(cs.startswith(p) for p in dep_prefixes)

    for _, row in pl_mom.iterrows():
        abs_delta = abs(row["Delta"])
        pct_change = row["Pct Change"]
        code = row["Account Code"]

        # Skip if both change amount and percent are 0
        if abs_delta == 0 and (pd.isna(pct_change) or pct_change == 0):
            continue

        account_class = classify_pl_account(code, CONFIG)
        trigger = ""

        if is_dep_pct_only(code):
            if (pd.notna(pct_change) and abs(pct_change) > CONFIG["recurring_pct_threshold"]):
                trigger = "Depreciation % change > threshold"
        elif account_class == "Recurring":
            if (abs_delta >= materiality and pd.notna(pct_change) and abs(pct_change) > CONFIG["recurring_pct_threshold"]):
                trigger = "Recurring >5% & ≥1B"
        else:
            if ((pd.notna(pct_change) and abs(pct_change) > CONFIG["revenue_opex_pct_threshold"]) or abs_delta >= materiality):
                trigger = "Revenue/OPEX >10% or ≥1B"

        if trigger:
            anomalies.append({
                "Subsidiary": sub,
                "Account": f'{row["Account Code"]}-{row["Account Name"]}',
                "Period": row["Period"],
                "Pct Change": round(pct_change * 100, 2) if pd.notna(pct_change) else "",
                "Abs Change (VND)": int(row["Delta"]),
                "Trigger(s)": trigger,
                "Suggested likely cause": get_threshold_cause("PL", row["Account Code"], CONFIG),
                "Status": "Needs Review",
                "Notes": "",
            })

    # Gross Margin anomalies
    anomalies.extend(build_gross_margin_anoms(sub, pl_data, pl_cols, CONFIG))

    # Revenue-by-customer anomalies (if a customer column exists)
    anomalies.extend(build_revenue_by_customer_anoms(sub, pl_data, pl_cols, CONFIG))

    # Correlation rules (optional)
    combined = pd.concat([
        bs_mom[["Account Code","Period","Delta"]],
        pl_mom[["Account Code","Period","Delta"]],
    ], ignore_index=True)
    combined["Norm_Period"] = combined["Period"].astype(str).map(normalize_period_label)
    periods = sorted(set(combined["Norm_Period"]), key=month_key)
    anomalies.extend(build_corr_anoms(sub, combined, corr_rules, periods, materiality))

    # Accounting-focused anomalies (wrapper)
    from .accounting_rules import run_accounting_rules_on_frames
    acct_anoms_df = run_accounting_rules_on_frames(pl_data, bs_data, subsidiary=sub)

    # Build final DataFrame
    main_df = pd.DataFrame(anomalies)
    if acct_anoms_df is not None and not acct_anoms_df.empty:
        if not main_df.empty:
            main_df = pd.concat([main_df, acct_anoms_df], ignore_index=True)
        else:
            main_df = acct_anoms_df

    # Filter out rows where both "Pct Change" and "Abs Change (VND)" are 0 or empty
    if not main_df.empty:
        def is_zero_or_empty(val):
            if pd.isna(val) or val == "" or val == 0 or val == "0":
                return True
            try:
                return float(val) == 0
            except (ValueError, TypeError):
                return False

        # Keep only rows where at least one of the change fields is non-zero
        mask = ~(main_df["Pct Change"].apply(is_zero_or_empty) & main_df["Abs Change (VND)"].apply(is_zero_or_empty))
        main_df = main_df[mask].reset_index(drop=True)

    return main_df

# -----------------------------------------------------------------------------
# Accounting-specific analysis functions
# -----------------------------------------------------------------------------

def check_gross_margin(pl_pivot: pd.DataFrame, entity_col: str = "Entity") -> list[dict]:
    """
    Needs rows for 511* (Revenue) and 632* (COGS) per entity. pl_pivot index should let us filter by account prefix.
    Returns anomaly records where gross margin % moves by >= threshold m/m.
    """
    out = []
    months = _months(pl_pivot)
    if not months:
        return out

    # group by entity
    for entity, dfE in pl_pivot.groupby(entity_col):
        # total 511 and 632 for the entity
        rev = dfE[dfE["Account"].apply(_is_511)][months].sum(numeric_only=True)
        cogs = dfE[dfE["Account"].apply(_is_632)][months].sum(numeric_only=True)

        # margin %
        with np.errstate(divide='ignore', invalid='ignore'):
            margin = (rev - cogs) / rev.replace({0: np.nan})

        # m/m change in percentage points
        for i in range(1, len(months)):
            m, pm = months[i], months[i-1]
            if pd.notna(margin[m]) and pd.notna(margin[pm]):
                delta = float(margin[m] - margin[pm])  # absolute change in fraction
                if abs(delta) >= ACCT_THRESH["gross_margin_pct_delta"]:
                    out.append(_anom_record(
                        "GrossMargin Δ≥1pp",
                        entity, "Gross Margin", m,
                        value=float(margin[m]),
                        detail={
                            "PrevMonth": pm,
                            "PrevValue": float(margin[pm]),
                            "DeltaPctPoints": round(delta*100, 2)
                        }
                    ))
    return out

def check_depreciation_variance(pl_pivot: pd.DataFrame, bs_pivot: pd.DataFrame, entity_col: str = "Entity") -> list[dict]:
    """
    Track % changes for 217*, 632* depreciation, and 214 (SCC).
    - P&L: 632* lines that are depreciation (we treat all 632* for simplicity; refine if you have sub-accounts)
    - BS: 217* and 214* balances delta month-over-month
    """
    out = []
    months = _months(pl_pivot)
    if not months:
        return out

    for entity, dfE in pl_pivot.groupby(entity_col):
        # P&L depreciation proxy: total 632*
        pl_dep = dfE[dfE["Account"].apply(_is_632)][months].sum(numeric_only=True)
        pl_pct = _pct_change(pl_dep)

        for i, m in enumerate(months):
            if i == 0:
                continue
            if pd.notna(pl_pct[m]) and abs(pl_pct[m]) >= ACCT_THRESH["depr_pct_delta"]:
                out.append(_anom_record(
                    "P&L Depreciation Δ%",
                    entity, "632* (Depreciation proxy)", m,
                    value=float(pl_dep[m]),
                    detail={"PctChange": float(pl_pct[m])}
                ))

        # BS 217*
        if entity in bs_pivot[entity_col].values:
            dfB = bs_pivot[bs_pivot[entity_col] == entity]
        else:
            dfB = bs_pivot  # fallback if bs not entity-split

        bal_217 = dfB[dfB["Account"].apply(_is_217)][months].sum(numeric_only=True)
        bal_217_pct = _pct_change(bal_217)
        for i, m in enumerate(months):
            if i == 0:
                continue
            if pd.notna(bal_217_pct[m]) and abs(bal_217_pct[m]) >= ACCT_THRESH["depr_pct_delta"]:
                out.append(_anom_record(
                    "BS 217* Δ%",
                    entity, "217* (Acc. Depreciation)", m,
                    value=float(bal_217[m]),
                    detail={"PctChange": float(bal_217_pct[m])}
                ))

        # BS 214* (esp. SCC)
        bal_214 = dfB[dfB["Account"].apply(_is_214)][months].sum(numeric_only=True)
        if not bal_214.empty:
            bal_214_pct = _pct_change(bal_214)
            for i, m in enumerate(months):
                if i == 0:
                    continue
                if pd.notna(bal_214_pct[m]) and abs(bal_214_pct[m]) >= ACCT_THRESH["depr_pct_delta"]:
                    out.append(_anom_record(
                        "BS 214* Δ%",
                        entity, "214* (Acc. Depreciation SCC)", m,
                        value=float(bal_214[m]),
                        detail={"PctChange": float(bal_214_pct[m])}
                    ))
    return out

def check_cogs_vs_revenue_ratio(pl_pivot: pd.DataFrame, entity_col: str = "Entity") -> list[dict]:
    """
    COGS/Revenue ratio drift > threshold vs historical mean.
    """
    out = []
    months = _months(pl_pivot)
    if not months:
        return out

    for entity, dfE in pl_pivot.groupby(entity_col):
        rev = dfE[dfE["Account"].apply(_is_511)][months].sum(numeric_only=True)
        cogs = dfE[dfE["Account"].apply(_is_632)][months].sum(numeric_only=True)
        ratio = (cogs / rev.replace({0: np.nan})).astype(float)

        hist_mean = float(ratio.mean(skipna=True)) if ratio.notna().any() else np.nan
        if pd.isna(hist_mean):
            continue

        for m in months:
            if pd.notna(ratio[m]) and abs(ratio[m] - hist_mean) >= ACCT_THRESH["cogs_ratio_delta"]:
                out.append(_anom_record(
                    "COGS/Revenue ratio drift",
                    entity, "632* vs 511*", m,
                    value=float(ratio[m]),
                    detail={"HistMean": hist_mean, "Delta": float(ratio[m] - hist_mean)}
                ))
    return out

def check_sga_as_pct_of_revenue(pl_pivot: pd.DataFrame, entity_col: str = "Entity") -> list[dict]:
    """
    SG&A as % of revenue exceeding historical mean by > threshold.
    """
    out = []
    months = _months(pl_pivot)
    if not months:
        return out

    for entity, dfE in pl_pivot.groupby(entity_col):
        rev = dfE[dfE["Account"].apply(_is_511)][months].sum(numeric_only=True)
        sga = dfE[dfE["Account"].apply(lambda a: _is_641(a) or _is_642(a))][months].sum(numeric_only=True)

        mean_pct, std_pct = _series_hist_pct_of_rev(sga, rev)
        if pd.isna(mean_pct):
            continue

        # Flag if current % > mean + 10% points of mean (relative), or simply > mean*(1+delta)
        for m in months:
            if rev.get(m, np.nan) == 0 or pd.isna(rev.get(m, np.nan)):
                continue
            pct = float(sga.get(m, np.nan) / rev.get(m, np.nan))
            if pd.notna(pct) and pct > (mean_pct * (1 + ACCT_THRESH["sga_pct_of_rev_delta"])):
                out.append(_anom_record(
                    "SG&A % of Revenue spike",
                    entity, "641*/642*", m,
                    value=float(sga.get(m, np.nan)),
                    detail={"PctOfRevenue": pct, "HistMean": mean_pct}
                ))
    return out

def check_financial_items_swings(pl_pivot: pd.DataFrame, entity_col: str = "Entity") -> list[dict]:
    """
    Financial expenses (635*) and income (515*) percentage swings > threshold.
    """
    out = []
    months = _months(pl_pivot)
    if not months:
        return out

    for entity, dfE in pl_pivot.groupby(entity_col):
        for prefix, label, pred in [
            ("635*", "Financial expenses (635*)", _is_635),
            ("515*", "Financial income (515*)", _is_515),
        ]:
            series = dfE[dfE["Account"].apply(pred)][months].sum(numeric_only=True)
            pct = _pct_change(series)
            for i, m in enumerate(months):
                if i == 0:
                    continue
                if pd.notna(pct[m]) and abs(pct[m]) >= ACCT_THRESH["fin_swing_pct"]:
                    out.append(_anom_record(
                        f"{label} swing",
                        entity, prefix, m,
                        value=float(series[m]),
                        detail={"PctChange": float(pct[m])}
                    ))
    return out

def check_bs_pl_dep_consistency(pl_pivot: pd.DataFrame, bs_pivot: pd.DataFrame, entity_col: str = "Entity") -> list[dict]:
    """
    Compare BS accumulated depreciation (217* + 214*) Δ vs P&L 632* depreciation expense.
    Flag when mismatch > threshold %.
    """
    out = []
    months = _months(pl_pivot)
    if not months:
        return out

    for entity, dfE in pl_pivot.groupby(entity_col):
        # P&L dep expense proxy
        pl_dep = dfE[dfE["Account"].apply(_is_632)][months].sum(numeric_only=True)

        if entity in bs_pivot[entity_col].values:
            dfB = bs_pivot[bs_pivot[entity_col] == entity]
        else:
            dfB = bs_pivot

        bs_acc_dep = dfB[dfB["Account"].apply(lambda a: _is_217(a) or _is_214(a))][months].sum(numeric_only=True)

        # month-over-month deltas
        pl_delta = pl_dep.diff()
        bs_delta = bs_acc_dep.diff()

        for m in months:
            if m == months[0]:
                continue
            x = float(pl_delta.get(m, np.nan)) if pd.notna(pl_delta.get(m, np.nan)) else None
            y = float(bs_delta.get(m, np.nan)) if pd.notna(bs_delta.get(m, np.nan)) else None
            if x is None or y is None or y == 0:
                continue
            diff_pct = abs(x - y) / (abs(y) if y != 0 else np.nan)
            if pd.notna(diff_pct) and diff_pct > ACCT_THRESH["bs_pl_dep_diff_pct"]:
                out.append(_anom_record(
                    "BS↔PL Depreciation mismatch",
                    entity, "217*+214* vs 632*", m,
                    value={"PL_Dep_Delta": x, "BS_AccDep_Delta": y},
                    detail={"DiffPct": float(diff_pct)}
                ))
    return out