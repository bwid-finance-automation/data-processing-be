# app/data_utils.py
"""Data processing utilities and helper functions."""

from __future__ import annotations

import io
import re
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # Avoid noisy pandas dtype warnings in logs

# =====================================================================================
# CENTRALIZED CONFIGURATION - All tunable numbers in one place
# =====================================================================================

# === Revenue Analysis Thresholds ===
REVENUE_ANALYSIS = {
    # Revenue account analysis
    "revenue_change_threshold_vnd": 1_000_000,          # > 1M VND changes are significant
    "revenue_entity_threshold_vnd": 100_000,            # > 100K VND entity changes are tracked
    "revenue_account_prefixes": ["511"],                # Revenue account codes

    # COGS analysis (632* accounts)
    "cogs_change_threshold_vnd": 500_000,               # > 500K VND COGS changes are significant
    "cogs_entity_threshold_vnd": 50_000,                # > 50K VND COGS entity changes are tracked
    "cogs_account_prefixes": ["632"],                   # COGS account codes

    # SG&A analysis (641* and 642* accounts)
    "sga_change_threshold_vnd": 500_000,                # > 500K VND SG&A changes are significant
    "sga_entity_threshold_vnd": 50_000,                 # > 50K VND SG&A entity changes are tracked
    "sga_641_account_prefixes": ["641"],                # SG&A 641 account codes
    "sga_642_account_prefixes": ["642"],                # SG&A 642 account codes

    # Risk assessment thresholds
    "gross_margin_change_threshold_pct": 1.0,           # > 1% gross margin change triggers risk
    "high_gross_margin_risk_threshold_pct": -2.0,       # < -2% gross margin change = HIGH risk
    "sga_ratio_change_threshold_pct": 2.0,              # > 2% SG&A ratio change triggers risk
    "high_sga_ratio_threshold_pct": 3.0,                # > 3% SG&A ratio change = HIGH risk
    "revenue_pct_change_risk_threshold": 5.0,           # > 5% revenue change = risk
    "high_revenue_pct_change_threshold": 20.0,          # > 20% revenue change = HIGH risk

    # Analysis parameters
    "months_to_analyze": 8,                             # Number of months to analyze
    "top_entity_impacts": 5,                            # Top N entity impacts to show
    "lookback_periods": 10,                             # Periods to look back for account detection
}

# === Excel Processing Constants ===
EXCEL_PROCESSING = {
    "max_sheet_name_length": 31,                        # Excel sheet name character limit
    "header_scan_rows": 40,                             # Rows to scan for header detection
    "data_row_offset": 2,                               # Offset from header to data rows
    "account_code_min_digits": 4,                       # Minimum digits in account codes
    "progress_milestones": {
        "start": 10,
        "load": 15,
        "config": 20,
        "ai_thresholds": 25,
        "analysis_start": 30,
        "analysis_complete": 85,
        "storage": 90,
        "finalize": 95,
        "complete": 100
    }
}

# === File Processing Constants ===
FILE_PROCESSING = {
    "bytes_per_kb": 1024,                               # Bytes to KB conversion
    "progress_file_range": 50,                          # Progress range per file (30% to 80%)
    "progress_base_start": 30,                          # Base progress start percentage
    "file_progress_offset": {
        "extract": 2,
        "analysis": 5,
        "complete": 5
    }
}

# === Accounting Thresholds (Legacy - keep for backward compatibility) ===
ACCT_THRESH = {
    # the % change in gross margin compared to another period
    # (last month, last year, or budget).
    "gross_margin_pct_delta": 0.01,    # 1% point change m/m
    # the percentage change in depreciation compared to a prior period
    # (last month, last year, or budget).
    "depr_pct_delta": 0.10,            # 10% change m/m for 217*, 632*, 214
    # how much the cost-to-revenue ratio has increased or decreased compared
    # to a previous period.
    "cogs_ratio_delta": 0.02,          # 2% points drift vs hist
    # how much the % of revenue spent on overhead (SGA) has increased or decreased
    # compared to history.
    "sga_pct_of_rev_delta": 0.10,      # +10% vs hist % of revenue
    # the % change (or volatility) in financial income/expenses compared to a prior period,
    # used to flag unusual financial fluctuations.
    "fin_swing_pct": 0.50,             # >50% swings
    # the % difference between depreciation recorded in the Balance Sheet vs.
    # the depreciation shown in the P&L.
    "bs_pl_dep_diff_pct": 0.05,        # 5% mismatch between 214/217 Δ and 632 dep expense
}

# === Account prefix helpers (VN CoA style) ===
def _is_511(name: str) -> bool:
    s = str(name).replace(" ", "").lower()
    return s.startswith("511")

def _is_632(name: str) -> bool:
    s = str(name).replace(" ", "").lower()
    return s.startswith("632")

def _is_641(name: str) -> bool:
    s = str(name).replace(" ", "").lower()
    return s.startswith("641")

def _is_642(name: str) -> bool:
    s = str(name).replace(" ", "").lower()
    return s.startswith("642")

def _is_635(name: str) -> bool:
    s = str(name).replace(" ", "").lower()
    return s.startswith("635")

def _is_515(name: str) -> bool:
    s = str(name).replace(" ", "").lower()
    return s.startswith("515")

def _is_217(name: str) -> bool:
    s = str(name).replace(" ", "").lower()
    return s.startswith("217")

def _is_214(name: str) -> bool:
    s = str(name).replace(" ", "").lower()
    return s.startswith("214")

# === Safe % change ===
def _pct_change(a: pd.Series) -> pd.Series:
    b = pd.to_numeric(a, errors="coerce")
    return b.pct_change().replace([np.inf, -np.inf], np.nan)

def _series_hist_pct_of_rev(series: pd.Series, rev: pd.Series) -> tuple[float, float]:
    """Return (hist_mean, hist_std) for (series / rev)."""
    x = pd.to_numeric(series, errors="coerce")
    r = pd.to_numeric(rev, errors="coerce")
    ratio = x / r.replace({0: np.nan})
    return ratio.mean(skipna=True), ratio.std(skipna=True)

def _months(df: pd.DataFrame) -> list[str]:
    # Assumes your pipeline already normalized to monthly columns like 'Jan 2025' ... 'Dec 2025'
    year_range = DEFAULT_CONFIG.get("year_range", ["2024", "2025", "2026", "2027", "2028", "2029", "2030"])
    return [c for c in df.columns if isinstance(c, str) and c.strip().lower().endswith(tuple(year_range))]

# === Output record helper ===
def _anom_record(rule: str, entity: str, account: str, month: str, value, detail: dict) -> dict:
    rec = {
        "Rule": rule,
        "Entity": entity,
        "Account": account,
        "Month": month,
        "Value": value,
    }
    rec.update(detail or {})
    return rec

# -----------------------------------------------------------------------------
# Defaults & constants (shared between Python and AI modes)
# -----------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    # ========== Core Analysis Thresholds ==========
    # if an error or difference is bigger than this, it matters; if smaller, we can ignore it.
    "materiality_vnd": 1_000_000_000,      # absolute VND change threshold
    # the % cut-off used to decide if recurring revenue or costs are large enough (vs. total) to count as meaningful.
    "recurring_pct_threshold": 0.05,       # 5% for recurring P/L accounts
    # the % of revenue spent on operating expenses beyond which costs are considered too high.
    "revenue_opex_pct_threshold": 0.10,    # 10% for revenue/opex accounts
    # the % cut-off used to decide if a balance sheet account change is large enough (vs. total) to count as meaningful.
    "bs_pct_threshold": 0.05,              # 5% for balance sheet
    # list of account code prefixes that indicate recurring revenue or costs.
    "recurring_code_prefixes": ["6321", "635", "515"],
    # the minimum number of periods (months) with data required to perform trend analysis.
    "min_trend_periods": 3,
    # the % drop in gross margin (absolute points) that triggers an anomaly.
    "gm_drop_threshold_pct": 0.01,         # 1% absolute drop triggers (e.g., 0.01 = 1pp)
    # list of account code prefixes for which depreciation should be analyzed using only % change rules.
    "dep_pct_only_prefixes": ["217", "632"],  # treat these as %-only rules
    # list of keywords to identify a "customer" column in the P&L data.
    "customer_column_hints": ["customer", "khách", "khach", "client", "buyer", "entity", "company", "subsidiary", "parent company", "bwid", "vc1", "vc2", "vc3", "logistics"],  # for 511* drilldown

    # ========== Revenue Analysis Configuration ==========
    # Include all revenue analysis thresholds
    **REVENUE_ANALYSIS,

    # ========== Excel Processing Configuration ==========
    # Include all excel processing constants
    **EXCEL_PROCESSING,

    # ========== File Processing Configuration ==========
    # Include all file processing constants
    **FILE_PROCESSING,

    # ========== Legacy Accounting Thresholds ==========
    # Include legacy thresholds for backward compatibility
    **ACCT_THRESH,

    # ========== AI Analysis Configuration ==========
    # AI-specific configuration
    "use_llm_analysis": False,              # Whether to use AI analysis
    "llm_model": "gpt-4o",                  # LLM model for AI analysis

    # ========== Data Processing Constants ==========
    "year_range": ["2024", "2025", "2026", "2027", "2028", "2029", "2030"],  # Valid year suffixes for month detection
    "trend_window_max": 5,                  # Maximum trend window periods
    "zero_division_replacement": 0.0,       # Value to use when dividing by zero
    "numeric_fill_value": 0.0,              # Fill value for coerced numeric columns
    "percentage_multiplier": 100.0,         # Convert decimal to percentage
}

MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

# Patterns to recognize header row content that carries a period like "As of Feb-2024" etc.
BS_PAT = re.compile(r'^\s*as\s*of\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\.\-\s]*(\d{2,4})\s*$', re.I)
PL_PAT = re.compile(r'^\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\.\-\s]*(\d{2,4})\s*$', re.I)

# -----------------------------------------------------------------------------
# Helpers: find optional "customer" column
# -----------------------------------------------------------------------------
def find_customer_column(df: pd.DataFrame, CONFIG: dict) -> Optional[str]:
    """Try to find a customer dimension column in a cleaned PL frame.
    Heuristic: any column whose name contains one of the configured hints.
    """
    if df is None or df.empty:
        return None
    cols = [str(c).strip() for c in df.columns]
    hints = [h.lower() for h in CONFIG.get("customer_column_hints", [])]
    for c in cols:
        lc = c.lower()
        if any(h in lc for h in hints):
            return c
    return None

# -----------------------------------------------------------------------------
# Helpers: period parsing / ordering
# -----------------------------------------------------------------------------
def normalize_period_label(label: object) -> str:
    """Turn many month-year formats into 'Mon YYYY'."""
    if label is None:
        return ""
    s = str(label).strip()
    if s == "":
        return ""
    try:
        s_clean = re.sub(r'^\s*(as\s*of|tinh\s*den|tính\s*đến|den\s*ngay|đến\s*ngày)\s*', '', s, flags=re.I)

        m = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[^\w]?[\s\-\.]*([12]\d{3}|\d{2})\b',
                      s_clean, flags=re.I)
        if m:
            mon, yr = m.group(1), m.group(2)
            yr = int(yr)
            yr = yr + 2000 if yr < 100 else yr
            return f"{mon.title()} {yr}"

        m = re.search(r'\b(1[0-2]|0?[1-9])[./\-](\d{4})\b', s_clean)
        if m:
            mon = int(m.group(1)); yr = int(m.group(2))
            return f"{MONTHS[mon-1].title()} {yr}"

        m = re.search(r'\b(\d{4})[./\-](1[0-2]|0?[1-9])\b', s_clean)
        if m:
            yr = int(m.group(1)); mon = int(m.group(2))
            return f"{MONTHS[mon-1].title()} {yr}"

        m_year = re.search(r'(20\d{2}|19\d{2})', s_clean)
        m_mon  = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', s_clean, flags=re.I)
        if m_year and m_mon:
            yr  = int(m_year.group(1))
            mon = m_mon.group(0)
            return f"{mon.title()} {yr}"
    except Exception:
        pass
    return s

def month_key(label: object) -> tuple[int, int]:
    """Return (year, month) for sorting. Unknown -> (9999, 99)."""
    n = normalize_period_label(label)
    m = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})', n, re.I)
    if not m:
        return (9999, 99)
    y = int(m.group(2))
    mi = MONTHS.index(m.group(1).lower()) + 1
    return (y, mi)

# -----------------------------------------------------------------------------
# Excel reading / header detection / cleaning (IN-MEMORY)
# -----------------------------------------------------------------------------

def detect_header_row(xl_bytes: bytes, sheet: str) -> int:
    """Heuristically find the header row by scanning first ~40 rows for 'Financial row'."""
    try:
        probe = pd.read_excel(io.BytesIO(xl_bytes), sheet_name=sheet, header=None, nrows=DEFAULT_CONFIG.get("header_scan_rows", 40))
        for i in range(len(probe)):
            row_values = probe.iloc[i].astype(str).str.strip().str.lower()
            if any("financial row" in v for v in row_values):
                return i
    except Exception:
        pass
    return DEFAULT_CONFIG.get("zero_division_replacement", 0)  # fallback to first row

def normalize_financial_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the main descriptor column is exactly 'Financial row'."""
    for c in df.columns:
        if str(c).strip().lower() == "financial row":
            return df.rename(columns={c: "Financial row"})
    # Otherwise assume first column is the descriptor
    return df.rename(columns={df.columns[0]: "Financial row"})

def promote_row8(df: pd.DataFrame, mode: str = None, sub: str = None) -> tuple[pd.DataFrame, list[str]]:
    """Use the first data row as headers when period info is there; normalize month columns."""
    if len(df) < 1:
        return df, []
    row0 = df.iloc[0]
    new_cols: list[str] = []
    for c in df.columns:
        v = str(row0.get(c, "")).strip()
        if BS_PAT.match(v) or PL_PAT.match(v):
            new_cols.append(normalize_period_label(v))
        else:
            new_cols.append(str(c))
    df = df.copy()
    df.columns = new_cols
    df = df.iloc[1:].reset_index(drop=True)

    month_cols: list[str] = []
    for c in df.columns:
        normalized = normalize_period_label(c)
        if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}$', normalized, re.I):
            month_cols.append(c)
    month_cols = sorted(month_cols, key=month_key)
    return df, month_cols

def fill_down_assign(df: pd.DataFrame) -> pd.DataFrame:
    """Extract account code / name from descriptor and forward-fill."""
    ser = df["Financial row"].astype(str)

    min_digits = DEFAULT_CONFIG.get("account_code_min_digits", 4)
    code_extract = ser.str.extract(rf'(\d{{{min_digits},}})', expand=False)
    name_extract = ser.str.replace(rf'.*?(\d{{{min_digits},}})\s*[-:]*\s*', '', regex=True).str.strip()

    row_has_code     = code_extract.notna()
    is_total_word    = ser.str.strip().str.lower().str.startswith(("total","subtotal","cộng","tong","tổng"))
    is_total_with_code = is_total_word & row_has_code
    is_section       = ser.str.match(r'^\s*([IVX]+\.|[A-Z]\.)\s')
    is_empty         = ser.str.strip().eq("")

    df["Account Code"] = code_extract.ffill()
    df["Account Name"] = name_extract.where(row_has_code).ffill()
    df["RowHadOwnCode"] = row_has_code
    df["IsTotal"] = is_total_with_code

    keep_mask = ~(is_section | is_empty)
    df = df[keep_mask & df["Account Code"].notna()].copy()
    return df

def coerce_numeric(df: pd.DataFrame, month_cols: list[str]) -> pd.DataFrame:
    """Coerce month columns to numeric."""
    out = df.copy()
    for c in month_cols:
        if c in out.columns:
            series = out[c].astype(str)
            series = (
                series
                .str.replace("\u00a0","", regex=False)    # nbsp
                .str.replace(",","", regex=False)
                .str.replace(r"\((.*)\)", r"-\1", regex=True)  # (100) -> -100
                .str.replace(r"[^0-9\.\-]", "", regex=True)
            )
            out[c] = pd.to_numeric(series, errors="coerce").fillna(DEFAULT_CONFIG.get("numeric_fill_value", 0.0))
    return out

def aggregate_totals(df: pd.DataFrame, month_cols: list[str]) -> pd.DataFrame:
    """Aggregate to a single row per account code; prefer explicit 'total' lines when present."""
    if df.empty:
        return pd.DataFrame(columns=["Account Code","Account Name"] + month_cols)

    nm_src   = df[df["RowHadOwnCode"]] if "RowHadOwnCode" in df.columns else df
    name_map = (
        nm_src.dropna(subset=["Account Code"])[["Account Code","Account Name"]]
             .drop_duplicates("Account Code")
             .set_index("Account Code")["Account Name"]
    )

    totals_df = df[df.get("IsTotal", False)]
    codes_with_total = set(totals_df["Account Code"].dropna().astype(str).unique())

    cols = ["Account Code"] + [c for c in month_cols if c in df.columns]
    parts = []
    if not totals_df.empty:
        parts.append(totals_df[cols].groupby("Account Code", as_index=False).sum())

    no_total_df = df[~df["Account Code"].astype(str).isin(codes_with_total)]
    if not no_total_df.empty:
        parts.append(no_total_df[cols].groupby("Account Code", as_index=False).sum())

    agg = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    agg["Account Name"] = agg["Account Code"].map(name_map).fillna("")
    return agg[["Account Code","Account Name"] + [c for c in month_cols if c in agg.columns]]

# -----------------------------------------------------------------------------
# MoM + trend signals
# -----------------------------------------------------------------------------

def compute_mom_with_trends(df: pd.DataFrame, month_cols: list[str], CONFIG: dict) -> pd.DataFrame:
    """Compute MoM deltas and a simple rolling average signal."""
    if len(month_cols) < 2:
        return pd.DataFrame(columns=[
            "Account Code","Account Name","Prior","Current","Delta","Pct Change","Period",
            "Trend_3M_Avg","Trend_Deviation"
        ])

    out: list[pd.DataFrame] = []
    for i in range(1, len(month_cols)):
        cur, prev = month_cols[i], month_cols[i-1]
        if cur not in df.columns or prev not in df.columns:
            continue

        tmp = df[["Account Code","Account Name", prev, cur]].copy()
        tmp = tmp.rename(columns={prev: "Prior", cur: "Current"})
        tmp["Delta"] = tmp["Current"] - tmp["Prior"]
        tmp["Pct Change"] = np.where(tmp["Prior"] == CONFIG.get("zero_division_replacement", 0.0), np.nan, tmp["Delta"] / tmp["Prior"])
        tmp["Period"] = normalize_period_label(cur)

        # simple trend window using preceding up-to-5 periods (require min_trend_periods)
        if i >= CONFIG["min_trend_periods"]:
            start_idx = max(0, i - CONFIG.get("trend_window_max", 5))
            trend_cols = month_cols[start_idx:i]
            if len(trend_cols) >= CONFIG["min_trend_periods"]:
                trend_data = df[trend_cols]
                tmp["Trend_3M_Avg"] = trend_data.mean(axis=1)
                tmp["Trend_Deviation"] = tmp["Current"] - tmp["Trend_3M_Avg"]
            else:
                tmp["Trend_3M_Avg"] = np.nan
                tmp["Trend_Deviation"] = np.nan
        else:
            tmp["Trend_3M_Avg"] = np.nan
            tmp["Trend_Deviation"] = np.nan

        out.append(tmp)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[
        "Account Code","Account Name","Prior","Current","Delta","Pct Change","Period",
        "Trend_3M_Avg","Trend_Deviation"
    ])

# -----------------------------------------------------------------------------
# Rules & anomaly builders
# -----------------------------------------------------------------------------

def classify_pl_account(code: object, CONFIG: dict) -> str:
    """Return 'Recurring' if code starts with any configured prefix; else 'Revenue/OPEX'."""
    code_str = str(code)
    return "Recurring" if any(code_str.startswith(p) for p in CONFIG["recurring_code_prefixes"]) else "Revenue/OPEX"

def get_threshold_cause(statement: str, code: object, CONFIG: dict) -> str:
    """Human-friendly cause suggestion based on statement type and code classification."""
    if statement == "BS":
        return "Balance changed materially — check reclass/missing offset."
    return ("Recurring moved — check accruals/timing."
            if classify_pl_account(code, CONFIG) == "Recurring"
            else "Revenue/OPEX moved — check billing/cut-off.")

def match_codes(series: pd.Series, pattern_str: str | float | int | None) -> pd.Series:
    """Return boolean mask for 'code patterns' like '111*,112*|515'."""
    if pd.isna(pattern_str) or pattern_str == "":
        return pd.Series(False, index=series.index)
    patterns = [p.strip() for p in str(pattern_str).split("|") if p.strip()]
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            mask |= series.astype(str).str.startswith(prefix)
        else:
            mask |= (series.astype(str) == pattern)
    return mask