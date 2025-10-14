# app/accounting_rules.py
"""Accounting-specific rule analysis functions."""

from __future__ import annotations

import re
import pandas as pd
import numpy as np

from ..data.data_utils import _months
from .anomaly_detection import (
    check_gross_margin, check_depreciation_variance, check_cogs_vs_revenue_ratio,
    check_sga_as_pct_of_revenue, check_financial_items_swings, check_bs_pl_dep_consistency
)

def _month_cols(df: pd.DataFrame) -> list[str]:
    patt = re.compile(r"^[A-Za-z]{3}\s+\d{4}$")
    return [c for c in df.columns if isinstance(c, str) and patt.match(c)]

def _ensure_account_and_entity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Entity" not in out.columns:
        if "Subsidiary" in out.columns:
            out.rename(columns={"Subsidiary": "Entity"}, inplace=True)
        else:
            out["Entity"] = "ALL"
    if "Account" not in out.columns:
        # Fix: ensure "Financial row" matches the standardized casing
        for cand in ["Account (Line): Mã số - Code", "Account (Line)", "Account Code", "Financial row"]:
            if cand in out.columns:
                out["Account"] = out[cand].astype(str)
                break
        if "Account" not in out.columns:
            out["Account"] = out.iloc[:,0].astype(str)
    return out[["Entity","Account"] + _month_cols(out)].copy()

def _map_accounting_records_to_summary(acct_df: pd.DataFrame, subsidiary: str) -> pd.DataFrame:
    if acct_df.empty:
        return acct_df

    def _get_pct(row):
        for k in ["PctChange","DeltaPct","DeltaPctPoints","DiffPct"]:
            if k in row and pd.notna(row[k]):
                return float(row[k])
        return np.nan

    def _get_abs(row):
        v = row.get("Value", np.nan)
        if isinstance(v, dict):
            return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan

    def _guess_cause(rule):
        rule = str(rule).lower()
        if "grossmargin" in rule:
            return "Pricing/COGS timing or one-off cost"
        if "depreciation" in rule:
            return "New asset, disposal, or misclassification"
        if "cogs/revenue" in rule:
            return "Inefficiency or pricing mismatch"
        if "sg&a" in rule:
            return "Overhead spike vs revenue"
        if "financial" in rule:
            return "Debt/investment swing or FX"
        if "dep mismatch" in rule:
            return "BS–PL inconsistency in depreciation"
        return ""

    rows = []
    for _, r in acct_df.iterrows():
        rows.append({
            "Subsidiary": subsidiary,
            "Account": r.get("Account",""),
            "Period": r.get("Month",""),
            "Pct Change": _get_pct(r),
            "Abs Change (VND)": _get_abs(r),
            "Trigger(s)": r.get("Rule",""),
            "Suggested likely cause": _guess_cause(r.get("Rule","")),
            "Status": "",
            "Notes": ""
        })
    return pd.DataFrame(rows)

def run_accounting_rules_on_frames(pl_df: pd.DataFrame, bs_df: pd.DataFrame, *, subsidiary: str) -> pd.DataFrame:
    if pl_df is None or pl_df.empty:
        return pd.DataFrame()

    pl_norm = _ensure_account_and_entity(pl_df)
    months = _month_cols(pl_norm)
    if not months:
        return pd.DataFrame()

    if bs_df is None or bs_df.empty:
        bs_norm = pd.DataFrame(columns=["Entity","Account"] + months)
    else:
        bs_norm = _ensure_account_and_entity(bs_df)

    acct_records = []
    acct_records += check_gross_margin(pl_norm)
    acct_records += check_depreciation_variance(pl_norm, bs_norm)
    acct_records += check_cogs_vs_revenue_ratio(pl_norm)
    acct_records += check_sga_as_pct_of_revenue(pl_norm)
    acct_records += check_financial_items_swings(pl_norm)
    acct_records += check_bs_pl_dep_consistency(pl_norm, bs_norm)

    acct_df = pd.DataFrame(acct_records)
    return _map_accounting_records_to_summary(acct_df, subsidiary)