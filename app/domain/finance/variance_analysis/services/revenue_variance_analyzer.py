"""
Revenue Variance Analyzer - MoM variance analysis with UFL linkage.

This module implements comprehensive revenue variance analysis for BW Industrial,
comparing monthly revenue data with Unit for Lease (UFL) activity to explain
revenue movements and flag unusual items.

Output: 13-sheet Excel with Flags, Executive Summary, and 11 monthly comparison sheets.
"""

import io
import re
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

logger = logging.getLogger(__name__)

# Revenue type classification based on account codes
REVENUE_TYPE_MAP = {
    "511710001": "Leasing",
    "511600001": "Service/Mgmt Fee",
    "511600002": "Service/Mgmt Fee",
    "511600003": "Service/Mgmt Fee",
    "511600004": "Service/Mgmt Fee",
    "511600005": "Service/Mgmt Fee",
    "511600006": "Service/Mgmt Fee",
    "511600007": "Service/Mgmt Fee",
    "511800001": "Utilities",
    "511800002": "Others",
}

# Variance threshold in VND (100 million = 0.1B)
VARIANCE_THRESHOLD = 100_000_000
# GLA threshold for UFL NO REVENUE flag (sqm)
GLA_THRESHOLD = 5000

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class RevenueRecord:
    """Single revenue record from NetSuite export."""
    subsidiary: str = ""
    tenant: str = ""
    tenant_code: str = ""
    account_code: str = ""
    revenue_type: str = ""
    monthly_amounts: Dict[int, float] = field(default_factory=dict)  # month (1-12) -> amount
    ltm: float = 0.0


@dataclass
class UFLRecord:
    """Unit for Lease record from NetSuite export."""
    tenant: str = ""
    tenant_code: str = ""
    subsidiary: str = ""
    status: str = ""  # Handed Over, Terminated, Voided, Open
    start_date: Optional[datetime] = None
    termination_date: Optional[datetime] = None
    gla: float = 0.0
    unit_name: str = ""


@dataclass
class VarianceFlag:
    """A flagged item requiring investigation."""
    period: str  # e.g., "Dec vs Nov"
    flag_type: str  # INCREASE NO UFL, DECREASE NO UFL, NEGATIVE NO UFL, REVERSAL, UFL NO REVENUE
    subsidiary: str
    tenant: str
    prior_amount: float  # in billions
    current_amount: float  # in billions
    variance: float  # in billions
    detail: str


@dataclass
class MonthlyComparison:
    """Data for a single month-over-month comparison."""
    current_month: int
    prior_month: int
    current_month_name: str
    prior_month_name: str

    # Level 1: By Revenue Stream
    stream_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    net_change: float = 0.0
    net_change_pct: float = 0.0

    # Level 2: Leasing drill-down
    leasing_variance: float = 0.0
    leasing_increases: List[Dict[str, Any]] = field(default_factory=list)
    leasing_decreases: List[Dict[str, Any]] = field(default_factory=list)
    total_increases: float = 0.0
    total_decreases: float = 0.0

    # Flags for this period
    flags: List[VarianceFlag] = field(default_factory=list)


class RevenueVarianceAnalyzer:
    """
    Analyzes revenue variance with UFL linkage.
    """

    def __init__(self):
        self.ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        self.revenue_records: List[RevenueRecord] = []
        self.ufl_records: List[UFLRecord] = []
        self.year = 2025  # Default year, will be detected from data

    def analyze(self, revenue_bytes: bytes, ufl_bytes: bytes) -> bytes:
        """
        Perform full revenue variance analysis.

        Args:
            revenue_bytes: Raw bytes of RevenueBreakdown XML-Excel file
            ufl_bytes: Raw bytes of UnitForLeaseList XML-Excel file

        Returns:
            Excel file bytes with 13 sheets
        """
        logger.info("Starting revenue variance analysis...")

        # Step 1: Parse revenue data
        logger.info("Parsing revenue data...")
        self.revenue_records = self._parse_revenue_data(revenue_bytes)
        logger.info(f"Parsed {len(self.revenue_records)} revenue records")

        # Step 2: Parse UFL data
        logger.info("Parsing UFL data...")
        self.ufl_records = self._parse_ufl_data(ufl_bytes)
        logger.info(f"Parsed {len(self.ufl_records)} UFL records")

        # Step 3: Build monthly comparisons (Dec vs Nov, Nov vs Oct, ..., Feb vs Jan)
        logger.info("Building monthly comparisons...")
        comparisons = self._build_monthly_comparisons()

        # Step 4: Collect all flags
        all_flags = []
        for comp in comparisons:
            all_flags.extend(comp.flags)

        # Step 5: Check for UFL NO REVENUE flags
        ufl_no_revenue_flags = self._check_ufl_no_revenue()
        all_flags.extend(ufl_no_revenue_flags)

        # Sort flags by variance magnitude
        all_flags.sort(key=lambda f: abs(f.variance), reverse=True)

        logger.info(f"Total flags: {len(all_flags)}")

        # Step 6: Build monthly revenue summary for executive summary
        monthly_totals = self._calculate_monthly_totals()

        # Step 7: Generate Excel output
        logger.info("Generating Excel output...")
        xlsx_bytes = self._generate_excel(comparisons, all_flags, monthly_totals)

        logger.info("Revenue variance analysis complete")
        return xlsx_bytes

    def _parse_revenue_data(self, file_bytes: bytes) -> List[RevenueRecord]:
        """Parse RevenueBreakdown XML-Excel file."""
        records = []

        try:
            content = file_bytes.decode('utf-8')
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse revenue XML: {e}")
            raise ValueError(f"Invalid XML format: {e}")

        worksheet = root.find('.//ss:Worksheet', self.ns)
        if worksheet is None:
            raise ValueError("No worksheet found in revenue file")

        table = worksheet.find('ss:Table', self.ns)
        if table is None:
            raise ValueError("No table found in revenue worksheet")

        rows = table.findall('ss:Row', self.ns)

        # Detect year from header
        self._detect_year(rows)

        # Find the data start row (after headers)
        data_start_idx = self._find_data_start(rows)

        current_subsidiary = ""
        current_account_code = ""

        for row in rows[data_start_idx:]:
            cells = row.findall('ss:Cell', self.ns)
            cell_values = self._extract_cell_values_with_index(cells)

            if not cell_values:
                continue

            first_val = cell_values.get(0, "").strip()

            # Check if this is an account code row (e.g., "511710001 - Revenue...")
            if first_val and re.match(r'^\d{9}\s*-', first_val):
                match = re.match(r'^(\d{9})', first_val)
                if match:
                    current_account_code = match.group(1)
                continue

            # Check if this is a Total row - skip
            if first_val.startswith("Total -"):
                continue

            # Check if this is a data row (empty first cell, subsidiary in col 1)
            if first_val == "" and cell_values.get(1, ""):
                subsidiary = cell_values.get(1, "").strip()
                if subsidiary and subsidiary != "- No Entity -":
                    current_subsidiary = subsidiary

                tenant = cell_values.get(4, "").strip()  # Entity name contains code like "C00000259 SPX"

                # Skip intercompany entries (IE/ prefix)
                if "IE/" in tenant:
                    continue

                # Extract tenant code from tenant name using regex
                # Format: "C00000259 SPX" or "S00001552 VIETMY"
                tenant_code = ""
                code_match = re.search(r'([CS]\d+)', tenant)
                if code_match:
                    tenant_code = code_match.group(1)

                # Get revenue type from account code
                revenue_type = REVENUE_TYPE_MAP.get(current_account_code, "Others")

                # Parse monthly amounts (columns 7-18 are Jan-Dec)
                monthly_amounts = {}
                for month in range(1, 13):
                    col_idx = 6 + month  # Column 7 is Jan (month 1)
                    val = cell_values.get(col_idx, "")
                    try:
                        amount = float(val) if val else 0.0
                        monthly_amounts[month] = amount
                    except (ValueError, TypeError):
                        monthly_amounts[month] = 0.0

                # Get LTM (last column before LTM is column 19, LTM is 20)
                ltm = 0.0
                try:
                    ltm_val = cell_values.get(19, "") or cell_values.get(20, "")
                    ltm = float(ltm_val) if ltm_val else 0.0
                except (ValueError, TypeError):
                    pass

                # Skip if no activity
                if ltm == 0 and all(v == 0 for v in monthly_amounts.values()):
                    continue

                record = RevenueRecord(
                    subsidiary=current_subsidiary,
                    tenant=tenant,
                    tenant_code=tenant_code,
                    account_code=current_account_code,
                    revenue_type=revenue_type,
                    monthly_amounts=monthly_amounts,
                    ltm=ltm
                )
                records.append(record)

        return records

    def _parse_ufl_data(self, file_bytes: bytes) -> List[UFLRecord]:
        """Parse UnitForLeaseList XML-Excel file."""
        records = []

        try:
            content = file_bytes.decode('utf-8')
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse UFL XML: {e}")
            raise ValueError(f"Invalid XML format: {e}")

        worksheet = root.find('.//ss:Worksheet', self.ns)
        if worksheet is None:
            raise ValueError("No worksheet found in UFL file")

        table = worksheet.find('ss:Table', self.ns)
        if table is None:
            raise ValueError("No table found in UFL worksheet")

        rows = table.findall('ss:Row', self.ns)

        if len(rows) < 2:
            return records

        # First row is header
        header_row = rows[0]
        headers = self._extract_header_row(header_row)
        header_map = {h.lower(): i for i, h in enumerate(headers)}

        for row in rows[1:]:
            cells = row.findall('ss:Cell', self.ns)
            values = self._extract_cell_values_with_index(cells)

            # Get values by header position
            tenant = self._get_by_header(values, header_map, "tenant", "")
            status = self._get_by_header(values, header_map, "unit for lease status", "")
            subsidiary = self._get_by_header(values, header_map, "subsidiary", "")
            gla_str = self._get_by_header(values, header_map, "gla for lease", "0")
            start_date_str = self._get_by_header(values, header_map, "start leasing date", "")
            term_date_str = self._get_by_header(values, header_map, "termination date", "")
            unit_name = self._get_by_header(values, header_map, "unit", "")

            # Extract tenant code
            tenant_code = ""
            code_match = re.search(r'([CS]\d+)', tenant)
            if code_match:
                tenant_code = code_match.group(1)

            # Parse GLA
            try:
                gla = float(gla_str) if gla_str else 0.0
            except (ValueError, TypeError):
                gla = 0.0

            # Parse dates
            start_date = self._parse_date(start_date_str)
            term_date = self._parse_date(term_date_str)

            # Only keep relevant statuses
            if status in ["Handed Over", "Terminated"]:
                record = UFLRecord(
                    tenant=tenant,
                    tenant_code=tenant_code,
                    subsidiary=subsidiary,
                    status=status,
                    start_date=start_date,
                    termination_date=term_date,
                    gla=gla,
                    unit_name=unit_name
                )
                records.append(record)

        return records

    def _build_monthly_comparisons(self) -> List[MonthlyComparison]:
        """Build all 11 monthly comparisons (Dec vs Nov through Feb vs Jan)."""
        comparisons = []

        # Dec vs Nov, Nov vs Oct, ..., Feb vs Jan
        for current_month in range(12, 1, -1):  # 12 down to 2
            prior_month = current_month - 1

            comp = MonthlyComparison(
                current_month=current_month,
                prior_month=prior_month,
                current_month_name=MONTH_NAMES[current_month - 1],
                prior_month_name=MONTH_NAMES[prior_month - 1]
            )

            # Calculate Level 1: By Revenue Stream
            stream_data = self._calculate_stream_breakdown(current_month, prior_month)
            comp.stream_breakdown = stream_data["breakdown"]
            comp.net_change = stream_data["net_change"]
            comp.net_change_pct = stream_data["net_change_pct"]

            # Calculate Level 2: Leasing drill-down with UFL linkage
            leasing_data = self._calculate_leasing_drilldown(current_month, prior_month)
            comp.leasing_variance = leasing_data["variance"]
            comp.leasing_increases = leasing_data["increases"]
            comp.leasing_decreases = leasing_data["decreases"]
            comp.total_increases = leasing_data["total_increases"]
            comp.total_decreases = leasing_data["total_decreases"]
            comp.flags = leasing_data["flags"]

            comparisons.append(comp)

        return comparisons

    def _calculate_stream_breakdown(self, current_month: int, prior_month: int) -> Dict[str, Any]:
        """Calculate variance by revenue stream."""
        streams = ["Leasing", "Service/Mgmt Fee", "Utilities", "Others"]
        breakdown = []

        total_prior = 0.0
        total_current = 0.0

        for stream in streams:
            stream_records = [r for r in self.revenue_records if r.revenue_type == stream]

            prior_sum = sum(r.monthly_amounts.get(prior_month, 0) for r in stream_records)
            current_sum = sum(r.monthly_amounts.get(current_month, 0) for r in stream_records)
            variance = current_sum - prior_sum

            total_prior += prior_sum
            total_current += current_sum

            breakdown.append({
                "stream": stream,
                "prior": prior_sum,
                "current": current_sum,
                "variance": variance
            })

        net_change = total_current - total_prior
        net_change_pct = (net_change / abs(total_prior) * 100) if total_prior != 0 else 0

        # Calculate % of net for each stream
        for item in breakdown:
            if net_change != 0:
                item["pct_of_net"] = (item["variance"] / net_change * 100)
            else:
                item["pct_of_net"] = 0

        # Add total row
        breakdown.append({
            "stream": "TOTAL",
            "prior": total_prior,
            "current": total_current,
            "variance": net_change,
            "pct_of_net": 100
        })

        return {
            "breakdown": breakdown,
            "net_change": net_change,
            "net_change_pct": net_change_pct
        }

    def _calculate_leasing_drilldown(self, current_month: int, prior_month: int) -> Dict[str, Any]:
        """Calculate leasing drill-down with UFL linkage."""
        leasing_records = [r for r in self.revenue_records if r.revenue_type == "Leasing"]

        # Aggregate by subsidiary + tenant
        tenant_data = defaultdict(lambda: {"prior": 0, "current": 0, "subsidiary": "", "tenant": "", "tenant_code": ""})

        for r in leasing_records:
            key = (r.subsidiary, r.tenant_code or r.tenant)
            tenant_data[key]["prior"] += r.monthly_amounts.get(prior_month, 0)
            tenant_data[key]["current"] += r.monthly_amounts.get(current_month, 0)
            tenant_data[key]["subsidiary"] = r.subsidiary
            tenant_data[key]["tenant"] = r.tenant
            tenant_data[key]["tenant_code"] = r.tenant_code

        increases = []
        decreases = []
        flags = []
        period_name = f"{MONTH_NAMES[current_month - 1]} vs {MONTH_NAMES[prior_month - 1]}"

        for key, data in tenant_data.items():
            variance = data["current"] - data["prior"]

            if abs(variance) < VARIANCE_THRESHOLD:
                continue

            # Check for REVERSAL (prior negative, current positive or zero)
            is_reversal = data["prior"] < 0 and data["current"] >= 0

            # UFL matching
            ufl_action = "-"
            ufl_month = "-"
            ufl_gla = ""
            flag_type = None

            tenant_code = data["tenant_code"]

            if variance > 0:  # Increase
                if is_reversal:
                    flag_type = "REVERSAL"
                else:
                    # Look for UFL handover
                    ufl_match = self._find_ufl_handover(tenant_code, current_month)
                    if ufl_match:
                        ufl_action = "NEW LEASE"
                        ufl_month = MONTH_NAMES[ufl_match.start_date.month - 1] if ufl_match.start_date else "-"
                        ufl_gla = f"{ufl_match.gla:,.0f}" if ufl_match.gla else ""
                    else:
                        flag_type = "INCREASE NO UFL"

                increases.append({
                    "subsidiary": data["subsidiary"],
                    "tenant": data["tenant"],
                    "tenant_code": tenant_code,
                    "prior": data["prior"],
                    "current": data["current"],
                    "variance": variance,
                    "ufl_action": ufl_action,
                    "ufl_month": ufl_month,
                    "gla": ufl_gla,
                    "flag": flag_type or ""
                })

            else:  # Decrease
                # Check for negative revenue (straight-line reversal)
                if data["current"] < 0:
                    # Look for UFL termination
                    ufl_match = self._find_ufl_termination(tenant_code, current_month)
                    if ufl_match:
                        ufl_action = "TERMINATED"
                        ufl_month = MONTH_NAMES[ufl_match.termination_date.month - 1] if ufl_match.termination_date else "-"
                        ufl_gla = f"{ufl_match.gla:,.0f}" if ufl_match.gla else ""
                    else:
                        flag_type = "NEGATIVE NO UFL"
                else:
                    # Regular decrease
                    ufl_match = self._find_ufl_termination(tenant_code, current_month)
                    if ufl_match:
                        ufl_action = "TERMINATED"
                        ufl_month = MONTH_NAMES[ufl_match.termination_date.month - 1] if ufl_match.termination_date else "-"
                        ufl_gla = f"{ufl_match.gla:,.0f}" if ufl_match.gla else ""
                    else:
                        flag_type = "DECREASE NO UFL"

                decreases.append({
                    "subsidiary": data["subsidiary"],
                    "tenant": data["tenant"],
                    "tenant_code": tenant_code,
                    "prior": data["prior"],
                    "current": data["current"],
                    "variance": variance,
                    "ufl_action": ufl_action,
                    "ufl_month": ufl_month,
                    "gla": ufl_gla,
                    "flag": flag_type or ""
                })

            # Create flag if needed
            if flag_type:
                flags.append(VarianceFlag(
                    period=period_name,
                    flag_type=flag_type,
                    subsidiary=data["subsidiary"],
                    tenant=data["tenant"],
                    prior_amount=data["prior"] / 1e9,  # Convert to billions
                    current_amount=data["current"] / 1e9,
                    variance=variance / 1e9,
                    detail=self._get_flag_detail(flag_type, current_month)
                ))

        # Sort by variance magnitude
        increases.sort(key=lambda x: abs(x["variance"]), reverse=True)
        decreases.sort(key=lambda x: abs(x["variance"]), reverse=True)

        # Calculate totals
        total_variance = sum(d["current"] - d["prior"] for d in tenant_data.values())
        total_increases = sum(i["variance"] for i in increases)
        total_decreases = sum(d["variance"] for d in decreases)

        return {
            "variance": total_variance,
            "increases": increases[:15],  # Top 15
            "decreases": decreases[:15],
            "total_increases": total_increases,
            "total_decreases": total_decreases,
            "flags": flags
        }

    def _find_ufl_handover(self, tenant_code: str, current_month: int) -> Optional[UFLRecord]:
        """Find UFL handover for a tenant within 0-2 months of current month."""
        if not tenant_code:
            return None

        for ufl in self.ufl_records:
            if ufl.tenant_code != tenant_code:
                continue
            if ufl.status != "Handed Over":
                continue
            if not ufl.start_date:
                continue

            # Check if start date is within 0-2 months of current month
            if ufl.start_date.year == self.year:
                month_diff = current_month - ufl.start_date.month
                if 0 <= month_diff <= 2:
                    return ufl

        return None

    def _find_ufl_termination(self, tenant_code: str, current_month: int) -> Optional[UFLRecord]:
        """Find UFL termination for a tenant within 0-1 month of current month."""
        if not tenant_code:
            return None

        for ufl in self.ufl_records:
            if ufl.tenant_code != tenant_code:
                continue
            if ufl.status != "Terminated":
                continue
            if not ufl.termination_date:
                continue

            # Check if termination date is within 0-1 month of current month
            if ufl.termination_date.year == self.year:
                month_diff = current_month - ufl.termination_date.month
                if 0 <= month_diff <= 1:
                    return ufl

        return None

    def _check_ufl_no_revenue(self) -> List[VarianceFlag]:
        """Check for large UFL handovers without corresponding revenue increase."""
        flags = []

        # Group UFL handovers by tenant code and month
        handovers = [ufl for ufl in self.ufl_records
                     if ufl.status == "Handed Over"
                     and ufl.start_date
                     and ufl.start_date.year == self.year
                     and ufl.gla >= GLA_THRESHOLD]

        for ufl in handovers:
            handover_month = ufl.start_date.month

            # Check if there's revenue increase within 3 months
            has_revenue_increase = False
            for check_month in range(handover_month, min(handover_month + 3, 13)):
                prior_month = check_month - 1 if check_month > 1 else 1

                # Find revenue for this tenant
                tenant_revenue = [r for r in self.revenue_records
                                  if r.tenant_code == ufl.tenant_code
                                  and r.revenue_type == "Leasing"]

                for r in tenant_revenue:
                    current = r.monthly_amounts.get(check_month, 0)
                    prior = r.monthly_amounts.get(prior_month, 0)
                    if current > prior + VARIANCE_THRESHOLD:
                        has_revenue_increase = True
                        break

                if has_revenue_increase:
                    break

            if not has_revenue_increase:
                flags.append(VarianceFlag(
                    period=f"{MONTH_NAMES[handover_month - 1]} Handover",
                    flag_type="UFL NO REVENUE",
                    subsidiary=ufl.subsidiary,
                    tenant=ufl.tenant,
                    prior_amount=0,
                    current_amount=0,
                    variance=0,
                    detail=f"UFL handover {ufl.gla:,.0f} sqm but no revenue increase within 3 months"
                ))

        return flags

    def _calculate_monthly_totals(self) -> Dict[int, float]:
        """Calculate total revenue for each month."""
        totals = {}
        for month in range(1, 13):
            total = sum(r.monthly_amounts.get(month, 0) for r in self.revenue_records)
            totals[month] = total
        return totals

    def _get_flag_detail(self, flag_type: str, current_month: int) -> str:
        """Get detail message for a flag type."""
        details = {
            "INCREASE NO UFL": f"No UFL handover found (checked 0-2 months prior)",
            "DECREASE NO UFL": f"No UFL termination found (checked 0-1 month prior)",
            "NEGATIVE NO UFL": f"Revenue went negative but no termination found",
            "REVERSAL": "Prior month negative, now positive - billing correction",
            "UFL NO REVENUE": "UFL handover but no corresponding revenue increase"
        }
        return details.get(flag_type, "")

    def _detect_year(self, rows: List) -> None:
        """Detect year from header rows."""
        for row in rows[:10]:
            cells = row.findall('ss:Cell', self.ns)
            for cell in cells:
                data = cell.find('ss:Data', self.ns)
                if data is not None and data.text:
                    # Look for patterns like "Jan 2025" or "From Jan 2025"
                    match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4})', data.text)
                    if match:
                        self.year = int(match.group(2))
                        logger.info(f"Detected year: {self.year}")
                        return

    def _find_data_start(self, rows: List) -> int:
        """Find where data rows start."""
        for idx, row in enumerate(rows):
            cells = row.findall('ss:Cell', self.ns)
            values = self._extract_cell_values_with_index(cells)
            first_val = values.get(0, "").strip()

            # Data starts after account code rows
            if first_val and re.match(r'^\d{9}\s*-', first_val):
                return idx

        return 10  # Default

    def _extract_cell_values_with_index(self, cells: List) -> Dict[int, str]:
        """Extract cell values handling sparse cells with Index attribute."""
        values = {}
        col_idx = 0

        for cell in cells:
            # Check for Index attribute
            idx_attr = cell.get('{urn:schemas-microsoft-com:office:spreadsheet}Index')
            if idx_attr:
                col_idx = int(idx_attr) - 1  # Convert to 0-based

            data = cell.find('ss:Data', self.ns)
            if data is not None and data.text:
                values[col_idx] = data.text.strip()
            else:
                values[col_idx] = ""

            col_idx += 1

        return values

    def _extract_header_row(self, row) -> List[str]:
        """Extract header values from a row."""
        cells = row.findall('ss:Cell', self.ns)
        headers = []
        col_idx = 0

        for cell in cells:
            idx_attr = cell.get('{urn:schemas-microsoft-com:office:spreadsheet}Index')
            if idx_attr:
                # Fill gaps with empty strings
                target_idx = int(idx_attr) - 1
                while col_idx < target_idx:
                    headers.append("")
                    col_idx += 1

            data = cell.find('ss:Data', self.ns)
            if data is not None and data.text:
                headers.append(data.text.strip())
            else:
                headers.append("")
            col_idx += 1

        return headers

    def _get_by_header(self, values: Dict[int, str], header_map: Dict[str, int], header: str, default: str = "") -> str:
        """Get value by header name."""
        idx = header_map.get(header.lower())
        if idx is not None:
            return values.get(idx, default)
        return default

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse ISO date string."""
        if not date_str:
            return None
        try:
            # Format: 2025-11-08T00:00:00
            return datetime.fromisoformat(date_str.replace("T", " ").split(".")[0])
        except ValueError:
            return None

    def _generate_excel(self, comparisons: List[MonthlyComparison],
                        all_flags: List[VarianceFlag],
                        monthly_totals: Dict[int, float]) -> bytes:
        """Generate the 13-sheet Excel output."""
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Flags - Unusual Items
            self._write_flags_sheet(writer, all_flags)

            # Sheet 2: Executive Summary
            self._write_executive_summary(writer, monthly_totals, all_flags)

            # Sheets 3-13: Monthly comparisons
            for comp in comparisons:
                sheet_name = f"{comp.current_month_name} vs {comp.prior_month_name}"
                self._write_monthly_sheet(writer, comp, sheet_name)

        return output.getvalue()

    def _write_flags_sheet(self, writer, flags: List[VarianceFlag]):
        """Write the Flags - Unusual Items sheet."""
        ws = writer.book.create_sheet("Flags - Unusual Items", 0)

        # Styles
        header_fill = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF', size=14)
        title_font = Font(bold=True, size=16)

        # Title
        ws['A1'] = "REVENUE VS UFL MISMATCH FLAGS"
        ws['A1'].font = title_font
        ws['A2'] = "Leasing Revenue (511) Only - Items requiring investigation"

        # Legend
        ws['A4'] = "FLAG TYPES:"
        ws['A4'].font = Font(bold=True)

        flag_legend = [
            ("INCREASE NO UFL", "Revenue increased but no UFL handover found", "FFEB9C"),
            ("DECREASE NO UFL", "Revenue decreased but no UFL termination found", "FFEB9C"),
            ("NEGATIVE NO UFL", "Revenue went negative but no termination found", "FFC7CE"),
            ("REVERSAL", "Prior month negative, now positive - billing correction", "B4C6E7"),
            ("UFL NO REVENUE", "UFL handover but no corresponding revenue increase", "C6EFCE"),
        ]

        for i, (flag_type, desc, color) in enumerate(flag_legend):
            ws.cell(row=5+i, column=1, value=flag_type)
            ws.cell(row=5+i, column=1).fill = PatternFill(start_color=color, end_color=color, fill_type='solid')
            ws.cell(row=5+i, column=2, value=desc)

        ws['A11'] = "NOTE: Flags apply to LEASING (511) revenue only - other revenue types do not have UFL linkage"

        # Summary counts
        flag_counts = defaultdict(int)
        for f in flags:
            flag_counts[f.flag_type] += 1

        ws['A13'] = f"TOTAL FLAGS: {len(flags)}"
        ws['A13'].font = Font(bold=True)

        row = 14
        for flag_type, count in sorted(flag_counts.items()):
            ws.cell(row=row, column=1, value=f"  {flag_type}: {count}")
            row += 1

        # Detail table
        detail_start = row + 2
        headers = ["Period", "Flag Type", "Subsidiary", "Tenant", "Prior (B)", "Current (B)", "Δ (B)", "Detail"]

        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=detail_start, column=col, value=header)
            cell.fill = header_fill
            cell.font = Font(bold=True, color='FFFFFF')

        for i, flag in enumerate(flags):
            row = detail_start + 1 + i
            ws.cell(row=row, column=1, value=flag.period)
            ws.cell(row=row, column=2, value=flag.flag_type)
            ws.cell(row=row, column=3, value=flag.subsidiary)
            ws.cell(row=row, column=4, value=flag.tenant)
            ws.cell(row=row, column=5, value=round(flag.prior_amount, 2) if flag.prior_amount else "-")
            ws.cell(row=row, column=6, value=round(flag.current_amount, 2) if flag.current_amount else "-")
            ws.cell(row=row, column=7, value=round(flag.variance, 2))
            ws.cell(row=row, column=8, value=flag.detail)

            # Color based on flag type
            color_map = {
                "INCREASE NO UFL": "FFEB9C",
                "DECREASE NO UFL": "FFEB9C",
                "NEGATIVE NO UFL": "FFC7CE",
                "REVERSAL": "B4C6E7",
                "UFL NO REVENUE": "C6EFCE"
            }
            fill_color = color_map.get(flag.flag_type, "FFFFFF")
            for col in range(1, 9):
                ws.cell(row=row, column=col).fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type='solid')

        # Set column widths
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 18
        ws.column_dimensions['C'].width = 12
        ws.column_dimensions['D'].width = 35
        ws.column_dimensions['E'].width = 12
        ws.column_dimensions['F'].width = 12
        ws.column_dimensions['G'].width = 10
        ws.column_dimensions['H'].width = 50

    def _write_executive_summary(self, writer, monthly_totals: Dict[int, float], flags: List[VarianceFlag]):
        """Write the Executive Summary sheet."""
        ws = writer.book.create_sheet("Executive Summary", 1)

        # Title
        ws['A1'] = "BW INDUSTRIAL - REVENUE VARIANCE ANALYSIS"
        ws['A1'].font = Font(bold=True, size=16)
        ws['A2'] = f"MoM Waterfall | {self.year} | UFL Linkage for Leasing (511)"

        # Monthly Revenue table
        ws['A5'] = "MONTHLY REVENUE (VND Billion)"
        ws['A5'].font = Font(bold=True, size=12)

        # Headers
        header_fill = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')

        ws['A7'] = ""
        for i, month in enumerate(MONTH_NAMES):
            cell = ws.cell(row=7, column=i+2, value=month)
            cell.fill = header_fill
            cell.font = header_font

        # Revenue row
        ws['A8'] = "Revenue"
        for i in range(1, 13):
            val = monthly_totals.get(i, 0) / 1e9  # Convert to billions
            ws.cell(row=8, column=i+1, value=round(val, 1))

        # MoM Change row
        ws['A9'] = "MoM Δ"
        ws.cell(row=9, column=2, value="-")  # Jan has no prior
        for i in range(2, 13):
            prior = monthly_totals.get(i-1, 0)
            current = monthly_totals.get(i, 0)
            change = (current - prior) / 1e9
            cell = ws.cell(row=9, column=i+1, value=round(change, 1))

            # Color code
            if change > 0:
                cell.fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
            elif change < 0:
                cell.fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')

        # Set column widths
        for i in range(1, 14):
            ws.column_dimensions[chr(64+i)].width = 10

    def _write_monthly_sheet(self, writer, comp: MonthlyComparison, sheet_name: str):
        """Write a monthly comparison sheet."""
        ws = writer.book.create_sheet(sheet_name)

        # Styles
        header_fill = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')
        section_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        increase_fill = PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid')
        decrease_fill = PatternFill(start_color='FCE4D6', end_color='FCE4D6', fill_type='solid')
        flag_fill = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
        reversal_fill = PatternFill(start_color='B4C6E7', end_color='B4C6E7', fill_type='solid')

        # Title
        ws['A1'] = f"VARIANCE WATERFALL: {comp.current_month_name} vs {comp.prior_month_name}"
        ws['A1'].font = Font(bold=True, size=14)

        flag_count = len(comp.flags)
        ws['A2'] = f"Net Change: {comp.net_change/1e9:+.1f}B VND ({comp.net_change_pct:+.1f}%) | Flags: {flag_count}"

        # Level 1: By Revenue Stream
        ws['A4'] = "LEVEL 1: BY REVENUE STREAM"
        ws['A4'].font = Font(bold=True, size=12)

        level1_headers = ["Revenue Stream", f"{comp.prior_month_name} (B)", f"{comp.current_month_name} (B)", "Variance (B)", "% of Net"]
        for col, header in enumerate(level1_headers, 1):
            cell = ws.cell(row=5, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font

        for i, stream_data in enumerate(comp.stream_breakdown):
            row = 6 + i
            ws.cell(row=row, column=1, value=stream_data["stream"])
            ws.cell(row=row, column=2, value=round(stream_data["prior"]/1e9, 2) if stream_data["prior"] else "")
            ws.cell(row=row, column=3, value=round(stream_data["current"]/1e9, 2) if stream_data["current"] else "")
            ws.cell(row=row, column=4, value=round(stream_data["variance"]/1e9, 2))
            ws.cell(row=row, column=5, value=f"{stream_data['pct_of_net']:+.0f}%")

            # Bold the TOTAL row
            if stream_data["stream"] == "TOTAL":
                for col in range(1, 6):
                    ws.cell(row=row, column=col).font = Font(bold=True)

        # Level 2: Leasing Drill Down
        level2_start = 6 + len(comp.stream_breakdown) + 2
        ws.cell(row=level2_start, column=1, value="LEVEL 2: LEASING (511) DRILL DOWN - UFL Linkage")
        ws.cell(row=level2_start, column=1).font = Font(bold=True, size=12)

        ws.cell(row=level2_start + 2, column=1, value=f"Leasing: {comp.leasing_variance/1e9:+.2f}B")
        ws.cell(row=level2_start + 2, column=1).fill = section_fill
        ws.cell(row=level2_start + 2, column=1).font = Font(bold=True, color='FFFFFF')

        # Increases section
        inc_start = level2_start + 3
        ws.cell(row=inc_start, column=1, value=f"  (+) INCREASES: {comp.total_increases/1e9:+.2f}B")
        ws.cell(row=inc_start, column=1).font = Font(bold=True)

        inc_headers = ["#", "Sub", "Tenant", comp.prior_month_name, comp.current_month_name, "Δ (B)", "UFL Action", "UFL Month", "GLA", "Flag"]
        for col, header in enumerate(inc_headers, 1):
            cell = ws.cell(row=inc_start + 1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font

        for i, inc in enumerate(comp.leasing_increases):
            row = inc_start + 2 + i
            ws.cell(row=row, column=1, value=i+1)
            ws.cell(row=row, column=2, value=inc["subsidiary"])
            ws.cell(row=row, column=3, value=inc["tenant"][:30] + ".." if len(inc["tenant"]) > 30 else inc["tenant"])
            ws.cell(row=row, column=4, value=round(inc["prior"]/1e9, 2) if inc["prior"] else "-")
            ws.cell(row=row, column=5, value=round(inc["current"]/1e9, 2) if inc["current"] else "")
            ws.cell(row=row, column=6, value=round(inc["variance"]/1e9, 2))
            ws.cell(row=row, column=7, value=inc["ufl_action"])
            ws.cell(row=row, column=8, value=inc["ufl_month"])
            ws.cell(row=row, column=9, value=inc["gla"])
            ws.cell(row=row, column=10, value=inc["flag"])

            # Color UFL Action
            if inc["ufl_action"] == "NEW LEASE":
                ws.cell(row=row, column=7).fill = increase_fill

            # Color flags
            if inc["flag"] == "REVERSAL":
                ws.cell(row=row, column=10).fill = reversal_fill
            elif inc["flag"]:
                ws.cell(row=row, column=10).fill = flag_fill

        # Decreases section
        dec_start = inc_start + 2 + len(comp.leasing_increases) + 2
        ws.cell(row=dec_start, column=1, value=f"  (-) DECREASES: {comp.total_decreases/1e9:.2f}B")
        ws.cell(row=dec_start, column=1).font = Font(bold=True)

        dec_headers = ["#", "Sub", "Tenant", comp.prior_month_name, comp.current_month_name, "Δ (B)", "UFL Action", "Term Month", "GLA", "Flag"]
        for col, header in enumerate(dec_headers, 1):
            cell = ws.cell(row=dec_start + 1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font

        for i, dec in enumerate(comp.leasing_decreases):
            row = dec_start + 2 + i
            ws.cell(row=row, column=1, value=i+1)
            ws.cell(row=row, column=2, value=dec["subsidiary"])
            ws.cell(row=row, column=3, value=dec["tenant"][:30] + ".." if len(dec["tenant"]) > 30 else dec["tenant"])
            ws.cell(row=row, column=4, value=round(dec["prior"]/1e9, 2) if dec["prior"] else "-")
            ws.cell(row=row, column=5, value=round(dec["current"]/1e9, 2) if dec["current"] else "-")
            ws.cell(row=row, column=6, value=round(dec["variance"]/1e9, 2))
            ws.cell(row=row, column=7, value=dec["ufl_action"])
            ws.cell(row=row, column=8, value=dec["ufl_month"])
            ws.cell(row=row, column=9, value=dec["gla"])
            ws.cell(row=row, column=10, value=dec["flag"])

            # Color UFL Action
            if dec["ufl_action"] == "TERMINATED":
                ws.cell(row=row, column=7).fill = decrease_fill

            # Color flags
            if dec["flag"]:
                ws.cell(row=row, column=10).fill = flag_fill

        # Set column widths
        ws.column_dimensions['A'].width = 5
        ws.column_dimensions['B'].width = 8
        ws.column_dimensions['C'].width = 30
        ws.column_dimensions['D'].width = 10
        ws.column_dimensions['E'].width = 10
        ws.column_dimensions['F'].width = 10
        ws.column_dimensions['G'].width = 12
        ws.column_dimensions['H'].width = 12
        ws.column_dimensions['I'].width = 10
        ws.column_dimensions['J'].width = 18


def analyze_revenue_variance(revenue_bytes: bytes, ufl_bytes: bytes) -> bytes:
    """
    Convenience function to analyze revenue variance.

    Args:
        revenue_bytes: Raw bytes of RevenueBreakdown file
        ufl_bytes: Raw bytes of UnitForLeaseList file

    Returns:
        Excel file bytes with 13 sheets
    """
    analyzer = RevenueVarianceAnalyzer()
    return analyzer.analyze(revenue_bytes, ufl_bytes)
