"""
Account 511 Parser - Parses RevenueBreakdown and UnitForLeaseList XML-Excel files.

This module handles the parsing of NetSuite-exported XML-based Excel files
for Account 511 (Revenue) drill-down analysis.
"""

import io
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


# Account 511 sub-account hierarchy
ACCOUNT_511_HIERARCHY = {
    "511600001": {"name": "Revenue: Service charge - Third parties", "parent": "511600000"},
    "511600002": {"name": "Revenue: Management fee Parent/Subsidiaries", "parent": "511600000"},
    "511600003": {"name": "Revenue: Management fee Related Parties", "parent": "511600000"},
    "511600004": {"name": "Revenue: Sales and tenant management fee", "parent": "511600000"},
    "511600005": {"name": "Revenue: Operation management fee", "parent": "511600000"},
    "511600006": {"name": "Revenue: Facility management fee", "parent": "511600000"},
    "511600007": {"name": "Revenue: Development Management fee", "parent": "511600000"},
    "511600000": {"name": "Revenue: Service charge", "parent": "511000000"},
    "511710001": {"name": "Revenue: Investment Properties for lease", "parent": "511710000"},
    "511710000": {"name": "Revenue: Investment properties", "parent": "511000000"},
    "511800001": {"name": "Revenue: Utilities", "parent": "511800000"},
    "511800002": {"name": "Revenue: Others.", "parent": "511800000"},
    "511800000": {"name": "Revenue: Others", "parent": "511000000"},
    "511000000": {"name": "Revenue from sale and service provider", "parent": None},
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class RevenueLineItem:
    """Single revenue line item from RevenueBreakdown."""
    subsidiary_code: str = ""
    project: str = ""
    phase: str = ""
    entity_name: str = ""
    entity_code: str = ""
    account_code: str = ""
    monthly_amounts: Dict[Tuple[int, int], float] = field(default_factory=dict)  # (year, month) -> amount
    ytd: float = 0.0
    ltm: float = 0.0


@dataclass
class Account511SubAccount:
    """Aggregated sub-account data."""
    account_code: str
    account_name: str
    parent_code: Optional[str]
    current_month_amount: float = 0.0
    previous_month_amount: float = 0.0
    variance: float = 0.0
    variance_pct: float = 0.0
    ytd: float = 0.0
    ltm: float = 0.0
    line_items: List[RevenueLineItem] = field(default_factory=list)
    # By project breakdown
    by_project: Dict[str, float] = field(default_factory=dict)


@dataclass
class UnitForLease:
    """Single unit from UnitForLeaseList."""
    internal_id: str = ""
    unit_id: str = ""
    unit_name: str = ""
    gla: float = 0.0
    plc_id: str = ""
    region: str = ""
    status: str = ""  # Handed Over, Open, Terminated, Voided
    novation_date: Optional[datetime] = None
    tenant: str = ""
    tenant_code: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    rental_rate: float = 0.0
    billing_contract_value: float = 0.0
    project_name: str = ""
    project_code: str = ""
    phase: str = ""
    subsidiary: str = ""
    product_type: str = ""


@dataclass
class Account511Analysis:
    """Complete Account 511 analysis result."""
    # Sub-account breakdown
    sub_accounts: Dict[str, Account511SubAccount] = field(default_factory=dict)
    # Period info
    current_period: Tuple[int, int] = (2026, 1)  # (year, month)
    previous_period: Tuple[int, int] = (2025, 12)  # (year, month)
    # Total variance
    total_current: float = 0.0
    total_previous: float = 0.0
    total_variance: float = 0.0
    total_variance_pct: float = 0.0
    # By project
    by_project: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Unit correlation
    units_by_tenant: Dict[str, List[UnitForLease]] = field(default_factory=dict)
    # AI-generated narrative
    narrative: str = ""


class RevenueBreakdownParser:
    """Parser for NetSuite RevenueBreakdown XML-Excel files."""

    def __init__(self):
        self.ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

    def parse(self, file_bytes: bytes) -> Tuple[List[RevenueLineItem], Dict[str, Account511SubAccount], Tuple[int, int], Tuple[int, int]]:
        """
        Parse RevenueBreakdown XML-Excel file.

        Args:
            file_bytes: Raw bytes of the XML-Excel file

        Returns:
            Tuple of (line_items, sub_accounts, current_period, previous_period)
        """
        try:
            content = file_bytes.decode('utf-8')
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            raise ValueError(f"Invalid XML format: {e}")

        worksheet = root.find('.//ss:Worksheet', self.ns)
        if worksheet is None:
            raise ValueError("No worksheet found in file")

        table = worksheet.find('ss:Table', self.ns)
        if table is None:
            raise ValueError("No table found in worksheet")

        rows = table.findall('ss:Row', self.ns)

        # Detect periods from header rows
        current_period, previous_period = self._detect_periods(rows)

        # Find header row and parse data
        header_row_idx = self._find_header_row(rows)
        if header_row_idx is None:
            raise ValueError("Could not find header row")

        # Parse month columns from header
        month_columns = self._parse_month_columns(rows, header_row_idx)

        # Parse line items and aggregate by sub-account
        line_items = []
        sub_accounts: Dict[str, Account511SubAccount] = {}

        # Initialize sub-accounts
        for code, info in ACCOUNT_511_HIERARCHY.items():
            sub_accounts[code] = Account511SubAccount(
                account_code=code,
                account_name=info["name"],
                parent_code=info["parent"]
            )

        current_sub_account = None

        # First pass: find all Total rows to extract sub-account amounts
        for row_idx, row in enumerate(rows):
            cells = row.findall('ss:Cell', self.ns)
            cell_values = self._extract_cell_values(cells)

            if not cell_values or len(cell_values) < 2:
                continue

            # Check if this is a total row for a sub-account
            first_cell = cell_values[0] if cell_values else ""
            if first_cell.startswith("Total - 511"):
                # Extract account code from "Total - 511600001 - Revenue: ..."
                match = re.match(r"Total - (\d+) -", first_cell)
                if match:
                    account_code = match.group(1)
                    if account_code in sub_accounts:
                        sub_account = sub_accounts[account_code]

                        # In Total rows, the first 6 columns are: Account name, then empty columns for
                        # Subsidiary, Project, Phase, Entity, EntityCode, Account
                        # Monthly amounts start at column index 7 (Jan, Feb, Mar... Dec, YTD, LTM)
                        # Find the first non-empty numeric value after the account name
                        amounts = []
                        for val in cell_values[1:]:  # Skip account name
                            if val:
                                try:
                                    amounts.append(float(val))
                                except (ValueError, TypeError):
                                    # Skip non-numeric values
                                    pass

                        # The first 12 values are monthly amounts (Jan-Dec)
                        # Current period is (2026, 1) = January = index 0
                        file_year = current_period[0]
                        current_month_idx = current_period[1] - 1  # 0 for Jan, 11 for Dec

                        # For previous month:
                        if previous_period[0] == file_year:
                            previous_month_idx = previous_period[1] - 1
                        else:
                            previous_month_idx = -1  # Different year - no data

                        # Try to get current month amount
                        if 0 <= current_month_idx < len(amounts):
                            sub_account.current_month_amount = amounts[current_month_idx]

                        # Try to get previous month amount (only if same year)
                        if 0 <= previous_month_idx < len(amounts):
                            sub_account.previous_month_amount = amounts[previous_month_idx]

                        sub_account.variance = sub_account.current_month_amount - sub_account.previous_month_amount
                        if sub_account.previous_month_amount != 0:
                            sub_account.variance_pct = (sub_account.variance / abs(sub_account.previous_month_amount)) * 100

        # Second pass: parse line items for project breakdown
        # Line item rows have format: [empty], Subsidiary, Project, Phase, Entity Name, Entity Code, Account, [monthly amounts...]
        # Example: ['', 'VC3', 'PVC3: VC3', 'VC3_00: VC3', 'CÔNG TY...', 'S00002874...', '01', '0.0', ...]
        for row_idx, row in enumerate(rows):
            cells = row.findall('ss:Cell', self.ns)
            cell_values = self._extract_cell_values(cells)

            if not cell_values or len(cell_values) < 7:
                continue

            # Skip Total rows and header-like rows
            first_cell = cell_values[0] if cell_values else ""
            if first_cell.startswith("Total -") or first_cell.startswith("511") or first_cell in ["Financial Row", "1. Doanh thu"]:
                continue

            # Line items have empty first cell, subsidiary in [1], project in [2] (format "PXXX: Name")
            # Check if this looks like a data row
            if (first_cell == "" and
                len(cell_values) >= 7 and
                cell_values[2] and ":" in cell_values[2]):  # Project format "PXXX: Name"

                line_item = RevenueLineItem(
                    subsidiary_code=cell_values[1] if len(cell_values) > 1 else "",
                    project=cell_values[2] if len(cell_values) > 2 else "",
                    phase=cell_values[3] if len(cell_values) > 3 else "",
                    entity_name=cell_values[4] if len(cell_values) > 4 else "",
                    entity_code=cell_values[5] if len(cell_values) > 5 else "",
                    account_code=cell_values[6] if len(cell_values) > 6 else ""
                )

                # Parse monthly amounts (columns 7 onwards - Jan through Dec)
                if len(cell_values) > 7:
                    for i, val in enumerate(cell_values[7:]):
                        if i < 12:  # Only first 12 are months
                            try:
                                amount = float(val) if val else 0.0
                                line_item.monthly_amounts[(current_period[0], i + 1)] = amount
                            except (ValueError, TypeError):
                                pass

                if line_item.project:
                    line_items.append(line_item)

        logger.info(f"Parsed {len(line_items)} line items, {len([s for s in sub_accounts.values() if s.current_month_amount != 0 or s.previous_month_amount != 0])} active sub-accounts")

        return line_items, sub_accounts, current_period, previous_period

    def _detect_periods(self, rows: List) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Detect current and previous periods from header rows."""
        # Look for month header row (contains "Jan YYYY", "Feb YYYY", etc.)
        for row in rows[:15]:
            cells = row.findall('ss:Cell', self.ns)
            cell_values = self._extract_cell_values(cells)

            # Check if this row has month names
            months_found = []
            for val in cell_values:
                if val:
                    match = re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4})$', val.strip())
                    if match:
                        month_str = match.group(1)
                        year = int(match.group(2))
                        month = MONTH_NAMES.index(month_str) + 1
                        months_found.append((year, month))

            # If we found multiple months in one row, this is the month header
            if len(months_found) >= 6:  # At least 6 months
                # First month is current, assume previous is the one before
                current = months_found[0]  # Jan 2026
                if current[1] == 1:
                    previous = (current[0] - 1, 12)  # Dec 2025
                else:
                    previous = (current[0], current[1] - 1)

                logger.info(f"Detected periods from header: {previous} → {current}")
                return current, previous

        # Default to Jan 2026 / Dec 2025
        logger.warning("Could not detect periods, using default Jan 2026 / Dec 2025")
        return (2026, 1), (2025, 12)

    def _find_header_row(self, rows: List) -> Optional[int]:
        """Find the header row containing column names."""
        for idx, row in enumerate(rows):
            cells = row.findall('ss:Cell', self.ns)
            cell_values = self._extract_cell_values(cells)

            # Look for header indicators
            if any(v in cell_values for v in ["Financial Row", "Subsidiary", "BWID Project.", "Entity"]):
                return idx
        return None

    def _parse_month_columns(self, rows: List, header_row_idx: int) -> Dict[int, Tuple[int, int]]:
        """Parse month columns from header rows."""
        month_columns: Dict[int, Tuple[int, int]] = {}

        # Check header row and row below for month names
        if header_row_idx + 1 < len(rows):
            month_row = rows[header_row_idx + 1]
            cells = month_row.findall('ss:Cell', self.ns)

            col_idx = 0
            for cell in cells:
                # Handle cell index attribute if present
                idx_attr = cell.get('{urn:schemas-microsoft-com:office:spreadsheet}Index')
                if idx_attr:
                    col_idx = int(idx_attr) - 1

                data = cell.find('ss:Data', self.ns)
                if data is not None and data.text:
                    text = data.text.strip()
                    # Match patterns like "Jan 2026", "Feb 2026"
                    match = re.match(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4})', text)
                    if match:
                        month_str = match.group(1)
                        year = int(match.group(2))
                        month = MONTH_NAMES.index(month_str) + 1
                        month_columns[col_idx] = (year, month)

                col_idx += 1

        return month_columns

    def _extract_cell_values(self, cells: List) -> List[str]:
        """Extract text values from cells."""
        values = []
        for cell in cells:
            data = cell.find('ss:Data', self.ns)
            if data is not None and data.text:
                values.append(data.text.strip())
            else:
                values.append("")
        return values

    def _parse_amounts(self, values: List[str], month_columns: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """Parse numeric amounts from cell values."""
        amounts: Dict[Tuple[int, int], float] = {}

        for col_idx, period in month_columns.items():
            # Adjust for offset (values starts after first column)
            val_idx = col_idx - 1  # Accounting for the first column being Financial Row
            if 0 <= val_idx < len(values):
                try:
                    # Handle scientific notation (e.g., -5.6704E8)
                    amount = float(values[val_idx]) if values[val_idx] else 0.0
                    amounts[period] = amount
                except (ValueError, IndexError):
                    pass

        return amounts

    def _parse_line_item(self, cell_values: List[str], month_columns: Dict[int, Tuple[int, int]]) -> Optional[RevenueLineItem]:
        """Parse a single line item from cell values."""
        if len(cell_values) < 5:
            return None

        # Expected format: Subsidiary, Project, Phase, Entity Name, Entity Code, Account, amounts...
        # But format varies, try to detect
        item = RevenueLineItem()

        # First few columns are identifiers
        idx = 0

        # Check if first value is subsidiary code (short code like "VC3", "BBA")
        if cell_values[idx] and len(cell_values[idx]) <= 5:
            item.subsidiary_code = cell_values[idx]
            idx += 1

        # Project (format: "PVC3: VC3" or "PBBA: Bau Bang")
        if idx < len(cell_values) and ":" in cell_values[idx]:
            item.project = cell_values[idx]
            idx += 1

        # Phase (format: "VC3_00: VC3")
        if idx < len(cell_values) and "_" in cell_values[idx]:
            item.phase = cell_values[idx]
            idx += 1

        # Entity name (Vietnamese company name)
        if idx < len(cell_values) and cell_values[idx]:
            item.entity_name = cell_values[idx]
            idx += 1

        # Entity code (format: "S000056 IE/BWID...")
        if idx < len(cell_values) and cell_values[idx]:
            item.entity_code = cell_values[idx]
            idx += 1

        # Account code (usually "01" for revenue)
        if idx < len(cell_values) and cell_values[idx]:
            item.account_code = cell_values[idx]
            idx += 1

        # Remaining are amounts
        amounts = self._parse_amounts(cell_values[idx:], {k - idx: v for k, v in month_columns.items() if k >= idx})
        item.monthly_amounts = amounts

        return item if item.project else None


class UnitForLeaseListParser:
    """Parser for NetSuite UnitForLeaseList XML-Excel files."""

    def __init__(self):
        self.ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        self.header_mapping = {}

    def parse(self, file_bytes: bytes) -> List[UnitForLease]:
        """
        Parse UnitForLeaseList XML-Excel file.

        Args:
            file_bytes: Raw bytes of the XML-Excel file

        Returns:
            List of UnitForLease objects
        """
        try:
            content = file_bytes.decode('utf-8')
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            raise ValueError(f"Invalid XML format: {e}")

        worksheet = root.find('.//ss:Worksheet', self.ns)
        if worksheet is None:
            raise ValueError("No worksheet found in file")

        table = worksheet.find('ss:Table', self.ns)
        if table is None:
            raise ValueError("No table found in worksheet")

        rows = table.findall('ss:Row', self.ns)

        if len(rows) < 2:
            raise ValueError("File has insufficient rows")

        # First row is header
        header_row = rows[0]
        headers = self._parse_header_row(header_row)

        # Build column index mapping
        self.header_mapping = {h.lower(): i for i, h in enumerate(headers)}

        # Parse data rows
        units = []
        for row in rows[1:]:
            unit = self._parse_unit_row(row, headers)
            if unit and unit.unit_name:
                units.append(unit)

        logger.info(f"Parsed {len(units)} units from UnitForLeaseList")
        return units

    def _parse_header_row(self, row) -> List[str]:
        """Parse header row to get column names."""
        cells = row.findall('ss:Cell', self.ns)
        headers = []
        for cell in cells:
            data = cell.find('ss:Data', self.ns)
            if data is not None and data.text:
                headers.append(data.text.strip())
            else:
                headers.append("")
        return headers

    def _parse_unit_row(self, row, headers: List[str]) -> Optional[UnitForLease]:
        """Parse a single unit row."""
        cells = row.findall('ss:Cell', self.ns)
        values = []

        col_idx = 0
        for cell in cells:
            # Handle cell index attribute
            idx_attr = cell.get('{urn:schemas-microsoft-com:office:spreadsheet}Index')
            if idx_attr:
                # Fill gaps with empty strings
                target_idx = int(idx_attr) - 1
                while col_idx < target_idx:
                    values.append("")
                    col_idx += 1

            data = cell.find('ss:Data', self.ns)
            if data is not None and data.text:
                values.append(data.text.strip())
            else:
                values.append("")
            col_idx += 1

        # Pad to match headers
        while len(values) < len(headers):
            values.append("")

        unit = UnitForLease()

        # Map values to unit fields
        unit.internal_id = self._get_value(values, "internal id")
        unit.unit_id = self._get_value(values, "id")
        unit.unit_name = self._get_value(values, "unit")
        unit.gla = self._get_float(values, "gla for lease")
        unit.plc_id = self._get_value(values, "plc id")
        unit.region = self._get_value(values, "region")
        unit.status = self._get_value(values, "unit for lease status")
        unit.tenant = self._get_value(values, "tenant")
        unit.rental_rate = self._get_float(values, "1st year rental rate")
        unit.billing_contract_value = self._get_float(values, "billing contract value")
        unit.project_name = self._get_value(values, "project name")
        unit.subsidiary = self._get_value(values, "subsidiary")
        unit.product_type = self._get_value(values, "ccs_product type")
        unit.phase = self._get_value(values, "phase")

        # Parse dates
        unit.start_date = self._parse_date(self._get_value(values, "start leasing date"))
        unit.end_date = self._parse_date(self._get_value(values, "end leasing date"))
        unit.novation_date = self._parse_date(self._get_value(values, "novation date"))

        # Extract project code from unit name (e.g., "P-LV1_01_A1A" -> "LV1")
        if unit.unit_name and unit.unit_name.startswith("P-"):
            parts = unit.unit_name.split("_")
            if parts:
                unit.project_code = parts[0].replace("P-", "")

        # Extract tenant code from tenant field (e.g., "C00000164 CTTV" -> "C00000164")
        if unit.tenant:
            tenant_parts = unit.tenant.split(" ", 1)
            if tenant_parts:
                unit.tenant_code = tenant_parts[0]

        return unit

    def _get_value(self, values: List[str], header: str) -> str:
        """Get value by header name."""
        idx = self.header_mapping.get(header.lower())
        if idx is not None and idx < len(values):
            return values[idx]
        return ""

    def _get_float(self, values: List[str], header: str) -> float:
        """Get float value by header name."""
        val = self._get_value(values, header)
        try:
            return float(val) if val else 0.0
        except ValueError:
            return 0.0

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse ISO date string."""
        if not date_str:
            return None
        try:
            # Format: 2022-11-08T00:00:00
            return datetime.fromisoformat(date_str.replace("T", " ").split(".")[0])
        except ValueError:
            return None


def parse_revenue_breakdown(file_bytes: bytes) -> Tuple[List[RevenueLineItem], Dict[str, Account511SubAccount], Tuple[int, int], Tuple[int, int]]:
    """
    Parse RevenueBreakdown file.

    Args:
        file_bytes: Raw bytes of the file

    Returns:
        Tuple of (line_items, sub_accounts, current_period, previous_period)
    """
    parser = RevenueBreakdownParser()
    return parser.parse(file_bytes)


def parse_unit_for_lease_list(file_bytes: bytes) -> List[UnitForLease]:
    """
    Parse UnitForLeaseList file.

    Args:
        file_bytes: Raw bytes of the file

    Returns:
        List of UnitForLease objects
    """
    parser = UnitForLeaseListParser()
    return parser.parse(file_bytes)
