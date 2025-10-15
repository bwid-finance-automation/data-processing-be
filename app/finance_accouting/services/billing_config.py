"""
Configuration for utility billing processing (adapted from original config.py)
"""
from pathlib import Path
from datetime import datetime, timedelta

# === FOLDER PATHS ===
# These will be overridden by SessionManager in the BillingProcessor
BASE_FOLDER = Path(__file__).parent.parent.parent
INPUT_FOLDER = BASE_FOLDER / "uploads" / "input"
MASTER_DATA_FOLDER = BASE_FOLDER / "uploads" / "master_data"
OUTPUT_FOLDER = BASE_FOLDER / "uploads" / "output"
ARCHIVE_FOLDER = BASE_FOLDER / "uploads" / "archive"
LOG_FOLDER = BASE_FOLDER / "uploads" / "logs"

# === MASTER DATA FILES ===
CUSTOMER_MASTER_FILE = MASTER_DATA_FOLDER / "Customers_Master.xlsx"
UNIT_MASTER_FILE = MASTER_DATA_FOLDER / "UnitForLease_Master.xlsx"
CONFIG_MAPPING_FILE = MASTER_DATA_FOLDER / "Config_Mapping.xlsx"

# === COLUMN MAPPINGS ===

# SOUTH FORMAT: CS Input File Columns (with Vietnamese headers)
CS_COLUMNS = {
    'site': 'Site',
    'tenant': 'Tenant\n(Khách hàng)',
    'unit': 'Unit\n(Xưởng/Kho)',
    'utility_type': 'Utility type\n(Loại)',
    'utility_subtype': 'Utility sub-type\n(Loại)',
    'start_date': 'Start date\n(từ ngày)',
    'end_date': 'End date\n(đến ngày)',
    'quantity': 'Actual Comsumption\n(Số tiêu thụ thực tế)',
    'rate': 'Price',
    'amount': 'Amount'
}

# NORTH FORMAT: Column mappings for Tenant's Utilities files (2 tabs)
CS_COLUMNS_NORTH_WATER = {
    'site': 'Project',
    'lot': 'Lot',
    'tenant': 'Tenant',
    'unit': 'Unit',
    'duration': 'Duration',
    'wsc_ip': 'WSC (IP)',
    'wwt_ip': 'WWT (IP)',
    'wwt_bw': ' WWT (BW)',
    'stp_min': 'Minimum STP',
    'old_meter': 'Old Meter Reading ',
    'new_meter': 'New Meter Reading',
    'wsc_m3': 'WSC (m3)',
    'wtp_m3': 'WTP (m3)',
    'wsc_fee': 'WSC fee (vnd)',
    'wwt_fee': 'WWT fee (vnd)',
    'stp_fee': 'STP',
    'amount': 'WSC+WWT+STP',
    'note': 'Note'
}

CS_COLUMNS_NORTH_ELECTRIC = {
    'site': 'Project',
    'lot': 'Lot',
    'tenant': 'Tenant',
    'unit': 'Unit',
    'duration': 'Duration',
    'peak_hour': 'Peak hour',
    'plc': 'PLC',
    'old_meter': 'Old Meter Reading ',
    'new_meter': 'New Meter Reading',
    'ti_rate': 'Ti rate',
    'quantity': 'Total kwh',
    'amount': 'Total amount',
    'note': 'Note'
}

# North format sheet names and settings
NORTH_WATER_SHEET = "Tenant's water (FIN)"
NORTH_ELECTRIC_SHEET = "Tenant's electric (FIN)"
NORTH_HEADER_ROW = 1  # Headers in row 2 (0-indexed row 1)

# South format settings
SOUTH_HEADER_ROW = 1  # Headers in row 2 (0-indexed row 1)

# Customer Master Columns (from Customers_Master.xlsx)
CUSTOMER_COLUMNS = {
    'customer_id': 'Customer ID (Use this for importing)',
    'name': 'Name',
    'subsidiary': 'Subsidiary'
}

# Unit Master Columns (from UnitForLease_Master.xlsx)
UNIT_COLUMNS = {
    'unit': 'Unit',
    'unit_name': 'Unit Name',
    'tenant': 'Tenant',
    'project': 'BWID Project.',
    'phase': 'Phase',
    'internal_id': 'Internal ID',
    'master_record_id': 'Master Record ID',
    'plc_id': 'PLC ID',
    'subsidiary': 'Subsidiary'
}

# ERP Output Columns (45 columns)
ERP_COLUMNS = [
    'EXTERNAL ID',
    'E-INVOICE PREFIX',
    'E-INVOICE NUMBER',
    'E-INVOICE INTEGRATION',
    'CUSTOMER',
    'PHASE',
    'PROJECT',
    'SUBSIDIARY',
    'DEPARTMENT',
    'BUDGET CODE',
    'CURRENCY',
    'EXCHANGE RATE',
    'PROFORMA INVOICE TYPE',
    'MEMO MAIN',
    'DATE',
    'DOCUMENT DATE',
    'DUE DATE',
    'BUDGET PERIOD',
    'BUDGET PERIOD(1)',
    'APPROVAL STATUS',
    'E-INVOICE FORM',
    'LINE ID',
    'ITEM',
    '  QUANTITY  ',
    'UNITS',
    ' RATE ',
    ' AMOUNT ',
    'TAX CODE',
    'MEMO (LINE) - Anh',
    'MEMO (VI)',
    'Mô tả hàng hóa, vật tư (changed)',
    'UNIT NO',
    'MASTER RECORD ID',
    'Unit for lease',
    'PREMISE LEASE CONTRACT',
    'BILL START DATE',
    'BILL END DATE',
    'BILL START DATE(1)',
    'BILL END DATE(1)',
    'SUB. BENEFICIARY NAME',
    'BANK ACCOUNT',
    'SWIFT CODE',
    'PHASE(1)',
    'PROJECT(1)',
    'BUDGET CODE(1)'
]

# === PROCESSING SETTINGS ===

# Fuzzy matching threshold for customer names (0-100)
FUZZY_MATCH_THRESHOLD = 70

# Invoice numbering
INVOICE_PREFIX = "INV"


def get_processing_date():
    """Get the processing date (today)"""
    return datetime.now()


def get_due_date(processing_date):
    """Calculate due date (7 days after processing date)"""
    return processing_date + timedelta(days=7)
