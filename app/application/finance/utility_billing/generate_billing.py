"""
Utility Billing Service - Main processing logic (adapted from utility_billing.py)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from rapidfuzz import fuzz, process
import shutil
from typing import Dict, List, Tuple, Optional
from app.shared.utils.logging_config import get_logger
from app.core.session import SessionManager
from . import billing_config as config

logger = get_logger(__name__)


class BillingProcessor:
    """Main class for processing utility billing"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.master_data = None
        self.processing_stats = {}

        # Set paths based on session
        if session_id:
            self.input_folder = SessionManager.get_input_dir(session_id)
            self.master_data_folder = SessionManager.get_master_data_dir(session_id)
            self.output_folder = SessionManager.get_output_dir(session_id)
            self.log_folder = SessionManager.get_log_dir(session_id)
        else:
            # Fallback to default paths
            self.input_folder = config.INPUT_FOLDER
            self.master_data_folder = config.MASTER_DATA_FOLDER
            self.output_folder = config.OUTPUT_FOLDER
            self.log_folder = config.LOG_FOLDER

    # === UTILITY FUNCTIONS ===
    @staticmethod
    def safe_str(value):
        """Convert value to string, handling NaN/None values"""
        if pd.isna(value) or value is None:
            return None
        return str(value).strip()

    @staticmethod
    def get_subsidiary_full_name(site, subsidiary_config_df):
        """Map CS Site to Subsidiary Full Name using Config_Mapping"""
        if pd.isna(site):
            return None, None

        site_str = str(site).strip()

        # Look up in subsidiary config
        match = subsidiary_config_df[
            subsidiary_config_df['Subsidiary'].str.strip() == site_str
        ]

        if not match.empty:
            subsidiary_code = site_str
            subsidiary_full_name = match.iloc[0]['Subsidiary_Full_Name']
            logger.debug(f"Mapped site '{site_str}' → '{subsidiary_full_name}'")
            return subsidiary_code, subsidiary_full_name

        logger.warning(f"Site '{site_str}' not found in subsidiary config")
        return None, None

    @staticmethod
    def fuzzy_match_unit_by_subsidiary(cs_unit_name, unit_master_df, subsidiary_full_name, threshold=70):
        """Fuzzy match CS unit name against UFL unit names within a specific subsidiary"""
        if pd.isna(cs_unit_name) or not subsidiary_full_name:
            return None

        cs_unit_clean = str(cs_unit_name).strip().upper()

        # Filter units by FULL subsidiary name (exact match)
        subsidiary_units = unit_master_df[
            unit_master_df[config.UNIT_COLUMNS['subsidiary']].str.strip() == subsidiary_full_name
        ]

        if subsidiary_units.empty:
            logger.warning(f"No units found for subsidiary '{subsidiary_full_name}'")
            return None

        logger.debug(f"Found {len(subsidiary_units)} units in subsidiary")

        # Get unit names for this subsidiary
        unit_names = subsidiary_units[config.UNIT_COLUMNS['unit_name']].dropna().tolist()

        if not unit_names:
            return None

        # Fuzzy match
        result = process.extractOne(
            cs_unit_clean,
            unit_names,
            scorer=fuzz.ratio
        )

        if result and result[1] >= threshold:
            matched_unit_name = result[0]

            # Get the full UFL record
            ufl_record = subsidiary_units[
                subsidiary_units[config.UNIT_COLUMNS['unit_name']] == matched_unit_name
            ].iloc[0]

            logger.info(f"Matched unit '{cs_unit_name}' → '{matched_unit_name}' (score: {result[1]})")
            return ufl_record

        logger.warning(f"Unit '{cs_unit_name}' not matched (best score: {result[1] if result else 0})")
        return None

    @staticmethod
    def parse_duration_north(duration_str):
        """Parse North format Duration field (e.g., "01/09/2025-30/09/2025")"""
        if pd.isna(duration_str) or not isinstance(duration_str, str):
            return None, None

        try:
            parts = duration_str.split('-')
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        except:
            pass

        return None, None

    # === MAIN PROCESSING FUNCTIONS ===
    def load_master_data(self):
        """Load all master data files"""
        logger.info("Loading master data files...")

        try:
            # Load customer master
            customer_file = self.master_data_folder / "Customers_Master.xlsx"
            customer_df = pd.read_excel(customer_file)
            logger.info(f"Loaded {len(customer_df)} customers")

            # Load unit master (UFL)
            unit_file = self.master_data_folder / "UnitForLease_Master.xlsx"
            unit_df = pd.read_excel(unit_file)

            # Filter out Terminated and Voided units
            original_count = len(unit_df)
            unit_df = unit_df[~unit_df['Unit for Lease Status'].isin(['Terminated', 'Voided'])]
            logger.info(f"Loaded {len(unit_df)} active units (excluded {original_count - len(unit_df)} terminated/voided)")

            # Load config mapping
            config_file = self.master_data_folder / "Config_Mapping.xlsx"
            subsidiary_config = pd.read_excel(config_file, sheet_name='Subsidiary_Config')
            utility_mapping = pd.read_excel(config_file, sheet_name='Utility_Mapping')

            logger.info("All master data loaded successfully")

            self.master_data = {
                'customers': customer_df,
                'units': unit_df,
                'subsidiary_config': subsidiary_config,
                'utility_mapping': utility_mapping
            }

            return self.master_data

        except Exception as e:
            logger.error(f"Error loading master data: {e}")
            raise

    def detect_file_format(self, file_path: Path):
        """Detect if file is North or South format"""
        try:
            xl_file = pd.ExcelFile(file_path)
            sheets = xl_file.sheet_names

            # North format has these specific sheet names
            if config.NORTH_WATER_SHEET in sheets and config.NORTH_ELECTRIC_SHEET in sheets:
                logger.info(f"  → Detected NORTH format (2 tabs)")
                return 'NORTH'
            else:
                logger.info(f"  → Detected SOUTH format (single sheet)")
                return 'SOUTH'
        except Exception as e:
            logger.warning(f"  → Error detecting format: {e}, defaulting to SOUTH")
            return 'SOUTH'

    def read_north_format(self, file_path: Path):
        """Read North format file (2 tabs: Water + Electric)"""
        logger.info(f"  → Reading NORTH format with 2 tabs...")

        all_records = []

        # Read Water tab
        logger.info(f"    → Processing '{config.NORTH_WATER_SHEET}' tab...")
        water_df = pd.read_excel(file_path, sheet_name=config.NORTH_WATER_SHEET, header=config.NORTH_HEADER_ROW)
        logger.info(f"      Found {len(water_df)} water records")

        # Map North columns to standardized South columns
        water_standardized = pd.DataFrame()
        water_standardized[config.CS_COLUMNS['site']] = water_df.get(config.CS_COLUMNS_NORTH_WATER['site'])
        water_standardized[config.CS_COLUMNS['tenant']] = water_df.get(config.CS_COLUMNS_NORTH_WATER['tenant'])
        water_standardized[config.CS_COLUMNS['unit']] = water_df.get(config.CS_COLUMNS_NORTH_WATER['unit'])
        water_standardized[config.CS_COLUMNS['amount']] = water_df.get(config.CS_COLUMNS_NORTH_WATER['amount'])
        water_standardized[config.CS_COLUMNS['quantity']] = water_df.get(config.CS_COLUMNS_NORTH_WATER['wsc_m3'])

        # Parse duration for start/end dates
        if config.CS_COLUMNS_NORTH_WATER['duration'] in water_df.columns:
            duration_parsed = water_df[config.CS_COLUMNS_NORTH_WATER['duration']].apply(self.parse_duration_north)
            water_standardized[config.CS_COLUMNS['start_date']] = duration_parsed.apply(lambda x: x[0] if x else None)
            water_standardized[config.CS_COLUMNS['end_date']] = duration_parsed.apply(lambda x: x[1] if x else None)

        # Set utility type
        water_standardized[config.CS_COLUMNS['utility_type']] = 'Nước'
        water_standardized['_format'] = 'NORTH'
        water_standardized['_utility_source'] = 'Water'

        # Remove empty rows
        water_standardized = water_standardized.dropna(subset=[config.CS_COLUMNS['site'], config.CS_COLUMNS['unit'], config.CS_COLUMNS['amount']], how='all')
        logger.info(f"      Processed {len(water_standardized)} valid water records")
        all_records.append(water_standardized)

        # Read Electric tab
        logger.info(f"    → Processing '{config.NORTH_ELECTRIC_SHEET}' tab...")
        electric_df = pd.read_excel(file_path, sheet_name=config.NORTH_ELECTRIC_SHEET, header=config.NORTH_HEADER_ROW)
        logger.info(f"      Found {len(electric_df)} electric records")

        # Map North columns to standardized South columns
        electric_standardized = pd.DataFrame()
        electric_standardized[config.CS_COLUMNS['site']] = electric_df.get(config.CS_COLUMNS_NORTH_ELECTRIC['site'])
        electric_standardized[config.CS_COLUMNS['tenant']] = electric_df.get(config.CS_COLUMNS_NORTH_ELECTRIC['tenant'])
        electric_standardized[config.CS_COLUMNS['unit']] = electric_df.get(config.CS_COLUMNS_NORTH_ELECTRIC['unit'])
        electric_standardized[config.CS_COLUMNS['amount']] = electric_df.get(config.CS_COLUMNS_NORTH_ELECTRIC['amount'])
        electric_standardized[config.CS_COLUMNS['quantity']] = electric_df.get(config.CS_COLUMNS_NORTH_ELECTRIC['quantity'])

        # Parse duration for start/end dates
        if config.CS_COLUMNS_NORTH_ELECTRIC['duration'] in electric_df.columns:
            duration_parsed = electric_df[config.CS_COLUMNS_NORTH_ELECTRIC['duration']].apply(self.parse_duration_north)
            electric_standardized[config.CS_COLUMNS['start_date']] = duration_parsed.apply(lambda x: x[0] if x else None)
            electric_standardized[config.CS_COLUMNS['end_date']] = duration_parsed.apply(lambda x: x[1] if x else None)

        # Set utility type
        electric_standardized[config.CS_COLUMNS['utility_type']] = 'Điện'
        electric_standardized['_format'] = 'NORTH'
        electric_standardized['_utility_source'] = 'Electric'

        # Remove empty rows
        electric_standardized = electric_standardized.dropna(subset=[config.CS_COLUMNS['site'], config.CS_COLUMNS['unit'], config.CS_COLUMNS['amount']], how='all')
        logger.info(f"      Processed {len(electric_standardized)} valid electric records")
        all_records.append(electric_standardized)

        # Combine all records
        combined_df = pd.concat(all_records, ignore_index=True)
        logger.info(f"  → Total NORTH records: {len(combined_df)}")

        return combined_df

    def read_south_format(self, file_path: Path):
        """Read South format file (single sheet with Vietnamese headers)"""
        logger.info(f"  → Reading SOUTH format (single sheet)...")

        df = pd.read_excel(file_path, header=config.SOUTH_HEADER_ROW)
        df['_format'] = 'SOUTH'

        logger.info(f"  → Total SOUTH records: {len(df)}")

        return df

    def load_cs_input_files(self, specific_files: Optional[List[str]] = None):
        """Load all CS input files from input folder"""
        logger.info(f"Scanning input folder (recursively): {self.input_folder}")

        # Recursively search for Excel files
        input_files = list(self.input_folder.rglob("*.xlsx")) + list(self.input_folder.rglob("*.xls"))
        input_files = [f for f in input_files if not f.name.startswith('~$')]

        # Filter by specific files if provided
        if specific_files:
            input_files = [f for f in input_files if f.name in specific_files]

        logger.info(f"Found {len(input_files)} files to process")

        all_data = []

        for file_path in input_files:
            logger.info(f"Reading: {file_path.name}")

            try:
                # Detect format
                format_type = self.detect_file_format(file_path)

                # Read based on format
                if format_type == 'NORTH':
                    df = self.read_north_format(file_path)
                else:
                    df = self.read_south_format(file_path)

                # Add metadata
                df['_source_file'] = file_path.name

                all_data.append(df)
                logger.info(f"  Successfully loaded {len(df)} records")

            except Exception as e:
                logger.error(f"  Error reading {file_path.name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        if not all_data:
            raise ValueError("No valid input files found!")

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total rows loaded: {len(combined_df)}")

        return combined_df

    def enrich_data(self, cs_df):
        """Enrich CS data with UFL lookups"""
        logger.info("Enriching data with UFL lookups...")
        logger.info("=== MATCHING FLOW: Site → Subsidiary → Unit → UFL Data ===")

        enriched_rows = []
        validation_issues = []

        for idx, row in cs_df.iterrows():
            site = row.get(config.CS_COLUMNS['site'])
            tenant = row.get(config.CS_COLUMNS['tenant'])
            unit = row.get(config.CS_COLUMNS['unit'])

            enriched_row = row.copy()

            # Store original CS data
            enriched_row['_cs_tenant'] = tenant
            enriched_row['_cs_unit'] = unit

            # Step 1: Map Site to Subsidiary Full Name
            subsidiary_code, subsidiary_full_name = self.get_subsidiary_full_name(
                site, self.master_data['subsidiary_config']
            )

            if not subsidiary_full_name:
                validation_issues.append({
                    'row': idx + 2,
                    'issue': 'Subsidiary not found',
                    'site': self.safe_str(site)
                })
                logger.error(f"Row {idx + 2}: Subsidiary not found for site '{site}'")
                continue

            # Step 2: Match Unit within Subsidiary in UFL
            ufl_match = self.fuzzy_match_unit_by_subsidiary(
                unit, self.master_data['units'], subsidiary_full_name, config.FUZZY_MATCH_THRESHOLD
            )

            if ufl_match is not None:
                # Step 3: Get EVERYTHING from UFL record
                enriched_row['_customer_id'] = ufl_match[config.UNIT_COLUMNS['tenant']]
                enriched_row['_ufl_unit_name'] = ufl_match[config.UNIT_COLUMNS['unit_name']]
                enriched_row['_unit_internal_id'] = ufl_match[config.UNIT_COLUMNS['internal_id']]
                enriched_row['_master_record_id'] = ufl_match[config.UNIT_COLUMNS['master_record_id']]
                enriched_row['_plc_id'] = ufl_match[config.UNIT_COLUMNS['plc_id']]
                enriched_row['_project_name'] = ufl_match[config.UNIT_COLUMNS['project']]
                enriched_row['_phase_name'] = ufl_match[config.UNIT_COLUMNS['phase']]
                enriched_row['_subsidiary'] = ufl_match.get(config.UNIT_COLUMNS['subsidiary'], '')

                logger.debug(f"Row {idx}: Customer={ufl_match[config.UNIT_COLUMNS['tenant']]}, Unit={ufl_match[config.UNIT_COLUMNS['unit_name']]}")

            else:
                validation_issues.append({
                    'row': idx + 2,
                    'issue': 'Unit not matched in UFL',
                    'unit': self.safe_str(unit),
                    'subsidiary': self.safe_str(subsidiary_code),
                    'cs_tenant': self.safe_str(tenant)
                })
                logger.warning(f"Row {idx + 2}: Unit '{unit}' not matched in UFL for subsidiary '{subsidiary_code}'")

            # Step 4: Get utility item mapping
            utility_type = row.get(config.CS_COLUMNS['utility_type'])
            utility_subtype = row.get(config.CS_COLUMNS.get('utility_subtype'))

            if pd.notna(utility_type):
                utility_type_upper = str(utility_type).strip().upper()
                utility_info = self.master_data['utility_mapping'][
                    self.master_data['utility_mapping']['Utility_Type_VI'].str.upper() == utility_type_upper
                ]
            else:
                utility_info = pd.DataFrame()

            # Match subtype if provided
            if utility_subtype and not pd.isna(utility_subtype) and str(utility_subtype).strip() != '':
                utility_subtype_upper = str(utility_subtype).strip().upper()
                utility_with_subtype = utility_info[
                    utility_info['Utility_SubType_VI'].str.upper() == utility_subtype_upper
                ]

                if not utility_with_subtype.empty:
                    utility_info = utility_with_subtype
                else:
                    utility_info = utility_info[
                        utility_info['Utility_SubType_VI'].isna() | (utility_info['Utility_SubType_VI'] == '')
                    ]
            else:
                if not utility_info.empty:
                    utility_info = utility_info[
                        utility_info['Utility_SubType_VI'].isna() | (utility_info['Utility_SubType_VI'] == '')
                    ]

            if not utility_info.empty:
                enriched_row['_item_erp'] = utility_info.iloc[0]['Item_ERP']
                enriched_row['_units'] = utility_info.iloc[0]['Units']
                enriched_row['_memo_main_prefix'] = utility_info.iloc[0]['Memo_Main_Prefix']
                enriched_row['_memo_line_en_template'] = utility_info.iloc[0]['Memo_Line_EN_Template']
                enriched_row['_memo_line_vi_template'] = utility_info.iloc[0]['Memo_Line_VI_Template']
                enriched_row['_memo_detail_vi_template'] = utility_info.iloc[0]['Memo_Detail_VI_Template']
            else:
                logger.warning(f"Row {idx}: No utility mapping found for type '{utility_type}'")

            enriched_rows.append(enriched_row)

        enriched_df = pd.DataFrame(enriched_rows)

        logger.info(f"Enrichment complete. {len(validation_issues)} validation issues found.")

        return enriched_df, validation_issues

    def generate_invoice_numbers(self, df):
        """Generate invoice numbers in format: INV/SUB/YYYYMMDD_XX"""
        logger.info("Generating invoice numbers...")

        processing_date = config.get_processing_date()
        date_str = processing_date.strftime('%Y%m%d')

        # Group by subsidiary and file
        df['_invoice_base'] = 'INV/' + df[config.CS_COLUMNS['site']].astype(str) + '/' + date_str

        # Sequential numbering by customer
        df['_customer_first_row'] = df.groupby(['_invoice_base', '_customer_id']).cumcount()
        df['_customer_seq'] = df[df['_customer_first_row'] == 0].groupby('_invoice_base').cumcount() + 1
        df['_customer_seq'] = df.groupby(['_invoice_base', '_customer_id'])['_customer_seq'].transform('first')
        df['_customer_seq'] = df['_customer_seq'].ffill().astype(int)

        df['_invoice_number'] = df['_invoice_base'] + '_' + df['_customer_seq'].astype(str).str.zfill(2)

        logger.info(f"Generated {df['_invoice_number'].nunique()} unique invoice numbers")

        return df

    def build_erp_output(self, df):
        """Build final ERP CSV output with all 45 columns"""
        logger.info("Building ERP output format...")

        processing_date = config.get_processing_date()
        doc_date_str = processing_date.strftime('%d/%m/%Y')
        due_date_str = config.get_due_date(processing_date).strftime('%d/%m/%Y')

        output_rows = []

        # Group by invoice number
        for invoice_num, group in df.groupby('_invoice_number', sort=False):

            # Get subsidiary config
            site = str(group.iloc[0][config.CS_COLUMNS['site']]).strip()
            sub_config_match = self.master_data['subsidiary_config'][
                self.master_data['subsidiary_config']['Subsidiary'].str.strip() == site
            ]

            if sub_config_match.empty:
                logger.warning(f"No subsidiary config found for: {site}")
                continue

            sub_config = sub_config_match.iloc[0]

            # Generate line items
            line_id = 1
            for _, row in group.iterrows():

                # Format dates
                try:
                    start_date_obj = pd.to_datetime(row[config.CS_COLUMNS['start_date']], dayfirst=True)
                    end_date_obj = pd.to_datetime(row[config.CS_COLUMNS['end_date']], dayfirst=True)
                    start_date = start_date_obj.strftime('%d/%m/%Y')
                    end_date = end_date_obj.strftime('%d/%m/%Y')
                except:
                    start_date = doc_date_str
                    end_date = doc_date_str

                # Use UFL unit name
                unit = row.get('_ufl_unit_name')
                if pd.isna(unit):
                    unit = str(row.get(config.CS_COLUMNS['unit']))
                else:
                    unit = str(unit)

                # Build memo fields
                memo_main_prefix = row.get('_memo_main_prefix', '')
                if pd.isna(memo_main_prefix):
                    memo_main_prefix = 'Utility'
                memo_main_prefix = str(memo_main_prefix).replace(' from', '').strip()
                memo_main = f"{memo_main_prefix} from {start_date} to {end_date}"

                memo_line_en_template = row.get('_memo_line_en_template', '')
                if pd.isna(memo_line_en_template) or not isinstance(memo_line_en_template, str):
                    memo_line_en = f"Utility of {unit} From {start_date} To {end_date}"
                else:
                    memo_line_en = memo_line_en_template.format(
                        unit=unit, start=start_date, end=end_date
                    )

                memo_line_vi_template = row.get('_memo_line_vi_template', '')
                if pd.isna(memo_line_vi_template) or not isinstance(memo_line_vi_template, str):
                    memo_line_vi = f"Chi phí tiện ích {unit} Từ ngày {start_date} Đến ngày {end_date}"
                else:
                    memo_line_vi = memo_line_vi_template.format(
                        unit=unit, start=start_date, end=end_date
                    )

                memo_detail_vi_template = row.get('_memo_detail_vi_template', '')
                if pd.isna(memo_detail_vi_template) or not isinstance(memo_detail_vi_template, str):
                    memo_detail_vi = f"Phí quản lý tiện ích {unit} từ ngày {start_date} đến ngày {end_date}"
                else:
                    memo_detail_vi = memo_detail_vi_template.format(
                        unit=unit, start=start_date, end=end_date
                    )

                # Get numeric values
                quantity_raw = row.get(config.CS_COLUMNS['quantity'])
                quantity = float(quantity_raw) if pd.notna(quantity_raw) and quantity_raw != '' else 0

                rate = row.get(config.CS_COLUMNS.get('rate'), 0)
                amount = row.get(config.CS_COLUMNS['amount'], 0)

                try:
                    rate = float(rate) if not pd.isna(rate) else 0
                except:
                    rate = 0

                try:
                    amount = float(amount) if not pd.isna(amount) else 0
                except:
                    amount = 0

                # Get UFL fields
                phase_str = str(row.get('_phase_name', '')) if not pd.isna(row.get('_phase_name')) else ''
                project_str = str(row.get('_project_name', '')) if not pd.isna(row.get('_project_name')) else ''

                master_record_id = row.get('_master_record_id', '')
                unit_internal_id = row.get('_unit_internal_id', '')
                plc_id = row.get('_plc_id', '')

                master_record_id = str(int(master_record_id)) if pd.notna(master_record_id) and master_record_id != '' else ''
                unit_internal_id = str(int(unit_internal_id)) if pd.notna(unit_internal_id) and unit_internal_id != '' else ''
                plc_id = str(plc_id) if pd.notna(plc_id) else ''

                # Build ERP row
                erp_row = {
                    'EXTERNAL ID': invoice_num,
                    'E-INVOICE PREFIX': '',
                    'E-INVOICE NUMBER': '',
                    'E-INVOICE INTEGRATION': 'T',
                    'CUSTOMER': row.get('_customer_id', ''),
                    'PHASE': phase_str,
                    'PROJECT': project_str,
                    'SUBSIDIARY': sub_config['Subsidiary_Full_Name'],
                    'DEPARTMENT': sub_config['Department'],
                    'BUDGET CODE': sub_config['Budget_Code'],
                    'CURRENCY': sub_config.get('Currency', 'VND'),
                    'EXCHANGE RATE': sub_config.get('Exchange_Rate', 1),
                    'PROFORMA INVOICE TYPE': 'Utility',
                    'MEMO MAIN': memo_main,
                    'DATE': doc_date_str,
                    'DOCUMENT DATE': doc_date_str,
                    'DUE DATE': due_date_str,
                    'BUDGET PERIOD': doc_date_str,
                    'BUDGET PERIOD(1)': doc_date_str,
                    'APPROVAL STATUS': 'Pending Approval',
                    'E-INVOICE FORM': 1,
                    'LINE ID': line_id,
                    'ITEM': row.get('_item_erp', ''),
                    '  QUANTITY  ': quantity,
                    'UNITS': row.get('_units', ''),
                    ' RATE ': rate,
                    ' AMOUNT ': amount,
                    'TAX CODE': sub_config.get('Tax_Code', 'HHDV:HHDV_VAT10'),
                    'MEMO (LINE) - Anh': memo_line_en,
                    'MEMO (VI)': memo_line_vi,
                    'Mô tả hàng hóa, vật tư (changed)': memo_detail_vi,
                    'UNIT NO': unit,
                    'MASTER RECORD ID': master_record_id,
                    'Unit for lease': unit_internal_id,
                    'PREMISE LEASE CONTRACT': plc_id,
                    'BILL START DATE': start_date,
                    'BILL END DATE': end_date,
                    'BILL START DATE(1)': start_date,
                    'BILL END DATE(1)': end_date,
                    'SUB. BENEFICIARY NAME': sub_config.get('Beneficiary_Name', ''),
                    'BANK ACCOUNT': sub_config.get('Bank_Account', ''),
                    'SWIFT CODE': sub_config.get('Swift_Code', ''),
                    'PHASE(1)': phase_str,
                    'PROJECT(1)': project_str,
                    'BUDGET CODE(1)': sub_config['Budget_Code']
                }

                output_rows.append(erp_row)
                line_id += 1

        output_df = pd.DataFrame(output_rows, columns=config.ERP_COLUMNS)

        logger.info(f"Generated {len(output_df)} line items across {df['_invoice_number'].nunique()} invoices")

        return output_df

    def generate_validation_report(self, validation_issues: List[Dict], enriched_df: pd.DataFrame) -> Path:
        """Generate Excel validation report"""
        logger.info("Generating validation report...")

        report_file = self.output_folder / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        with pd.ExcelWriter(report_file, engine='openpyxl') as writer:

            # Sheet 1: Issues summary
            if validation_issues:
                issues_df = pd.DataFrame(validation_issues)
                issues_df.to_excel(writer, sheet_name='Issues', index=False)

            # Sheet 2: UFL matching summary
            if '_ufl_unit_name' in enriched_df.columns:
                ufl_cols = ['_source_file', '_cs_tenant', '_cs_unit', '_ufl_unit_name',
                           '_customer_id', '_project_name', '_phase_name', '_master_record_id', '_plc_id']
                ufl_matches = enriched_df[[col for col in ufl_cols if col in enriched_df.columns]].drop_duplicates()
                ufl_matches.to_excel(writer, sheet_name='UFL_Matches', index=False)

            # Sheet 3: Preview
            preview_cols = ['_source_file', config.CS_COLUMNS['site'],
                           '_cs_tenant', '_cs_unit', '_customer_id', '_ufl_unit_name']
            available_preview_cols = [col for col in preview_cols if col in enriched_df.columns]
            preview_df = enriched_df[available_preview_cols].head(50)
            preview_df.to_excel(writer, sheet_name='Preview', index=False)

        logger.info(f"Validation report saved: {report_file}")

        return report_file

    def process_billing(self, specific_files: Optional[List[str]] = None) -> Dict:
        """Main processing flow"""
        start_time = datetime.now()

        try:
            # Step 1: Load master data
            self.load_master_data()

            # Step 2: Load CS input files
            cs_df = self.load_cs_input_files(specific_files)
            total_input_records = len(cs_df)

            # Step 3: Enrich data
            enriched_df, validation_issues = self.enrich_data(cs_df)

            # Step 4: Generate validation report
            validation_report = self.generate_validation_report(validation_issues, enriched_df)

            # Step 5: Generate invoice numbers
            enriched_df = self.generate_invoice_numbers(enriched_df)

            # Step 6: Build ERP output
            erp_df = self.build_erp_output(enriched_df)

            # Step 7: Export CSV
            output_file = self.output_folder / f"INV_ERP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # Format numbers
            numeric_cols = ['  QUANTITY  ', ' RATE ', ' AMOUNT ']
            for col in numeric_cols:
                if col in erp_df.columns:
                    erp_df[col] = erp_df[col].apply(lambda x: f"{float(x):.2f}".strip() if pd.notna(x) else "0.00")

            # Export
            erp_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            # Calculate stats
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'message': 'Processing completed successfully',
                'stats': {
                    'total_input_records': total_input_records,
                    'total_invoices': enriched_df['_invoice_number'].nunique(),
                    'total_line_items': len(erp_df),
                    'validation_issues_count': len(validation_issues),
                    'processing_time_seconds': processing_time
                },
                'output_file': output_file.name,
                'validation_report': validation_report.name,
                'validation_issues': validation_issues
            }

        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            return {
                'success': False,
                'message': 'Processing failed',
                'error': str(e)
            }
