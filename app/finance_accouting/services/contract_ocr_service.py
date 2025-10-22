"""Contract OCR service for extracting information from contract documents."""
import json
import time
import os
from pathlib import Path
from typing import Union, Optional, List

from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF
import pytesseract

from ..models.contract_schemas import ContractInfo, ContractExtractionResult, RatePeriod
from ..utils.logging_config import get_logger
from .contract_validator import ContractValidator

logger = get_logger(__name__)


class ContractOCRService:
    """Service for reading and extracting information from contract documents using Tesseract OCR and OpenAI text extraction."""

    def __init__(self, api_key: Optional[str] = None, enable_validation: bool = True):
        """Initialize the OCR service with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.pdf_max_pages = 10  # Maximum pages to process from PDF

        # Configure Tesseract path if needed (for Windows or custom installations)
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Chunking configuration for long contracts
        self.chunk_size = 15000  # Characters per chunk (~3,750 words)
        self.chunk_overlap = 500  # Overlap between chunks to avoid missing context

        # Validation
        self.enable_validation = enable_validation
        self.validator = ContractValidator() if enable_validation else None

        # Log and print model information prominently
        model_info = f"Contract OCR Service initialized with Tesseract OCR + {self.model} (validation: {enable_validation})"
        logger.info(model_info)
        print("\n" + "="*80)
        print(f"🤖 AI MODEL: {self.model}")
        print("="*80 + "\n")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for processing long contracts.

        Args:
            text: The full text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def _are_all_fields_extracted(self, data: dict) -> bool:
        """
        Check if all required fields have been extracted.
        Includes both legacy fields and new multilingual fields.

        Args:
            data: Extracted data dictionary

        Returns:
            True if all fields are present and non-null
        """
        # Core required fields (original + new critical fields)
        required_fields = [
            'type',
            'tenant',
            'gla_for_lease',
            'rate_periods',
            'customer_name',
            'contract_number',
            'contract_date'
        ]

        # Check basic fields
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == '':
                return False

        # Check rate_periods has at least one entry with required fields
        if not isinstance(data['rate_periods'], list) or len(data['rate_periods']) == 0:
            return False

        # Check that at least one rate period has all required fields
        for period in data['rate_periods']:
            if not isinstance(period, dict):
                continue
            period_fields = ['start_date', 'end_date', 'monthly_rate_per_sqm', 'total_monthly_rate']
            has_all = all(period.get(f) is not None and period.get(f) != '' for f in period_fields)
            if has_all:
                return True

        return False

    def _calculate_service_charge(self, data: dict) -> str:
        """
        Calculate total service charge based on extracted data.

        Logic:
        - If service_charge_applies_to == "rent_free_only": Calculate for rent-free periods only
        - If service_charge_applies_to == "all_periods": Calculate for all periods
        - If service_charge_applies_to == "not_applicable": Return None

        Formula:
        service_charge_total = service_charge_rate × number_of_months × (gfa or gla_for_lease)

        Args:
            data: Extracted contract data with rate_periods, service_charge_rate, etc.

        Returns:
            Total service charge as string, or None if not applicable
        """
        try:
            # Check if service charge is applicable
            applies_to = data.get('service_charge_applies_to')
            if not applies_to or applies_to == 'not_applicable':
                return None

            # Get service charge rate
            service_charge_rate = data.get('service_charge_rate')
            if not service_charge_rate:
                return None

            # Get area (prefer gfa, fallback to gla_for_lease)
            area = data.get('gfa') or data.get('gla_for_lease')
            if not area:
                logger.warning("Cannot calculate service charge: no GFA or GLA found")
                return None

            # Convert to float for calculation
            rate = float(service_charge_rate)
            area_float = float(area)

            # Get rate periods
            rate_periods = data.get('rate_periods', [])
            if not rate_periods:
                return None

            total_months = 0

            if applies_to == 'rent_free_only':
                # Calculate months where foc_from and foc_to are set (rent-free periods)
                for period in rate_periods:
                    if not isinstance(period, dict):
                        continue

                    foc_from = period.get('foc_from')
                    foc_to = period.get('foc_to')

                    # Check if this period has FOC dates
                    if foc_from and foc_to:
                        try:
                            from datetime import datetime
                            start = datetime.strptime(foc_from, '%m-%d-%Y')
                            end = datetime.strptime(foc_to, '%m-%d-%Y')
                            months = ((end.year - start.year) * 12 + end.month - start.month) + 1
                            total_months += months
                            logger.info(f"Found FOC period: {foc_from} to {foc_to} ({months} month(s))")
                        except Exception as e:
                            logger.warning(f"Error calculating FOC months for period {foc_from} to {foc_to}: {e}")

            elif applies_to == 'all_periods':
                # Calculate total months across all periods
                for period in rate_periods:
                    if not isinstance(period, dict):
                        continue

                    start_date = period.get('start_date')
                    end_date = period.get('end_date')

                    if start_date and end_date:
                        try:
                            from datetime import datetime
                            start = datetime.strptime(start_date, '%m-%d-%Y')
                            end = datetime.strptime(end_date, '%m-%d-%Y')
                            months = ((end.year - start.year) * 12 + end.month - start.month) + 1
                            total_months += months
                        except Exception as e:
                            logger.warning(f"Error calculating months for period {start_date} to {end_date}: {e}")

            if total_months == 0:
                logger.info(f"Service charge applicable to '{applies_to}' but no matching periods found")
                return "0"

            # Calculate total: rate × months × area
            total_service_charge = rate * total_months * area_float

            logger.info(f"Calculated service charge: {rate}/sqm/month × {total_months} months × {area_float} sqm = {total_service_charge}")

            return str(round(total_service_charge, 2))

        except Exception as e:
            logger.error(f"Error calculating service charge: {e}", exc_info=True)
            return None

    def _merge_extracted_data(self, existing: dict, new_data: dict) -> dict:
        """
        Merge newly extracted data with existing data, filling in missing fields.

        Args:
            existing: Previously extracted data
            new_data: Newly extracted data from current chunk

        Returns:
            Merged data dictionary
        """
        for key, value in new_data.items():
            if key == 'rate_periods':
                # Merge rate_periods arrays
                if 'rate_periods' not in existing or not existing['rate_periods']:
                    existing['rate_periods'] = value if isinstance(value, list) else []
                elif isinstance(value, list):
                    # Add new periods that don't already exist
                    existing_periods = existing.get('rate_periods', [])
                    for new_period in value:
                        # Check if this period already exists (by start_date)
                        if not any(p.get('start_date') == new_period.get('start_date')
                                  for p in existing_periods):
                            existing_periods.append(new_period)
                    existing['rate_periods'] = existing_periods
            else:
                # Only update if existing field is null/empty and new value is not
                if (key not in existing or existing[key] is None or existing[key] == '') and value:
                    existing[key] = value

        return existing

    def _extract_text_from_image(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from image using Tesseract OCR.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text string
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config='--psm 6')  # PSM 6: Assume uniform block of text
            logger.info(f"Extracted {len(text)} characters from image")

            # Print OCR output to terminal
            print("\n" + "="*80)
            print("📄 OCR EXTRACTED TEXT FROM IMAGE:")
            print("="*80)
            print(text)
            print("="*80 + "\n")

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""

    def _extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from PDF using PyMuPDF and Tesseract OCR.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Combined text from all pages
        """
        pdf_document = fitz.open(pdf_path)
        all_text = []

        total_pages = len(pdf_document)
        pages_to_process = min(total_pages, self.pdf_max_pages)

        logger.info(f"Processing {pages_to_process} page(s) from PDF with {total_pages} total pages")

        for page_num in range(pages_to_process):
            page = pdf_document[page_num]

            # First try to extract text directly (for text-based PDFs)
            page_text = page.get_text()

            # If no text found or very little text, use OCR on the page image
            if not page_text or len(page_text.strip()) < 50:
                logger.info(f"Page {page_num + 1}: Using OCR (insufficient text found)")
                # Render page to image with high DPI for better OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Extract text using Tesseract
                page_text = pytesseract.image_to_string(img, config='--psm 6')
                logger.info(f"Page {page_num + 1}: Extracted {len(page_text)} characters via OCR")

                # Print OCR output to terminal
                print("\n" + "="*80)
                print(f"📄 OCR EXTRACTED TEXT FROM PDF PAGE {page_num + 1}:")
                print("="*80)
                print(page_text)
                print("="*80 + "\n")
            else:
                logger.info(f"Page {page_num + 1}: Extracted {len(page_text)} characters (text-based PDF)")

                # Print direct text extraction to terminal
                print("\n" + "="*80)
                print(f"📄 DIRECT TEXT EXTRACTION FROM PDF PAGE {page_num + 1}:")
                print("="*80)
                print(page_text)
                print("="*80 + "\n")

            all_text.append(page_text)

        pdf_document.close()

        combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
        logger.info(f"Total extracted text: {len(combined_text)} characters")

        # Print complete combined text
        print("\n" + "="*80)
        print(f"📋 COMPLETE COMBINED TEXT ({len(combined_text)} characters):")
        print("="*80)
        print(combined_text)
        print("="*80 + "\n")

        return combined_text.strip()

    def _create_extraction_prompt(self, extracted_text: str) -> str:
        """
        Create the prompt for information extraction from OCR text.
        Supports multilingual contracts (Vietnamese, English, Chinese).

        Args:
            extracted_text: The OCR-extracted text from the contract

        Returns:
            Formatted prompt string
        """
        return f"""You are an expert contract analyst specializing in multilingual lease contracts (Vietnamese, English, Chinese). Analyze the following contract text that was extracted via OCR and extract ALL of the following information in JSON format:

MULTILINGUAL SUPPORT:
- Contracts may be in Vietnamese, English, Chinese, or mixed languages
- Extract the exact text as written (preserve original language in text fields)
- Convert dates to MM-DD-YYYY format regardless of original format
- Strip currency symbols from all monetary values

REQUIRED FIELDS (EXISTING):
1. type: Type of lease: Rent, Fit out, Service Charge in Rent Free, Rent Free, or Other
2. tenant: Tenant name (person or company name) - also check "bên đi thuê" (Vietnamese)
3. gla_for_lease: GLA (Gross Leasable Area) for lease in square meters (as string, numeric value only)
4. rate_periods: Array of rate periods, where each period has:
   - start_date: Period start date (MM-DD-YYYY as string)
   - end_date: Period end date (MM-DD-YYYY as string)
   - monthly_rate_per_sqm: Monthly rate per sqm for this period (as string, no currency symbol)
     * IMPORTANT: Extract the RAW monthly rate per sqm for this period (do NOT calculate totals, Python will do that)
     * This is the rent rate charged per square meter per month
     * Vietnamese keywords: "Đơn Giá Thuê", "giá thuê/m²/tháng", "VNĐ/m²/tháng"
     * English keywords: "Rental rate", "rent per sqm", "per square meter per month"
     * Chinese keywords: "租金单价", "每平方米每月"
   - total_monthly_rate: DEPRECATED - set to null (Python will calculate: monthly_rate_per_sqm × gfa)
   - num_months: Number of billing months in this period (as string)
     * Calculate based on the contract's billing structure (e.g., "12", "11", "1")
     * For periods like "09-15-2025 to 09-14-2026", this is typically "12" (12 monthly payments)
   - foc_from: FOC start date within this period (MM-DD-YYYY as string, optional)
   - foc_to: FOC end date within this period (MM-DD-YYYY as string, optional)
   - foc_num_months: Number of FOC months (as string, optional)
     * Calculate the number of months covered by the FOC period
     * For "10-01-2026 to 10-31-2026", this is "1"
     * For "09-15-2025 to 11-14-2025", this is "2"
   - service_charge_rate_per_sqm: Service charge rate per sqm per month for this period (as string, no currency symbol)
     * CRITICAL: Extract the RAW service charge rate for EVERY PERIOD - do NOT leave any period blank
     * IMPORTANT: Extract the service charge rate for THIS SPECIFIC PERIOD (do NOT calculate totals, Python will do that)
     * The service charge rate should be extracted for ALL periods, regardless of whether it's included in rent or not
     * Look for the service charge rate in the contract and apply it to ALL rate periods:
       - Find the base service charge rate (e.g., "10288" VND/m²/month)
       - Extract this rate for EVERY period, including:
         * Fit-out periods
         * FOC (rent-free) periods
         * Regular paid rent periods
     * Look for escalation clauses: "Phí Dịch Vụ sẽ tăng X% mỗi năm" (service charge increases X% per year)
       - If escalation exists, calculate DIFFERENT rates for each year/period
       - Example: Year 1: "10288", Year 2: "10699.52" (10288×1.04), Year 3: "11127.50" (10699.52×1.04)
       - Calculate the escalated rate for each period/year and extract as a string
     * Vietnamese keywords: "Phí Dịch Vụ", "phí quản lý", "VNĐ/m²/tháng"
     * English keywords: "Service charge", "management fee", "per sqm per month"
     * Chinese keywords: "服务费", "管理费", "每平方米每月"
     * NOTE: Even if the contract states "Phí Dịch Vụ đã được bao gồm trong Đơn Giá Thuê" (service charge included in rent),
             STILL extract the service charge rate so the user can see it separately
     * Only use null if the contract genuinely has NO service charge mentioned at all
     * NOTE: Python will calculate the total monthly amount: service_charge_rate_per_sqm × gfa

   CRITICAL: LOOK FOR RENT-FREE PERIODS SECTION AND DISTRIBUTION RULES!
   - Vietnamese contracts often have a dedicated section: "THỜI HẠN MIỄN TIỀN THUÊ" or "Thời hạn miễn giảm tiền thuê"
   - This section specifies when FOC (Free of Charge) months occur during the lease term
   - IMPORTANT: Read HOW the rent-free periods are DISTRIBUTED throughout the lease term
   - Common patterns:
     * "4 tháng và được miễn sẽ được phân bổ vào tháng thuê thứ 13, 25, 37, và 49 của Thời Hạn Thuê"
       = 4 FOC months distributed at lease months 13, 25, 37, and 49
     * "miễn tiền thuê 2 tháng đầu" = first 2 months are FOC
     * "1 tháng miễn thuê sau mỗi 12 tháng" = 1 FOC month after every 12 months

   CRITICAL FORMATTING RULE - FOC AS FIELDS WITHIN RATE_PERIODS:
   - DO NOT create separate rate_period entries for FOC months
   - DO NOT split a rate period just because it contains a FOC month
   - INSTEAD: Add foc_from and foc_to fields to the rate_period that CONTAINS the FOC month
   - The monthly_rate_per_sqm should be the PAID rent rate (not "0")
   - Keep the original rate period dates intact - only add foc_from/foc_to fields
   - Each rate_period represents a rental rate escalation period (typically 12 months)
   - FOC months occur WITHIN these periods and should not cause period splits

   STEP-BY-STEP FOC EXTRACTION:
   1. FIRST: Extract the rate escalation schedule from the rent table (usually shows Year 1, Year 2, etc. with rates)
   2. SECOND: Find "THỜI HẠN MIỄN TIỀN THUÊ" section and identify which lease months are FOC (e.g., months 13, 25, 37, 49)
   3. Calculate the calendar dates for each FOC month:
      - Month 13 from handover 11-01-2024 = 11-01-2024 + 12 months = 12-01-2025 to 12-31-2025
      - Month 25 from handover 11-01-2024 = 11-01-2024 + 24 months = 12-01-2026 to 12-31-2026
      - Month 37 from handover 11-01-2024 = 11-01-2024 + 36 months = 12-01-2027 to 12-31-2027
      - Month 49 from handover 11-01-2024 = 11-01-2024 + 48 months = 12-01-2028 to 12-31-2028
   4. For each rate_period from the rent table, check if any FOC month falls within its date range
   5. If a FOC month is within the period, add foc_from and foc_to to that rate_period
   6. If a rate_period contains NO FOC months, leave foc_from and foc_to as null
   7. DO NOT modify the start_date or end_date of rate periods - keep them as shown in the rent table

   EXAMPLE: Contract with rate table showing Year 1-8 and "4 FOC months at lease months 13, 25, 37, 49":
   - Rate table shows: Year 2 from 12-01-2024 to 10-31-2025 at rate 131440
   - Month 13 (12-01-2025) does NOT fall in this period (which ends 10-31-2025)
   - So Year 2 period has: foc_from: null, foc_to: null

   - Rate table shows: Year 3 from 11-01-2025 to 10-31-2026 at rate 138012
   - Month 13 (12-01-2025 to 12-31-2025) DOES fall in this period
   - So Year 3 period has: foc_from: "12-01-2025", foc_to: "12-31-2025"

NEW REQUIRED FIELDS:
5. customer_name: Customer/tenant name from tenant section
   - Vietnamese: Look for "bên đi thuê", "bên thuê", "khách hàng"
   - English: "tenant", "lessee", "customer"
   - Chinese: "承租方", "租户"

6. contract_number: Contract number (extract exactly as written)
   - Vietnamese: "Số hợp đồng", "HỢP ĐỒNG SỐ", "Mã HĐ"
   - English: "Contract No", "Contract Number", "Agreement No"
   - Chinese: "合同编号", "合同号"
   - Usually on first page, often in header or title

7. contract_date: Contract signing date (MM-DD-YYYY format)
   - Vietnamese: "Ngày ký", "Ngày hợp đồng", "Ký ngày"
   - English: "Date", "Signing Date", "Contract Date"
   - Chinese: "签订日期", "合同日期"
   - Convert formats like "15/01/2025", "01-15-2025", "2025年1月15日" to "01-15-2025"

8. payment_terms_details: Payment frequency ONLY - return "monthly" or "quarterly" (lowercase string)
   - Vietnamese:
     * "Thanh toán hàng tháng", "Tháng", "mỗi tháng" → return "monthly"
     * "Thanh toán hàng quý", "Quý", "mỗi quý", "3 tháng/lần" → return "quarterly"
   - English:
     * "Monthly", "per month", "each month" → return "monthly"
     * "Quarterly", "per quarter", "every 3 months" → return "quarterly"
   - Chinese:
     * "每月", "月付" → return "monthly"
     * "每季度", "季付" → return "quarterly"
   - IMPORTANT: Return ONLY "monthly" or "quarterly" (no other text, no original language)

9. deposit_amount: Total deposit amount ONLY (numeric string, no currency symbols)
   - Vietnamese: "Tiền đặt cọc", "Tiền cọc", "Tiền bảo đảm"
   - English: "Deposit", "Security Deposit", "Guarantee"
   - Chinese: "押金", "保证金"
   - If total stated: extract total only (e.g., "150000000")
   - If only installments listed: sum them (e.g., "75000000 + 75000000" = "150000000")
   - If expressed as months of rent (e.g., "3 tháng tiền thuê"): calculate (3 × monthly_rate × gfa)

10. handover_date: Property handover date (MM-DD-YYYY format)
    - Vietnamese: "Ngày bàn giao", "Ngày giao nhà", "Ngày nhận bàn giao"
    - English: "Handover Date", "Delivery Date", "Possession Date"
    - Chinese: "交接日期", "交付日期"

11. gfa: Gross Floor Area (leasable construction floor area in sqm - DIFFERENT from GLA)
    - Vietnamese: "Diện tích sàn xây dựng", "Diện tích sàn", "Diện tích thuê"
    - English: "GFA", "Gross Floor Area", "Floor Area"
    - Chinese: "建筑面积", "楼面面积"
    - Extract numeric value only (as string)
    - NOTE: GFA and GLA are different fields - extract both if both are mentioned

12. service_charge_rate: Service charge rate per sqm per month (numeric string, no currency)
    - Vietnamese: "Phí dịch vụ", "Chi phí dịch vụ", "Phí quản lý"
    - English: "Service Charge", "Management Fee", "Service Fee"
    - Chinese: "服务费", "管理费"
    - Extract the rate per sqm per month (e.g., "50000")
    - Look for tables or fee schedules

13. service_charge_applies_to: When service charge applies (extract as one of these values ONLY)
    - Return "rent_free_only" if wording indicates:
      * Vietnamese: "Phí dịch vụ áp dụng trong thời gian miễn giảm tiền thuê", "chỉ áp dụng khi miễn thuê"
      * English: "Service charge applies during rent-free period", "applies only during rent-free months"
      * Chinese: "服务费仅适用于免租期"
    - Return "all_periods" if wording indicates:
      * Vietnamese: "Phí dịch vụ áp dụng cho toàn bộ thời gian thuê", "áp dụng suốt thời gian hợp đồng"
      * English: "Service charge applies throughout the entire lease term", "applies for all periods"
      * Chinese: "服务费适用于整个租赁期"
    - Return "not_applicable" if service charge not mentioned or unclear

IMPORTANT INSTRUCTIONS:
- Return the information as a valid JSON object
- ALL values must be strings (even numbers - wrap them in quotes: "250" not 250)
- If a field cannot be determined, use null
- Extract all monetary values as strings without currency symbols (VND, USD, CNY, THB, etc.)
- Extract all dates in MM-DD-YYYY format as strings (NOT YYYY-MM-DD)
- The text may contain OCR errors, so use context to interpret unclear words
- IMPORTANT: Contracts often have different rates for different years/periods - extract ALL rate periods
- Each rate period should have its own start_date, end_date, and rates
- CRITICAL FOR RENT-FREE/FOC PERIODS - READ CAREFULLY:
  * DO NOT create separate rate_period entries for FOC months
  * DO NOT set monthly_rate_per_sqm to "0" for any period
  * INSTEAD: Use foc_from and foc_to fields within the relevant rate_period

  * STEP 1: Find the section "THỜI HẠN MIỄN TIỀN THUÊ" or "Thời hạn miễn giảm tiền thuê"
  * STEP 2: Read HOW the rent-free months are distributed. Look for phrases like:
    - "phân bổ vào tháng thuê thứ X, Y, Z" (distributed to lease months X, Y, Z)
    - "tháng đầu tiên" (first months)
    - "sau mỗi X tháng" (after every X months)
  * STEP 3: Calculate the EXACT calendar dates for each FOC month based on:
    - The handover_date (starting point of the lease, month 1)
    - The distribution rule (which lease months are FOC)
    - Example: handover 11-01-2024, month 13 = 11-01-2024 + 12 months = 12-01-2025 to 12-31-2025
  * STEP 4: For each rate_period, determine if any FOC month falls within that period's date range
  * STEP 5: If yes, add foc_from and foc_to to that rate_period (one FOC month per period typically)
  * STEP 6: Keep monthly_rate_per_sqm as the PAID rate (not "0"), FOC is tracked separately

  * EXAMPLE: "4 tháng miễn thuê phân bổ vào tháng 13, 25, 37, 49" with handover 11-01-2024:
    - Month 13 = 12-01-2025 to 12-31-2025 → Add to period covering Dec 2025
    - Month 25 = 12-01-2026 to 12-31-2026 → Add to period covering Dec 2026
    - Month 37 = 12-01-2027 to 12-31-2027 → Add to period covering Dec 2027
    - Month 49 = 12-01-2028 to 12-31-2028 → Add to period covering Dec 2028

CRITICAL: FIT-OUT PERIOD / GRACE PERIOD / SETUP PERIOD (miễn giảm thời gian lắp đặt):
- Many contracts have a FIT-OUT or SETUP period at the BEGINNING (right after handover)
- This is a grace period where the tenant can install equipment, renovate, setup operations
- During this period, NO RENT is charged (it's a type of FOC)
- This fit-out period should be captured as the FIRST rate_period with FOC fields

IDENTIFYING FIT-OUT PERIOD:
- Vietnamese keywords:
  * "Thời gian lắp đặt hoàn thiện", "Thời Hạn Lắp Đặt", "giai đoạn lắp đặt"
  * "Miễn tiền thuê đợt lắp đặt", "không tính tiền thuê trong thời gian lắp đặt"
  * "Tiền Đặt Cọc cho giai đoạn lắp đặt hoàn thiện" (deposit for fit-out period)
  * "Trong vòng X tháng/ngày... lắp đặt hoàn thiện"
  * "Thời Hạn Miễn Tiền Thuê đối với Bên Thuê để lắp đặt hoàn thiện"
- English keywords:
  * "Fit-out period", "Setup period", "Grace period", "Installation period"
  * "Rent-free for fit-out", "No rent during setup"
  * "Fit-out deposit", "Setup deposit"
- Chinese keywords:
  * "装修期", "免租装修期", "设置期"

HOW TO EXTRACT FIT-OUT PERIOD:
1. Look for the fit-out duration (e.g., "1 tháng" = 1 month, "01 tháng" = 1 month)
2. The fit-out period starts from handover_date
3. Calculate the end date: handover_date + fit-out duration - 1 day
4. Create a rate_period entry with:
   - start_date: handover_date
   - end_date: handover_date + fit-out_months - 1 day (last day of fit-out month)
   - monthly_rate_per_sqm: The REGULAR rent rate (NOT "0")
   - total_monthly_rate: Calculated from regular rate
   - num_months: The fit-out duration (e.g., "1")
   - foc_from: Same as start_date
   - foc_to: Same as end_date
   - foc_num_months: Same as num_months
5. This should be the FIRST rate_period in chronological order
6. The actual paid rent periods start AFTER the fit-out period ends

EXAMPLE: Right Weigh contract
- Handover date: "11-01-2024"
- Fit-out period: "01 tháng" (1 month) mentioned in "Thời Hạn Miễn Tiền Thuê"
- Regular rate for Year 1: "131440" per sqm
- FIT-OUT RATE PERIOD (should be FIRST in rate_periods array):
  {{
    "start_date": "11-01-2024",
    "end_date": "11-30-2024",
    "monthly_rate_per_sqm": "131440",
    "total_monthly_rate": "164800048",
    "num_months": "1",
    "foc_from": "11-01-2024",
    "foc_to": "11-30-2024",
    "foc_num_months": "1"
  }}
- Then Year 1 paid rent period starts from "12-01-2024"

MULTILINGUAL KEYWORDS TO LOOK FOR:
- Vietnamese: "bên đi thuê", "tiền thuê", "diện tích", "m²", "tháng", "năm", "ngày", "tiền đặt cọc", "phí dịch vụ", "miễn giảm", "miễn phí", "không tính tiền", "THỜI HẠN MIỄN TIỀN THUÊ", "lắp đặt hoàn thiện", "giai đoạn lắp đặt", "thời gian lắp đặt"
- English: "tenant", "lessee", "rent", "area", "sqm", "month", "year", "deposit", "service charge", "rent-free", "FOC", "free of charge", "no rent", "RENT-FREE PERIOD", "fit-out", "setup period", "grace period", "installation period"
- Chinese: "承租方", "租金", "面积", "平方米", "月", "年", "押金", "服务费", "免租期", "免费", "装修期", "免租装修期"

EXAMPLE OUTPUT FORMAT:
IMPORTANT: rate_periods MUST be in CHRONOLOGICAL ORDER by start_date!
FOC months are tracked using foc_from and foc_to fields WITHIN each rate_period (NOT as separate periods).
Each rate_period should correspond to ONE rental rate from the contract's rent table.
DO NOT create additional periods or split periods just because they contain FOC months.

Example: RIGHT WEIGH contract - "1 month fit-out + 4 FOC months at lease months 13, 25, 37, 49" with handover 11-01-2024:
The contract has a 1-month fit-out period from handover (11-01-2024 to 11-30-2024).
Then the rent table shows 8 years of rates starting from 12-01-2024.
Month 13 (12-01-2025) falls in Year 2, Month 25 (12-01-2026) falls in Year 3, etc.
Service charge is 9920 VND/m²/month with 4% annual escalation.

{{
  "type": "Rent",
  "tenant": "CONG TY TNHH RIGHT WEIGH",
  "gla_for_lease": "1254.2",
  "rate_periods": [
    {{
      "start_date": "11-01-2024",
      "end_date": "11-30-2024",
      "monthly_rate_per_sqm": "131440",
      "total_monthly_rate": null,
      "num_months": "1",
      "foc_from": "11-01-2024",
      "foc_to": "11-30-2024",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "9920"
    }},
    {{
      "start_date": "12-01-2024",
      "end_date": "10-31-2025",
      "monthly_rate_per_sqm": "131440",
      "total_monthly_rate": null,
      "num_months": "11",
      "foc_from": null,
      "foc_to": null,
      "foc_num_months": null,
      "service_charge_rate_per_sqm": "9920"
    }},
    {{
      "start_date": "11-01-2025",
      "end_date": "10-31-2026",
      "monthly_rate_per_sqm": "138012",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": "12-01-2025",
      "foc_to": "12-31-2025",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "10316.8"
    }},
    {{
      "start_date": "11-01-2026",
      "end_date": "10-31-2027",
      "monthly_rate_per_sqm": "144913",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": "12-01-2026",
      "foc_to": "12-31-2026",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "10729.472"
    }},
    {{
      "start_date": "11-01-2027",
      "end_date": "10-31-2028",
      "monthly_rate_per_sqm": "152158",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": "12-01-2027",
      "foc_to": "12-31-2027",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "11158.650880"
    }},
    {{
      "start_date": "11-01-2028",
      "end_date": "10-31-2029",
      "monthly_rate_per_sqm": "159766",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": "12-01-2028",
      "foc_to": "12-31-2028",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "11604.996915"
    }},
    {{
      "start_date": "11-01-2029",
      "end_date": "10-31-2030",
      "monthly_rate_per_sqm": "167754",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": null,
      "foc_to": null,
      "foc_num_months": null,
      "service_charge_rate_per_sqm": "12069.196792"
    }},
    {{
      "start_date": "11-01-2030",
      "end_date": "10-31-2031",
      "monthly_rate_per_sqm": "176142",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": null,
      "foc_to": null,
      "foc_num_months": null,
      "service_charge_rate_per_sqm": "12551.964664"
    }},
    {{
      "start_date": "11-01-2031",
      "end_date": "10-31-2032",
      "monthly_rate_per_sqm": "184949",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": null,
      "foc_to": null,
      "foc_num_months": null,
      "service_charge_rate_per_sqm": "13054.043251"
    }}
  ],
  "customer_name": "CONG TY TNHH RIGHT WEIGH",
  "contract_number": "BWSCC/PLC/24022",
  "contract_date": "10-01-2024",
  "payment_terms_details": "quarterly",
  "deposit_amount": "824260240",
  "handover_date": "11-01-2024",
  "gfa": "1254.2",
  "service_charge_rate": "9920",
  "service_charge_applies_to": "rent_free_only"
}}

CONTRACT TEXT:
---
{extracted_text}
---

Based on the contract text above, extract ALL 13 required fields and return as valid JSON. Remember to preserve original language in text fields while converting dates to MM-DD-YYYY and removing currency symbols from numbers."""

    def _extract_from_text_chunk(self, text_chunk: str) -> dict:
        """
        Extract data from a single text chunk using LLM.

        Args:
            text_chunk: Text chunk to process

        Returns:
            Extracted data dictionary
        """
        prompt = self._create_extraction_prompt(text_chunk)

        # Print prompt for debugging
        print("\n" + "="*80)
        print("📝 PROMPT SENT TO AI (for debugging):")
        print("="*80)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print(f"\n[Full prompt length: {len(prompt)} characters]")
        print("="*80 + "\n")

        # Log AI call
        logger.info(f"Calling OpenAI API with model: {self.model}")
        print(f"🤖 Calling AI model: {self.model}")

        # Configure parameters based on model type
        try:
            if self.model.startswith("gpt-5"):
                # GPT-5 uses new parameters and extended reasoning
                print(f"   Using GPT-5 parameters: max_completion_tokens=2000, reasoning_effort=high")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=2000,
                    reasoning_effort="high",
                )
            else:
                # GPT-4 and earlier models use traditional parameters
                print(f"   Using GPT-4 parameters: max_tokens=2000, temperature=0.1")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.1,
                )

            # Log response details
            print(f"   Response received. Finish reason: {response.choices[0].finish_reason}")
            content = response.choices[0].message.content

            if not content:
                print(f"   ⚠️ WARNING: Empty response content from {self.model}")
                logger.warning(f"Empty response from {self.model}")
                return {}

            print(f"   Response length: {len(content)} characters")

            # Print AI response for debugging
            print("\n" + "="*80)
            print("🤖 AI RESPONSE (for debugging):")
            print("="*80)
            print(content)
            print("="*80 + "\n")

        except Exception as e:
            print(f"   ❌ ERROR calling {self.model}: {str(e)}")
            logger.error(f"Error calling {self.model}: {str(e)}")
            raise

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def _process_text_with_chunking(self, extracted_text: str) -> dict:
        """
        Process text in chunks, stopping when all fields are found or all text is processed.

        Args:
            extracted_text: The full OCR-extracted text

        Returns:
            Extracted data dictionary with all found fields
        """
        chunks = self._chunk_text(extracted_text)
        merged_data = {}

        logger.info(f"Processing {len(chunks)} chunk(s) of text...")

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}...")

            try:
                chunk_data = self._extract_from_text_chunk(chunk)
                merged_data = self._merge_extracted_data(merged_data, chunk_data)

                # Check if we have all fields
                if self._are_all_fields_extracted(merged_data):
                    num_periods = len(merged_data.get('rate_periods', []))
                    logger.info(f"✓ All required fields found with {num_periods} rate period(s) after processing {i}/{len(chunks)} chunks. Stopping early.")
                    print("\n" + "="*80)
                    print(f"✓ ALL FIELDS EXTRACTED - {num_periods} RATE PERIOD(S) (stopped at chunk {i}/{len(chunks)})")
                    print("="*80 + "\n")
                    break
                else:
                    missing = []
                    for f in ['type', 'tenant', 'gla_for_lease']:
                        if f not in merged_data or not merged_data[f]:
                            missing.append(f)
                    if 'rate_periods' not in merged_data or not merged_data.get('rate_periods'):
                        missing.append('rate_periods')
                    else:
                        num_periods = len(merged_data.get('rate_periods', []))
                        logger.info(f"Found {num_periods} rate period(s) so far...")
                    logger.info(f"Missing fields: {missing}. Continuing to next chunk...")

            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {str(e)}. Continuing...")
                continue

        # Final check
        if self._are_all_fields_extracted(merged_data):
            num_periods = len(merged_data.get('rate_periods', []))
            logger.info(f"✓ Successfully extracted all required fields with {num_periods} rate period(s)")
        else:
            missing = []
            for f in ['type', 'tenant', 'gla_for_lease']:
                if f not in merged_data or not merged_data[f]:
                    missing.append(f)
            if 'rate_periods' not in merged_data or not merged_data.get('rate_periods'):
                missing.append('rate_periods (no valid periods found)')
            logger.warning(f"⚠ Processed all chunks but still missing fields: {missing}")

        return merged_data

    def process_contract(self, file_path: Union[str, Path]) -> ContractExtractionResult:
        """
        Read and extract information from a contract document using Tesseract OCR + OpenAI text extraction.

        Args:
            file_path: Path to the contract document (image or PDF)

        Returns:
            ContractExtractionResult with extracted contract information
        """
        start_time = time.time()
        file_path = Path(file_path)

        logger.info(f"Processing contract: {file_path.name}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return ContractExtractionResult(
                success=False,
                error=f"File not found: {file_path}",
                source_file=str(file_path.name)
            )

        try:
            # Step 1: Extract text using Tesseract OCR
            if file_path.suffix.lower() == '.pdf':
                logger.info(f"Extracting text from PDF using OCR...")
                extracted_text = self._extract_text_from_pdf(file_path)
            else:
                logger.info(f"Extracting text from image using OCR...")
                extracted_text = self._extract_text_from_image(file_path)

            if not extracted_text:
                logger.warning("No text extracted from document")
                return ContractExtractionResult(
                    success=False,
                    error="No text could be extracted from the document",
                    source_file=str(file_path.name),
                    processing_time=time.time() - start_time
                )

            logger.info(f"Successfully extracted {len(extracted_text)} characters of text")

            # Step 2: Process text with chunking (stops when all 7 fields found or all text processed)
            logger.info(f"Processing text with intelligent chunking...")
            extracted_data = self._process_text_with_chunking(extracted_text)

            # Add raw extracted text to the data
            extracted_data["raw_text"] = extracted_text

            # Convert gla_for_lease to string if numeric
            if 'gla_for_lease' in extracted_data and extracted_data['gla_for_lease'] is not None:
                if isinstance(extracted_data['gla_for_lease'], (int, float)):
                    extracted_data['gla_for_lease'] = str(extracted_data['gla_for_lease'])

            # Process rate_periods array
            if 'rate_periods' in extracted_data and isinstance(extracted_data['rate_periods'], list):
                gla = extracted_data.get('gla_for_lease')
                processed_periods = []

                for period in extracted_data['rate_periods']:
                    if not isinstance(period, dict):
                        continue

                    # Convert numeric fields to strings
                    for field in ['monthly_rate_per_sqm', 'total_monthly_rate']:
                        if field in period and period[field] is not None:
                            if isinstance(period[field], (int, float)):
                                period[field] = str(period[field])

                    # Calculate total_monthly_rate if missing but we have the components
                    if (not period.get('total_monthly_rate') and
                        period.get('monthly_rate_per_sqm') and gla):
                        try:
                            rate_per_sqm = float(period['monthly_rate_per_sqm'])
                            gla_float = float(gla)
                            total = rate_per_sqm * gla_float
                            period['total_monthly_rate'] = str(round(total, 2))
                            logger.info(f"Calculated total_monthly_rate for period {period.get('start_date')}: {period['total_monthly_rate']}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not calculate total_monthly_rate for period: {e}")

                    # Convert dict to RatePeriod object
                    try:
                        rate_period_obj = RatePeriod(**period)
                        processed_periods.append(rate_period_obj)
                    except Exception as e:
                        logger.warning(f"Could not create RatePeriod object from {period}: {e}")

                # Replace the list of dicts with list of RatePeriod objects
                extracted_data['rate_periods'] = processed_periods
                logger.info(f"Successfully converted {len(processed_periods)} rate periods to RatePeriod objects")

            # Step 3: Calculate service charge (Python calculation layer)
            logger.info("Calculating service charge based on extracted data...")
            service_charge_total = self._calculate_service_charge(extracted_data)
            if service_charge_total:
                extracted_data['service_charge_total'] = service_charge_total
                logger.info(f"Service charge calculated: {service_charge_total}")
            else:
                extracted_data['service_charge_total'] = None
                logger.info("Service charge not calculated (not applicable or insufficient data)")

            # Convert new numeric fields to strings
            for field in ['gfa', 'deposit_amount', 'service_charge_rate']:
                if field in extracted_data and extracted_data[field] is not None:
                    if isinstance(extracted_data[field], (int, float)):
                        extracted_data[field] = str(extracted_data[field])

            # Sanitize data - convert empty arrays to None for string fields
            string_fields = [
                'contract_title', 'contract_type', 'effective_date', 'expiration_date',
                'contract_value', 'payment_terms', 'termination_clauses', 'governing_law',
                'internal_id', 'id', 'master_record_id', 'plc_id', 'unit_for_lease',
                'type', 'start_date', 'end_date', 'tenant', 'monthly_rate_per_sqm',
                'gla_for_lease', 'total_monthly_rate', 'total_rate', 'status',
                'historical_journal_entry', 'amortization_journal_entry',
                'related_billing_schedule', 'subsidiary', 'ccs_product_type',
                'bwid_project', 'phase', 'raw_text',
                # New fields
                'customer_name', 'contract_number', 'contract_date', 'payment_terms_details',
                'deposit_amount', 'handover_date', 'gfa', 'service_charge_rate',
                'service_charge_applies_to', 'service_charge_total'
            ]

            for field in string_fields:
                if field in extracted_data:
                    value = extracted_data[field]
                    # Convert empty arrays, empty strings, or invalid types to None
                    if isinstance(value, list) and len(value) == 0:
                        extracted_data[field] = None
                    elif isinstance(value, str) and value.strip() == '':
                        extracted_data[field] = None
                    elif value is not None and not isinstance(value, (str, int, float, bool)):
                        extracted_data[field] = None

            # Create ContractInfo object
            contract_info = ContractInfo(**extracted_data)

            # Step 4: Validate extracted data (if enabled)
            if self.enable_validation and self.validator:
                logger.info("Running validation on extracted data...")
                validation_result = self.validator.validate_contract(contract_info)

                errors = validation_result.get('errors', [])
                warnings = validation_result.get('warnings', [])

                if errors:
                    logger.warning(f"Validation errors found: {len(errors)}")
                    for error in errors:
                        logger.warning(f"  - {error}")

                if warnings:
                    logger.info(f"Validation warnings: {len(warnings)}")
                    for warning in warnings:
                        logger.info(f"  - {warning}")

                # Log validation summary
                summary = self.validator.get_validation_summary(validation_result)
                print("\n" + "="*80)
                print("📋 VALIDATION SUMMARY:")
                print("="*80)
                print(summary)
                print("="*80 + "\n")

            processing_time = time.time() - start_time

            logger.info(f"Successfully processed contract: {file_path.name} in {processing_time:.2f}s")

            return ContractExtractionResult(
                success=True,
                data=contract_info,
                processing_time=processing_time,
                source_file=str(file_path.name)
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            logger.error(error_msg)
            return ContractExtractionResult(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time,
                source_file=str(file_path.name)
            )
        except Exception as e:
            error_msg = f"Error processing contract: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ContractExtractionResult(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time,
                source_file=str(file_path.name)
            )

    def process_contracts_batch(self, file_paths: List[Union[str, Path]]) -> List[ContractExtractionResult]:
        """
        Process multiple contract documents in batch.

        Args:
            file_paths: List of paths to contract documents

        Returns:
            List of ContractExtractionResult objects
        """
        logger.info(f"Processing batch of {len(file_paths)} contracts")
        results = []
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing contract {i}/{len(file_paths)}")
            result = self.process_contract(file_path)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch processing complete: {successful}/{len(results)} successful")
        return results
