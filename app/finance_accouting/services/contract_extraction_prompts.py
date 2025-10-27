"""Prompts for contract OCR extraction service."""


def get_contract_extraction_prompt(extracted_text: str) -> str:
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
  {{{{
    "start_date": "11-01-2024",
    "end_date": "11-30-2024",
    "monthly_rate_per_sqm": "131440",
    "total_monthly_rate": "164800048",
    "num_months": "1",
    "foc_from": "11-01-2024",
    "foc_to": "11-30-2024",
    "foc_num_months": "1"
  }}}}
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

{{{{
  "type": "Rent",
  "tenant": "CONG TY TNHH RIGHT WEIGH",
  "gla_for_lease": "1254.2",
  "rate_periods": [
    {{{{
      "start_date": "11-01-2024",
      "end_date": "11-30-2024",
      "monthly_rate_per_sqm": "131440",
      "total_monthly_rate": null,
      "num_months": "1",
      "foc_from": "11-01-2024",
      "foc_to": "11-30-2024",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "9920"
    }}}},
    {{{{
      "start_date": "12-01-2024",
      "end_date": "10-31-2025",
      "monthly_rate_per_sqm": "131440",
      "total_monthly_rate": null,
      "num_months": "11",
      "foc_from": null,
      "foc_to": null,
      "foc_num_months": null,
      "service_charge_rate_per_sqm": "9920"
    }}}},
    {{{{
      "start_date": "11-01-2025",
      "end_date": "10-31-2026",
      "monthly_rate_per_sqm": "138012",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": "12-01-2025",
      "foc_to": "12-31-2025",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "10316.8"
    }}}},
    {{{{
      "start_date": "11-01-2026",
      "end_date": "10-31-2027",
      "monthly_rate_per_sqm": "144913",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": "12-01-2026",
      "foc_to": "12-31-2026",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "10729.472"
    }}}},
    {{{{
      "start_date": "11-01-2027",
      "end_date": "10-31-2028",
      "monthly_rate_per_sqm": "152158",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": "12-01-2027",
      "foc_to": "12-31-2027",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "11158.650880"
    }}}},
    {{{{
      "start_date": "11-01-2028",
      "end_date": "10-31-2029",
      "monthly_rate_per_sqm": "159766",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": "12-01-2028",
      "foc_to": "12-31-2028",
      "foc_num_months": "1",
      "service_charge_rate_per_sqm": "11604.996915"
    }}}},
    {{{{
      "start_date": "11-01-2029",
      "end_date": "10-31-2030",
      "monthly_rate_per_sqm": "167754",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": null,
      "foc_to": null,
      "foc_num_months": null,
      "service_charge_rate_per_sqm": "12069.196792"
    }}}},
    {{{{
      "start_date": "11-01-2030",
      "end_date": "10-31-2031",
      "monthly_rate_per_sqm": "176142",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": null,
      "foc_to": null,
      "foc_num_months": null,
      "service_charge_rate_per_sqm": "12551.964664"
    }}}},
    {{{{
      "start_date": "11-01-2031",
      "end_date": "10-31-2032",
      "monthly_rate_per_sqm": "184949",
      "total_monthly_rate": null,
      "num_months": "12",
      "foc_from": null,
      "foc_to": null,
      "foc_num_months": null,
      "service_charge_rate_per_sqm": "13054.043251"
    }}}}
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
}}}}

CONTRACT TEXT:
---
{extracted_text}
---

Based on the contract text above, extract ALL 13 required fields and return as valid JSON. Remember to preserve original language in text fields while converting dates to MM-DD-YYYY and removing currency symbols from numbers."""
