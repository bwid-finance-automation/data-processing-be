import sys
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.application.finance.cash_report.cash_report_service import (
    CashReportService,
    DEF_REFERENCE_FILE,
    DEF_ENTITY_CODES_FILE,
    ENTITY_CODES,
)
from app.application.finance.cash_report.movement_data_writer import MovementTransaction


def make_service() -> CashReportService:
    service = CashReportService.__new__(CashReportService)
    csv_codes = set()
    for csv_path in (DEF_REFERENCE_FILE, DEF_ENTITY_CODES_FILE):
        csv_codes.update(CashReportService._load_entity_codes_from_csv(csv_path))
    service._entity_codes = frozenset(
        {str(code).strip().lower() for code in ENTITY_CODES if str(code).strip()} | csv_codes
    )
    return service


def assert_dual_entity(service: CashReportService, description: str, is_receipt: bool, expected: str) -> None:
    actual = service._classify_dual_entity_transfer(description, is_receipt)
    print(f"{description} -> {actual}")
    assert actual == expected


def assert_guardrail(service: CashReportService, description: str, debit, credit, expected: str) -> None:
    tx = MovementTransaction(
        source="NS",
        bank="BIDV",
        account="1234567890",
        date=None,
        description=description,
        debit=Decimal(str(debit)) if debit is not None else None,
        credit=Decimal(str(credit)) if credit is not None else None,
        nature="Receipt from tenants" if debit is not None else "Operating expense",
    )
    service._apply_classification_guardrails(tx)
    print(f"guardrail -> {tx.nature}")
    assert tx.nature == expected


def main() -> None:
    service = make_service()

    receipt_cases = [
        "NT2_VC3 chuyen tien hop tac kinh doanh (BCC transfer)",
        "REF222A262144UPNPNS B/O 8620004064 CTY CP PHAT TRIEN CONG NGHIEP HAI PHONG (VIET NAM) F/O 114002874189 CTY CP SAO HOA TOAN QUOC NHH24201003 JHP CONG NGHIEP HP VN_VC3_Payment for Development management fee 2025_INVOICE 00000025",
        "MISC CREDIT | MIR602274581C01 | S4C-VC3-REPAYMENT OF SHL INTEREST/",
        "MISC CREDIT | MIR602274582C01 | VC3-H4H-TRA 1 PHAN LAI VAY SHL/PAR",
        "MISC CREDIT | MIR602274585C01 | BB5REPAYMENT BCC TO VC3",
    ]
    payment_cases = [
        "So GD goc: 10000018 TH2-VC3-TRANSFER TO VC3 PER LOAN AGREEMENT/ CHUYEN TIEN THEO HD VAY",
        "So GD goc: 10000026 BHI-TRANSFER TO VC3 PER LOAN AGREEMENT/ BHI CHO VC3 VAY",
        "VC3 THA TRANSFER TO VC3 PER LOAN AGREEMENT",
        "BBA_VC3_TRANSFER TO VC3 PER LOAN AGREEMENT/CHUYEN TIEN THEO HD VAY",
        "Fund transfer - So GD: 900A26214495UW8B TH2-VC3-TRANSFER TO VC3 PER LOAN AGREEMENT/ CHUYEN TIEN THEO HD VAY",
        "Fund transfer - So GD: 900A262144CB5Y9B BHI-TRANSFER TO VC3 PER LOAN AGREEMENT/ BHI CHO VC3 VAY",
    ]

    for description in receipt_cases:
        assert_dual_entity(service, description, True, "Internal transfer in")

    for description in payment_cases:
        assert_dual_entity(service, description, False, "Internal transfer out")

    assert_guardrail(
        service,
        "MISC CREDIT | MIR602274585C01 | BB5REPAYMENT BCC TO VC3",
        "894888022",
        None,
        "Internal transfer in",
    )
    assert_guardrail(
        service,
        "So GD goc: 10000018 TH2-VC3-TRANSFER TO VC3 PER LOAN AGREEMENT/ CHUYEN TIEN THEO HD VAY",
        None,
        "1000000000",
        "Internal transfer out",
    )

    print("PASS")


if __name__ == "__main__":
    main()
