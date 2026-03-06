import sys
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.application.finance.cash_report.cash_report_service import (
    CashReportService,
    SETTLEMENT_PATTERNS_COMPILED,
)
from app.application.finance.cash_report.movement_data_writer import MovementTransaction


def main() -> None:
    service = CashReportService.__new__(CashReportService)
    service._settlement_patterns_compiled = SETTLEMENT_PATTERNS_COMPILED

    description = "Rut tien gui online tren BIDV iBank"
    account = "2221133456"
    amount = Decimal("4000087671")

    nature = service._classify_settlement_receipt_nature(
        description=description,
        amount=amount,
        account=account,
    )
    principal, interest = CashReportService._split_settlement_amount(amount)

    print("Target case")
    print(f"nature={nature}")
    print(f"principal={principal}")
    print(f"interest={interest}")

    assert nature == "Internal transfer in"
    assert principal == Decimal("4000000000")
    assert interest == Decimal("87671")

    settlement_tx = MovementTransaction(
        source="NS",
        bank="BIDV",
        account=account,
        date=None,
        description=description,
        debit=principal,
        credit=None,
        nature=nature,
    )
    can_run_settlement = service._is_settlement_candidate(settlement_tx)

    print(f"can_run_settlement={can_run_settlement}")
    assert can_run_settlement is True

    exception_amount = Decimal("8000526028")
    exception_nature = service._classify_settlement_receipt_nature(
        description=description,
        amount=exception_amount,
        account=account,
    )

    print("Known outlier")
    print(f"nature={exception_nature}")
    assert exception_nature == "Receipt from tenants"

    print("PASS")


if __name__ == "__main__":
    main()
