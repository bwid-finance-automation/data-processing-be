"""Regression test for VIB PDF parser using real Gemini OCR snapshots.

Usage:
  python scripts/test_vib_pdf_regression.py
  python scripts/test_vib_pdf_regression.py --refresh-ocr

Default behavior:
- Load OCR snapshots from sample_for_test/bank_statement/.ocr_snapshots
- Parse with VIB parser
- Validate section-level consistency against statement summary/opening/closing

With --refresh-ocr:
- Re-run Gemini OCR for each PDF and overwrite snapshots first
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is importable when running script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.application.finance.bank_statement_parser.bank_parsers.vib_parser import VIBParser
from app.application.finance.bank_statement_parser.gemini_ocr_service import GeminiOCRService


@dataclass
class SectionResult:
    file_name: str
    section_index: int
    acc_no: str
    currency: str
    tx_headers: int
    tx_matched: int
    summary_count: Optional[int]
    parser_withdrawal: float
    parser_deposit: float
    summary_withdrawal: Optional[float]
    summary_deposit: Optional[float]
    opening: Optional[float]
    closing: Optional[float]
    predicted_closing: Optional[float]
    wd_err: Optional[float]
    dep_err: Optional[float]
    close_err: Optional[float]
    direction_violations: int
    missing_tx_ids: List[str]


def _load_snapshot(snapshot_path: Path) -> Optional[str]:
    if not snapshot_path.exists():
        return None
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return data.get("ocr_text", "")


def _save_snapshot(snapshot_path: Path, file_name: str, text: str, metrics: dict) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "file_name": file_name,
        "ocr_text": text,
        "metrics": metrics,
    }
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _refresh_ocr_snapshots(pdf_files: List[Path], snapshot_dir: Path) -> Dict[str, str]:
    svc = GeminiOCRService()
    ocr_texts: Dict[str, str] = {}

    for pdf in pdf_files:
        pdf_bytes = pdf.read_bytes()
        text, metrics = await svc.extract_text_from_pdf(pdf_bytes, pdf.name)

        snapshot_path = snapshot_dir / f"{pdf.stem}.json"
        _save_snapshot(
            snapshot_path=snapshot_path,
            file_name=pdf.name,
            text=text,
            metrics={
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "total_tokens": metrics.total_tokens,
                "processing_time_ms": metrics.processing_time_ms,
                "model_name": metrics.model_name,
            },
        )
        ocr_texts[pdf.name] = text
        print(
            f"[OCR] {pdf.name}: chars={len(text)} "
            f"tokens={metrics.total_tokens} time_ms={metrics.processing_time_ms:.0f}"
        )

    return ocr_texts


def _get_section_opening_closing(parser: VIBParser, section: str, currency: str) -> Tuple[Optional[float], Optional[float]]:
    opening = parser._extract_section_balance(section, ["số dư đầu", "opening balance"])
    closing = parser._extract_section_balance(section, ["số dư cuối", "ending balance"])

    if opening is None or opening == 0:
        opening = parser._extract_balance_fallback_section(section, ["số dư đầu", "opening balance"], currency)
    if opening is None or opening == 0:
        opening = parser._extract_opening_before_withdrawal(section, currency)

    if closing is None or closing == 0:
        closing = parser._extract_balance_fallback_section(section, ["số dư cuối", "ending balance"], currency)

    return opening, closing


def _direction_violation_count(
    parser: VIBParser,
    tx_headers: List[tuple],
    tx_map: Dict[str, object],
) -> int:
    violations = 0
    for tx_id, _, code in tx_headers:
        tx = tx_map.get(tx_id)
        if not tx:
            continue
        if not parser._is_amount_direction_consistent(code, tx.debit, tx.credit):
            violations += 1
    return violations


def _evaluate_file(parser: VIBParser, file_name: str, text: str) -> List[SectionResult]:
    txs = parser.parse_transactions_from_text(text, file_name)
    sections = parser._split_into_account_sections(text)

    # Group tx by account to avoid collisions when a file has multi-account sections.
    tx_by_acc: Dict[str, Dict[str, object]] = {}
    for tx in txs:
        acc = tx.acc_no or "__UNKNOWN__"
        tx_by_acc.setdefault(acc, {})
        if tx.transaction_id:
            tx_by_acc[acc][tx.transaction_id] = tx

    results: List[SectionResult] = []
    tol = 5_000.0

    for i, section in enumerate(sections):
        acc_no, currency = parser._extract_account_info_from_section(section)
        tx_headers = parser._extract_tx_headers_from_section(section)
        sum_wd, sum_dep, sum_cnt = parser._extract_summary_totals_from_section(section, currency)
        opening, closing = _get_section_opening_closing(parser, section, currency)

        section_tx_map = tx_by_acc.get(acc_no, {})
        if not section_tx_map:
            # Fallback for parsers that may not fill acc_no consistently
            merged = {}
            for m in tx_by_acc.values():
                merged.update(m)
            section_tx_map = merged

        matched = 0
        parser_wd = 0.0
        parser_dep = 0.0
        missing_ids: List[str] = []
        for tx_id, _, _ in tx_headers:
            tx = section_tx_map.get(tx_id)
            if not tx:
                missing_ids.append(tx_id)
                continue
            matched += 1
            parser_wd += tx.debit or 0.0
            parser_dep += tx.credit or 0.0

        predicted_close = None if opening is None else (opening + parser_dep - parser_wd)
        wd_err = None if sum_wd is None else abs(parser_wd - sum_wd)
        dep_err = None if sum_dep is None else abs(parser_dep - sum_dep)
        close_err = None if (closing is None or predicted_close is None) else abs(predicted_close - closing)

        direction_viol = _direction_violation_count(parser, tx_headers, section_tx_map)

        results.append(
            SectionResult(
                file_name=file_name,
                section_index=i + 1,
                acc_no=acc_no,
                currency=currency,
                tx_headers=len(tx_headers),
                tx_matched=matched,
                summary_count=sum_cnt,
                parser_withdrawal=parser_wd,
                parser_deposit=parser_dep,
                summary_withdrawal=sum_wd,
                summary_deposit=sum_dep,
                opening=opening,
                closing=closing,
                predicted_closing=predicted_close,
                wd_err=wd_err,
                dep_err=dep_err,
                close_err=close_err,
                direction_violations=direction_viol,
                missing_tx_ids=missing_ids,
            )
        )

    return results


def _result_failed(r: SectionResult) -> bool:
    tol = 5_000.0
    if r.tx_headers and r.tx_matched != r.tx_headers:
        return True
    if r.summary_count is not None and r.tx_headers and r.summary_count != r.tx_headers:
        return True
    if r.wd_err is not None and r.wd_err > tol:
        return True
    if r.dep_err is not None and r.dep_err > tol:
        return True
    if r.close_err is not None and r.close_err > tol:
        return True
    if r.direction_violations > 0:
        return True
    return False


def _print_report(results: List[SectionResult]) -> int:
    failures = 0
    for r in results:
        failed = _result_failed(r)
        status = "FAIL" if failed else "PASS"
        if failed:
            failures += 1

        print(
            f"[{status}] {r.file_name} | section={r.section_index} "
            f"acc={r.acc_no}/{r.currency} headers={r.tx_headers} matched={r.tx_matched}"
        )
        print(
            f"       parser(wd={r.parser_withdrawal:.0f}, dep={r.parser_deposit:.0f}) "
            f"summary(wd={r.summary_withdrawal}, dep={r.summary_deposit}, cnt={r.summary_count})"
        )
        print(
            f"       balance(open={r.opening}, close={r.closing}, predicted={r.predicted_closing}) "
            f"errors(wd={r.wd_err}, dep={r.dep_err}, close={r.close_err}) "
            f"direction_viol={r.direction_violations}"
        )
        if r.missing_tx_ids:
            print(f"       missing_tx_ids={r.missing_tx_ids}")

    print(f"\nSections checked: {len(results)}, failures: {failures}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="VIB PDF regression test with OCR snapshots")
    parser.add_argument(
        "--folder",
        default="sample_for_test/bank_statement",
        help="Folder containing VIB PDF samples",
    )
    parser.add_argument(
        "--refresh-ocr",
        action="store_true",
        help="Refresh OCR snapshots from Gemini before running checks",
    )
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    pdf_files = sorted(root.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {root}")

    snapshot_dir = root / ".ocr_snapshots"
    ocr_texts: Dict[str, str] = {}

    if args.refresh_ocr:
        ocr_texts = asyncio.run(_refresh_ocr_snapshots(pdf_files, snapshot_dir))
    else:
        for pdf in pdf_files:
            snapshot_path = snapshot_dir / f"{pdf.stem}.json"
            text = _load_snapshot(snapshot_path)
            if text is None:
                raise SystemExit(
                    f"Missing snapshot: {snapshot_path}\n"
                    "Run with --refresh-ocr first."
                )
            ocr_texts[pdf.name] = text

    vib = VIBParser()
    all_results: List[SectionResult] = []
    for pdf in pdf_files:
        text = ocr_texts[pdf.name]
        all_results.extend(_evaluate_file(vib, pdf.name, text))

    failures = _print_report(all_results)
    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
