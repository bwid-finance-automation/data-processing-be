"""
Quick proof: classify two 'CA - TARGET' transactions to show correct Nature.
"""
import sys
import re
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load settlement patterns from CSV
SETTLEMENT_CSV = PROJECT_ROOT / "movement_nature_filter" / "Settlement.csv"
import csv

settlement_patterns = []
with SETTLEMENT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        regex = (row.get("Regex") or "").strip()
        if regex:
            try:
                settlement_patterns.append(re.compile(regex, re.IGNORECASE))
            except re.error:
                pass

HDTG_DEPOSIT_KEYWORDS = [
    "hdtg", "tien gui", "tiet kiem", "timemo", "timect",
    "ky han", "saving", "term deposit", "fixed deposit",
]
_UNIT_1B = Decimal("1000000000")

def classify(description: str, debit: Decimal):
    is_receipt = debit > 0
    print(f"\n{'='*60}")
    print(f"  Description: {description}")
    print(f"  Debit:       {debit:>25,.0f} VND")
    print(f"  Is Receipt:  {is_receipt}")
    print(f"{'='*60}")

    # Step 1: Check settlement patterns
    settlement_match = False
    for p in settlement_patterns:
        if p.search(description):
            settlement_match = True
            print(f"  ✓ Settlement pattern matched: {p.pattern}")
            break

    if not settlement_match:
        print(f"  ✗ No settlement pattern matched")

    # Step 2: Check HDTG keywords + amount >= 1B
    hdtg_match = False
    if debit >= _UNIT_1B:
        desc_lower = description.lower()
        for kw in HDTG_DEPOSIT_KEYWORDS:
            if kw in desc_lower:
                hdtg_match = True
                print(f"  ✓ HDTG keyword '{kw}' + debit >= 1B")
                break

    # Step 3: _classify_from_tfidf logic
    print(f"\n  --- _classify_from_tfidf ---")
    if is_receipt and debit >= _UNIT_1B:
        if hdtg_match:
            print(f"  → HDTG keyword + debit >= 1B → 'Internal transfer in'")
            nature_tfidf = "Internal transfer in"
        elif settlement_match:
            print(f"  → Settlement pattern + debit >= 1B → 'Internal transfer in'")
            nature_tfidf = "Internal transfer in"
        else:
            print(f"  → No keyword/pattern match, falls to TF-IDF similarity")
            nature_tfidf = None
    else:
        print(f"  → debit < 1B, no forced override, falls to TF-IDF similarity")
        nature_tfidf = None

    # Step 4: Guardrail (runs AFTER classification)
    print(f"\n  --- _apply_classification_guardrails ---")
    if settlement_match:
        # principal > 0 for any positive debit
        print(f"  → Settlement pattern matched, principal > 0 → 'Internal transfer in'")
        nature_guardrail = "Internal transfer in"
    elif is_receipt and debit >= _UNIT_1B:
        if hdtg_match:
            print(f"  → HDTG keyword + debit >= 1B → 'Internal transfer in'")
            nature_guardrail = "Internal transfer in"
        else:
            # Round amount check
            unit = Decimal("100000000")
            is_round = debit >= unit and debit % unit == 0
            if is_round:
                print(f"  → Round amount ({debit:,.0f} % 100M == 0) → 'Internal transfer in'")
                nature_guardrail = "Internal transfer in"
            else:
                print(f"  → Non-round amount → no override")
                nature_guardrail = nature_tfidf
    else:
        print(f"  → No guardrail override")
        nature_guardrail = nature_tfidf

    final = nature_guardrail or nature_tfidf or "(would go to AI/other tier)"
    print(f"\n  ★ FINAL NATURE: {final}")
    return final


print("=" * 60)
print("  CLASSIFICATION PROOF TEST")
print("=" * 60)

# Transaction 1: CA - TARGET with 6B VND
classify("CA - TARGET", Decimal("6000000000"))

# Transaction 2: CA - TARGET with 3.42M VND
classify("CA - TARGET", Decimal("3420000"))
