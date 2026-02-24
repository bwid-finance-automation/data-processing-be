"""
Test script for the TF-IDF Nature classifier.

Loads Transactions.csv, splits 80/20 train/test, fits the TF-IDF classifier
on the training set, and evaluates classification accuracy on the test set.

Usage:
    python scripts/test_tfidf_classifier.py
"""
import csv
import sys
import random
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.finance.cash_report.services.tfidf_classifier import (
    TfidfNatureClassifier,
)

TRANSACTIONS_FILE = PROJECT_ROOT / "movement_nature_filter" / "Transactions.csv"


def load_transactions(csv_path: Path):
    """Load all labeled rows from Transactions.csv."""
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)  # skip header

        for row in reader:
            if not row or len(row) < 2:
                continue
            desc = row[0].strip()
            nature = row[1].strip()
            if desc and nature:
                rows.append((desc, nature))
    return rows


def main():
    if not TRANSACTIONS_FILE.exists():
        print(f"ERROR: {TRANSACTIONS_FILE} not found")
        sys.exit(1)

    print(f"Loading {TRANSACTIONS_FILE}...")
    all_rows = load_transactions(TRANSACTIONS_FILE)
    print(f"Total rows: {len(all_rows)}")

    # Receipt vs Payment natures
    RECEIPT_NATURES = {
        "Receipt from tenants", "Other receipts", "Internal transfer in",
        "Refinancing", "Loan receipts", "VAT refund",
        "Dividend receipt (inside group)", "Corporate Loan drawdown",
        "Loan drawdown", "Loan repayment",
    }
    PAYMENT_NATURES = {
        "Operating expense", "Internal transfer out", "Loan repayment",
        "Loan interest", "Construction expense", "Deal payment",
        "Land acquisition", "Dividend paid (inside group)",
        "Payment for acquisition",
    }

    # Filter to valid natures
    valid_rows = []
    for desc, nature in all_rows:
        if nature in RECEIPT_NATURES or nature in PAYMENT_NATURES:
            valid_rows.append((desc, nature))
    print(f"Valid rows: {len(valid_rows)}")

    # Show distribution
    nature_counts = Counter(n for _, n in valid_rows)
    print("\n=== Nature Distribution ===")
    for nature, count in nature_counts.most_common():
        print(f"  {nature:40s} {count:6d}")

    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(valid_rows)
    split_idx = int(len(valid_rows) * 0.8)
    train_rows = valid_rows[:split_idx]
    test_rows = valid_rows[split_idx:]
    print(f"\nTrain: {len(train_rows)}, Test: {len(test_rows)}")

    # Fit on train set
    descs = [r[0] for r in train_rows]
    natures = [r[1] for r in train_rows]
    is_receipts = [n in RECEIPT_NATURES for n in natures]

    # Test with different thresholds
    for threshold in [0.40, 0.50, 0.55, 0.60, 0.70]:
        print(f"\n{'=' * 60}")
        print(f"=== Threshold: {threshold} ===")
        print(f"{'=' * 60}")

        classifier = TfidfNatureClassifier(min_similarity=threshold)
        corpus_size = classifier.fit(descs, natures, is_receipts)
        print(f"Corpus size: {corpus_size}")

        # Evaluate on test set
        correct = 0
        wrong = 0
        no_prediction = 0
        per_nature_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "no_pred": 0, "total": 0})

        for test_desc, true_nature in test_rows:
            is_receipt = true_nature in RECEIPT_NATURES
            predicted = classifier.predict(test_desc, is_receipt)
            per_nature_stats[true_nature]["total"] += 1

            if predicted is None:
                no_prediction += 1
                per_nature_stats[true_nature]["no_pred"] += 1
            elif predicted == true_nature:
                correct += 1
                per_nature_stats[true_nature]["correct"] += 1
            else:
                wrong += 1
                per_nature_stats[true_nature]["wrong"] += 1

        total = len(test_rows)
        coverage = (correct + wrong) / total * 100
        accuracy_of_predicted = correct / (correct + wrong) * 100 if (correct + wrong) > 0 else 0
        overall_accuracy = correct / total * 100

        print(f"\n  Total test:      {total}")
        print(f"  Predicted:       {correct + wrong} ({coverage:.1f}% coverage)")
        print(f"  No prediction:   {no_prediction}")
        print(f"  Correct:         {correct}")
        print(f"  Wrong:           {wrong}")
        print(f"  Accuracy (pred): {accuracy_of_predicted:.1f}%")
        print(f"  Accuracy (all):  {overall_accuracy:.1f}%")

        print(f"\n  {'Nature':40s} {'Total':>6s} {'Correct':>8s} {'Wrong':>6s} {'NoPred':>7s} {'Prec%':>6s}")
        print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")
        for nature in sorted(per_nature_stats.keys()):
            s = per_nature_stats[nature]
            prec = s["correct"] / (s["correct"] + s["wrong"]) * 100 if (s["correct"] + s["wrong"]) > 0 else 0
            print(f"  {nature:40s} {s['total']:6d} {s['correct']:8d} {s['wrong']:6d} {s['no_pred']:7d} {prec:5.1f}%")


if __name__ == "__main__":
    main()
