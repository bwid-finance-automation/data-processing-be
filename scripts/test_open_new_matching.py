"""
Integration test: trace open_new matching with real data.
Simulates the matching logic using the actual VTB lookup file and
the target entities' transaction data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# Import the actual parsing function
from app.application.finance.cash_report.cash_report_service import CashReportService

VTB_FILE = r"D:\Project\data-processing-be\sample_for_test\cash_report_automation\lookup_file\VTB_Saving_Nhi.xls"
BIDV_FILE = r"D:\Project\data-processing-be\sample_for_test\cash_report_automation\lookup_file\BIDV_Saving_VND_Nhi.xls"
VCB_FILE = r"D:\Project\data-processing-be\sample_for_test\cash_report_automation\lookup_file\VCB_Saving.xlsx"
WOORI_FILES = [
    r"D:\Project\data-processing-be\sample_for_test\cash_report_automation\lookup_file\Wooribank_Saving_200700045749.pdf",
    r"D:\Project\data-processing-be\sample_for_test\cash_report_automation\lookup_file\Wooribank_Saving_200700050679.pdf",
    r"D:\Project\data-processing-be\sample_for_test\cash_report_automation\lookup_file\Wooribank_Saving_200700052415.pdf",
    r"D:\Project\data-processing-be\sample_for_test\cash_report_automation\lookup_file\Wooribank_Saving_200700052504.pdf",
]

# Read all lookup files
service = CashReportService.__new__(CashReportService)

all_lookup_accounts = {}
all_account_details = {}

for filepath in [VTB_FILE, BIDV_FILE, VCB_FILE] + WOORI_FILES:
    with open(filepath, "rb") as f:
        content = f.read()
    parsed, details = service._parse_saving_lookup_file_with_metadata(content)
    for key, accounts in parsed.items():
        bucket = all_lookup_accounts.setdefault(key, [])
        for acc in accounts:
            if acc not in bucket:
                bucket.append(acc)
    for acc, meta in details.items():
        if acc:
            dst = all_account_details.setdefault(acc, {})
            for mk, mv in meta.items():
                if mv not in (None, ""):
                    if dst.get(mk) in (None, ""):
                        dst[mk] = mv

# Normalize account text
def _normalize_account_text(raw):
    if not raw:
        return ""
    s = str(raw).strip().replace("\xa0", "").replace("'", "")
    try:
        if "." in s and float(s) == int(float(s)):
            s = str(int(float(s)))
    except (ValueError, OverflowError):
        pass
    return s

# Normalize details keys
normalized_details = {}
for acc, meta in all_account_details.items():
    acc_norm = _normalize_account_text(acc)
    if acc_norm:
        dst = normalized_details.setdefault(acc_norm, {})
        for mk, mv in (meta or {}).items():
            if mv not in (None, ""):
                if dst.get(mk) in (None, ""):
                    dst[mk] = mv
all_account_details = normalized_details

# Normalize lookup keys
for key, accounts in list(all_lookup_accounts.items()):
    norm_accounts = []
    for acc in accounts:
        acc_norm = _normalize_account_text(acc)
        if acc_norm and acc_norm not in norm_accounts:
            norm_accounts.append(acc_norm)
    all_lookup_accounts[key] = norm_accounts

print("=" * 80)
print("PARSED LOOKUP DATA")
print("=" * 80)
print(f"Total (entity, amount) keys: {len(all_lookup_accounts)}")
print(f"Total account details: {len(all_account_details)}")

# Build secondary indices
providers = {str(m.get("provider") or "").strip().upper() for m in all_account_details.values() if str(m.get("provider") or "").strip()}
print(f"Providers: {providers}")

# Build opening date index
accounts_by_date = {}
for acc, meta in all_account_details.items():
    od = meta.get("opening_date")
    if isinstance(od, date):
        bucket = accounts_by_date.setdefault(od, [])
        if acc not in bucket:
            bucket.append(acc)

target_date = date(2026, 2, 12)
print(f"\nAccounts opened on {target_date}:")
for acc in accounts_by_date.get(target_date, []):
    meta = all_account_details.get(acc, {})
    print(f"  {acc}: entity={meta.get('entity', '?')}, amount={meta.get('amount')}, "
          f"provider={meta.get('provider')}, term={meta.get('term_months')}M/{meta.get('term_days')}D, "
          f"rate={meta.get('interest_rate')}")

# Entity matching functions (copied from the code)
ENTITY_NOISE = {
    "CT", "CTY", "CONG", "TY", "TNHH", "MTV", "CP", "CO", "PHAN", "VA",
    "PHAT", "TRIEN", "NGHIEP", "DAU", "TU", "DU", "AN", "MOT", "THANH",
    "VIEN", "HUU", "HAN", "JSC", "PTCN", "BW",
}

def _normalize_entity_text(value):
    text = str(value or "").upper()
    text = text.replace("BWID", "BW")
    text = re.sub(r'[^A-Z0-9]+', ' ', text)
    tokens = [tok for tok in text.split() if tok and tok not in ENTITY_NOISE]
    return "".join(tokens)

def _entity_similarity_score(left, right):
    l = _normalize_entity_text(left)
    r = _normalize_entity_text(right)
    if not l or not r:
        return 0.0
    if l == r:
        return 1.0
    base = SequenceMatcher(None, l, r).ratio()
    if l in r or r in l:
        base = max(base, 0.85)
    return base

def _bank_matches(a, b):
    return CashReportService._bank_matches(a, b)

# Simulate matching for target entities
print("\n" + "=" * 80)
print("SIMULATING MATCHING FOR TARGET ENTITIES")
print("=" * 80)

# These represent the open-new candidate transactions
targets = [
    {"entity": "WINLOCK 2A",  "bank": "VTB",   "date": date(2026,2,12), "amount": 4_000_000_000,
     "desc": "GUI TIEN THEO HOP DONG TIEN GUI CO KY HAN SO 285.2026.49236 NGAY 12.02.2026",
     "current_acc": "114628748888"},
    {"entity": "THUAN DAO",   "bank": "VTB",   "date": date(2026,2,12), "amount": 3_000_000_000,
     "desc": "HDTG 900/2026/49150",
     "current_acc": "116002949864"},
    {"entity": "XENIA 1",    "bank": "VTB",   "date": date(2026,2,12), "amount": 3_000_000_000,
     "desc": "HOP DONG TIEN GUI SO 285/2026/49235",
     "current_acc": "116646026868"},
    {"entity": "XENIA 2",    "bank": "VTB",   "date": date(2026,2,12), "amount": 3_000_000_000,
     "desc": "GUI TIEN THEO HDTG CO KY HAN SO 285.2026.49237 NGAY 12.02.2026",
     "current_acc": "118625469999"},
    {"entity": "SAO HOA",    "bank": "TCB",   "date": date(2026,2,9),  "amount": 60_000_000_000,
     "desc": "CONG TY CP SAO HOA TOAN QUOC GUI TIEN NGAY 09/02/2026 THEO",
     "current_acc": "19037134677017"},
    {"entity": "BW SUPLLY CHAIN CITY", "bank": "TCB", "date": date(2026,2,12), "amount": 6_000_000_000,
     "desc": "CT TNHH BW SUPPLY CHAIN CITY GUI TIEN NGAY 12/02/2026 THEO HDTG",
     "current_acc": "19040114651012"},
    {"entity": "BW MY PHUOC 3", "bank": "WOORI", "date": date(2026,2,2), "amount": 45_000_000_000,
     "desc": "Withdrawal - Withdrawal",
     "current_acc": "100700488704"},
]

existing_saving_acc_set = set()

for tx in targets:
    entity = tx["entity"]
    bank = tx["bank"]
    tx_date = tx["date"]
    amount = tx["amount"]
    desc = tx["desc"]
    current_acc = tx["current_acc"]

    print(f"\n--- {entity} ---")
    print(f"  Bank={bank}, Date={tx_date}, Amount={amount:,.0f}")
    print(f"  Description: {desc}")

    # Step 1: Extract from description
    saving_acc = CashReportService._extract_saving_account_for_open_new(desc)
    saving_acc = _normalize_account_text(saving_acc)
    print(f"  Step 1 (description extract): {saving_acc or 'None'}")

    matched_lookup = saving_acc if saving_acc and saving_acc in all_account_details else None
    if saving_acc and not matched_lookup:
        print(f"    -> extracted '{saving_acc}' but NOT in lookup details")

    # Check bank availability
    bank_norm = bank.strip().upper()
    bank_available = any(_bank_matches(bank_norm, p) for p in providers)
    print(f"  Bank '{bank_norm}' in lookup providers: {bank_available}")

    if saving_acc:
        print(f"  -> Using extracted account: {saving_acc}")
    elif not bank_available:
        print(f"  -> SKIP LOOKUP (no provider for {bank_norm})")
        # Falls through to suffix
    else:
        # Step 2a: Opening date match
        date_candidates = accounts_by_date.get(tx_date, [])
        candidates = [_normalize_account_text(a) for a in date_candidates if _normalize_account_text(a)]
        candidates = list(dict.fromkeys(candidates))
        print(f"  Step 2a (opening date {tx_date}): {len(candidates)} candidates")

        # Bank filter (strict)
        bank_filtered = [a for a in candidates
                         if _bank_matches(bank_norm, str(all_account_details.get(a,{}).get("provider","")).upper())]
        print(f"    After bank filter: {len(bank_filtered)} candidates")
        if bank_filtered:
            candidates = bank_filtered
        else:
            print(f"    -> No bank match, step 2a fails")
            candidates = []

        # Entity filter (strict)
        if candidates and entity:
            scored = []
            for acc in candidates:
                lookup_ent = str(all_account_details.get(acc, {}).get("entity", ""))
                score = _entity_similarity_score(entity, lookup_ent)
                scored.append((score, acc, lookup_ent))
            scored.sort(key=lambda x: x[0], reverse=True)
            print(f"    Entity scores:")
            for s, a, e in scored:
                print(f"      {s:.4f}: {a} (entity={e}, norm={_normalize_entity_text(e)})")

            top_score = scored[0][0] if scored else 0
            if top_score <= 0:
                print(f"    -> All scores 0, entity filter fails")
                candidates = []
            else:
                cutoff = max(0.35, top_score - 0.12)
                filtered = [a for s, a, _ in scored if s >= cutoff and s > 0]
                print(f"    -> cutoff={cutoff:.4f}, filtered: {filtered}")
                candidates = filtered if filtered else []

        # Amount filter
        if candidates:
            exact = []
            greater = []
            for acc in candidates:
                la = all_account_details.get(acc, {}).get("amount")
                if la is None:
                    continue
                if abs(la - amount) <= 1:
                    exact.append(acc)
                elif la > amount + 1:
                    greater.append((la - amount, acc))
            if exact:
                candidates = exact
                print(f"    Amount exact match: {candidates}")
            elif greater:
                greater.sort()
                min_gap = greater[0][0]
                candidates = [a for g, a in greater if abs(g - min_gap) <= 1]
                print(f"    Amount closest greater: {candidates}")
            else:
                print(f"    -> No amount match")
                candidates = []

        if candidates:
            # Select: prefer non-existing
            selected = None
            for acc in candidates:
                if acc not in existing_saving_acc_set:
                    selected = acc
                    break
            if not selected and candidates:
                selected = candidates[0]
            saving_acc = selected
            matched_lookup = selected
            print(f"  Step 2a RESULT: {saving_acc}")
        else:
            print(f"  Step 2a: No match found")

            # Step 2b-2e: abbreviated - check entity-aligned
            print(f"  Steps 2b-2f: trying other matching steps...")
            # Step 2b: entity substring match on lookup_accounts keys
            entity_norm = re.sub(r'\s+', '', entity.upper())
            found_2b = False
            for (ent, _amt), accs in all_lookup_accounts.items():
                ent_norm = re.sub(r'\s+', '', ent)
                if ent_norm == entity_norm or ent_norm in entity_norm or entity_norm in ent_norm:
                    print(f"    2b: Entity substring match: '{ent}' -> {accs}")
                    found_2b = True
            if not found_2b:
                print(f"    2b: No entity substring match for '{entity}'")

    # Final: suffix fallback
    if not saving_acc:
        suffix = 1
        while f"{current_acc}_{suffix}" in existing_saving_acc_set:
            suffix += 1
        saving_acc = f"{current_acc}_{suffix}"
        print(f"  FALLBACK: suffix -> {saving_acc}")

    # Check if already in existing set
    if saving_acc in existing_saving_acc_set:
        base = saving_acc.split("_")[0]
        suffix = 1
        while f"{base}_{suffix}" in existing_saving_acc_set:
            suffix += 1
        old_acc = saving_acc
        saving_acc = f"{base}_{suffix}"
        print(f"  DEDUP: {old_acc} already taken -> {saving_acc}")

    existing_saving_acc_set.add(saving_acc)
    print(f"  FINAL RESULT: {saving_acc}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for acc in sorted(existing_saving_acc_set):
    print(f"  {acc}")
