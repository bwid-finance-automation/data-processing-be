"""Diagnostic: simulate Big X relocation + row inserts on Acc_Char, check for corruption."""
import sys, os, re, zipfile, io
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.application.finance.cash_report.openpyxl_handler import OpenpyxlHandler

template_path = os.path.join(
    os.path.dirname(__file__), "..", "templates", "cash_report", "[0] Cash report Template.xlsx"
)
handler = OpenpyxlHandler()

with open(template_path, "rb") as f:
    zip_data = f.read()

sheet_paths = handler._get_sheet_xml_paths(zip_data)
ac_path = sheet_paths.get("Acc_Char")

with zipfile.ZipFile(io.BytesIO(zip_data), "r") as z:
    ac_xml = z.read(ac_path)

# Simulate init_session processing (strip shared formulas)
ac_xml = handler._strip_shared_formulas_bytes(ac_xml)

# === Big X relocation ===
x_row = handler._find_x_marker_row(ac_xml, "J", zip_data=zip_data)
print(f"Big X at row: {x_row}")

old_x_row = x_row
ac_xml = handler._clear_x_marker_cell(ac_xml, "J", x_row)
last_xml_row = handler._find_last_xml_row_num(ac_xml)
last_data = handler._find_last_data_row_before(ac_xml, last_xml_row + 1, "B")
if last_data:
    ac_xml = handler._set_x_marker_on_row(ac_xml, "J", last_data)
    x_row = last_data
print(f"X relocated from {old_x_row} to {x_row}")

# === Source row ===
source_row = handler._find_last_data_row_before(ac_xml, x_row + 1, "B")
if not source_row:
    source_row = 4
print(f"Source row: {source_row}")

src_pat = rb'<row[^>]*\sr="' + str(source_row).encode() + rb'"[^>]*>.*?</row>'
src_m = re.search(src_pat, ac_xml, re.DOTALL)
src_xml = src_m.group(0) if src_m else b""


def _get_style(src_xml_bytes, col, fallback="999"):
    m = re.search(rb'<c\s[^>]*r="' + col.encode() + rb'\d+"[^>]*s="(\d+)"', src_xml_bytes)
    return m.group(1).decode() if m else fallback


def _esc(v):
    return v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# === Templates (insertable rows) ===
search_window_end = x_row + 1000
small_x_rows = handler._find_small_x_rows(
    ac_xml, "B", x_row + 1, zip_data=zip_data, end_row=search_window_end
)
ceiling = small_x_rows[0] if small_x_rows else search_window_end
missing = handler._find_missing_row_slots(ac_xml, x_row + 1, end_row=ceiling - 1)
insertable = handler._find_all_insertable_rows(
    ac_xml, "B", x_row + 1, zip_data=zip_data, end_row=ceiling - 1
)
templates = sorted(set(missing + insertable + small_x_rows))
print(f"Templates: {templates[:10]} (total {len(templates)})")
print(f"Small-x rows: {small_x_rows[:5]}")

# === Insert test rows ===
test_entries = [
    {
        "saving_acc": "TEST001",
        "code": "CODE01",
        "bank": "TEST BANK",
        "currency": "VND",
        "account_type": "Saving Account",
        "entity": "TEST_ENTITY",
    },
    {
        "saving_acc": "TEST002",
        "code": "CODE02",
        "bank": "TEST BANK2",
        "currency": "USD",
        "account_type": "Saving Account",
        "entity": "TEST_ENTITY2",
    },
]

tpl_idx = 0
for entry in test_entries:
    acc = entry["saving_acc"]
    if tpl_idx >= len(templates):
        print(f"No more template rows for {acc}")
        break
    row_num = templates[tpl_idx]
    tpl_idx += 1
    rn = str(row_num)
    prev_row = max(row_num - 1, 1)

    overrides = {
        "A": f'<c r="A{rn}" s="{_get_style(src_xml, "A", "998")}"><f>A{prev_row}+1</f></c>'.encode(),
        "B": f'<c r="B{rn}" s="{_get_style(src_xml, "B")}" t="inlineStr"><is><t>{_esc(acc)}</t></is></c>'.encode(),
        "C": f'<c r="C{rn}" s="{_get_style(src_xml, "C")}" t="inlineStr"><is><t>{_esc(entry.get("code", ""))}</t></is></c>'.encode(),
        "E": f'<c r="E{rn}" s="{_get_style(src_xml, "E")}" t="inlineStr"><is><t>{_esc(entry.get("bank", ""))}</t></is></c>'.encode(),
        "F": f'<c r="F{rn}" s="{_get_style(src_xml, "F")}" t="inlineStr"><is><t>{_esc(entry.get("currency", "VND"))}</t></is></c>'.encode(),
        "G": f'<c r="G{rn}" s="{_get_style(src_xml, "G")}" t="inlineStr"><is><t>{_esc(entry.get("account_type", "Saving Account"))}</t></is></c>'.encode(),
        "J": f'<c r="J{rn}" s="{_get_style(src_xml, "J", "1142")}"/>'.encode(),
    }

    row_xml = handler._clone_row_for_insert(ac_xml, source_row, row_num, overrides)
    ac_cached = {
        "D": _esc(entry.get("entity", "")),
        "H": "Subsidiaries",
        "I": _esc(entry.get("bank", "")),
    }
    row_xml = handler._inject_formula_cached_values(row_xml, row_num, ac_cached)

    ac_xml, replaced = handler._replace_row_xml(ac_xml, row_num, row_xml)
    if not replaced:
        ac_xml = handler._insert_row_sorted(ac_xml, row_num, row_xml)
    print(f"Inserted {acc} at row {row_num} (replaced={replaced})")

# Apply dedupe (as in batch code)
ac_xml = handler._dedupe_rows_by_number(ac_xml)

# Apply full repair pipeline (as in _write_multiple_sheets)
ac_xml_final = handler._repair_worksheet_xml_for_safe_open(ac_xml)

# === COMPREHENSIVE CHECKS ===
print()
print("=== Comprehensive corruption checks ===")

# Check 1: Self-closing cells with t=
sc_with_t = list(re.finditer(rb'<c\s[^>]*\st="[^"]+"[^>]*/\s*>', ac_xml_final))
if sc_with_t:
    print(f"  FAIL: {len(sc_with_t)} self-closing cells with t= found")
    for sc in sc_with_t[:5]:
        print(f"    {sc.group(0)[:120]}")
else:
    print("  OK: No self-closing cells with t=")

# Check 2: Full cells with t=(str|n|b|e|s) but no <v> and no <is>
bad_cells = []
for cm in re.finditer(rb'<c\s[^>]*(?<!/)>.*?</c>', ac_xml_final, re.DOTALL):
    cell = cm.group(0)
    t_m = re.search(rb'<c[^>]*\st="(str|n|b|e|s)"', cell)
    if not t_m:
        continue
    has_v = bool(re.search(rb'<v>[^<]+</v>', cell))
    has_is = b'<is>' in cell
    if not has_v and not has_is:
        bad_cells.append(cell)

if bad_cells:
    print(f"  FAIL: {len(bad_cells)} cells with t= but no value")
    for bc in bad_cells[:5]:
        print(f"    {bc[:200]}")
else:
    print("  OK: All typed cells have values")

# Check 3: Row order
row_nums = [int(m.group(1)) for m in re.finditer(rb'<row[^>]*\sr="(\d+)"', ac_xml_final)]
sorted_check = all(row_nums[i] < row_nums[i + 1] for i in range(len(row_nums) - 1))
if not sorted_check:
    for i in range(len(row_nums) - 1):
        if row_nums[i] >= row_nums[i + 1]:
            print(f"  FAIL: Row order broken: row {row_nums[i]} >= {row_nums[i+1]}")
            break
else:
    print(f"  OK: Row order ({len(row_nums)} rows)")

# Check 4: Cell order within rows (check all modified rows)
rows_to_check = [x_row] + templates[: len(test_entries)]
for target_row in rows_to_check:
    if target_row == 0:
        continue
    row_pat = rb'<row[^>]*\sr="' + str(target_row).encode() + rb'"[^>]*>.*?</row>'
    row_m = re.search(row_pat, ac_xml_final, re.DOTALL)
    if not row_m:
        continue
    row_xml_str = row_m.group(0)
    cols = [m.group(1).decode() for m in re.finditer(rb'<c[^>]*r="([A-Z]+)\d+"', row_xml_str)]
    col_nums = []
    for c in cols:
        n = 0
        for ch in c:
            n = n * 26 + (ord(ch) - ord("A") + 1)
        col_nums.append(n)
    is_col_sorted = all(col_nums[i] < col_nums[i + 1] for i in range(len(col_nums) - 1))
    if not is_col_sorted:
        print(f"  FAIL: Cell order in row {target_row}: {cols}")
    else:
        print(f"  OK: Cell order in row {target_row}: {cols}")

# Check 5: Duplicate rows
row_counter = Counter(row_nums)
dupes = {k: v for k, v in row_counter.items() if v > 1}
if dupes:
    print(f"  FAIL: Duplicate rows: {dupes}")
else:
    print("  OK: No duplicate rows")

# Check 6: Orphan cells outside rows
sd_m = re.search(rb'<sheetData[^>]*>(.*?)</sheetData>', ac_xml_final, re.DOTALL)
if sd_m:
    body = sd_m.group(1)
    orphan = re.search(rb'</row>\s*<c\s', body)
    if orphan:
        print("  FAIL: Orphan cells found outside rows")
    else:
        print("  OK: No orphan cells")

# Check 7: Invalid style references (check if styles.xml has enough entries)
with zipfile.ZipFile(io.BytesIO(zip_data), "r") as z:
    styles_xml = z.read("xl/styles.xml")
xf_count = len(re.findall(rb'<xf ', styles_xml))
print(f"  INFO: styles.xml has {xf_count} xf entries")

# Find max style index used in our modified Acc_Char
max_style = 0
for sm in re.finditer(rb'\ss="(\d+)"', ac_xml_final):
    s = int(sm.group(1))
    if s > max_style:
        max_style = s
print(f"  INFO: Max style index used in Acc_Char: {max_style}")
if max_style >= xf_count:
    print(f"  FAIL: Style {max_style} >= {xf_count} total styles (INVALID)")
else:
    print(f"  OK: All styles valid (max {max_style} < {xf_count})")

# Check 8: Look for cells with empty <v></v> which can cause issues
empty_v = re.findall(rb'<c[^>]*>.*?<v\s*/>.*?</c>', ac_xml_final, re.DOTALL)
empty_v2 = re.findall(rb'<c[^>]*>.*?<v></v>.*?</c>', ac_xml_final, re.DOTALL)
if empty_v or empty_v2:
    print(f"  WARN: {len(empty_v) + len(empty_v2)} cells with empty <v>")
    for ev in (empty_v + empty_v2)[:3]:
        print(f"    {ev[:150]}")
else:
    print("  OK: No empty <v> tags")

# Check 9: Look for cm= attribute on cloned cells (dynamic array metadata)
cm_cells = re.findall(rb'<c[^>]*cm="\d+"[^>]*(?:>.*?</c>|/>)', ac_xml_final, re.DOTALL)
print(f"  INFO: {len(cm_cells)} cells with cm= attribute")
# Show cm cells in newly inserted rows
for target_row in templates[: len(test_entries)]:
    row_pat = rb'<row[^>]*\sr="' + str(target_row).encode() + rb'"[^>]*>.*?</row>'
    row_m = re.search(row_pat, ac_xml_final, re.DOTALL)
    if row_m:
        cm_in_row = re.findall(rb'<c[^>]*cm="\d+"[^>]*(?:>.*?</c>|/>)', row_m.group(0), re.DOTALL)
        if cm_in_row:
            for c in cm_in_row:
                print(f"  INFO: Row {target_row} has cm= cell: {c[:150]}")

print()
print("Done.")
