"""
Test script for Big X marker clear/relocate/place logic.

Tests:
1. _clear_x_marker_cell  — clears the X value from a cell
2. _set_x_marker_on_row  — places an X value on a cell
3. End-to-end flow: clear from original → place on last data row
"""
import sys
import re

sys.path.insert(0, r"d:\Project\data-processing-be")

from app.application.finance.cash_report.openpyxl_handler import OpenpyxlHandler

handler = OpenpyxlHandler()

# ─────────────────────────────────────────────────────────────────
# Helper to build minimal sheet XML for testing
# ─────────────────────────────────────────────────────────────────
def make_sheet_xml(rows_dict):
    """Build a minimal sheet XML from {row_num: {col: (value, style)}}."""
    parts = [b'<worksheet><sheetData>']
    for rn in sorted(rows_dict):
        cells = []
        for col, (value, style) in sorted(rows_dict[rn].items()):
            s = f's="{style}"' if style else ''
            if value:
                cells.append(
                    f'<c r="{col}{rn}" {s} t="inlineStr"><is><t>{value}</t></is></c>'
                )
            else:
                cells.append(f'<c r="{col}{rn}" {s}/>')
        row_xml = f'<row r="{rn}">' + ''.join(cells) + '</row>'
        parts.append(row_xml.encode())
    parts.append(b'</sheetData></worksheet>')
    return b''.join(parts)


def get_cell_value(xml, col, row):
    """Extract cell value from XML."""
    rn = str(row).encode()
    c = col.encode()
    ref = c + rn  # e.g. b"J10"

    # Check self-closing FIRST (empty cell): <c ... r="J10" ... />
    # Must check before content pattern because content pattern can
    # accidentally match through a self-closing tag into the next cell.
    m_self = re.search(
        rb'<c\s[^>]*r="' + ref + rb'"[^>]*/\s*>',
        xml,
    )
    if m_self:
        return ""

    # Then try to find cell with content: <c ... r="J10" ...>...</c>
    m_content = re.search(
        rb'<c\s[^>]*r="' + ref + rb'"[^>]*>(.*?)</c>',
        xml, re.DOTALL,
    )
    if m_content:
        inner = m_content.group(1)
        t_m = re.search(rb'<t>([^<]*)</t>', inner)
        if t_m:
            return t_m.group(1).decode()
        return ""

    return None  # cell not found


# ─────────────────────────────────────────────────────────────────
# Test 1: _clear_x_marker_cell
# ─────────────────────────────────────────────────────────────────
def test_clear_x_marker_cell():
    """Clear the Big X value from a cell, leaving an empty cell."""
    xml = make_sheet_xml({
        
        10: {"B": ("ACC001", "999"), "J": ("X", "35")},
        11: {"B": ("ACC002", "999"), "J": ("", "35")},
    })

    # Before: row 10, col J has "X"
    assert get_cell_value(xml, "J", 10) == "X", f"Pre-check failed: J10 = {get_cell_value(xml, 'J', 10)}"

    # Clear X from J10
    result = OpenpyxlHandler._clear_x_marker_cell(xml, "J", 10)

    # After: J10 should be empty
    val = get_cell_value(result, "J", 10)
    assert val == "", f"FAIL: J10 should be empty after clear, got '{val}'"

    # Row 10 should still exist with B column
    assert get_cell_value(result, "B", 10) == "ACC001", "FAIL: B10 should be preserved"

    # Row 11 should be unchanged
    assert get_cell_value(result, "J", 11) == "", f"FAIL: J11 should still be empty"
    assert get_cell_value(result, "B", 11) == "ACC002", "FAIL: B11 should be preserved"

    print("  ✅ test_clear_x_marker_cell PASSED")


# ─────────────────────────────────────────────────────────────────
# Test 2: _set_x_marker_on_row — existing empty cell
# ─────────────────────────────────────────────────────────────────
def test_set_x_marker_on_empty_cell():
    """Place X on a row that has an empty cell in the target column."""
    xml = make_sheet_xml({
        10: {"B": ("ACC001", "999"), "J": ("", "35")},
    })

    # Before: J10 is empty
    assert get_cell_value(xml, "J", 10) == "", f"Pre-check failed"

    # Set X on J10
    result = OpenpyxlHandler._set_x_marker_on_row(xml, "J", 10)

    # After: J10 should be "X"
    val = get_cell_value(result, "J", 10)
    assert val == "X", f"FAIL: J10 should be 'X', got '{val}'"

    # B10 preserved
    assert get_cell_value(result, "B", 10) == "ACC001"

    print("  ✅ test_set_x_marker_on_empty_cell PASSED")


# ─────────────────────────────────────────────────────────────────
# Test 3: _set_x_marker_on_row — no existing cell (append)
# ─────────────────────────────────────────────────────────────────
def test_set_x_marker_no_existing_cell():
    """Place X on a row that does NOT have the target column at all."""
    xml = make_sheet_xml({
        10: {"B": ("ACC001", "999"), "C": ("VND", "100")},
    })

    # Before: J10 doesn't exist
    assert get_cell_value(xml, "J", 10) is None, "Pre-check failed: J10 shouldn't exist"

    # Set X on J10
    result = OpenpyxlHandler._set_x_marker_on_row(xml, "J", 10)

    # After: J10 should be "X"
    val = get_cell_value(result, "J", 10)
    assert val == "X", f"FAIL: J10 should be 'X', got '{val}'"

    print("  ✅ test_set_x_marker_no_existing_cell PASSED")


# ─────────────────────────────────────────────────────────────────
# Test 4: _set_x_marker_on_row — overwrite existing value
# ─────────────────────────────────────────────────────────────────
def test_set_x_marker_overwrite():
    """Place X on a row that already has a value in the target column."""
    xml = make_sheet_xml({
        10: {"B": ("ACC001", "999"), "J": ("old_value", "35")},
    })

    assert get_cell_value(xml, "J", 10) == "old_value"

    result = OpenpyxlHandler._set_x_marker_on_row(xml, "J", 10)

    val = get_cell_value(result, "J", 10)
    assert val == "X", f"FAIL: J10 should be 'X', got '{val}'"

    print("  ✅ test_set_x_marker_overwrite PASSED")


# ─────────────────────────────────────────────────────────────────
# Test 5: End-to-end — clear from row 5, place on row 8 (last data row)
# ─────────────────────────────────────────────────────────────────
def test_e2e_clear_and_relocate():
    """Simulate the full flow: clear Big X from original → place on last data row."""
    xml = make_sheet_xml({
        3: {"B": ("ACC001", "999"), "J": ("", "35")},
        5: {"B": ("ACC002", "999"), "J": ("X", "35")},   # ← Big X here
        6: {"B": ("", "999"), "J": ("", "35")},            # empty/template
        7: {"B": ("", "999"), "J": ("", "35")},            # empty/template
        8: {"B": ("ACC003", "999"), "J": ("", "35")},     # last data row
    })

    # Step 1: X is at row 5
    assert get_cell_value(xml, "J", 5) == "X"

    # Step 2: Clear X from row 5
    xml = OpenpyxlHandler._clear_x_marker_cell(xml, "J", 5)
    assert get_cell_value(xml, "J", 5) == "", "After clear, J5 should be empty"

    # Step 3: Place X on last data row (row 8)
    xml = OpenpyxlHandler._set_x_marker_on_row(xml, "J", 8)
    assert get_cell_value(xml, "J", 8) == "X", "After relocate, J8 should be 'X'"

    # Verify row 5 is still empty, row 3 still intact
    assert get_cell_value(xml, "J", 5) == ""
    assert get_cell_value(xml, "B", 3) == "ACC001"
    assert get_cell_value(xml, "B", 5) == "ACC002"

    print("  ✅ test_e2e_clear_and_relocate PASSED")


# ─────────────────────────────────────────────────────────────────
# Test 6: All 3 column variants (J, AD, Z)
# ─────────────────────────────────────────────────────────────────
def test_all_columns():
    """Verify clear/set works for J (Acc_Char), AD (Saving), Z (Cash Balance)."""
    for col, name in [("J", "Acc_Char"), ("AD", "Saving"), ("Z", "Cash Balance")]:
        xml = make_sheet_xml({
            10: {"B": ("ACC001", "999"), col: ("X", "35")},
            12: {"B": ("ACC002", "999"), col: ("", "35")},
        })
        # Clear X from row 10
        xml = OpenpyxlHandler._clear_x_marker_cell(xml, col, 10)
        assert get_cell_value(xml, col, 10) == "", f"{name}: clear failed"

        # Place X on row 12
        xml = OpenpyxlHandler._set_x_marker_on_row(xml, col, 12)
        assert get_cell_value(xml, col, 12) == "X", f"{name}: set failed"

    print("  ✅ test_all_columns (J, AD, Z) PASSED")


# ─────────────────────────────────────────────────────────────────
# Test 7: No-op when no rows inserted (X stays on last data row)
# ─────────────────────────────────────────────────────────────────
def test_no_inserts_x_stays():
    """If no new rows are inserted, Big X stays on the relocated position."""
    xml = make_sheet_xml({
        3: {"B": ("ACC001", "999"), "J": ("X", "35")},
        5: {"B": ("ACC002", "999"), "J": ("", "35")},
    })

    # Clear and relocate to row 5
    xml = OpenpyxlHandler._clear_x_marker_cell(xml, "J", 3)
    xml = OpenpyxlHandler._set_x_marker_on_row(xml, "J", 5)

    # Verify: J3 empty, J5 has X
    assert get_cell_value(xml, "J", 3) == ""
    assert get_cell_value(xml, "J", 5) == "X"

    # Simulate "no inserts" — ac_added == 0, so no final X placement
    # X should remain on row 5
    assert get_cell_value(xml, "J", 5) == "X"

    print("  ✅ test_no_inserts_x_stays PASSED")


# ─────────────────────────────────────────────────────────────────
# Test 8: Simulate full batch flow — only ONE Big X at the end
# ─────────────────────────────────────────────────────────────────
def test_batch_flow_x_stays_on_top():
    """Simulate the batch flow: relocate X to last data row, insert below.
    X should stay on the last OLD data row (on top of newly inserted rows).
    """
    # Template: X at row 10 (last data row), templates at 11-13
    xml = make_sheet_xml({
        4: {"B": ("ACC001", "999"), "J": ("", "35")},
        8: {"B": ("ACC002", "999"), "J": ("", "35")},
        10: {"B": ("ACC003", "999"), "J": ("X", "35")},  # Big X here
        11: {"B": ("", "999"), "J": ("", "35")},           # template
        12: {"B": ("", "999"), "J": ("", "35")},           # template
        13: {"B": ("", "999"), "J": ("", "35")},           # template
    })

    # Step 1: Find X at row 10
    x_row = 10
    assert get_cell_value(xml, "J", 10) == "X"

    # Step 2: Relocate — clear X, find last data row (entire sheet), place X there
    xml = OpenpyxlHandler._clear_x_marker_cell(xml, "J", x_row)
    # Last data row in entire sheet = row 10 (B has ACC003)
    last_data = 10
    xml = OpenpyxlHandler._set_x_marker_on_row(xml, "J", last_data)
    x_row = last_data

    # Step 3: Simulate inserting 2 accounts at rows 11 and 12
    xml = re.sub(
        rb'<c r="B11"[^/]*/>', b'<c r="B11" s="999" t="inlineStr"><is><t>NEW001</t></is></c>', xml
    )
    xml = re.sub(
        rb'<c r="B12"[^/]*/>', b'<c r="B12" s="999" t="inlineStr"><is><t>NEW002</t></is></c>', xml
    )

    # Step 4: X stays at row 10 (on top of newly inserted rows 11-12)
    # NO final X placement — X does NOT move to last inserted row

    # Verify: X is at row 10, NOT at 11 or 12
    assert get_cell_value(xml, "J", 10) == "X", f"FAIL: X should stay at row 10"
    assert get_cell_value(xml, "J", 11) == "", f"FAIL: J11 should be empty"
    assert get_cell_value(xml, "J", 12) == "", f"FAIL: J12 should be empty"

    # Only 1 Big X in the sheet
    x_count = sum(1 for rn in [4, 8, 10, 11, 12, 13]
                  if (get_cell_value(xml, "J", rn) or "").upper() == "X")
    assert x_count == 1, f"FAIL: Expected exactly 1 Big X, found {x_count}"

    print("  OK test_batch_flow_x_stays_on_top PASSED")


def test_second_batch_x_walks_to_last_data():
    """After first batch, X is at old last row. Second batch should
    relocate X to the new last data row (including first batch's rows).
    """
    # After first batch: X at row 10, data at 11-12
    xml = make_sheet_xml({
        4: {"B": ("ACC001", "999"), "J": ("", "35")},
        10: {"B": ("ACC003", "999"), "J": ("X", "35")},  # Big X
        11: {"B": ("NEW001", "999"), "J": ("", "35")},    # from batch 1
        12: {"B": ("NEW002", "999"), "J": ("", "35")},    # from batch 1
        13: {"B": ("", "999"), "J": ("", "35")},           # template
    })

    # Step 1: Find X at row 10
    x_row = 10

    # Step 2: Relocate — search entire sheet for last data row
    xml = OpenpyxlHandler._clear_x_marker_cell(xml, "J", x_row)
    # Last data row in entire sheet = row 12 (NEW002)
    last_data = 12
    xml = OpenpyxlHandler._set_x_marker_on_row(xml, "J", last_data)
    x_row = last_data

    # Verify: X moved from 10 to 12 (the real last data row)
    assert get_cell_value(xml, "J", 10) == "", f"FAIL: J10 should be cleared"
    assert get_cell_value(xml, "J", 12) == "X", f"FAIL: X should now be at row 12"

    # New insertions would go at row 13+ (after X at 12)

    print("  OK test_second_batch_x_walks_to_last_data PASSED")


# ─────────────────────────────────────────────────────────────────
# Test 9: Reproduce the old bug — without clearing, TWO X markers
# ─────────────────────────────────────────────────────────────────
def test_old_bug_duplicate_x():
    """Reproduce the old bug: final X placement without clearing old position
    leaves duplicate X markers."""
    xml = make_sheet_xml({
        10: {"B": ("ACC003", "999"), "J": ("X", "35")},
        11: {"B": ("", "999"), "J": ("", "35")},
        12: {"B": ("", "999"), "J": ("", "35")},
    })

    x_row = 10
    # Relocate: clear X from 10, find last data = 10, place back on 10
    xml = OpenpyxlHandler._clear_x_marker_cell(xml, "J", 10)
    xml = OpenpyxlHandler._set_x_marker_on_row(xml, "J", 10)  # same spot

    # OLD BUG: final placement WITHOUT clearing x_row first
    last_inserted_row = 12
    # xml = OpenpyxlHandler._clear_x_marker_cell(xml, "J", x_row)  # THIS WAS MISSING
    xml = OpenpyxlHandler._set_x_marker_on_row(xml, "J", last_inserted_row)

    # Both row 10 AND row 12 have X — this is the bug!
    j10 = get_cell_value(xml, "J", 10)
    j12 = get_cell_value(xml, "J", 12)
    assert j10 == "X" and j12 == "X", f"Bug not reproduced: J10='{j10}', J12='{j12}'"

    print("  OK test_old_bug_duplicate_x PASSED (bug confirmed)")


# ─────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Testing Big X Marker Logic ===\n")

    test_clear_x_marker_cell()
    test_set_x_marker_on_empty_cell()
    test_set_x_marker_no_existing_cell()
    test_set_x_marker_overwrite()
    test_e2e_clear_and_relocate()
    test_all_columns()
    test_no_inserts_x_stays()
    test_batch_flow_x_stays_on_top()
    test_second_batch_x_walks_to_last_data()
    test_old_bug_duplicate_x()

    print("\n=== All 10 tests passed! ===\n")
