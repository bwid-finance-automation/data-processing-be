"""
Openpyxl Handler for Cash Report - Cross-platform Excel handler.
Replaces COM automation with optimized openpyxl operations.
Works on Windows, Linux, and macOS.
"""
import io
import re
import zipfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from xml.etree import ElementTree as ET

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)
 

class OpenpyxlHandler:
    """
    Cross-platform Excel handler using openpyxl.
    Optimized for performance with large files.

    Features:
    - Formula template copying with row adjustment
    - Batch operations for better performance
    - Preserve formatting and formulas
    """

    # Column mapping for Movement sheet
    INPUT_COLUMNS = {
        1: 'source',    # A
        2: 'bank',      # B
        3: 'account',   # C
        4: 'date',      # D
        5: 'description',  # E
        6: 'debit',     # F
        7: 'credit',    # G
        9: 'nature',    # I
    }

    FORMULA_COLUMNS = [8, 10, 11, 12, 13, 14, 15, 16]  # H, J, K, L, M, N, O, P

    FIRST_DATA_ROW = 4

    def __init__(self):
        self._formula_cache: Dict[str, Dict[int, str]] = {}

    @staticmethod
    def is_available() -> bool:
        """Always available - openpyxl is cross-platform."""
        return True

    # ========== Byte-level XML helpers for Movement sheet ==========
    # These avoid parsing the full sheet XML (which hangs for 1M-row templates)
    # by scanning/modifying raw bytes and only parsing individual rows as needed.

    def _read_movement_xml(self, file_path: Path) -> Tuple[bytes, str, bytes]:
        """Read ZIP data and extract Movement sheet XML bytes."""
        with open(file_path, "rb") as f:
            zip_data = f.read()
        sheet_paths = self._get_sheet_xml_paths(zip_data)
        movement_path = sheet_paths.get("Movement")
        if not movement_path:
            raise ValueError("Movement sheet not found in workbook")
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as z:
            mov_xml = z.read(movement_path)
        return zip_data, movement_path, mov_xml

    @staticmethod
    def _write_sheet_xml(
        file_path: Path, zip_data: bytes, sheet_path: str, new_xml: bytes
    ) -> None:
        """Write modified sheet XML back to the ZIP file, removing calcChain."""
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as src_zip:
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as dst_zip:
                for entry in src_zip.namelist():
                    # Skip calcChain to force Excel full recalculation
                    if "calcChain" in entry:
                        continue
                    if entry == sheet_path:
                        dst_zip.writestr(entry, new_xml)
                    elif entry == "[Content_Types].xml":
                        ct_xml = src_zip.read(entry)
                        ct_xml = re.sub(
                            rb'<Override[^>]*calcChain[^>]*/>', b'', ct_xml
                        )
                        dst_zip.writestr(entry, ct_xml)
                    elif entry == "xl/workbook.xml":
                        wb_xml = src_zip.read(entry)
                        wb_xml = OpenpyxlHandler._set_full_calc_on_load(wb_xml)
                        dst_zip.writestr(entry, wb_xml)
                    else:
                        dst_zip.writestr(entry, src_zip.read(entry))
        with open(file_path, "wb") as f:
            f.write(output.getvalue())

    @staticmethod
    def _set_full_calc_on_load(wb_xml: bytes) -> bytes:
        """Set fullCalcOnLoad='1' in workbook.xml so Excel recalculates all formulas on open."""
        if b"fullCalcOnLoad" in wb_xml:
            # Already has the attribute, ensure it's set to "1"
            wb_xml = re.sub(rb'fullCalcOnLoad="[^"]*"', b'fullCalcOnLoad="1"', wb_xml)
        elif b"<calcPr" in wb_xml:
            # calcPr exists, add the attribute
            wb_xml = wb_xml.replace(b"<calcPr", b'<calcPr fullCalcOnLoad="1"', 1)
        else:
            # No calcPr element, insert one before </workbook>
            wb_xml = wb_xml.replace(
                b"</workbook>",
                b'<calcPr fullCalcOnLoad="1"/></workbook>',
            )
        return wb_xml

    @classmethod
    def _find_row_byte_positions(
        cls, xml_bytes: bytes, target_rows: set
    ) -> Dict[int, Tuple[int, int]]:
        """
        Find byte (start, end) positions for specific row numbers in one scan.
        Stops early once all targets are found (fast for rows near the start).
        """
        positions: Dict[int, Tuple[int, int]] = {}
        target_set = set(target_rows) if not isinstance(target_rows, set) else target_rows
        if not target_set:
            return positions

        for m in re.finditer(rb'<row\s[^>]*r="(\d+)"', xml_bytes):
            row_num = int(m.group(1))
            if row_num not in target_set:
                continue

            start = m.start()
            tag_end = xml_bytes.index(b'>', m.end())

            if xml_bytes[tag_end - 1:tag_end] == b'/':
                end = tag_end + 1
            else:
                close = xml_bytes.find(b'</row>', tag_end)
                end = close + 6  # len('</row>')

            positions[row_num] = (start, end)
            if len(positions) == len(target_set):
                break

        return positions

    @classmethod
    def _parse_single_row(cls, row_bytes: bytes) -> ET.Element:
        """Parse a single <row> element from raw bytes with proper namespace handling."""
        cls._register_all_ns()
        ns_decls = ' '.join(
            f'xmlns:{prefix}="{uri}"' if prefix else f'xmlns="{uri}"'
            for prefix, uri in cls._ALL_NS.items()
        )
        wrapped = f'<_w {ns_decls}>'.encode() + row_bytes + b'</_w>'
        wrapper = ET.fromstring(wrapped)
        return wrapper[0]

    @classmethod
    def _serialize_row(cls, row_el: ET.Element) -> bytes:
        """Serialize a row element, stripping inherited namespace declarations."""
        cls._register_all_ns()
        row_bytes = ET.tostring(row_el, encoding="unicode").encode("utf-8")
        for prefix, uri in cls._ALL_NS.items():
            if prefix:
                decl = f' xmlns:{prefix}="{uri}"'.encode()
            else:
                decl = f' xmlns="{uri}"'.encode()
            row_bytes = row_bytes.replace(decl, b'', 1)
        return row_bytes

    @staticmethod
    def _cell_has_value(xml_bytes: bytes, match_end: int) -> bool:
        """Check if a <c> cell has a <v> or <is> value (not self-closing)."""
        gt = xml_bytes.find(b'>', match_end)
        if gt < 0:
            return False
        # Self-closing tag: <c .../>
        if xml_bytes[gt - 1:gt] == b'/':
            return False
        # Find </c> closing tag
        close = xml_bytes.find(b'</c>', gt)
        if close < 0:
            return False
        content = xml_bytes[gt + 1:close]
        return b'<v>' in content or b'<is>' in content

    def _find_last_data_row_bytes(self, xml_bytes: bytes) -> int:
        """Find last row with data in column C using byte scanning."""
        last_row = self.FIRST_DATA_ROW - 1
        for m in re.finditer(rb'<c\s[^>]*?r="C(\d+)"', xml_bytes):
            row_num = int(m.group(1))
            if self._cell_has_value(xml_bytes, m.end()):
                last_row = max(last_row, row_num)
        return last_row

    def _extract_row4_templates_bytes(
        self, xml_bytes: bytes
    ) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Extract formula and style templates from row 4 via byte-level extraction."""
        positions = self._find_row_byte_positions(xml_bytes, {4})
        if 4 not in positions:
            return {}, {}

        start, end = positions[4]
        row4_el = self._parse_single_row(xml_bytes[start:end])

        ns = self._NS
        formula_templates: Dict[int, str] = {}
        style_templates: Dict[int, str] = {}

        for cell in row4_el:
            ref = cell.get("r", "")
            col_letter = re.match(r"([A-Z]+)", ref)
            if not col_letter:
                continue
            col_num = column_index_from_string(col_letter.group(1))

            f_el = cell.find(f"{{{ns}}}f")
            if f_el is not None and f_el.text:
                formula_templates[col_num] = f_el.text

            style_id = cell.get("s")
            if style_id:
                style_templates[col_num] = style_id

        return formula_templates, style_templates

    # ========== End byte-level helpers ==========

    def update_config(
        self,
        file_path: Path,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal,
        period_name: str,
    ) -> None:
        """
        Update Summary sheet configuration via direct XML/ZIP manipulation.

        Summary sheet structure:
        - B1: Date (ending date)
        - B3: FX rate (VND/USD)
        - B4: Week/Period name
        - B5: Opening date
        - B6: Ending date
        """
        with open(file_path, "rb") as f:
            zip_data = f.read()

        sheet_paths = self._get_sheet_xml_paths(zip_data)
        summary_path = sheet_paths.get("Summary")
        if not summary_path:
            raise ValueError("Summary sheet not found")

        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as src_zip:
            summary_xml = src_zip.read(summary_path)
            new_summary_xml = self._modify_summary_xml(
                summary_xml, opening_date, ending_date, fx_rate, period_name
            )
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as dst_zip:
                for entry in src_zip.namelist():
                    if "calcChain" in entry:
                        continue
                    if entry == summary_path:
                        dst_zip.writestr(entry, new_summary_xml)
                    elif entry == "[Content_Types].xml":
                        ct_xml = src_zip.read(entry)
                        ct_xml = re.sub(
                            rb'<Override[^>]*calcChain[^>]*/>', b'', ct_xml
                        )
                        dst_zip.writestr(entry, ct_xml)
                    else:
                        dst_zip.writestr(entry, src_zip.read(entry))

        with open(file_path, "wb") as f:
            f.write(output.getvalue())
        logger.info(f"Updated config via XML: {file_path.name}")

    def clear_movement_data(self, file_path: Path) -> int:
        """
        Clear Movement sheet data rows (row 5+) via byte-level XML manipulation.
        Keeps rows 1-3 as-is, clears input values from row 4.
        Does NOT parse the full XML tree (handles 1M-row templates efficiently).
        """
        zip_data, movement_path, mov_xml = self._read_movement_xml(file_path)
        ns = self._NS
        self._register_all_ns()

        # Find row 4 for modification
        row4_pos = self._find_row_byte_positions(mov_xml, {4})

        # Find first row >= 5
        first_ge5_start = None
        for m in re.finditer(rb'<row\s[^>]*r="(\d+)"', mov_xml):
            rn = int(m.group(1))
            if rn >= 5:
                first_ge5_start = m.start()
                break

        # Modify row 4 (clear input cells)
        new_row4 = None
        if 4 in row4_pos:
            r4_start, r4_end = row4_pos[4]
            row4_el = self._parse_single_row(mov_xml[r4_start:r4_end])
            input_cols = {"A", "B", "C", "D", "E", "F", "G", "I"}
            for cell in list(row4_el):
                ref = cell.get("r", "")
                col_match = re.match(r"([A-Z]+)", ref)
                if col_match and col_match.group(1) in input_cols:
                    f_el = cell.find(f"{{{ns}}}f")
                    if f_el is None:
                        # Clear value but keep cell with style for template extraction
                        for child in list(cell):
                            cell.remove(child)
                        cell.attrib.pop("t", None)
            new_row4 = self._serialize_row(row4_el)

        if first_ge5_start is None and new_row4 is None:
            return 0

        # Count rows being cleared (fast byte count)
        sd_close = mov_xml.rfind(b'</sheetData>')
        rows_cleared = 0
        if first_ge5_start is not None:
            section = mov_xml[first_ge5_start:sd_close]
            rows_cleared = len(re.findall(rb'<row\s', section))

        # Build result
        parts = []
        if new_row4 and 4 in row4_pos:
            r4_start, r4_end = row4_pos[4]
            parts.append(mov_xml[:r4_start])
            parts.append(new_row4)
            if first_ge5_start is not None:
                parts.append(b'\n')
                parts.append(mov_xml[sd_close:])
            else:
                parts.append(mov_xml[r4_end:])
        elif first_ge5_start is not None:
            parts.append(mov_xml[:first_ge5_start])
            parts.append(mov_xml[sd_close:])

        new_mov_xml = b''.join(parts)
        self._write_sheet_xml(file_path, zip_data, movement_path, new_mov_xml)

        logger.info(f"Cleared {rows_cleared} rows via byte-level XML")
        return rows_cleared

    def append_transactions(
        self,
        file_path: Path,
        transactions: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Append transactions to Movement sheet via byte-level XML manipulation.

        Instead of parsing the full sheet XML (which hangs for 1M-row templates),
        this method:
        1. Scans raw bytes to find row 4 (templates) and last data row
        2. Finds only the target rows' byte positions
        3. Parses/modifies individual rows (fast, small XML chunks)
        4. Splices modified row bytes into the original XML
        """
        if not transactions:
            return 0, 0

        zip_data, movement_path, mov_xml = self._read_movement_xml(file_path)
        ns = self._NS
        self._register_all_ns()

        # Step 1: Extract templates from row 4
        formula_templates, style_templates = self._extract_row4_templates_bytes(mov_xml)

        # Always look up correct date style from styles.xml for column D(4).
        # Row 4 may have a style, but it could be a general style, not a date format.
        date_style = self._find_date_style_index(zip_data)
        if date_style:
            style_templates[4] = date_style
            logger.info(f"Applied date style s={date_style} from styles.xml")

        # Step 2: Find last data row (scan column C for values)
        last_data_row = self._find_last_data_row_bytes(mov_xml)
        start_row = max(last_data_row + 1, self.FIRST_DATA_ROW)

        # Step 3: Find byte positions of target rows (only the ones we need)
        target_row_nums = set(range(start_row, start_row + len(transactions)))
        row_positions = self._find_row_byte_positions(mov_xml, target_row_nums)

        # Step 4: Build modifications
        # modifications: (byte_start, byte_end, new_row_bytes) for existing rows
        # new_rows: bytes for rows that don't exist in the template
        modifications = []
        new_rows = []

        for i, tx in enumerate(transactions):
            row_num = start_row + i

            if row_num in row_positions:
                # Existing row (has formula cells) - merge input data
                start, end = row_positions[row_num]
                row_el = self._parse_single_row(mov_xml[start:end])
                self._merge_input_into_row_xml(ns, row_el, row_num, tx, style_templates)
                new_bytes = self._serialize_row(row_el)
                modifications.append((start, end, new_bytes))
            else:
                # Row doesn't exist - build complete new row
                row_el = self._build_movement_row_xml(
                    ns, row_num, tx, formula_templates, style_templates
                )
                new_rows.append(self._serialize_row(row_el))

        # Step 5: Apply modifications in one pass (preserves byte positions)
        modifications.sort(key=lambda x: x[0])
        parts = []
        last_end = 0
        for start, end, new_bytes in modifications:
            parts.append(mov_xml[last_end:start])
            parts.append(new_bytes)
            last_end = end

        # Insert new rows (if any) before </sheetData>
        remaining = mov_xml[last_end:]
        if new_rows:
            sd_close = remaining.rfind(b'</sheetData>')
            if sd_close >= 0:
                parts.append(remaining[:sd_close])
                parts.extend(new_rows)
                parts.append(remaining[sd_close:])
            else:
                parts.append(remaining)
                parts.extend(new_rows)
        else:
            parts.append(remaining)

        new_mov_xml = b''.join(parts)

        # Step 6: Update dimension to reflect actual data range
        last_written_row = start_row + len(transactions) - 1
        new_mov_xml = self._update_dimension(new_mov_xml, last_written_row)

        # Step 7: Write back
        self._write_sheet_xml(file_path, zip_data, movement_path, new_mov_xml)

        rows_added = len(transactions)
        total_rows = (start_row - self.FIRST_DATA_ROW) + rows_added
        logger.info(f"Appended {rows_added} transactions via byte-level XML, total: {total_rows}")
        return rows_added, total_rows

    def _merge_input_into_row_xml(
        self,
        ns: str,
        row_el: ET.Element,
        row_num: int,
        tx: Dict[str, Any],
        style_templates: Dict[int, str],
    ) -> None:
        """
        Merge input data (columns A-G, I) into an existing row that already
        has formula cells. Preserves all existing formula cells.
        """
        input_data = {
            1: ("str", tx.get("source", "")),
            2: ("str", tx.get("bank", "")),
            3: ("str", str(tx.get("account", ""))),
            4: ("date", tx.get("date")),
            5: ("str", tx.get("description", "")),
            6: ("num", tx.get("debit")),
            7: ("num", tx.get("credit")),
            9: ("str", tx.get("nature", "")),
        }

        # Collect existing cell styles as fallback (before removing them)
        existing_styles: Dict[int, str] = {}
        for cell in row_el:
            ref = cell.get("r", "")
            col_match = re.match(r"([A-Z]+)", ref)
            if col_match:
                col_num = column_index_from_string(col_match.group(1))
                s = cell.get("s")
                if s:
                    existing_styles[col_num] = s

        # Build new input cells
        new_cells = []
        for col_num, (dtype, value) in input_data.items():
            col_letter = get_column_letter(col_num)
            ref = f"{col_letter}{row_num}"
            cell = ET.Element(f"{{{ns}}}c")
            cell.set("r", ref)

            style = style_templates.get(col_num) or existing_styles.get(col_num)
            if style:
                cell.set("s", style)

            if dtype == "str" and value:
                cell.set("t", "inlineStr")
                is_el = ET.SubElement(cell, f"{{{ns}}}is")
                t_el = ET.SubElement(is_el, f"{{{ns}}}t")
                t_el.text = str(value)
            elif dtype == "num" and value is not None:
                v_el = ET.SubElement(cell, f"{{{ns}}}v")
                v_el.text = str(float(value))
            elif dtype == "date" and value is not None:
                v_el = ET.SubElement(cell, f"{{{ns}}}v")
                if isinstance(value, (date, datetime)):
                    v_el.text = str(self._date_to_serial(
                        value if isinstance(value, date) else value.date()
                    ))
                else:
                    v_el.text = str(value)

            new_cells.append((col_num, cell))

        # Collect existing cells, replacing input columns
        existing_by_col: Dict[int, ET.Element] = {}
        for cell in list(row_el):
            ref = cell.get("r", "")
            col_match = re.match(r"([A-Z]+)", ref)
            if col_match:
                col_num = column_index_from_string(col_match.group(1))
                existing_by_col[col_num] = cell

        # Remove existing input column cells (will be replaced)
        input_col_nums = set(input_data.keys())
        for col_num in input_col_nums:
            if col_num in existing_by_col:
                row_el.remove(existing_by_col[col_num])
                del existing_by_col[col_num]

        # Remove cached <v> from formula cells to force Excel recalculation
        for col_num, cell in existing_by_col.items():
            f_el = cell.find(f"{{{ns}}}f")
            if f_el is not None:
                v_el = cell.find(f"{{{ns}}}v")
                if v_el is not None:
                    cell.remove(v_el)

        # Merge: add new input cells + keep existing formula cells
        all_cells = list(existing_by_col.items()) + new_cells
        all_cells.sort(key=lambda x: x[0])

        # Clear and rebuild row children in order
        for child in list(row_el):
            row_el.remove(child)
        for _, cell in all_cells:
            row_el.append(cell)

    def _build_movement_row_xml(
        self,
        ns: str,
        row_num: int,
        tx: Dict[str, Any],
        formula_templates: Dict[int, str],
        style_templates: Dict[int, str],
    ) -> ET.Element:
        """Build a complete <row> XML element for a transaction."""
        row_el = ET.Element(f"{{{ns}}}row")
        row_el.set("r", str(row_num))

        cells = []

        # Input columns: A(1), B(2), C(3), D(4), E(5), F(6), G(7), I(9)
        input_data = {
            1: ("str", tx.get("source", "")),
            2: ("str", tx.get("bank", "")),
            3: ("str", str(tx.get("account", ""))),
            4: ("date", tx.get("date")),
            5: ("str", tx.get("description", "")),
            6: ("num", tx.get("debit")),
            7: ("num", tx.get("credit")),
            9: ("str", tx.get("nature", "")),
        }

        for col_num, (dtype, value) in input_data.items():
            col_letter = get_column_letter(col_num)
            ref = f"{col_letter}{row_num}"
            cell = ET.Element(f"{{{ns}}}c")
            cell.set("r", ref)

            # Apply style from template
            style = style_templates.get(col_num)
            if style:
                cell.set("s", style)

            if dtype == "str" and value:
                cell.set("t", "inlineStr")
                is_el = ET.SubElement(cell, f"{{{ns}}}is")
                t_el = ET.SubElement(is_el, f"{{{ns}}}t")
                t_el.text = str(value)
            elif dtype == "num" and value is not None:
                v_el = ET.SubElement(cell, f"{{{ns}}}v")
                v_el.text = str(float(value))
            elif dtype == "date" and value is not None:
                v_el = ET.SubElement(cell, f"{{{ns}}}v")
                if isinstance(value, (date, datetime)):
                    v_el.text = str(self._date_to_serial(value if isinstance(value, date) else value.date()))
                else:
                    v_el.text = str(value)

            cells.append((col_num, cell))

        # Formula columns
        for col_num, formula in formula_templates.items():
            col_letter = get_column_letter(col_num)
            ref = f"{col_letter}{row_num}"
            cell = ET.Element(f"{{{ns}}}c")
            cell.set("r", ref)

            style = style_templates.get(col_num)
            if style:
                cell.set("s", style)

            adjusted = self._adjust_formula(
                f"={formula}", self.FIRST_DATA_ROW, row_num
            )
            f_el = ET.SubElement(cell, f"{{{ns}}}f")
            f_el.text = adjusted[1:]  # Remove leading '='

            cells.append((col_num, cell))

        # Sort by column number and add to row
        cells.sort(key=lambda x: x[0])
        for _, cell in cells:
            row_el.append(cell)

        return row_el

    def _adjust_formula(self, formula: str, source_row: int, target_row: int) -> str:
        """
        Adjust formula row references from source_row to target_row.
        Handles both relative and absolute ($) references.

        Examples:
            =F4-G4 -> =F23-G23 (when target_row=23)
            =VLOOKUP(I4,Lookup!$A:$B,2,FALSE) -> =VLOOKUP(I23,Lookup!$A:$B,2,FALSE)
            =$A$1 -> =$A$1 (absolute reference unchanged)
        """
        if not formula or not formula.startswith('='):
            return formula

        # Pattern to match cell references
        pattern = r'(\$?[A-Z]+)(\$?)(\d+)'

        def replace_ref(match):
            col = match.group(1)
            dollar = match.group(2)
            row_num = int(match.group(3))

            # If row is absolute ($), don't adjust
            if dollar == '$':
                return f"{col}${row_num}"

            # Only adjust if row matches source_row
            if row_num == source_row:
                return f"{col}{target_row}"

            return match.group(0)

        return re.sub(pattern, replace_ref, formula)

    @staticmethod
    def _expand_shared_formula(master_formula: str, master_row: int, target_row: int) -> str:
        """
        Expand a shared formula from master_row to target_row.
        Shifts ALL relative row references by (target_row - master_row).
        Absolute references ($) are preserved unchanged.

        Unlike _adjust_formula (which only shifts references matching source_row),
        this shifts every relative reference — required for shared formula expansion
        where cross-row references (e.g. SUM ranges) must also shift.
        """
        if not master_formula:
            return master_formula

        delta = target_row - master_row
        if delta == 0:
            return master_formula

        pattern = r'(\$?[A-Z]+)(\$?)(\d+)'

        def replace_ref(match):
            col = match.group(1)
            dollar = match.group(2)
            row_num = int(match.group(3))

            if dollar == '$':
                return f"{col}${row_num}"

            return f"{col}{row_num + delta}"

        return re.sub(pattern, replace_ref, master_formula)

    def _read_sheet_xml(self, file_path: Path, sheet_name: str) -> Tuple[bytes, str, bytes]:
        """Read ZIP data and extract a named sheet's XML bytes."""
        with open(file_path, "rb") as f:
            zip_data = f.read()
        sheet_paths = self._get_sheet_xml_paths(zip_data)
        sheet_path = sheet_paths.get(sheet_name)
        if not sheet_path:
            raise ValueError(f"{sheet_name} sheet not found in workbook")
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as z:
            sheet_xml = z.read(sheet_path)
        return zip_data, sheet_path, sheet_xml

    def remove_zero_closing_balance_saving_rows(self, file_path: Path) -> int:
        """
        Remove rows from Saving Account sheet where CLOSING BALANCE (VND) = 0.

        Phase 1: Find CLOSING BALANCE (VND) column letter from headers (openpyxl).
        Phase 2: Read XML directly to detect zero values + collect shared formulas.
        Phase 3: Remove rows, convert shared formulas to regular, renumber.

        Returns:
            Number of rows removed
        """
        SAVING_SHEET = "Saving Account"
        DATA_START_ROW = 4

        # Phase 1: Find CLOSING BALANCE (VND) column letter from headers
        wb = load_workbook(file_path, data_only=True, read_only=True)
        try:
            if SAVING_SHEET not in wb.sheetnames:
                logger.warning("Saving Account sheet not found, skipping zero-balance cleanup")
                return 0

            ws = wb[SAVING_SHEET]
            closing_vnd_col = None  # column letter (e.g. "F")
            for row in ws.iter_rows(min_row=1, max_row=3, values_only=False):
                for cell in row:
                    val = str(cell.value or "").upper()
                    if "CLOSING BALANCE" in val and "VND" in val:
                        closing_vnd_col = get_column_letter(cell.column)
                        break
                if closing_vnd_col:
                    break
        finally:
            wb.close()

        if not closing_vnd_col:
            logger.warning("CLOSING BALANCE (VND) column not found in Saving Account headers")
            return 0

        # Phase 2: Read sheet XML, detect zero-balance rows + collect shared formulas
        zip_data, sheet_path, sheet_xml = self._read_sheet_xml(file_path, SAVING_SHEET)
        self._register_all_ns()
        ns = self._NS

        last_data_row = self._find_last_data_row_bytes(sheet_xml)
        if last_data_row < DATA_START_ROW:
            return 0

        target_rows = set(range(DATA_START_ROW, last_data_row + 1))
        row_positions = self._find_row_byte_positions(sheet_xml, target_rows)
        if not row_positions:
            return 0

        rows_to_remove = set()
        shared_formulas = {}  # si -> (formula_text, master_row)

        for row_num in sorted(row_positions.keys()):
            start, end = row_positions[row_num]
            row_el = self._parse_single_row(sheet_xml[start:end])

            has_account = False
            closing_is_zero = False

            for cell in row_el:
                ref = cell.get("r", "")
                col_match = re.match(r"([A-Z]+)", ref)
                if not col_match:
                    continue
                col = col_match.group(1)

                # Check column C for account data
                if col == "C":
                    v_el = cell.find(f"{{{ns}}}v")
                    is_el = cell.find(f"{{{ns}}}is")
                    if (v_el is not None and v_el.text) or is_el is not None:
                        has_account = True

                # Check CLOSING BALANCE (VND) column for zero via <v> element
                if col == closing_vnd_col:
                    v_el = cell.find(f"{{{ns}}}v")
                    if v_el is not None and v_el.text:
                        try:
                            if float(v_el.text) == 0:
                                closing_is_zero = True
                        except (ValueError, TypeError):
                            pass

                # Collect shared formula masters (needed for Phase 3)
                f_el = cell.find(f"{{{ns}}}f")
                if f_el is not None and f_el.get("t") == "shared" and f_el.text:
                    si = f_el.get("si")
                    if si is not None and si not in shared_formulas:
                        shared_formulas[si] = (f_el.text, row_num)

            if has_account and closing_is_zero:
                rows_to_remove.add(row_num)

        if not rows_to_remove:
            logger.info("No zero-closing-balance rows found in Saving Account sheet")
            return 0

        # Phase 3: Remove rows, convert shared formulas, renumber
        remaining_rows = []
        new_row_num = DATA_START_ROW

        for row_num in sorted(row_positions.keys()):
            if row_num in rows_to_remove:
                continue

            start, end = row_positions[row_num]
            row_el = self._parse_single_row(sheet_xml[start:end])

            # Renumber row
            row_el.set("r", str(new_row_num))
            for cell in row_el:
                ref = cell.get("r", "")
                col_match = re.match(r"([A-Z]+)", ref)
                if col_match:
                    cell.set("r", f"{col_match.group(1)}{new_row_num}")

                # Handle formulas
                f_el = cell.find(f"{{{ns}}}f")
                if f_el is not None:
                    if f_el.get("t") == "shared":
                        # Convert shared formula → regular formula
                        si = f_el.get("si")
                        if f_el.text:
                            # Master cell: expand from own row to new row
                            f_el.text = self._expand_shared_formula(
                                f_el.text, row_num, new_row_num
                            )
                        elif si and si in shared_formulas:
                            # Dependent cell: expand from master row to new row
                            master_formula, master_row = shared_formulas[si]
                            f_el.text = self._expand_shared_formula(
                                master_formula, master_row, new_row_num
                            )
                        # Strip shared formula attributes
                        for attr in ("t", "si", "ref"):
                            if attr in f_el.attrib:
                                del f_el.attrib[attr]
                    elif f_el.text and row_num != new_row_num:
                        # Regular formula: adjust row references
                        f_el.text = self._expand_shared_formula(
                            f_el.text, row_num, new_row_num
                        )

            row_bytes = self._serialize_row(row_el)
            remaining_rows.append(row_bytes)
            new_row_num += 1

        # Replace data row region
        sorted_positions = sorted(row_positions.values())
        first_start = sorted_positions[0][0]
        last_end = sorted_positions[-1][1]

        new_sheet_xml = b''.join([
            sheet_xml[:first_start],
            *remaining_rows,
            sheet_xml[last_end:],
        ])

        # Update dimension
        if remaining_rows:
            new_sheet_xml = self._update_dimension(new_sheet_xml, new_row_num - 1)

        self._write_sheet_xml(file_path, zip_data, sheet_path, new_sheet_xml)

        logger.info(f"Removed {len(rows_to_remove)} zero-closing-balance rows from Saving Account sheet")
        return len(rows_to_remove)

    def remove_rows_by_source(self, file_path: Path, source_name: str) -> int:
        """
        Remove rows with matching source name (column A) via byte-level manipulation.
        Only scans data rows (up to last data row), not the full 1M-row template.
        """
        zip_data, movement_path, mov_xml = self._read_movement_xml(file_path)
        self._register_all_ns()

        # Find last data row to limit our scan
        last_data_row = self._find_last_data_row_bytes(mov_xml)
        if last_data_row < self.FIRST_DATA_ROW:
            return 0

        # Find byte positions of all data rows
        target_rows = set(range(self.FIRST_DATA_ROW, last_data_row + 1))
        row_positions = self._find_row_byte_positions(mov_xml, target_rows)

        if not row_positions:
            return 0

        # Check column A for matching source
        rows_to_remove = set()
        for row_num in sorted(row_positions.keys()):
            start, end = row_positions[row_num]
            row_bytes = mov_xml[start:end]

            a_cell = re.search(
                rb'<c\s[^>]*r="A\d+"[^>]*>.*?</c>', row_bytes, re.DOTALL
            )
            if a_cell:
                t_match = re.search(rb'<t[^>]*>(.*?)</t>', a_cell.group(0), re.DOTALL)
                if t_match:
                    val = t_match.group(1).decode("utf-8", errors="replace").strip()
                    if val == source_name.strip():
                        rows_to_remove.add(row_num)

        if not rows_to_remove:
            return 0

        # Build remaining data rows with new numbering
        remaining_rows = []
        new_row_num = self.FIRST_DATA_ROW

        for row_num in sorted(row_positions.keys()):
            if row_num in rows_to_remove:
                continue

            start, end = row_positions[row_num]
            row_bytes = mov_xml[start:end]

            if row_num != new_row_num:
                # Renumber this row
                row_el = self._parse_single_row(row_bytes)
                row_el.set("r", str(new_row_num))
                for cell in row_el:
                    ref = cell.get("r", "")
                    col_match = re.match(r"([A-Z]+)", ref)
                    if col_match:
                        cell.set("r", f"{col_match.group(1)}{new_row_num}")
                row_bytes = self._serialize_row(row_el)

            remaining_rows.append(row_bytes)
            new_row_num += 1

        # Replace the data row region in the original XML
        # Everything before first data row + remaining rows + everything after last data row
        sorted_positions = sorted(row_positions.values())
        first_start = sorted_positions[0][0]
        last_end = sorted_positions[-1][1]

        parts = [
            mov_xml[:first_start],
            *remaining_rows,
            mov_xml[last_end:],  # preserves formula-only rows beyond data
        ]

        new_mov_xml = b''.join(parts)
        self._write_sheet_xml(file_path, zip_data, movement_path, new_mov_xml)

        logger.info(f"Removed {len(rows_to_remove)} rows with source '{source_name}' via byte-level XML")
        return len(rows_to_remove)

    def get_movement_row_count(self, file_path: Path) -> int:
        """Get the number of data rows in Movement sheet via byte scanning."""
        try:
            _, _, mov_xml = self._read_movement_xml(file_path)
        except ValueError:
            return 0

        count = 0
        for m in re.finditer(rb'<c\s[^>]*?r="C(\d+)"', mov_xml):
            row_num = int(m.group(1))
            if row_num >= self.FIRST_DATA_ROW:
                if self._cell_has_value(mov_xml, m.end()):
                    count += 1
        return count

    def append_transactions_batch(
        self,
        file_path: Path,
        transactions: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> Tuple[int, int]:
        """
        Append transactions in batches for very large datasets.

        This method saves periodically to prevent memory issues
        with very large transaction lists.

        Args:
            file_path: Path to Excel file
            transactions: List of transaction dicts
            batch_size: Number of transactions per batch

        Returns:
            Tuple of (rows_added, total_rows)
        """
        if not transactions:
            return 0, 0

        total_added = 0
        total_rows = 0

        # Process in batches
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            rows_added, total_rows = self.append_transactions(file_path, batch)
            total_added += rows_added
            logger.info(f"Batch {i // batch_size + 1}: added {rows_added} rows")

        return total_added, total_rows

    # Column mapping: Cash Balance -> Cash balance (Prior period)
    # Cash Balance columns (source)
    CB_COL_ENTITY = 1       # A
    CB_COL_BRANCH = 2       # B
    CB_COL_ACCOUNT = 3      # C
    CB_COL_ACCOUNT_TYPE = 4 # D
    CB_COL_CURRENCY = 5     # E
    CB_COL_CLOSING_VND = 12 # L - CLOSING BALANCE (VND)2
    CB_COL_CLOSING_USD = 20 # T - CLOSING BALANCE (USD)
    CB_COL_NAME = 23        # W
    CB_COL_BANK_1 = 24      # X
    CB_COL_BANK = 25        # Y

    # Prior period columns (target)
    PP_COL_ENTITY = 1       # A
    PP_COL_BRANCH = 2       # B
    PP_COL_ACCOUNT = 3      # C
    PP_COL_ACCOUNT_TYPE = 4 # D
    PP_COL_CURRENCY = 5     # E
    PP_COL_CLOSING_VND = 6  # F
    PP_COL_CLOSING_USD = 7  # G
    PP_COL_NAME = 8         # H
    PP_COL_BANK_1 = 9       # I
    PP_COL_BANK = 10        # J

    PP_SHEET_NAME = "Cash balance (Prior period)"
    CB_SHEET_NAME = "Cash Balance"

    def _read_cash_balance_data(self, file_path: Path) -> Tuple[str, List[tuple]]:
        """
        Read computed values from Cash Balance sheet using data_only + read_only mode.

        Returns:
            Tuple of (period_name, list_of_row_tuples)
        """
        wb = load_workbook(file_path, data_only=True, read_only=True)
        try:
            # Read period name from Summary
            summary_ws = wb["Summary"]
            period_name = ""
            for row in summary_ws.iter_rows(min_row=4, max_row=4, min_col=2, max_col=2):
                period_name = row[0].value or ""

            # Read Cash Balance data using iter_rows (fast in read_only mode)
            cb_ws = wb[self.CB_SHEET_NAME]
            rows_data = []

            # Read columns: A,B,C,D,E (1-5), L(12), T(20), W(23), X(24), Y(25)
            for row in cb_ws.iter_rows(min_row=4, max_col=25):
                account = row[self.CB_COL_ACCOUNT - 1].value  # C (index 2)
                if not account or not str(account).strip():
                    continue

                # Closing balances might be non-numeric (e.g. "x" for inactive accounts)
                closing_vnd = row[self.CB_COL_CLOSING_VND - 1].value
                closing_usd = row[self.CB_COL_CLOSING_USD - 1].value
                if not isinstance(closing_vnd, (int, float)):
                    closing_vnd = 0
                if not isinstance(closing_usd, (int, float)):
                    closing_usd = 0

                rows_data.append((
                    row[self.CB_COL_ENTITY - 1].value,
                    row[self.CB_COL_BRANCH - 1].value,
                    row[self.CB_COL_ACCOUNT - 1].value,
                    row[self.CB_COL_ACCOUNT_TYPE - 1].value,
                    row[self.CB_COL_CURRENCY - 1].value,
                    closing_vnd,
                    closing_usd,
                    row[self.CB_COL_NAME - 1].value,
                    row[self.CB_COL_BANK_1 - 1].value,
                    row[self.CB_COL_BANK - 1].value,
                ))
        finally:
            wb.close()

        return period_name, rows_data

    # XML namespace used in xlsx sheet files
    _NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    _NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    # All namespaces used in Excel sheet XML - must register ALL to prevent
    # ElementTree from mangling prefix names (e.g. xr -> ns2)
    _ALL_NS = {
        "": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
        "x14ac": "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac",
        "xr": "http://schemas.microsoft.com/office/spreadsheetml/2014/revision",
        "xr2": "http://schemas.microsoft.com/office/spreadsheetml/2015/revision2",
        "xr3": "http://schemas.microsoft.com/office/spreadsheetml/2016/revision3",
    }

    @classmethod
    def _register_all_ns(cls):
        """Register all Excel XML namespaces to preserve prefixes during serialization."""
        for prefix, uri in cls._ALL_NS.items():
            ET.register_namespace(prefix, uri)

    @classmethod
    def _to_xml_bytes(cls, root: ET.Element) -> bytes:
        """Serialize XML element to bytes with proper Excel XML declaration."""
        xml_bytes = ET.tostring(root, xml_declaration=True, encoding="UTF-8")
        # ET uses standalone='no' or omits it; Excel requires standalone="yes"
        xml_bytes = xml_bytes.replace(
            b"<?xml version='1.0' encoding='UTF-8'?>",
            b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        )
        return xml_bytes

    @classmethod
    def _patch_sheet_data(cls, original_xml: bytes, root: ET.Element) -> bytes:
        """
        Replace only <sheetData>...</sheetData> in the original XML bytes,
        preserving the original XML declaration, worksheet tag, namespaces,
        and all other elements (sheetViews, pageSetup, etc.) byte-for-byte.

        This avoids ElementTree mangling namespace prefixes (xr->ns2, dropping xr2/xr3).
        """
        ns = cls._NS
        # Serialize only the <sheetData> element from the modified tree
        sheet_data = root.find(f"{{{ns}}}sheetData")
        cls._register_all_ns()
        new_sd_bytes = ET.tostring(sheet_data, encoding="unicode").encode("utf-8")

        # Find <sheetData...> and </sheetData> in original bytes
        sd_open = re.search(rb'<sheetData[^>]*/?>', original_xml)
        sd_close_pos = original_xml.rfind(b'</sheetData>')

        if sd_open is None:
            # No sheetData in original - fall back to full serialization
            return cls._to_xml_bytes(root)

        if sd_close_pos > sd_open.start():
            # Normal case: <sheetData>...</sheetData>
            return (
                original_xml[:sd_open.start()]
                + new_sd_bytes
                + original_xml[sd_close_pos + len(b'</sheetData>'):]
            )
        else:
            # Self-closing: <sheetData/>
            return (
                original_xml[:sd_open.start()]
                + new_sd_bytes
                + original_xml[sd_open.end():]
            )

    @staticmethod
    def _date_to_serial(d: date) -> int:
        """Convert Python date to Excel serial number (1900 date system)."""
        return (d - date(1899, 12, 30)).days

    @classmethod
    def _find_date_style_index(cls, zip_data: bytes) -> Optional[str]:
        """
        Find a cell style index (xf) that uses d/m/yyyy date format.
        Falls back to any date format. Returns style index as string, or None.
        """
        ns = cls._NS
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            styles_xml = z.read("xl/styles.xml")

        root = ET.fromstring(styles_xml)

        # Collect custom numFmtIds that look like date formats
        custom_date_ids: Dict[int, str] = {}
        for nf in root.iter(f"{{{ns}}}numFmt"):
            code = nf.get("formatCode", "")
            fid = int(nf.get("numFmtId", "0"))
            if ("d" in code.lower() or "y" in code.lower()) and "m" in code.lower():
                custom_date_ids[fid] = code

        # Built-in date format IDs (14-22)
        builtin_date_ids = set(range(14, 23))

        # Prefer d/m/yyyy (numFmtId=173 in this template)
        preferred_ids = [
            fid for fid, code in custom_date_ids.items()
            if "d/m/yyyy" in code.lower() or "dd/mm/yyyy" in code.lower()
        ]

        logger.info(
            f"Date style lookup: custom_date_ids={custom_date_ids}, "
            f"preferred_ids={preferred_ids}"
        )

        # Search cellXfs for matching style
        cellXfs = root.find(f"{{{ns}}}cellXfs")
        if cellXfs is None:
            return None

        fallback = None
        fallback_any = None
        for idx, xf in enumerate(cellXfs):
            nf_id = int(xf.get("numFmtId", "0"))
            if nf_id in preferred_ids:
                return str(idx)
            if nf_id in builtin_date_ids or nf_id in custom_date_ids:
                if fallback is None and xf.get("applyNumberFormat") == "1":
                    fallback = str(idx)
                elif fallback_any is None:
                    fallback_any = str(idx)

        return fallback or fallback_any

    @staticmethod
    def _update_dimension(xml_bytes: bytes, last_row: int) -> bytes:
        """
        Update <dimension ref="..."> to reflect actual data range.
        Keeps original start ref and end column, updates end row.
        """
        match = re.search(rb'<dimension ref="([A-Z]+\d+):([A-Z]+)\d+"', xml_bytes)
        if match:
            start_ref = match.group(1).decode()
            end_col = match.group(2).decode()
            new_tag = f'<dimension ref="{start_ref}:{end_col}{last_row}"'.encode()
            return xml_bytes[:match.start()] + new_tag + xml_bytes[match.end():]
        return xml_bytes

    @staticmethod
    def _get_sheet_xml_paths(zip_data: bytes) -> Dict[str, str]:
        """
        Map sheet names to their XML file paths inside the xlsx zip.
        Parses workbook.xml and workbook.xml.rels.
        """
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            wb_xml = z.read("xl/workbook.xml")
            wb_rels = z.read("xl/_rels/workbook.xml.rels")

        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

        # sheet name -> rId
        wb_root = ET.fromstring(wb_xml)
        sheet_rids = {}
        for sheet_el in wb_root.iter(f"{{{ns}}}sheet"):
            sheet_rids[sheet_el.get("name")] = sheet_el.get(f"{{{ns_r}}}id")

        # rId -> file path
        rels_root = ET.fromstring(wb_rels)
        rid_to_path = {}
        for rel in rels_root:
            rid_to_path[rel.get("Id")] = f"xl/{rel.get('Target')}"

        return {name: rid_to_path[rid] for name, rid in sheet_rids.items()
                if rid in rid_to_path}

    def _build_pp_cell(
        self, col_letter: str, row: int, value,
        is_number: bool = False, style: str = None
    ) -> ET.Element:
        """Build an XML <c> element for the Prior period sheet."""
        ns = self._NS
        ref = f"{col_letter}{row}"
        cell = ET.Element(f"{{{ns}}}c")
        cell.set("r", ref)

        if style:
            cell.set("s", style)

        if value is None:
            return cell

        if is_number:
            v = ET.SubElement(cell, f"{{{ns}}}v")
            v.text = str(value)
        else:
            # Use inline string to avoid shared string table manipulation
            cell.set("t", "inlineStr")
            is_el = ET.SubElement(cell, f"{{{ns}}}is")
            t_el = ET.SubElement(is_el, f"{{{ns}}}t")
            t_el.text = str(value) if value else ""

        return cell

    def _modify_prior_period_xml(
        self, xml_bytes: bytes, old_period_name: str, cb_rows: List[tuple]
    ) -> bytes:
        """
        Directly modify the Prior period sheet XML:
        - Update A2, B2 with period name
        - Replace data rows (4+) with Cash Balance closing balances
        """
        ns = self._NS
        ET.register_namespace("", ns)
        # Preserve other namespaces
        ET.register_namespace("r", self._NS_R)
        ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
        ET.register_namespace("x14ac", "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac")

        root = ET.fromstring(xml_bytes)
        sheet_data = root.find(f"{{{ns}}}sheetData")

        # Extract styles from existing row 4 (template row) before removing
        pp_col_styles: Dict[str, str] = {}  # col_letter -> style_id
        for row_el in sheet_data:
            if row_el.get("r") == "4":
                for cell in row_el:
                    ref = cell.get("r", "")
                    style = cell.get("s")
                    col = ref.rstrip("0123456789")
                    if style and col:
                        pp_col_styles[col] = style
                break

        # Separate rows: keep 1-3, remove 4+
        rows_to_keep = []
        for row_el in list(sheet_data):
            row_num = int(row_el.get("r"))
            if row_num <= 3:
                rows_to_keep.append(row_el)
            sheet_data.remove(row_el)

        # Modify row 2: update A2 and B2 cells
        for row_el in rows_to_keep:
            if row_el.get("r") == "2":
                for cell in list(row_el):
                    ref = cell.get("r")
                    if ref in ("A2", "B2"):
                        row_el.remove(cell)
                # Add new A2 and B2
                a2 = self._build_pp_cell("A", 2, old_period_name)
                b2 = self._build_pp_cell("B", 2, f"OB{old_period_name}" if old_period_name else "")
                # Insert at the beginning of row
                row_el.insert(0, b2)
                row_el.insert(0, a2)
                break

        # Re-add kept rows
        for row_el in rows_to_keep:
            sheet_data.append(row_el)

        # Column letters for Prior period: A,B,C,D,E,F,G,H,I,J
        pp_cols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        # Which are numbers: F(5) and G(6) are closing balances
        pp_is_number = [False, False, False, False, False, True, True, False, False, False]

        # Add new data rows with preserved styles
        for i, data in enumerate(cb_rows):
            row_num = 4 + i
            row_el = ET.SubElement(sheet_data, f"{{{ns}}}row")
            row_el.set("r", str(row_num))

            for col_idx, (col_letter, is_num) in enumerate(zip(pp_cols, pp_is_number)):
                value = data[col_idx]
                style = pp_col_styles.get(col_letter)
                cell = self._build_pp_cell(
                    col_letter, row_num, value,
                    is_number=is_num, style=style
                )
                row_el.append(cell)

        return self._patch_sheet_data(xml_bytes, root)

    def _modify_summary_xml(
        self,
        xml_bytes: bytes,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal,
        period_name: str,
    ) -> bytes:
        """
        Directly modify Summary sheet XML:
        - B1: ending_date, B3: fx_rate, B4: period_name
        - B5: opening_date, B6: ending_date
        """
        ns = self._NS
        self._register_all_ns()

        root = ET.fromstring(xml_bytes)
        sheet_data = root.find(f"{{{ns}}}sheetData")

        # Map of cell ref -> (value, is_number, style_id)
        updates = {
            "B1": (self._date_to_serial(ending_date), True),
            "B3": (float(fx_rate), True),
            "B4": (period_name, False),
            "B5": (self._date_to_serial(opening_date), True),
            "B6": (self._date_to_serial(ending_date), True),
        }

        for row_el in sheet_data:
            row_num = row_el.get("r")
            if row_num not in ("1", "2", "3", "4", "5", "6"):
                continue

            for cell in row_el:
                ref = cell.get("r")
                if ref not in updates:
                    continue

                value, is_number = updates[ref]
                style = cell.get("s")  # Preserve style

                # Clear existing content
                for child in list(cell):
                    cell.remove(child)

                if is_number:
                    cell.attrib.pop("t", None)  # Remove type attr for numbers
                    v = ET.SubElement(cell, f"{{{ns}}}v")
                    v.text = str(value)
                else:
                    cell.set("t", "inlineStr")
                    is_el = ET.SubElement(cell, f"{{{ns}}}is")
                    t_el = ET.SubElement(is_el, f"{{{ns}}}t")
                    t_el.text = str(value)

                if style:
                    cell.set("s", style)

        return self._patch_sheet_data(xml_bytes, root)

    def initialize_session_optimized(
        self,
        file_path: Path,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal,
        period_name: str,
    ) -> int:
        """
        Initialize session: Step 0 + config update via direct XML manipulation.

        Performance: ~1-2s total (vs ~55s with openpyxl load/save).
        1. read_only + data_only: read Cash Balance cached values (0.2s)
        2. ZIP/XML: modify Prior period + Summary sheets directly (0.5s)

        Movement sheet is untouched (template is already clean).

        Returns:
            Number of rows cleared (always 0 for fresh template)
        """
        # Phase 1: Read Cash Balance computed values (fast read_only mode)
        old_period_name, cb_rows = self._read_cash_balance_data(file_path)

        # Phase 2: Direct XML manipulation in the xlsx ZIP
        with open(file_path, "rb") as f:
            zip_data = f.read()

        # Get sheet name -> XML path mapping
        sheet_paths = self._get_sheet_xml_paths(zip_data)
        pp_path = sheet_paths[self.PP_SHEET_NAME]
        summary_path = sheet_paths["Summary"]

        # Modify sheets in-memory
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as src_zip:
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as dst_zip:
                for entry in src_zip.namelist():
                    # Skip calcChain to force Excel full recalculation
                    if "calcChain" in entry:
                        continue

                    data = src_zip.read(entry)

                    if entry == pp_path and cb_rows:
                        data = self._modify_prior_period_xml(
                            data, old_period_name, cb_rows
                        )
                    elif entry == summary_path:
                        data = self._modify_summary_xml(
                            data, opening_date, ending_date, fx_rate, period_name
                        )
                    elif entry == "[Content_Types].xml":
                        data = re.sub(
                            rb'<Override[^>]*calcChain[^>]*/>', b'', data
                        )
                    elif entry == "xl/workbook.xml":
                        data = self._set_full_calc_on_load(data)

                    dst_zip.writestr(entry, data)

        with open(file_path, "wb") as f:
            f.write(output.getvalue())

        logger.info(
            f"Session initialized (fast XML): "
            f"{len(cb_rows)} rows to Prior period (period: {old_period_name}), "
            f"Summary config updated to {period_name}"
        )

        return 0

    def highlight_rows(
        self, file_path: Path, sheet_name: str, row_numbers: List[int]
    ) -> None:
        """
        Apply orange settlement highlight to specific rows via XML manipulation.
        Does NOT use openpyxl load/save (which corrupts drawings/charts).

        Modifies only:
        - xl/styles.xml: adds a fill + cloned xf entries
        - Sheet XML: updates cell s= attributes on target rows
        All other ZIP entries (drawings, charts, etc.) are preserved byte-for-byte.
        """
        if not row_numbers:
            return

        with open(file_path, "rb") as f:
            zip_data = f.read()

        sheet_paths = self._get_sheet_xml_paths(zip_data)
        sheet_path = sheet_paths.get(sheet_name)
        if not sheet_path:
            raise ValueError(f"{sheet_name} sheet not found")

        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as z:
            styles_xml = z.read("xl/styles.xml")
            sheet_xml = z.read(sheet_path)

        self._register_all_ns()

        # --- Step 1: Collect unique style indices from target rows ---
        row_set = set(row_numbers)
        row_positions = self._find_row_byte_positions(sheet_xml, row_set)

        unique_styles = set()
        for row_num in row_numbers:
            if row_num not in row_positions:
                continue
            start, end = row_positions[row_num]
            for m in re.finditer(rb'<c\s[^>]*?s="(\d+)"', sheet_xml[start:end]):
                unique_styles.add(m.group(1).decode())

        # --- Step 2: Modify styles.xml via byte manipulation ---
        styles_xml, style_map = self._add_highlight_styles(
            styles_xml, unique_styles
        )

        # --- Step 3: Update cell styles in target rows ---
        modifications = []
        for row_num in sorted(row_numbers):
            if row_num not in row_positions:
                continue
            start, end = row_positions[row_num]
            row_el = self._parse_single_row(sheet_xml[start:end])

            for cell in row_el:
                old_s = cell.get("s")
                new_s = style_map.get(old_s, style_map.get(None))
                if new_s:
                    cell.set("s", new_s)

            new_bytes = self._serialize_row(row_el)
            modifications.append((start, end, new_bytes))

        # Apply modifications to sheet XML
        modifications.sort(key=lambda x: x[0])
        parts = []
        last_end = 0
        for start, end, new_bytes in modifications:
            parts.append(sheet_xml[last_end:start])
            parts.append(new_bytes)
            last_end = end
        parts.append(sheet_xml[last_end:])
        new_sheet_xml = b''.join(parts)

        # --- Step 4: Write back to ZIP (preserving drawings etc.) ---
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as src_zip:
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as dst_zip:
                for entry in src_zip.namelist():
                    if entry == "xl/styles.xml":
                        dst_zip.writestr(entry, styles_xml)
                    elif entry == sheet_path:
                        dst_zip.writestr(entry, new_sheet_xml)
                    else:
                        dst_zip.writestr(entry, src_zip.read(entry))

        with open(file_path, "wb") as f:
            f.write(output.getvalue())

        logger.info(f"Highlighted {len(row_numbers)} rows in '{sheet_name}' via XML")

    @staticmethod
    def _add_highlight_styles(
        styles_xml: bytes, unique_styles: set
    ) -> tuple:
        """
        Add orange fill and cloned xf entries to styles.xml via byte manipulation.
        Returns (modified_styles_xml, style_map).

        style_map: {old_s_str_or_None: new_s_str}
        """
        # --- Add fill (theme 5, tint 0.6 = orange) ---
        fill_xml = (
            b'<fill><patternFill patternType="solid">'
            b'<fgColor theme="5" tint="0.5999938962981048"/>'
            b'</patternFill></fill>'
        )

        fills_count_m = re.search(rb'<fills\s+count="(\d+)"', styles_xml)
        old_fill_count = int(fills_count_m.group(1))
        new_fill_id = old_fill_count

        # Insert fill before </fills>
        fills_close = styles_xml.find(b'</fills>')
        styles_xml = (
            styles_xml[:fills_close]
            + fill_xml
            + styles_xml[fills_close:]
        )
        # Update fills count attribute
        fills_count_m2 = re.search(rb'<fills\s+count="(\d+)"', styles_xml)
        new_fills_tag = f'<fills count="{old_fill_count + 1}"'.encode()
        styles_xml = (
            styles_xml[:fills_count_m2.start()]
            + new_fills_tag
            + styles_xml[fills_count_m2.end():]
        )

        # --- Extract existing xf entries from cellXfs ---
        cellXfs_m = re.search(rb'<cellXfs\s+count="(\d+)"', styles_xml)
        old_xf_count = int(cellXfs_m.group(1))
        cellXfs_start = cellXfs_m.start()
        cellXfs_close = styles_xml.find(b'</cellXfs>', cellXfs_start)
        cellXfs_section = styles_xml[cellXfs_start:cellXfs_close]

        # Extract individual <xf .../> or <xf ...>...</xf> entries
        xf_entries = []
        for m in re.finditer(rb'<xf\s', cellXfs_section):
            xf_start = m.start()
            tag_end = cellXfs_section.find(b'>', m.end())
            if cellXfs_section[tag_end - 1:tag_end] == b'/':
                xf_entries.append(cellXfs_section[xf_start:tag_end + 1])
            else:
                close = cellXfs_section.find(b'</xf>', tag_end)
                xf_entries.append(cellXfs_section[xf_start:close + 5])

        # Build style map and new xf bytes
        style_map = {}
        next_idx = old_xf_count
        new_xf_parts = []

        # Bare xf for cells without any style
        bare_xf = (
            f'<xf numFmtId="0" fontId="0" fillId="{new_fill_id}" '
            f'borderId="0" applyFill="1"/>'
        ).encode()
        new_xf_parts.append(bare_xf)
        style_map[None] = str(next_idx)
        next_idx += 1

        # Clone each unique existing style with the new fill
        for s in sorted(unique_styles):
            s_idx = int(s)
            if s_idx < len(xf_entries):
                cloned = xf_entries[s_idx]
                # Replace or add fillId
                if b'fillId=' in cloned:
                    cloned = re.sub(
                        rb'fillId="\d+"',
                        f'fillId="{new_fill_id}"'.encode(),
                        cloned,
                        count=1,
                    )
                else:
                    cloned = cloned.replace(
                        b'<xf ', f'<xf fillId="{new_fill_id}" '.encode(), 1
                    )
                # Add or update applyFill
                if b'applyFill=' in cloned:
                    cloned = re.sub(
                        rb'applyFill="[^"]*"', b'applyFill="1"', cloned, count=1
                    )
                else:
                    cloned = cloned.replace(
                        b'<xf ', b'<xf applyFill="1" ', 1
                    )
            else:
                cloned = (
                    f'<xf numFmtId="0" fontId="0" fillId="{new_fill_id}" '
                    f'borderId="0" applyFill="1"/>'
                ).encode()
            new_xf_parts.append(cloned)
            style_map[s] = str(next_idx)
            next_idx += 1

        # Insert new xf entries before </cellXfs>
        cellXfs_close_pos = styles_xml.find(b'</cellXfs>')
        insert_bytes = b''.join(new_xf_parts)
        styles_xml = (
            styles_xml[:cellXfs_close_pos]
            + insert_bytes
            + styles_xml[cellXfs_close_pos:]
        )

        # Update cellXfs count
        cellXfs_count_m = re.search(rb'<cellXfs\s+count="(\d+)"', styles_xml)
        new_xfs_tag = f'<cellXfs count="{next_idx}"'.encode()
        styles_xml = (
            styles_xml[:cellXfs_count_m.start()]
            + new_xfs_tag
            + styles_xml[cellXfs_count_m.end():]
        )

        return styles_xml, style_map


# Singleton instance for reuse
_handler: Optional[OpenpyxlHandler] = None


def get_openpyxl_handler() -> OpenpyxlHandler:
    """Get singleton OpenpyxlHandler instance."""
    global _handler
    if _handler is None:
        _handler = OpenpyxlHandler()
    return _handler
