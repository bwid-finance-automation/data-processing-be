"""Factory for auto-detecting and creating bank parsers."""

from typing import Optional, List, Tuple
from .base_parser import BaseBankParser
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)
from .acb_parser import ACBParser
from .vib_parser import VIBParser
from .kbank_parser import KBANKParser
from .sinopac_parser import SINOPACParser
from .mbb_parser import MBBParser
from .ocb_parser import OCBParser
from .bidv_parser import BIDVParser
from .vtb_parser import VTBParser
from .vcb_parser import VCBParser
from .woori_parser import WooriParser
from .scb_parser import SCBParser
from .tcb_parser import TCBParser
from .uob_parser import UOBParser


class ParserFactory:
    """Factory to detect bank and return appropriate parser."""

    # Register all available parsers here
    # NOTE: Order matters! More specific parsers should come before generic ones
    # - VCB must come early (VCBACCOUNTDETAIL sheet name is unique, but has bilingual "Opening balance :" that matches Woori)
    # - UOB must come before KBANK ("Account Activities" is unique to UOB)
    # - OCB and Woori must come before MBB (MBB has broad detection that can match others)
    # - Banks with unique markers (ACB, VIB, KBANK) can be in any order
    _parsers: List[BaseBankParser] = [
        ACBParser(),
        VIBParser(),
        VCBParser(),      # Must be early (has "Opening balance :" that can match Woori)
        UOBParser(),      # Must be before KBANK ("Account Activities" is unique to UOB)
        TCBParser(),      # Must be before KBANK (KBANK has broad "transaction date + debit + credit" detection)
        SCBParser(),      # Must be before KBANK (Standard Chartered)
        KBANKParser(),    # Has broader detection patterns for PDF
        SINOPACParser(),
        OCBParser(),      # Must be before MBB (MBB's "ACCOUNT STATEMENT" can match OCB)
        WooriParser(),    # Must be before MBB (MBB's "ACCOUNT STATEMENT" can match Woori)
        MBBParser(),      # Has broader detection patterns
        BIDVParser(),
        VTBParser(),
        # Add more parsers here as they are implemented
    ]

    @classmethod
    def get_parser(cls, file_bytes: bytes) -> Optional[BaseBankParser]:
        """
        Auto-detect bank from file content and return appropriate parser.

        Args:
            file_bytes: Excel file as binary

        Returns:
            Parser instance if bank detected, None otherwise
        """
        for parser in cls._parsers:
            if parser.can_parse(file_bytes):
                return parser

        return None

    @classmethod
    def get_parser_by_name(cls, bank_name: str) -> Optional[BaseBankParser]:
        """
        Get parser by bank name.

        Args:
            bank_name: Bank name (e.g., 'ACB', 'OCB')

        Returns:
            Parser instance if found, None otherwise
        """
        bank_name_upper = bank_name.upper()

        for parser in cls._parsers:
            if parser.bank_name.upper() == bank_name_upper:
                return parser

        return None

    @classmethod
    def get_supported_banks(cls) -> List[str]:
        """Get list of supported bank names."""
        return [parser.bank_name for parser in cls._parsers]

    @classmethod
    def get_all_parsers(cls, file_bytes: bytes) -> List[BaseBankParser]:
        """
        Detect ALL banks that can parse this file (not just the first match).

        Useful for multi-bank files where different parsers might match
        different sheets.

        Args:
            file_bytes: Excel file as binary

        Returns:
            List of parser instances that can handle this file
        """
        matching = []
        for parser in cls._parsers:
            if parser.can_parse(file_bytes):
                matching.append(parser)
        return matching

    @classmethod
    def detect_all_banks_in_file(
        cls, file_bytes: bytes
    ) -> List[Tuple[BaseBankParser, str, bytes]]:
        """
        Detect all banks across all sheets of a multi-sheet Excel file.

        Strategy:
        1. First try whole-file detection (existing parsers check sheet 0)
        2. If file has multiple sheets, try each sheet individually
        3. Return all unique (parser, sheet_name, sheet_bytes) tuples

        Args:
            file_bytes: Excel file as binary

        Returns:
            List of (parser, sheet_name, sheet_bytes) tuples.
            sheet_bytes is the isolated single-sheet Excel for that bank.
            If only one bank found on sheet 0, sheet_bytes = original file_bytes.
        """
        # Get sheet names
        sheet_names = BaseBankParser.get_sheet_names(file_bytes)

        if len(sheet_names) <= 1:
            # Single sheet: just return the standard detection
            parser = cls.get_parser(file_bytes)
            if parser:
                sheet_name = sheet_names[0] if sheet_names else "Sheet1"
                return [(parser, sheet_name, file_bytes)]
            return []

        # Multi-sheet: try each sheet individually
        results: List[Tuple[BaseBankParser, str, bytes]] = []
        detected_sheets = set()

        for sheet_name in sheet_names:
            sheet_bytes = BaseBankParser.extract_sheet_as_bytes(file_bytes, sheet_name)
            if sheet_bytes is None:
                continue

            parser = cls.get_parser(sheet_bytes)
            if parser and sheet_name not in detected_sheets:
                results.append((parser, sheet_name, sheet_bytes))
                detected_sheets.add(sheet_name)
                logger.info(
                    f"Multi-bank detection: sheet '{sheet_name}' -> {parser.bank_name}"
                )

        return results

    @classmethod
    def get_parser_for_text(cls, text: str) -> Optional[BaseBankParser]:
        """
        Auto-detect bank from OCR text content and return appropriate parser.

        Args:
            text: OCR extracted text from AI Builder

        Returns:
            Parser instance if bank detected and supports text parsing, None otherwise
        """
        for parser in cls._parsers:
            if parser.can_parse_text(text):
                return parser

        return None
