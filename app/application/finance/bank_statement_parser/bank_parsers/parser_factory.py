"""Factory for auto-detecting and creating bank parsers."""

from typing import Optional, List
from .base_parser import BaseBankParser
from .acb_parser import ACBParser
from .vib_parser import VIBParser
from .ctbc_parser import CTBCParser
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


class ParserFactory:
    """Factory to detect bank and return appropriate parser."""

    # Register all available parsers here
    # NOTE: Order matters! More specific parsers should come before generic ones
    # - VCB must come early (VCBACCOUNTDETAIL sheet name is unique, but has bilingual "Opening balance :" that matches Woori)
    # - OCB and Woori must come before MBB (MBB has broad detection that can match others)
    # - Banks with unique markers (ACB, VIB, CTBC, KBANK) can be in any order
    _parsers: List[BaseBankParser] = [
        ACBParser(),
        VIBParser(),
        VCBParser(),      # Must be early (has "Opening balance :" that can match Woori)
        CTBCParser(),
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
