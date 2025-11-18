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


class ParserFactory:
    """Factory to detect bank and return appropriate parser."""

    # Register all available parsers here
    _parsers: List[BaseBankParser] = [
        ACBParser(),
        VIBParser(),
        CTBCParser(),
        KBANKParser(),
        SINOPACParser(),
        MBBParser(),
        OCBParser(),
        BIDVParser(),
        VTBParser(),
        VCBParser(),
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
