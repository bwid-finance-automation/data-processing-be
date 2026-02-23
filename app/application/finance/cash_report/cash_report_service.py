"""
Cash Report Service - Main service for cash report automation.
Integrates all components: MasterTemplateManager, MovementDataWriter, BankStatementReader, AI Classifier.
"""
import asyncio
import csv
import re
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime
from difflib import SequenceMatcher
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from app.shared.utils.logging_config import get_logger
from app.infrastructure.database.models.cash_report_session import (
    CashReportSessionModel,
    CashReportUploadedFileModel,
    CashReportSessionStatus,
)
from app.domain.finance.cash_report.services.ai_classifier import (
    AITransactionClassifier,
    ALL_PAYMENT_CATEGORIES,
    ALL_RECEIPT_CATEGORIES,
)
from app.domain.finance.cash_report.models.key_payment import KeyPaymentClassifier

from .master_template_manager import MasterTemplateManager
from .movement_data_writer import MovementDataWriter, MovementTransaction
from .bank_statement_reader import BankStatementReader
from .progress_store import ProgressEvent

logger = get_logger(__name__)

# â”€â”€ Settlement detection patterns (Group A - Táº¥t toÃ¡n / RÃºt tiá»n) â”€â”€
# Used to detect settlement-eligible transactions by description.
# Requires BOTH: Nature = "Internal transfer in" AND keyword/pattern match.

# Regex patterns for settlement detection (compiled at module load)
SETTLEMENT_PATTERNS = [
    # GROUP A â€” Táº¥t toÃ¡n / RÃºt tiá»n (Current Account â† Savings)
    # 1-4: Táº¥t toÃ¡n / Terminate
    r"TAT\s*TOAN",                              # Táº¥t toÃ¡n
    r"TT\s+HDTG",                               # Viáº¿t táº¯t: Táº¥t toÃ¡n HDTG
    r"TERMINATE\s*(PARTIAL\s*)?SAVING",         # Terminate saving â€” VTB
    r"CA\s*-?\s*TARGET",                        # CA - TARGET â€” SINOPAC settlement
    # 5-10: Withdrawal / RÃºt tiá»n (EN)
    r"WITHDRAWAL\s*(OF\s*)?(THE\s*)?TERM\s*DEPOSIT",  # Withdrawal term deposit â€” VTB
    r"FULL\s*WITHDRAWAL\s*FROM\s*SAVINGS",      # Full withdrawal â€” VTB
    r"WITHDRAW\s*FIXED\s*DEPOSIT",              # Withdraw fixed deposit â€” BIDV
    r"PARTIALLY\s*WITHDRAW\s*TERM\s*DEPOSIT",   # Partially withdraw â€” BIDV
    r"CLOSE\s*TD\s*ACC",                        # Close Term Deposit â€” Kbank
    r"CLOSING\s*TERM\s*DEPOSIT",                # Closing term deposit â€” Escrow
    r"CLOSING\s*TDA",                           # Closing TDA â€” Woori
    # 11-17: RÃºt tiá»n (VI)
    r"RUT\s*TIEN\s*GUI",                        # RÃºt tiá»n gá»­i online â€” BIDV
    r"RUT\s*TIEN.*TIET\s*KIEM",                 # RÃºt tiá»n tá»« TK tiáº¿t kiá»‡m
    r"RUT\s*TIEN.*HDTG",                        # RÃºt tiá»n theo HDTG
    r"RUT\s*MOT\s*PHAN",                        # RÃºt má»™t pháº§n tiá»n gá»­i
    r"RUT\s*1\s*PHAN",                          # RÃºt 1 pháº§n â€” variant sá»‘
    r"RUT\s*GOC\s*MOT\s*PHAN",                  # RÃºt gá»‘c má»™t pháº§n HDTG
    r"RUT\s*TRUOC\s*HAN",                       # RÃºt trÆ°á»›c háº¡n
    # 18-22: ÄÃ³ng TK
    r"DONG\s*TKKH",                             # ÄÃ³ng TK ká»³ háº¡n
    r"DONG\s*TK.*KY\s*HAN",                     # ÄÃ³ng TK + ká»³ háº¡n
    r"DONG\s*TK.*TIET\s*KIEM",                  # ÄÃ³ng TK tiáº¿t kiá»‡m
    r"DONG\s*TK\s*(BIDV|THEO\s*DE|\d{3,})",     # ÄÃ³ng TK + mÃ£ NH/sá»‘ TK
    r"DONG\s*HDTG",                             # ÄÃ³ng HDTG
    # 23-24: Tráº£ gá»‘c
    r"TRA\s*GOC",                               # Tráº£ gá»‘c (standalone â€” catches all TRA GOC variants)
    # 25: Bare account number (SINOPAC-style, description = only digits)
    r"^\s*\d{10,}(?:\.0+)?\s*$",                # Pure account number (10+ digits, optional .0) â€” SINOPAC
]

# Simple keyword matching (faster, for common patterns)
# NOTE: Only include keywords specific to settling/closing saving deposits.
# Do NOT include intercompany transfer keywords (SHL, BCC, CHUYEN TIEN, etc.)
# â€” those are normal "Internal transfer in" but NOT settlement of savings.
SETTLEMENT_KEYWORDS = [
    "tat toan",         # Táº¥t toÃ¡n (lowercase fallback)
    "tra goc",          # Tráº£ gá»‘c (lowercase fallback)
    "rut tien gui",     # RÃºt tiá»n gá»­i (lowercase fallback)
]

# Compile regex patterns for efficiency
import re as _re
SETTLEMENT_PATTERNS_COMPILED = [_re.compile(p, _re.IGNORECASE) for p in SETTLEMENT_PATTERNS]

# Natures eligible for settlement detection.
SETTLEMENT_ELIGIBLE_NATURES = frozenset({
    "internal transfer in",   # Only Internal transfer in triggers settlement
})

# Natures that are NEVER settlement candidates, regardless of description.
# Counter entries and automation-generated interest rows must be excluded.
SETTLEMENT_BLOCKED_NATURES = frozenset({
    "internal transfer out",
    "other receipts",
})

# Special handling for numeric-only bank descriptions (account-like values).
# Flexible rule: any numeric-only description in this shape is processed.
NUMERIC_ACCOUNT_DESC_RE = _re.compile(r"^\d{10,20}$")

# Ground-truth transaction mapping file: description -> Nature.
# This file is curated by finance team and should be treated as highest-priority
# reference when an exact description+direction match exists.
TRANSACTIONS_REFERENCE_FILE = Path(__file__).resolve().parents[4] / "movement_nature_filter" / "Transactions.csv"
SETTLEMENT_REFERENCE_FILE = Path(__file__).resolve().parents[4] / "movement_nature_filter" / "Settlement.csv"
OPEN_NEW_REFERENCE_FILE = Path(__file__).resolve().parents[4] / "movement_nature_filter" / "OpenNew.csv"

# Keep these exclusions even when loading patterns from OpenNew.csv.
# They are known non-open-new movements and caused false positives in production.
OPEN_NEW_EXCLUDED_REGEXES = frozenset({
    r"AUTO\s*ROLLOVER",
    r"^WITHDRAWAL$",
})

# Normalize known legacy/variant category names from reference files.
TRANSACTION_NATURE_ALIASES = {
    "Operating Expense": "Operating expense",
    "Dividend paid (inside group)": "Dividend paid (inside group)",
    "Dividend paid (inside\xa0group)": "Dividend paid (inside group)",
    "Internal contribution": "Internal transfer in",
    "Internal Contribution": "Internal transfer in",
    "Internal contribution in": "Internal transfer in",
    "Internal Contribution in": "Internal transfer in",
    "Internal contribution out": "Internal transfer out",
    "Internal Contribution out": "Internal transfer out",
    "Refund land/deal deposit payment": "Receipt from tenants",
}

# Entity codes for dual-entity transfer detection.
# If description mentions 2+ DIFFERENT entity codes â†’ internal transfer between entities.
# E.g., "VC3_DC2_TRANSFER MONEY" contains VC3 and DC2 â†’ settlement candidate.
ENTITY_CODES = [
    "bwd", "bhd", "bbn", "hd2", "bhp", "bnt", "hd3", "bth",
    "th1hc", "th2hc", "th2", "btu", "bsg", "bnc", "bsl", "nt2",
    "bwb", "bwp", "dvl", "ndv", "tdh", "dal", "bla", "sgl",
    "hnm", "ire", "tdm", "ktr", "scc",
    "vc1", "vc2", "vc3",
    "bb3", "tpt", "bb4", "stl", "bha",
    "spv4a", "spv4b", "spv4c", "spv4d", "spv4e",
    "spv5a", "spv5b", "spv5c", "spv5d", "spv5e",
    "plc", "gtc", "bci", "bdg",
    "h4f", "h4h", "h4g", "h4i", "h4j",
    "h5f", "h5g", "h5h", "h5i", "h5j",
    "dvc", "byp", "bb5", "bb6", "blb",
    "xa1", "xa2", "wl1", "wl2",
    "bnh", "bdh", "btp", "mp3", "bba",
    "h4k", "h4l", "h4m", "h4n", "h4o", "h4p",
    "h5k", "h5l", "h5m",
    "wj1", "wj2", "wj3", "t3b", "msp", "tsp", "sc2",
    "h4q", "h4r", "h4s", "h4t", "h4u", "h4v", "h4w", "h4x", "h4y", "h4z",
    "h5q", "h5r", "h5s", "h5t", "h5u",
    "bhi", "btd", "w2a", "w2b",
    "hj1", "hj2", "yph", "hqv", "tha", "hhl", "bhl",
    "jyp", "jhp", "dtp",
    "h4c", "tcs", "h4d", "bmp", "h4a", "h5a", "h4b", "h5b", "h4e",
    "qv2", "w3a", "w3b",
    "pna", "bpd", "tna",
    "n4a", "n4b", "n4c", "n4d", "n4e",
    "lmx", "dc2", "pat", "st2", "tde", "mpl", "nas", "lbn",
    # Additional codes found in settlement descriptions
    "pae", "pdhd", "pde", "wl2b", "shl",
]

# â”€â”€ Open New Saving Account patterns (Group B - Gá»­i tiá»n / Má»Ÿ HDTG) â”€â”€
# Used to detect transactions that open new saving accounts.
# Requires BOTH: Nature = "Internal transfer out" AND pattern match.

OPEN_NEW_PATTERNS = [
    # GROUP B â€” Gá»­i tiá»n / Má»Ÿ HDTG (Current Account â†’ Saving Account)
    # Tiá»n gá»­i cÃ³ ká»³ háº¡n (reversed word order: "TIEN GUI CO KY HAN" not "GUI TIEN")
    r"TIEN\s*GUI\s*CO\s*KY\s*HAN",               # Tiá»n gá»­i cÃ³ ká»³ háº¡n â€” BIDV eFAST
    # Há»£p Ä‘á»“ng tiá»n gá»­i (without MO prefix)
    r"HOP\s*DONG\s*TIEN\s*GUI",                   # Há»£p Ä‘á»“ng tiá»n gá»­i sá»‘ â€” BIDV
    # Gá»­i tiá»n patterns
    r"GUI\s*TIEN.*HDTG",                          # Gá»­i tiá»n theo HDTG
    r"GUI\s*TIEN.*HOP\s*DONG",                    # Gá»­i tiá»n theo há»£p Ä‘á»“ng
    r"GUI\s*TIEN\s*VAO\s*HOP\s*DONG",             # Gá»­i tiá»n vÃ o HDTG
    r"GUI\s*TIEN\s*GUI\s*KY\s*HAN",               # Gá»­i tiá»n gá»­i ká»³ háº¡n
    r"GUI\s*TIEN\s*VAO\s*TAI\s*KHOAN\s*TIEN\s*GUI",  # Gá»­i tiá»n vÃ o TKTG cÃ³ ká»³ háº¡n
    r"GUI\s*TIEN\s*TK",                           # Gá»­i tiá»n TK â€” BIDV
    r"GUI\s*TIEN\s*GUI\s*CKH",                    # Gá»­i tiá»n gá»­i CKH â€” ACB
    r"GUI\s*TIEN\s*NGAY",                         # Gá»­i tiá»n ngÃ y â€” TCB
    # 32-38: Gá»­i HDTG/TK
    r"GUI\s*HDTG",                                # Gá»­i HDTG
    r"GUI\s*TK\s*THEO\s*HDTG",                    # Gá»­i TK theo HDTG
    r"GUI\s*TIET\s*KIEM",                         # Gá»­i tiáº¿t kiá»‡m
    r"GUI\s*KY\s*HAN",                            # Gá»­i ká»³ háº¡n
    r"GUI\s*TK\s*\d+\s*THANG",                    # Gá»­i TK + thÃ¡ng â€” VTB
    r"GUI\s*TKLH",                                # Gá»­i TK linh hoáº¡t â€” BIDV
    r"GUI\s*TGCKH",                               # Gá»­i tiá»n gá»­i CKH â€” ACB
    # 39-44: Má»Ÿ HDTG
    r"MO\s*HDTG",                                 # Má»Ÿ HDTG
    r"MO\s*HOP\s*DONG\s*TIEN\s*GUI",              # Má»Ÿ há»£p Ä‘á»“ng tiá»n gá»­i
    r"MO\s*HGTG",                                 # Má»Ÿ HGTG â€” typo = MO HDTG
    r"TRICH\s*TK\s*MO.*HDTG",                     # TrÃ­ch TK má»Ÿ HDTG
    r"HACH\s*TOAN\s*HDTG",                        # Háº¡ch toÃ¡n HDTG
    r"HT\s+HDTG",                                 # Viáº¿t táº¯t: Háº¡ch toÃ¡n HDTG
    # 45-54: HDTG patterns
    r"CHUYEN\s*KHOAN\s*VAO\s*HDTG",               # Chuyá»ƒn khoáº£n vÃ o HDTG
    r"HDTG\s*KH\s*\d",                            # HDTG ká»³ háº¡n + sá»‘
    r"HDTG\s*RGLH",                               # HDTG rÃºt gá»‘c linh hoáº¡t
    r"HDTG\s*\d+\s*(THANG|NGAY|TH|THT)",          # HDTG + ká»³ háº¡n thÃ¡ng/ngÃ y
    r"HDTG\s*SO\s*\d",                            # HDTG sá»‘ â€” BIDV
    r"HDTG\s+\d{3}[\s/]",                         # HDTG + mÃ£ chi nhÃ¡nh
    r"HDTG:\s*\d{3}",                             # HDTG: + mÃ£ HÄ
    r"HDTG\s+CKH",                                # HDTG cÃ³ ká»³ háº¡n
    r"HGTD\s*SO",                                 # HGTD â€” typo = HDTG
    r"\d+/\d+/HDTG",                              # MÃ£ HÄ dáº¡ng sá»‘/nÄƒm/HDTG
    # 55-57: VCB Time Deposit
    r"CK\s*SANG\s*TK\s*TIME",                     # CK sang TK TIME â€” VCB
    r"TIMEMO",                                    # TK TIMEMO â€” VCB savings
    r"TIMECT",                                    # TK TIMECT â€” VCB savings
    # 58-63: English patterns
    r"DEPOSIT\s*FOR\s*NEXT\s*PRINCIPAL",          # KEB Hana escrow savings
    r"OPEN\s*TD",                                 # Open Term Deposit â€” Kbank
    r"COMPLETED\s*TRANSFER\s*TO\s*BIDV\s*CD",     # Transfer to BIDV CD account
    r"SETTLEMENT\s*CONTRACT",                     # Settlement contract â€” VIB
    # NOTE: AUTO ROLLOVER removed â€” rollover renews existing deposit, not open new
    # Ká»³ háº¡n / RGLH / ÄÃ¡o háº¡n
    r"KY\s*HAN.*RGLH",                            # Ká»³ háº¡n + RGLH
    r"RGLH.*KY\s*HAN",                            # RGLH + ká»³ háº¡n
    r"DAO\s*HAN.*QUAY\s*VONG",                    # ÄÃ¡o háº¡n quay vÃ²ng
]

# Compile open new patterns for efficiency
OPEN_NEW_PATTERNS_COMPILED = [_re.compile(p, _re.IGNORECASE) for p in OPEN_NEW_PATTERNS]


class CashReportService:
    """
    Main service for cash report automation.

    Orchestrates the flow:
    1. Create session with config
    2. Upload parsed bank statements
    3. Classify transactions (rule-based first, AI for leftovers)
    4. Review/confirm before writing
    5. Write to Movement sheet
    6. Download result
    """

    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize the service.

        Args:
            db_session: Optional async database session for persistence
        """
        self.db_session = db_session
        self.template_manager = MasterTemplateManager()
        self.statement_reader = BankStatementReader()
        self.ai_classifier = AITransactionClassifier()
        self.rule_classifier = KeyPaymentClassifier()
        self._rules_file_mtime = self._get_rules_file_mtime()
        self._external_keyword_index = self._build_external_keyword_index()
        self._settlement_rules_mtime: Optional[float] = None
        self._open_new_rules_mtime: Optional[float] = None
        self._settlement_patterns_compiled = SETTLEMENT_PATTERNS_COMPILED
        self._open_new_patterns_compiled = OPEN_NEW_PATTERNS_COMPILED
        self._transactions_reference_mtime: Optional[float] = None
        self._transactions_reference_exact: Dict[Tuple[str, bool], str] = {}
        self._transactions_reference_majority: Dict[Tuple[str, bool], str] = {}
        self._transactions_reference_fuzzy_exact: Dict[Tuple[str, bool], str] = {}
        self._transactions_reference_fuzzy_majority: Dict[Tuple[str, bool], str] = {}
        self._reload_detection_pattern_indexes()
        self._rebuild_transactions_reference_index()

    async def _verify_session_owner(self, session_id: str, user_id: int) -> None:
        """
        Verify that the session belongs to the given user.
        Raises PermissionError if the session belongs to another user.
        Raises ValueError if session not found.
        """
        if not self.db_session:
            return

        result = await self.db_session.execute(
            select(CashReportSessionModel).where(
                CashReportSessionModel.session_id == session_id
            )
        )
        session = result.scalar_one_or_none()

        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.user_id is not None and session.user_id != user_id:
            raise PermissionError("You do not have access to this session")

    async def get_or_create_session(
        self,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal = Decimal("26175"),
        period_name: str = "",
        user_id: Optional[int] = None,
        template_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Get existing active session or create a new one.
        Only 1 session allowed per user.

        Args:
            opening_date: Report period start date
            ending_date: Report period end date
            fx_rate: VND/USD exchange rate
            period_name: Period name (e.g., "W3-4Jan26")
            user_id: Owner user ID

        Returns:
            Session info dict with 'is_existing' flag
        """
        # Check for existing active session for this user
        if self.db_session:
            query = select(CashReportSessionModel).where(
                CashReportSessionModel.status == CashReportSessionStatus.ACTIVE
            )
            if user_id is not None:
                query = query.where(CashReportSessionModel.user_id == user_id)
            query = query.order_by(CashReportSessionModel.created_at.desc())

            result = await self.db_session.execute(query)
            existing_session = result.scalar_one_or_none()

            if existing_session:
                # Return existing session info
                working_file = self.template_manager.get_working_file_path(existing_session.session_id)
                file_size_mb = 0
                if working_file and working_file.exists():
                    file_size_mb = round(working_file.stat().st_size / (1024 * 1024), 2)

                logger.info(f"Returning existing session: {existing_session.session_id}")
                return {
                    "session_id": existing_session.session_id,
                    "is_existing": True,
                    "working_file": str(working_file) if working_file else None,
                    "file_size_mb": file_size_mb,
                    "movement_rows": existing_session.total_transactions,
                    "config": {
                        "opening_date": existing_session.opening_date.isoformat() if existing_session.opening_date else None,
                        "ending_date": existing_session.ending_date.isoformat() if existing_session.ending_date else None,
                        "fx_rate": float(existing_session.fx_rate) if existing_session.fx_rate else None,
                        "period_name": existing_session.period_name,
                    },
                }

        # No existing session, create new one
        return await self._create_new_session(opening_date, ending_date, fx_rate, period_name, user_id=user_id, template_bytes=template_bytes)

    async def _create_new_session(
        self,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal,
        period_name: str,
        user_id: Optional[int] = None,
        template_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Internal method to create a new session."""
        # Create session with template manager
        session_info = self.template_manager.create_session(
            opening_date=opening_date,
            ending_date=ending_date,
            fx_rate=fx_rate,
            period_name=period_name,
            template_bytes=template_bytes,
        )

        # Persist to database if session available
        if self.db_session:
            db_model = CashReportSessionModel(
                session_id=session_info["session_id"],
                status=CashReportSessionStatus.ACTIVE,
                period_name=period_name,
                opening_date=opening_date,
                ending_date=ending_date,
                fx_rate=fx_rate,
                working_file_path=session_info["working_file"],
                total_transactions=0,
                total_files_uploaded=0,
                user_id=user_id,
                # Track whether Movement prep already ran (user-uploaded template)
                metadata_json={"movement_prepared": session_info.get("movement_prepared", False)},
            )
            self.db_session.add(db_model)
            await self.db_session.commit()

        logger.info(f"Created new cash report session: {session_info['session_id']}")
        return {
            **session_info,
            "is_existing": False,
        }

    async def upload_bank_statements(
        self,
        session_id: str,
        files: List[Tuple[str, bytes]],  # List of (filename, content)
        filter_by_date: bool = True,
        progress_callback: Optional[callable] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Upload and process parsed bank statement files.

        Args:
            session_id: The session ID
            files: List of (filename, file_content) tuples
            filter_by_date: Whether to filter transactions by session date range
            user_id: Owner user ID for access control

        Returns:
            Processing result summary
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        # Get session info
        session_info = self.template_manager.get_session_info(session_id)
        if not session_info:
            raise ValueError(f"Session {session_id} not found")

        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            raise ValueError(f"Working file for session {session_id} not found")

        # Get date range for filtering
        opening_date = None
        ending_date = None
        if filter_by_date and session_info.get("config"):
            config = session_info["config"]
            if config.get("opening_date"):
                date_str = config["opening_date"]
                # Handle both date and datetime strings
                if "T" in date_str:
                    opening_date = datetime.fromisoformat(date_str).date()
                else:
                    opening_date = date.fromisoformat(date_str)
            if config.get("ending_date"):
                date_str = config["ending_date"]
                if "T" in date_str:
                    ending_date = datetime.fromisoformat(date_str).date()
                else:
                    ending_date = date.fromisoformat(date_str)

        # Process each file
        all_transactions: List[MovementTransaction] = []
        file_results = []
        total_skipped = 0
        total_found = 0

        for file_idx, (filename, content) in enumerate(files):
            try:
                # Emit progress: reading file
                if progress_callback:
                    progress_callback(ProgressEvent(
                        event_type="step_start",
                        step="reading",
                        message=f"Reading {filename}...",
                        detail=f"Parsing transactions from file {file_idx + 1}/{len(files)}",
                        percentage=int((file_idx) / len(files) * 20),
                    ))

                # Read transactions from parsed Excel
                transactions = self.statement_reader.read_from_bytes(content, "Automation")
                found_count = len(transactions)
                total_found += found_count

                # Emit progress: file read complete
                if progress_callback:
                    progress_callback(ProgressEvent(
                        event_type="step_complete",
                        step="reading",
                        message=f"Found {found_count} transactions in {filename}",
                        percentage=int((file_idx + 1) / len(files) * 20),
                        data={"filename": filename, "count": found_count},
                    ))

                # Collect file date range before filtering
                file_dates = [tx.date for tx in transactions if tx.date]
                file_date_range = None
                if file_dates:
                    file_date_range = {
                        "start": min(file_dates).isoformat(),
                        "end": max(file_dates).isoformat(),
                    }

                # Filter by date if enabled
                skipped = 0
                if filter_by_date and opening_date and ending_date:
                    transactions, skipped = self.statement_reader.filter_by_date_range(
                        transactions, opening_date, ending_date
                    )

                all_transactions.extend(transactions)
                total_skipped += skipped

                file_result = {
                    "filename": filename,
                    "status": "success",
                    "transactions_found": found_count,
                    "transactions_added": len(transactions),
                    "transactions_skipped": skipped,
                }
                if file_date_range:
                    file_result["file_date_range"] = file_date_range

                file_results.append(file_result)

                # Track in database only when transactions were added
                if self.db_session and len(transactions) > 0:
                    await self._track_uploaded_file(
                        session_id=session_id,
                        filename=filename,
                        file_size=len(content),
                        transactions_count=len(transactions) + skipped,
                        transactions_added=len(transactions),
                        transactions_skipped=skipped,
                    )

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                file_results.append({
                    "filename": filename,
                    "status": "error",
                    "error": str(e),
                })

        # Emit filtering summary
        if progress_callback and filter_by_date and opening_date and ending_date:
            progress_callback(ProgressEvent(
                event_type="step_start",
                step="filtering",
                message=f"Filtering {total_found} transactions by date range",
                detail=f"{opening_date.strftime('%d/%m/%Y')} - {ending_date.strftime('%d/%m/%Y')}",
                percentage=25,
            ))
            progress_callback(ProgressEvent(
                event_type="step_complete",
                step="filtering",
                message=f"{len(all_transactions)} transactions within range, {total_skipped} skipped",
                percentage=30,
                data={"kept": len(all_transactions), "skipped": total_skipped},
            ))

        if not all_transactions:
            # Emit completion even when no transactions
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="complete",
                    step="done",
                    message="No transactions to process",
                    percentage=100,
                ))
            # Build a helpful warning message
            if total_skipped > 0 and total_found > 0:
                message = (
                    f"All {total_skipped} transactions were outside the session period "
                    f"({opening_date.strftime('%d/%m/%Y')} - {ending_date.strftime('%d/%m/%Y')}). "
                    f"Please check if the correct period was selected."
                )
                warning = "date_mismatch"
            else:
                message = "No valid transactions found in uploaded files"
                warning = None

            result = {
                "session_id": session_id,
                "files_processed": len(files),
                "total_transactions_added": 0,
                "total_transactions_found": total_found,
                "total_transactions_skipped": total_skipped,
                "file_results": file_results,
                "message": message,
            }
            if warning:
                result["warning"] = warning
                if opening_date and ending_date:
                    result["session_period"] = {
                        "start": opening_date.isoformat(),
                        "end": ending_date.isoformat(),
                    }
            return result

        # Classify transactions (rule-based first, AI for leftovers)
        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_start",
                step="classifying",
                message=f"Classifying {len(all_transactions)} transactions (rule-based + AI)...",
                detail="Rule-based keywords first, then AI for unmatched",
                percentage=35,
            ))
            await asyncio.sleep(0)  # Flush SSE

        logger.info(f"Classifying {len(all_transactions)} transactions...")
        import time as _time
        _classify_start = _time.monotonic()
        classified_transactions = await self._classify_transactions(
            all_transactions,
            progress_callback=progress_callback,
        )
        _classify_elapsed_ms = (_time.monotonic() - _classify_start) * 1000

        # Collect classification stats
        rule_count = sum(1 for tx in classified_transactions if getattr(tx, '_classified_by', '') == 'rule')
        ai_count = sum(1 for tx in classified_transactions if getattr(tx, '_classified_by', '') == 'ai')
        ai_cached_count = sum(1 for tx in classified_transactions if getattr(tx, '_classified_by', '') == 'ai_cached')
        unclassified_count = len(classified_transactions) - rule_count - ai_count - ai_cached_count

        logger.info(f"Classification stats: rule={rule_count}, ai={ai_count}, "
                     f"ai_cached={ai_cached_count}, unclassified={unclassified_count}")

        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_complete",
                step="classifying",
                message=f"Classified {len(classified_transactions)} transactions (rule: {rule_count}, AI: {ai_count}, cached: {ai_cached_count})",
                percentage=80,
                data={"rule_classified": rule_count, "ai_classified": ai_count, "unclassified": unclassified_count},
            ))
            await asyncio.sleep(0)  # Flush SSE

        # Prepare Movement sheet (copy Cash Balance â†’ Prior Period + clear old data)
        # Only on first upload (check DB total_transactions, not Excel rows which include template data)
        is_first_upload = await self._is_first_upload(session_id)
        if is_first_upload:
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="step_start",
                    step="preparing",
                    message="Copying Cash Balance to Prior Period and clearing Movement...",
                    percentage=82,
                ))
                await asyncio.sleep(0)

            from .openpyxl_handler import get_openpyxl_handler
            handler = get_openpyxl_handler()
            rows_cleared = await asyncio.to_thread(
                handler.prepare_movement_for_writing, working_file
            )

            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="step_complete",
                    step="preparing",
                    message=f"Cleared {rows_cleared} old rows, Cash Balance copied",
                    percentage=84,
                ))

        # Write to Movement sheet
        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_start",
                step="writing",
                message=f"Writing {len(classified_transactions)} transactions to Movement sheet...",
                detail="Appending data to Excel workbook",
                percentage=85,
            ))
            await asyncio.sleep(0)  # Flush SSE

        writer = MovementDataWriter(working_file)
        # Run blocking Excel write in thread to not block event loop
        rows_added, total_rows = await asyncio.to_thread(
            writer.append_transactions, classified_transactions
        )

        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_complete",
                step="writing",
                message=f"Written {rows_added} rows (total: {total_rows})",
                percentage=95,
            ))

        # Update session stats in database
        if self.db_session:
            await self._update_session_stats(session_id, rows_added, len(files))

        # Emit completion
        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="complete",
                step="done",
                message=f"Upload complete! {rows_added} transactions processed.",
                percentage=100,
                data={
                    "files_processed": len(files),
                    "total_transactions_added": rows_added,
                    "total_transactions_skipped": total_skipped,
                    "total_rows_in_movement": total_rows,
                },
            ))

        # Collect AI usage from classifier
        ai_usage = self.ai_classifier.get_and_reset_usage()
        ai_usage["processing_time_ms"] = _classify_elapsed_ms

        return {
            "session_id": session_id,
            "files_processed": len(files),
            "total_transactions_added": rows_added,
            "total_transactions_skipped": total_skipped,
            "total_rows_in_movement": total_rows,
            "file_results": file_results,
            "ai_usage": ai_usage,
            "classification_stats": {
                "rule_based": rule_count,
                "ai": ai_count,
                "ai_cached": ai_cached_count,
                "unclassified": unclassified_count,
            },
        }

    # ------------------------------------------------------------------ #
    #  Upload Movement NS/Manual file                                      #
    # ------------------------------------------------------------------ #

    async def upload_movement_file(
        self,
        session_id: str,
        file: Tuple[str, bytes],  # (filename, content)
        filter_by_date: bool = True,
        progress_callback: Optional[callable] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Upload pre-classified Movement data (Netsuite & Manual).

        Unlike bank statement uploads, these transactions already have their
        Nature category populated and do NOT require AI classification.

        Workflow:
        1. Read Movement file (filter out Automation rows)
        2. Filter by session date range (if enabled)
        3. Prepare Movement sheet on first upload (copy Cash Balance → Prior Period)
        4. Append transactions to Movement sheet
        5. Update session stats

        Args:
            session_id: The session ID
            file: Tuple of (filename, file_content)
            filter_by_date: Whether to filter transactions by session date range
            progress_callback: SSE progress callback
            user_id: Owner user ID for access control

        Returns:
            Upload result summary (no AI usage since pre-classified)
        """
        filename, content = file

        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        # Get session info
        session_info = self.template_manager.get_session_info(session_id)
        if not session_info:
            raise ValueError(f"Session {session_id} not found")

        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            raise ValueError(f"Working file for session {session_id} not found")

        # Get date range for filtering
        opening_date = None
        ending_date = None
        if filter_by_date and session_info.get("config"):
            config = session_info["config"]
            if config.get("opening_date"):
                date_str = config["opening_date"]
                if "T" in date_str:
                    opening_date = datetime.fromisoformat(date_str).date()
                else:
                    opening_date = date.fromisoformat(date_str)
            if config.get("ending_date"):
                date_str = config["ending_date"]
                if "T" in date_str:
                    ending_date = datetime.fromisoformat(date_str).date()
                else:
                    ending_date = date.fromisoformat(date_str)

        # --- Step 1: Read Movement file ---
        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_start",
                step="reading",
                message=f"Reading Movement file: {filename}...",
                percentage=10,
            ))
            await asyncio.sleep(0)

        transactions = self.statement_reader.read_movement_file(content, filename)
        found_count = len(transactions)

        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_complete",
                step="reading",
                message=f"Found {found_count} Movement transactions",
                percentage=30,
                data={"count": found_count},
            ))

        # --- Step 2: Filter by date range ---
        total_skipped = 0
        if filter_by_date and opening_date and ending_date:
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="step_start",
                    step="filtering",
                    message=f"Filtering by date range {opening_date} - {ending_date}...",
                    percentage=40,
                ))
            transactions, skipped = self.statement_reader.filter_by_date_range(
                transactions, opening_date, ending_date
            )
            total_skipped = skipped
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="step_complete",
                    step="filtering",
                    message=f"{len(transactions)} within range, {skipped} skipped",
                    percentage=50,
                ))

        if not transactions:
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="complete",
                    step="done",
                    message="No valid Movement transactions to process",
                    percentage=100,
                ))
            return {
                "session_id": session_id,
                "total_transactions_found": found_count,
                "total_transactions_added": 0,
                "total_transactions_skipped": total_skipped,
                "message": "No valid Movement transactions found after filtering",
            }

        # --- Step 3: Prepare Movement sheet on first upload ---
        is_first_upload = await self._is_first_upload(session_id)
        if is_first_upload:
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="step_start",
                    step="preparing",
                    message="First upload: Copying Cash Balance to Prior Period...",
                    percentage=55,
                ))
                await asyncio.sleep(0)

            from .openpyxl_handler import get_openpyxl_handler
            handler = get_openpyxl_handler()
            rows_cleared = await asyncio.to_thread(
                handler.prepare_movement_for_writing, working_file
            )

            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="step_complete",
                    step="preparing",
                    message=f"Cleared {rows_cleared} old rows, Cash Balance copied",
                    percentage=65,
                ))

        # --- Step 4: Write to Movement sheet ---
        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_start",
                step="writing",
                message=f"Writing {len(transactions)} transactions to Movement sheet...",
                percentage=70,
            ))
            await asyncio.sleep(0)

        writer = MovementDataWriter(working_file)
        rows_added, total_rows = await asyncio.to_thread(
            writer.append_transactions, transactions
        )

        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_complete",
                step="writing",
                message=f"Written {rows_added} rows (total: {total_rows})",
                percentage=90,
            ))

        # --- Step 5: Update session stats ---
        if self.db_session:
            await self._update_session_stats(session_id, rows_added, 1)

        # Emit completion
        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="complete",
                step="done",
                message=f"Upload complete! {rows_added} Movement transactions processed (from {found_count} found).",
                percentage=100,
                data={
                    "total_transactions_found": found_count,
                    "total_transactions_added": rows_added,
                    "total_transactions_skipped": total_skipped,
                    "total_rows_in_movement": total_rows,
                },
            ))

        return {
            "session_id": session_id,
            "total_transactions_found": found_count,
            "total_transactions_added": rows_added,
            "total_transactions_skipped": total_skipped,
            "total_rows_in_movement": total_rows,
            "message": f"Successfully uploaded {rows_added} Movement transactions",
        }

    # ------------------------------------------------------------------ #

    def _rule_based_classify(self, description: str, is_receipt: bool) -> Optional[str]:
        """
        Rule-based classification using KeyPaymentClassifier.

        Returns:
            Nature category string if confidently matched, None if needs AI.
        """
        key_payment, category, _ = self.rule_classifier.classify(description, is_receipt)
        # Default fallbacks mean no specific keyword matched â†' needs AI
        if key_payment in ("Other receipts", "Other payment"):
            return None
        return category

    @staticmethod
    def _normalize_text_for_match(text: str) -> str:
        """Normalize free text for robust case-insensitive keyword matching."""
        normalized = (text or "").lower().replace("_", " ").strip()
        normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _canonical_nature_for_direction(raw_category: str, is_receipt: bool) -> Optional[str]:
        """
        Map external keyword category to canonical Nature for receipt/payment direction.
        Returns None when the category is not valid for the transaction direction.
        """
        category = (raw_category or "").strip()
        if not category:
            return None

        category_lower = category.lower()
        if category_lower == "internal transfer":
            return "Internal transfer in" if is_receipt else "Internal transfer out"
        if category_lower == "other payment":
            category = "Operating expense"

        target = ALL_RECEIPT_CATEGORIES if is_receipt else ALL_PAYMENT_CATEGORIES
        for valid_category in target:
            if valid_category.lower() == category.lower():
                return valid_category
        return None

    @staticmethod
    def _normalize_reference_description(text: str) -> str:
        """Normalize bank description for exact reference matching (whitespace-insensitive)."""
        normalized = (text or "").replace("\u00A0", " ").strip().lower()
        return re.sub(r"\s+", " ", normalized)

    @staticmethod
    def _canonical_reference_nature(raw_nature: str) -> Optional[str]:
        """Normalize Nature values from reference CSV to canonical categories."""
        nature = (raw_nature or "").replace("\u00A0", " ").strip()
        if not nature:
            return None
        nature = TRANSACTION_NATURE_ALIASES.get(nature, nature)

        all_categories = ALL_PAYMENT_CATEGORIES | ALL_RECEIPT_CATEGORIES
        if nature in all_categories:
            return nature

        nature_lower = nature.lower()
        for valid_category in all_categories:
            if valid_category.lower() == nature_lower:
                return valid_category
        return None

    @staticmethod
    def _is_receipt_nature(nature: str) -> bool:
        """True if Nature belongs to receipt direction."""
        return nature in ALL_RECEIPT_CATEGORIES

    @staticmethod
    def _choose_majority_nature(counter: Counter) -> Optional[str]:
        """
        Resolve ambiguous reference labels by majority vote.
        Returns None when no clear winner exists.
        """
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def _rebuild_transactions_reference_index(self) -> None:
        """
        Build exact and fuzzy description->nature lookup tables from Transactions.csv.
        Direction is derived from nature (receipt/payment) to avoid cross-direction leakage.
        """
        path = TRANSACTIONS_REFERENCE_FILE
        self._transactions_reference_exact = {}
        self._transactions_reference_majority = {}
        self._transactions_reference_fuzzy_exact = {}
        self._transactions_reference_fuzzy_majority = {}
        self._transactions_reference_mtime = None

        if not path.exists():
            logger.warning(f"Transactions reference file not found: {path}")
            return

        exact_counters: Dict[Tuple[str, bool], Counter] = defaultdict(Counter)
        fuzzy_counters: Dict[Tuple[str, bool], Counter] = defaultdict(Counter)
        total_rows = 0
        skipped_rows = 0

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f, delimiter=";")
                next(reader, None)  # header

                for row in reader:
                    if not row or len(row) < 2:
                        continue

                    desc_raw = row[0]
                    nature_raw = row[1]
                    nature = self._canonical_reference_nature(nature_raw)
                    if not nature:
                        skipped_rows += 1
                        continue

                    is_receipt = self._is_receipt_nature(nature)
                    desc_exact = self._normalize_reference_description(desc_raw)
                    desc_fuzzy = self._normalize_text_for_match(desc_raw)
                    if not desc_exact and not desc_fuzzy:
                        skipped_rows += 1
                        continue

                    if desc_exact:
                        exact_counters[(desc_exact, is_receipt)][nature] += 1
                    if desc_fuzzy:
                        fuzzy_counters[(desc_fuzzy, is_receipt)][nature] += 1
                    total_rows += 1
        except Exception as e:
            logger.warning(f"Failed to load Transactions.csv reference data: {e}")
            return

        exact_ambiguous = 0
        for key, counts in exact_counters.items():
            if len(counts) == 1:
                self._transactions_reference_exact[key] = counts.most_common(1)[0][0]
                continue
            exact_ambiguous += 1
            majority_nature = self._choose_majority_nature(counts)
            if majority_nature:
                self._transactions_reference_majority[key] = majority_nature

        fuzzy_ambiguous = 0
        for key, counts in fuzzy_counters.items():
            if len(counts) == 1:
                self._transactions_reference_fuzzy_exact[key] = counts.most_common(1)[0][0]
                continue
            fuzzy_ambiguous += 1
            majority_nature = self._choose_majority_nature(counts)
            if majority_nature:
                self._transactions_reference_fuzzy_majority[key] = majority_nature

        try:
            self._transactions_reference_mtime = path.stat().st_mtime
        except OSError:
            self._transactions_reference_mtime = None

        logger.info(
            "Transactions reference loaded: rows=%s, exact=%s, exact_majority=%s, "
            "fuzzy=%s, fuzzy_majority=%s, ambiguous_exact=%s, ambiguous_fuzzy=%s, skipped=%s",
            total_rows,
            len(self._transactions_reference_exact),
            len(self._transactions_reference_majority),
            len(self._transactions_reference_fuzzy_exact),
            len(self._transactions_reference_fuzzy_majority),
            exact_ambiguous,
            fuzzy_ambiguous,
            skipped_rows,
        )

    def _get_transactions_reference_mtime(self) -> Optional[float]:
        """Get modification time for Transactions.csv reference file."""
        try:
            return TRANSACTIONS_REFERENCE_FILE.stat().st_mtime
        except OSError:
            return None

    def _refresh_transactions_reference_index_if_needed(self) -> None:
        """Reload Transactions.csv reference when file content changes."""
        current_mtime = self._get_transactions_reference_mtime()
        if current_mtime is None or current_mtime == self._transactions_reference_mtime:
            return
        self._rebuild_transactions_reference_index()

    def _classify_from_transactions_reference(self, tx: MovementTransaction) -> Tuple[Optional[str], Optional[str]]:
        """
        Classify by ground-truth Transactions.csv.
        Returns tuple (nature, source_tag).
        """
        is_receipt = bool(tx.debit)
        description = tx.description or ""

        desc_exact = self._normalize_reference_description(description)
        if desc_exact:
            key = (desc_exact, is_receipt)
            if key in self._transactions_reference_exact:
                return self._transactions_reference_exact[key], "reference_exact"
            if key in self._transactions_reference_majority:
                return self._transactions_reference_majority[key], "reference_majority"

        desc_fuzzy = self._normalize_text_for_match(description)
        if desc_fuzzy:
            key = (desc_fuzzy, is_receipt)
            if key in self._transactions_reference_fuzzy_exact:
                return self._transactions_reference_fuzzy_exact[key], "reference_fuzzy_exact"
            if key in self._transactions_reference_fuzzy_majority:
                return self._transactions_reference_fuzzy_majority[key], "reference_fuzzy_majority"

        return None, None

    def _build_external_keyword_index(self) -> List[Tuple[str, str]]:
        """
        Build a flattened keyword index from movement_nature_filter/Key Words.csv rules.
        Longest keywords are matched first for better specificity.
        """
        rules = getattr(self.ai_classifier, "_keyword_rules", {}) or {}
        index: List[Tuple[str, str]] = []
        for category, keywords in rules.items():
            for keyword in keywords or []:
                keyword_norm = self._normalize_text_for_match(keyword)
                if keyword_norm:
                    index.append((keyword_norm, category))
        index.sort(key=lambda item: len(item[0]), reverse=True)
        return index

    def _get_rules_file_mtime(self) -> Optional[float]:
        """Get current modification time for keyword-rules CSV used by AI classifier."""
        rules_file = getattr(self.ai_classifier, "_rules_file", None)
        if not rules_file:
            return None
        try:
            return Path(rules_file).stat().st_mtime
        except OSError:
            return None

    def _refresh_external_keyword_index_if_needed(self) -> None:
        """
        Reload keyword rules when `Key Words.csv` changes and rebuild the matching index.
        This keeps the flow aligned with latest CSV without restarting the service.
        """
        current_mtime = self._get_rules_file_mtime()
        if current_mtime is None or current_mtime == self._rules_file_mtime:
            return

        try:
            self.ai_classifier.reload_rules()
        except Exception as e:
            logger.warning(f"Failed to reload keyword rules file: {e}")
            return

        self._external_keyword_index = self._build_external_keyword_index()
        self._rules_file_mtime = current_mtime
        logger.info(
            f"Reloaded keyword rules and rebuilt index: {len(self._external_keyword_index)} keyword patterns"
        )

    @staticmethod
    def _normalize_regex_pattern(pattern: str) -> str:
        """Normalize regex text for dedupe/exclusion checks."""
        normalized = (pattern or "").replace("\u00A0", " ").strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.lower()

    def _load_detection_patterns_from_csv(
        self,
        file_path: Path,
        fallback_patterns: List[str],
        flow_name: str,
        excluded_patterns: Optional[set] = None,
    ) -> List[Any]:
        """
        Load regex patterns from CSV file and merge with fallback patterns in code.
        CSV rows are expected to use semicolon separator with a `Regex` column.
        """
        csv_patterns: List[str] = []

        if file_path.exists():
            try:
                with file_path.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f, delimiter=";")
                    for row in reader:
                        regex_value = ((row or {}).get("Regex") or (row or {}).get("regex") or "").strip()
                        if regex_value:
                            csv_patterns.append(regex_value)
            except Exception as e:
                logger.warning(f"Failed to load {flow_name} patterns from {file_path}: {e}")
        else:
            logger.warning(f"{flow_name} pattern file not found: {file_path}")

        merged_patterns: List[str] = []
        seen_norm = set()

        def _append_unique(pattern: str) -> None:
            norm = self._normalize_regex_pattern(pattern)
            if not norm or norm in seen_norm:
                return
            seen_norm.add(norm)
            merged_patterns.append(pattern.strip())

        for pattern in csv_patterns:
            _append_unique(pattern)
        for pattern in fallback_patterns:
            _append_unique(pattern)

        excluded_norm = {
            self._normalize_regex_pattern(pattern)
            for pattern in (excluded_patterns or set())
            if pattern
        }
        if excluded_norm:
            merged_patterns = [
                pattern for pattern in merged_patterns
                if self._normalize_regex_pattern(pattern) not in excluded_norm
            ]

        compiled_patterns: List[Any] = []
        for pattern in merged_patterns:
            try:
                compiled_patterns.append(_re.compile(pattern, _re.IGNORECASE))
            except _re.error as e:
                logger.warning(f"Invalid {flow_name} regex '{pattern}' in {file_path}: {e}")

        if not compiled_patterns:
            for pattern in fallback_patterns:
                norm = self._normalize_regex_pattern(pattern)
                if norm in excluded_norm:
                    continue
                try:
                    compiled_patterns.append(_re.compile(pattern, _re.IGNORECASE))
                except _re.error:
                    continue

        logger.info(
            "%s patterns loaded: compiled=%s, from_csv=%s, merged_total=%s, file=%s",
            flow_name,
            len(compiled_patterns),
            len(csv_patterns),
            len(merged_patterns),
            file_path.name,
        )
        return compiled_patterns

    @staticmethod
    def _get_file_mtime(path: Path) -> Optional[float]:
        """Get file modification time, or None when missing/unreadable."""
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _reload_detection_pattern_indexes(self) -> None:
        """Reload settlement/open-new detection regex from CSV and merge with in-code fallbacks."""
        self._settlement_patterns_compiled = self._load_detection_patterns_from_csv(
            file_path=SETTLEMENT_REFERENCE_FILE,
            fallback_patterns=SETTLEMENT_PATTERNS,
            flow_name="settlement",
        )
        self._open_new_patterns_compiled = self._load_detection_patterns_from_csv(
            file_path=OPEN_NEW_REFERENCE_FILE,
            fallback_patterns=OPEN_NEW_PATTERNS,
            flow_name="open_new",
            excluded_patterns=OPEN_NEW_EXCLUDED_REGEXES,
        )
        self._settlement_rules_mtime = self._get_file_mtime(SETTLEMENT_REFERENCE_FILE)
        self._open_new_rules_mtime = self._get_file_mtime(OPEN_NEW_REFERENCE_FILE)

    def _refresh_detection_pattern_indexes_if_needed(self) -> None:
        """Reload settlement/open-new detection regex when CSV files change."""
        settlement_mtime = self._get_file_mtime(SETTLEMENT_REFERENCE_FILE)
        open_new_mtime = self._get_file_mtime(OPEN_NEW_REFERENCE_FILE)
        if (
            settlement_mtime == self._settlement_rules_mtime
            and open_new_mtime == self._open_new_rules_mtime
        ):
            return
        self._reload_detection_pattern_indexes()

    @staticmethod
    def _keyword_matches_description(
        description_norm: str,
        description_tokens: set,
        keyword_norm: str,
    ) -> bool:
        """Match keyword in normalized description with boundary-safe handling for short tokens."""
        if not keyword_norm:
            return False
        if len(keyword_norm) <= 3 and " " not in keyword_norm:
            return keyword_norm in description_tokens
        return keyword_norm in description_norm

    def _classify_from_external_keyword_rules(
        self,
        description: str,
        is_receipt: bool,
    ) -> Optional[str]:
        """
        Deterministic classification using external keyword CSV rules.
        Returns canonical Nature for direction or None if no confident match.
        """
        if not self._external_keyword_index:
            return None

        description_norm = self._normalize_text_for_match(description or "")
        if not description_norm:
            return None
        description_tokens = set(description_norm.split())

        for keyword_norm, category in self._external_keyword_index:
            if self._keyword_matches_description(description_norm, description_tokens, keyword_norm):
                mapped = self._canonical_nature_for_direction(category, is_receipt)
                if mapped:
                    return mapped
        return None

    def _apply_classification_guardrails(self, tx: MovementTransaction) -> None:
        """
        Post-classification guardrails to prevent obvious misclassifications.
        - Numeric account-like descriptions follow deterministic round/non-round rule.
        - Settlement/open-new patterns enforce transfer directions.
        - Receipt amount constraints:
          * Round receipts => Internal transfer in
          * Other receipts must be non-round and < 100M
        - Empty nature always falls back by cash direction.
        """
        is_receipt = bool(tx.debit)
        description = (tx.description or "").strip()
        nature_now = (tx.nature or "").strip()

        special_nature = self._classify_special_bank_description(tx)
        if special_nature:
            tx.nature = special_nature
            tx._classified_by = "rule_guardrail"
            return

        # Don't override ground-truth classifications from Transactions.csv reference
        classified_by = getattr(tx, "_classified_by", "") or ""
        is_reference = classified_by.startswith("reference")

        if is_receipt:
            if not is_reference and any(pattern.search(description) for pattern in self._settlement_patterns_compiled):
                principal, _ = self._split_settlement_amount(tx.debit or Decimal("0"))
                tx.nature = "Internal transfer in" if principal > 0 else "Other receipts"
                tx._classified_by = "rule_guardrail"
                nature_now = tx.nature
        else:
            if not is_reference and any(pattern.search(description) for pattern in self._open_new_patterns_compiled):
                tx.nature = "Internal transfer out"
                tx._classified_by = "rule_guardrail"
                return

        if not nature_now:
            tx.nature = "Receipt from tenants" if is_receipt else "Operating expense"
            tx._classified_by = "fallback"
            nature_now = tx.nature

        adjusted_nature = self._apply_receipt_amount_nature_constraints(tx, nature_now)
        if adjusted_nature and adjusted_nature != nature_now:
            tx.nature = adjusted_nature
            tx._classified_by = "rule_guardrail"

    @staticmethod
    def _is_round_receipt_amount(amount: Decimal) -> bool:
        """
        Treat round receipts as internal transfer amounts.
        Round = >= 100M and divisible by 100M.
        """
        if not amount or amount <= 0:
            return False
        unit = Decimal("100000000")
        return amount >= unit and amount % unit == 0

    def _apply_receipt_amount_nature_constraints(self, tx: MovementTransaction, current_nature: str) -> Optional[str]:
        """
        Additional user-defined constraints for receipt-side natures:
        - Round (>=100M, divisible by 100M) → Internal transfer in
        - Non-round + >=100M                → Receipt from tenants
        - Internal transfer in + <100M      → Other receipts
        """
        if not tx.debit or tx.debit <= 0:
            return None

        # Keep explicit numeric-account rule behavior unchanged.
        desc_norm = self._normalize_bank_description(tx.description or "")
        if self._is_numeric_account_like_description(desc_norm):
            return None

        nature = (current_nature or "").strip().lower()
        if nature not in {"receipt from tenants", "other receipts", "internal transfer in"}:
            return None

        amount = tx.debit
        UNIT_100M = Decimal("100000000")
        is_round = amount >= UNIT_100M and amount % UNIT_100M == 0

        # Rule 1: Round → Internal transfer in
        if is_round:
            return "Internal transfer in"

        # Below: non-round

        # Rule 2: >=100M → Receipt from tenants
        if amount >= UNIT_100M:
            return "Receipt from tenants"

        # Rule 3: Internal transfer in + <100M → Other receipts
        if nature == "internal transfer in":
            return "Other receipts"

        return None

    @staticmethod
    def _normalize_bank_description(description: str) -> str:
        """Normalize numeric bank descriptions that may come as '<digits>.0'."""
        return re.sub(r"\.0+$", "", (description or "").strip())

    @staticmethod
    def _is_numeric_account_like_description(desc_norm: str) -> bool:
        """True for account-like numeric descriptions (10-20 digits)."""
        return bool(NUMERIC_ACCOUNT_DESC_RE.fullmatch(desc_norm or ""))

    def _classify_special_bank_description(self, tx: MovementTransaction) -> Optional[str]:
        """
        Special rules for account-like numeric bank descriptions.
        - round amount (>= 100M and divisible by 100M) => Internal transfer in
        - non-round amount => Other receipts
        """
        desc_norm = self._normalize_bank_description(tx.description or "")
        if not self._is_numeric_account_like_description(desc_norm):
            return None
        if not tx.debit or tx.debit <= 0:
            return None

        unit = Decimal("100000000")
        if tx.debit >= unit and tx.debit % unit == 0:
            return "Internal transfer in"
        return "Other receipts"

    async def _classify_transactions(
        self,
        transactions: List[MovementTransaction],
        progress_callback: Optional[callable] = None,
    ) -> List[MovementTransaction]:
        """
        Classify transactions using hybrid approach:
        Phase 1: Rule-based (keyword matching) - fast, deterministic
        Phase 2: AI (Gemini) - only for transactions rule-based couldn't classify

        Args:
            transactions: List of transactions without Nature
            progress_callback: Optional callback for progress updates

        Returns:
            List of transactions with Nature classified and classification_stats dict
        """
        if not transactions:
            return []

        self._refresh_detection_pattern_indexes_if_needed()
        self._refresh_external_keyword_index_if_needed()
        self._refresh_transactions_reference_index_if_needed()

        # â”€â”€ Phase 1: Rule-based classification â”€â”€
        needs_ai_indices = []
        rule_classified = 0

        for i, tx in enumerate(transactions):
            reference_nature, reference_source = self._classify_from_transactions_reference(tx)
            if reference_nature:
                tx.nature = reference_nature
                tx._classified_by = reference_source or "reference"
                rule_classified += 1
                continue

            special_nature = self._classify_special_bank_description(tx)
            if special_nature:
                tx.nature = special_nature
                tx._classified_by = "rule"
                rule_classified += 1
                continue

            is_receipt = bool(tx.debit)  # Debit = money in = Receipt
            external_rule_nature = self._classify_from_external_keyword_rules(tx.description or "", is_receipt)
            if external_rule_nature:
                tx.nature = external_rule_nature
                tx._classified_by = "rule"
                rule_classified += 1
                continue

            nature = self._rule_based_classify(tx.description or "", is_receipt)
            if nature:
                tx.nature = nature
                tx._classified_by = "rule"
                rule_classified += 1
            else:
                needs_ai_indices.append(i)

        logger.info(f"Rule-based classified {rule_classified}/{len(transactions)}, "
                     f"{len(needs_ai_indices)} need AI")

        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_update",
                step="classifying",
                message=f"Rule-based: {rule_classified} classified, {len(needs_ai_indices)} need AI...",
                percentage=40,
            ))
            await asyncio.sleep(0)

        # â”€â”€ Phase 2: AI classification for remaining â”€â”€
        if not needs_ai_indices:
            logger.info("All transactions classified by rules, no AI needed")
            for tx in transactions:
                self._apply_classification_guardrails(tx)
            return transactions

        if not getattr(self.ai_classifier, "client", None):
            logger.warning("AI classifier is unavailable; applying deterministic fallback for remaining transactions")
            for tx in transactions:
                self._apply_classification_guardrails(tx)
            return transactions

        # Prepare batch for AI (only unclassified)
        ai_batch = []
        for idx in needs_ai_indices:
            tx = transactions[idx]
            is_receipt = bool(tx.debit)
            ai_batch.append((tx.description, is_receipt))
        # Classify with per-batch progress updates
        try:
            batch_size = 50
            all_natures = []
            total_batches = (len(ai_batch) + batch_size - 1) // batch_size

            for i in range(0, len(ai_batch), batch_size):
                chunk = ai_batch[i:i + batch_size]
                batch_num = i // batch_size + 1

                if progress_callback:
                    pct = 45 + int((batch_num / total_batches) * 35)  # 45-80%
                    progress_callback(ProgressEvent(
                        event_type="step_update",
                        step="classifying",
                        message=f"AI batch {batch_num}/{total_batches} ({len(chunk)} transactions)...",
                        percentage=min(pct, 79),
                    ))
                    await asyncio.sleep(0)

                batch_natures = await asyncio.to_thread(
                    self.ai_classifier._classify_batch_internal, chunk
                )
                all_natures.extend(batch_natures)
                logger.info(f"AI classified batch {batch_num}/{total_batches}: {len(chunk)} transactions")

            # Apply AI classifications to the correct transactions
            ai_classified = 0
            ai_cached = 0
            for j, (nature, was_cached) in enumerate(all_natures):
                if j < len(needs_ai_indices):
                    idx = needs_ai_indices[j]
                    transactions[idx].nature = nature
                    transactions[idx]._classified_by = "ai_cached" if was_cached else "ai"
                    if nature:
                        ai_classified += 1
                        if was_cached:
                            ai_cached += 1

            logger.info(f"AI classified {ai_classified}/{len(needs_ai_indices)} "
                         f"(cached: {ai_cached}, API calls: {ai_classified - ai_cached})")

        except Exception as e:
            logger.error(f"AI classification error: {e}")
            # Leave nature empty for failed AI classifications

        for tx in transactions:
            self._apply_classification_guardrails(tx)

        return transactions

    def _get_pending_file(self, session_id: str) -> Path:
        """Get path for pending classifications JSON file."""
        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            raise ValueError(f"Session {session_id} not found")
        return working_file.parent / f"{session_id}_pending.json"

    def _save_pending(self, session_id: str, transactions: List[MovementTransaction], file_results: list, stats: dict) -> None:
        """Save pending classified transactions to JSON for review."""
        import json
        pending_file = self._get_pending_file(session_id)
        data = {
            "transactions": [
                {
                    "index": i,
                    "source": tx.source,
                    "bank": tx.bank,
                    "account": tx.account,
                    "date": tx.date.isoformat() if tx.date else None,
                    "description": tx.description,
                    "debit": float(tx.debit) if tx.debit else None,
                    "credit": float(tx.credit) if tx.credit else None,
                    "nature": tx.nature,
                    "classified_by": getattr(tx, '_classified_by', ''),
                }
                for i, tx in enumerate(transactions)
            ],
            "file_results": file_results,
            "stats": stats,
        }
        pending_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        logger.info(f"Saved {len(transactions)} pending classifications to {pending_file}")

    def _load_pending(self, session_id: str) -> Optional[dict]:
        """Load pending classified transactions from JSON."""
        import json
        pending_file = self._get_pending_file(session_id)
        if not pending_file.exists():
            return None
        data = json.loads(pending_file.read_text(encoding='utf-8'))
        return data

    def _clear_pending(self, session_id: str) -> None:
        """Delete pending classifications file."""
        try:
            pending_file = self._get_pending_file(session_id)
            if pending_file.exists():
                pending_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear pending file: {e}")

    async def upload_and_preview(
        self,
        session_id: str,
        files: List[Tuple[str, bytes]],
        filter_by_date: bool = True,
        progress_callback: Optional[callable] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Upload files, classify transactions, and return preview for review.
        Does NOT write to Excel yet. Call confirm_classifications() to write.

        Returns:
            Preview with classified transactions and stats
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        session_info = self.template_manager.get_session_info(session_id)
        if not session_info:
            raise ValueError(f"Session {session_id} not found")

        # Get date range for filtering
        opening_date = None
        ending_date = None
        if filter_by_date and session_info.get("config"):
            config = session_info["config"]
            if config.get("opening_date"):
                date_str = config["opening_date"]
                if "T" in date_str:
                    opening_date = datetime.fromisoformat(date_str).date()
                else:
                    opening_date = date.fromisoformat(date_str)
            if config.get("ending_date"):
                date_str = config["ending_date"]
                if "T" in date_str:
                    ending_date = datetime.fromisoformat(date_str).date()
                else:
                    ending_date = date.fromisoformat(date_str)

        # Process each file
        all_transactions: List[MovementTransaction] = []
        file_results = []
        total_skipped = 0
        total_found = 0

        for file_idx, (filename, content) in enumerate(files):
            try:
                if progress_callback:
                    progress_callback(ProgressEvent(
                        event_type="step_start",
                        step="reading",
                        message=f"Reading {filename}...",
                        percentage=int((file_idx) / len(files) * 20),
                    ))

                transactions = self.statement_reader.read_from_bytes(content, "Automation")
                found_count = len(transactions)
                total_found += found_count

                if progress_callback:
                    progress_callback(ProgressEvent(
                        event_type="step_complete",
                        step="reading",
                        message=f"Found {found_count} transactions in {filename}",
                        percentage=int((file_idx + 1) / len(files) * 20),
                    ))

                # Filter by date if enabled
                skipped = 0
                if filter_by_date and opening_date and ending_date:
                    transactions, skipped = self.statement_reader.filter_by_date_range(
                        transactions, opening_date, ending_date
                    )

                all_transactions.extend(transactions)
                total_skipped += skipped
                file_results.append({
                    "filename": filename,
                    "status": "success",
                    "transactions_found": found_count,
                    "transactions_added": len(transactions),
                    "transactions_skipped": skipped,
                })

                # Track in database
                if self.db_session and len(transactions) > 0:
                    await self._track_uploaded_file(
                        session_id=session_id, filename=filename,
                        file_size=len(content), transactions_count=len(transactions) + skipped,
                        transactions_added=len(transactions), transactions_skipped=skipped,
                    )

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                file_results.append({"filename": filename, "status": "error", "error": str(e)})

        if not all_transactions:
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="complete", step="done",
                    message="No transactions to process", percentage=100,
                ))
            return {
                "session_id": session_id,
                "status": "no_transactions",
                "files_processed": len(files),
                "total_transactions_found": total_found,
                "total_transactions_skipped": total_skipped,
                "file_results": file_results,
                "transactions": [],
            }

        # Classify (rule-based + AI)
        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_start", step="classifying",
                message=f"Classifying {len(all_transactions)} transactions...",
                percentage=35,
            ))
            await asyncio.sleep(0)

        import time as _time
        _classify_start = _time.monotonic()
        classified = await self._classify_transactions(all_transactions, progress_callback)
        _classify_elapsed_ms = (_time.monotonic() - _classify_start) * 1000

        # Collect stats
        rule_count = sum(1 for tx in classified if getattr(tx, '_classified_by', '') == 'rule')
        ai_count = sum(1 for tx in classified if getattr(tx, '_classified_by', '') == 'ai')
        ai_cached_count = sum(1 for tx in classified if getattr(tx, '_classified_by', '') == 'ai_cached')
        unclassified_count = len(classified) - rule_count - ai_count - ai_cached_count
        ai_usage = self.ai_classifier.get_and_reset_usage()
        ai_usage["processing_time_ms"] = _classify_elapsed_ms

        stats = {
            "rule_based": rule_count,
            "ai": ai_count,
            "ai_cached": ai_cached_count,
            "unclassified": unclassified_count,
        }

        logger.info(f"Classification stats: rule={rule_count}, ai={ai_count}, "
                     f"ai_cached={ai_cached_count}, unclassified={unclassified_count}")

        # Save pending (don't write to Excel yet)
        self._save_pending(session_id, classified, file_results, stats)

        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="complete", step="preview_ready",
                message=f"Classification complete â€” review {len(classified)} transactions before confirming",
                percentage=100,
                data={"classification_stats": stats},
            ))

        # Build preview response
        preview_transactions = [
            {
                "index": i,
                "source": tx.source,
                "bank": tx.bank,
                "account": tx.account,
                "date": tx.date.isoformat() if tx.date else None,
                "description": tx.description,
                "debit": float(tx.debit) if tx.debit else None,
                "credit": float(tx.credit) if tx.credit else None,
                "nature": tx.nature,
                "classified_by": getattr(tx, '_classified_by', ''),
            }
            for i, tx in enumerate(classified)
        ]

        return {
            "session_id": session_id,
            "status": "pending_review",
            "files_processed": len(files),
            "total_transactions": len(classified),
            "total_transactions_skipped": total_skipped,
            "file_results": file_results,
            "classification_stats": stats,
            "ai_usage": ai_usage,
            "transactions": preview_transactions,
        }

    async def confirm_classifications(
        self,
        session_id: str,
        modifications: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Confirm and write pending classifications to Excel.

        Args:
            session_id: The session ID
            modifications: Optional list of {index, nature} to override AI/rule classifications
            user_id: Owner user ID for access control

        Returns:
            Write result summary
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        # Load pending data
        pending = self._load_pending(session_id)
        if not pending:
            raise ValueError("No pending classifications found. Please upload and preview first.")

        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            raise ValueError(f"Session {session_id} not found")

        # Apply user modifications
        modifications_applied = 0
        if modifications:
            tx_map = {tx["index"]: tx for tx in pending["transactions"]}
            for mod in modifications:
                idx = mod.get("index")
                new_nature = mod.get("nature")
                if idx is not None and new_nature and idx in tx_map:
                    tx_map[idx]["nature"] = new_nature
                    tx_map[idx]["classified_by"] = "manual"
                    modifications_applied += 1
            logger.info(f"Applied {modifications_applied} manual modifications")

        # Convert back to MovementTransaction objects
        transactions = []
        for tx_data in pending["transactions"]:
            tx_date = None
            if tx_data["date"]:
                tx_date = date.fromisoformat(tx_data["date"])
            transactions.append(MovementTransaction(
                source=tx_data["source"],
                bank=tx_data["bank"],
                account=tx_data["account"],
                date=tx_date,
                description=tx_data["description"],
                debit=Decimal(str(tx_data["debit"])) if tx_data["debit"] else None,
                credit=Decimal(str(tx_data["credit"])) if tx_data["credit"] else None,
                nature=tx_data["nature"],
            ))

        # Prepare Movement (copy Cash Balance â†’ Prior Period + clear old data) on first write
        is_first_upload = await self._is_first_upload(session_id)
        if is_first_upload:
            from .openpyxl_handler import get_openpyxl_handler
            handler = get_openpyxl_handler()
            await asyncio.to_thread(handler.prepare_movement_for_writing, working_file)

        # Write to Excel
        writer = MovementDataWriter(working_file)
        rows_added, total_rows = await asyncio.to_thread(
            writer.append_transactions, transactions
        )

        # Update session stats
        if self.db_session:
            await self._update_session_stats(session_id, rows_added, len(pending["file_results"]))

        # Clear pending file
        self._clear_pending(session_id)

        return {
            "session_id": session_id,
            "status": "confirmed",
            "total_transactions_written": rows_added,
            "total_rows_in_movement": total_rows,
            "modifications_applied": modifications_applied,
            "classification_stats": pending.get("stats", {}),
        }

    async def _is_first_upload(self, session_id: str) -> bool:
        """Check if this is the first upload for a session.

        Returns False (skip prep) if:
        - total_transactions > 0 (already uploaded before), OR
        - metadata_json.movement_prepared == True (prep already ran at session init
          because user uploaded their own template)
        """
        if not self.db_session:
            return True  # No DB = always prepare (safe default)
        result = await self.db_session.execute(
            select(
                CashReportSessionModel.total_transactions,
                CashReportSessionModel.metadata_json,
            ).where(CashReportSessionModel.session_id == session_id)
        )
        row = result.one_or_none()
        if row is None:
            return True
        total, metadata = row
        # Skip prep if Movement was already prepared at session init
        if metadata and metadata.get("movement_prepared"):
            return False
        return total is None or total == 0

    async def _track_uploaded_file(
        self,
        session_id: str,
        filename: str,
        file_size: int,
        transactions_count: int,
        transactions_added: int,
        transactions_skipped: int,
    ) -> None:
        """Track uploaded file in database."""
        if not self.db_session:
            return

        # Find session in database
        result = await self.db_session.execute(
            select(CashReportSessionModel).where(
                CashReportSessionModel.session_id == session_id
            )
        )
        db_session = result.scalar_one_or_none()

        if db_session:
            file_model = CashReportUploadedFileModel(
                session_id=db_session.id,
                original_filename=filename,
                file_size=file_size,
                transactions_count=transactions_count,
                transactions_added=transactions_added,
                transactions_skipped=transactions_skipped,
                processed_at=datetime.utcnow(),
            )
            self.db_session.add(file_model)
            await self.db_session.commit()

    async def _update_session_stats(
        self,
        session_id: str,
        rows_added: int,
        files_added: int,
    ) -> None:
        """Update session statistics in database."""
        if not self.db_session:
            return

        await self.db_session.execute(
            update(CashReportSessionModel)
            .where(CashReportSessionModel.session_id == session_id)
            .values(
                total_transactions=CashReportSessionModel.total_transactions + rows_added,
                total_files_uploaded=CashReportSessionModel.total_files_uploaded + files_added,
            )
        )
        await self.db_session.commit()

    async def get_session_status(self, session_id: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get session status and statistics (fast - uses database).

        Args:
            session_id: The session ID
            user_id: Owner user ID for access control

        Returns:
            Session info dict or None if not found
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        # Get from database first (fast)
        if self.db_session:
            result = await self.db_session.execute(
                select(CashReportSessionModel)
                .options(selectinload(CashReportSessionModel.uploaded_files))
                .where(CashReportSessionModel.session_id == session_id)
            )
            db_session = result.scalar_one_or_none()

            if db_session:
                # Get file size without opening Excel
                working_file = self.template_manager.get_working_file_path(session_id)
                file_size_mb = 0
                if working_file and working_file.exists():
                    file_size_mb = round(working_file.stat().st_size / (1024 * 1024), 2)

                return {
                    "session_id": session_id,
                    "status": db_session.status.value,
                    "movement_rows": db_session.total_transactions,
                    "file_size_mb": file_size_mb,
                    "total_files_uploaded": db_session.total_files_uploaded,
                    "config": {
                        "opening_date": db_session.opening_date.isoformat() if db_session.opening_date else None,
                        "ending_date": db_session.ending_date.isoformat() if db_session.ending_date else None,
                        "fx_rate": float(db_session.fx_rate) if db_session.fx_rate else None,
                        "period_name": db_session.period_name,
                    },
                    "uploaded_files": [
                        {
                            "filename": f.original_filename,
                            "transactions_added": f.transactions_added,
                            "transactions_skipped": f.transactions_skipped,
                            "processed_at": f.processed_at.isoformat() if f.processed_at else None,
                        }
                        for f in db_session.uploaded_files
                    ],
                }

        # Fallback to reading file (slower)
        return self.template_manager.get_session_info(session_id)

    async def reset_session(self, session_id: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset session to clean state.

        Args:
            session_id: The session ID
            user_id: Owner user ID for access control

        Returns:
            Reset result
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        result = self.template_manager.reset_session(session_id)

        # Update database
        if self.db_session:
            # Delete uploaded files records
            db_result = await self.db_session.execute(
                select(CashReportSessionModel).where(
                    CashReportSessionModel.session_id == session_id
                )
            )
            db_session = db_result.scalar_one_or_none()

            if db_session:
                # Delete uploaded files
                await self.db_session.execute(
                    delete(CashReportUploadedFileModel).where(
                        CashReportUploadedFileModel.session_id == db_session.id
                    )
                )
                # Reset stats
                db_session.total_transactions = 0
                db_session.total_files_uploaded = 0
                await self.db_session.commit()

        logger.info(f"Reset session {session_id}")
        return result

    async def delete_session(self, session_id: str, user_id: Optional[int] = None) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session ID
            user_id: Owner user ID for access control

        Returns:
            True if deleted, False if not found
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        # Delete from file system
        file_deleted = self.template_manager.delete_session(session_id)

        # Delete from database
        db_deleted = False
        if self.db_session:
            result = await self.db_session.execute(
                delete(CashReportSessionModel).where(
                    CashReportSessionModel.session_id == session_id
                )
            )
            await self.db_session.commit()
            db_deleted = result.rowcount > 0

        return file_deleted or db_deleted

    def _save_step_snapshot(self, session_id: str, step_name: str) -> None:
        """Save a snapshot of the working file after a step completes.

        Creates a copy like ``snapshot_settlement.xlsx`` in the session directory
        so the user can download the result of each step independently.
        """
        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file or not working_file.exists():
            return
        snapshot_path = working_file.parent / f"snapshot_{step_name}.xlsx"
        import shutil
        shutil.copy2(working_file, snapshot_path)
        logger.info(f"Saved step snapshot: {snapshot_path.name}")

    async def get_working_file_path(
        self, session_id: str, user_id: Optional[int] = None, step: Optional[str] = None,
    ) -> Optional[Path]:
        """Get the working file path for download.

        Args:
            step: Optional step name (``settlement``, ``open_new``).
                  If given, returns the snapshot from that step.
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        if step:
            working_file = self.template_manager.get_working_file_path(session_id)
            if working_file:
                snapshot = working_file.parent / f"snapshot_{step}.xlsx"
                if snapshot.exists():
                    return snapshot
            # Fallback to current working file if snapshot not found
        return self.template_manager.get_working_file_path(session_id)

    async def list_sessions(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List active sessions from database, filtered by user."""
        if self.db_session:
            query = select(CashReportSessionModel).where(
                CashReportSessionModel.status == CashReportSessionStatus.ACTIVE
            )
            if user_id is not None:
                query = query.where(CashReportSessionModel.user_id == user_id)
            query = query.order_by(CashReportSessionModel.created_at.desc())

            result = await self.db_session.execute(query)
            db_sessions = result.scalars().all()

            return [
                {
                    "session_id": s.session_id,
                    "movement_rows": s.total_transactions,
                    "file_size_mb": 0,  # Don't read file for listing
                    "config": {
                        "opening_date": s.opening_date.isoformat() if s.opening_date else None,
                        "ending_date": s.ending_date.isoformat() if s.ending_date else None,
                        "fx_rate": float(s.fx_rate) if s.fx_rate else None,
                        "period_name": s.period_name,
                    }
                }
                for s in db_sessions
            ]

        # Fallback to file system (slower)
        return self.template_manager.list_sessions()

    async def get_data_preview(self, session_id: str, limit: int = 20, user_id: Optional[int] = None) -> List[dict]:
        """
        Get preview of Movement data.

        Args:
            session_id: The session ID
            limit: Maximum number of rows
            user_id: Owner user ID for access control

        Returns:
            List of row dicts
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)
        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            return []

        writer = MovementDataWriter(working_file)
        return writer.get_data_preview(limit)

    async def generate_movement_data(
        self,
        file: Tuple[str, bytes],
        period_name: str = "",
        progress_callback: Optional[callable] = None,
    ) -> bytes:
        """
        Generate a Movement Data Export file from a Movement Data Upload file.

        Upload format (no header, 7 columns):
            A=Source, B=Bank, C=Account, D=Date, E=Description, F=Debit(Nợ), G=Credit(Có)

        Export format (with header row, 16 columns):
            A=Source, B=Bank, C=Account, D=Date, E=Description,
            F=Nợ, G=Có, H=Net(F-G), I=Nature, J=Entity,
            K=Grouping, L=Key payment, M=Currency, N=Account type,
            O=Period, P=Text

        Args:
            file: (filename, bytes) tuple of the uploaded Upload-format file
            period_name: Period label to fill in column O (e.g. "W3-4Jan26")
            progress_callback: Optional progress callback

        Returns:
            bytes of the generated Export-format .xlsx file
        """
        import openpyxl
        from openpyxl import Workbook
        from io import BytesIO
        from datetime import datetime

        def emit(event_type, step, message, percentage=0):
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type=event_type, step=step,
                    message=message, percentage=percentage,
                ))

        filename, content = file
        emit("progress", "reading", f"Reading upload file: {filename}", percentage=10)

        # ── Read upload file ──────────────────────────────────────────────────
        wb_in = openpyxl.load_workbook(BytesIO(content), data_only=True, read_only=True)
        ws_in = wb_in.active

        transactions: List[MovementTransaction] = []
        for row in ws_in.iter_rows(min_row=1, values_only=True):
            # Skip completely empty rows
            if not any(c for c in row):
                continue

            # Parse each column (0-based)
            def _s(v):
                return str(v).strip() if v is not None else ""

            def _d(v):
                """Parse a decimal value (debit/credit). 0 or None → None."""
                if v is None:
                    return None
                try:
                    dec = Decimal(str(v))
                    return dec if dec > 0 else None
                except Exception:
                    return None

            source = _s(row[0]) if len(row) > 0 else ""
            bank = _s(row[1]) if len(row) > 1 else ""
            account = _s(row[2]) if len(row) > 2 else ""
            date_raw = row[3] if len(row) > 3 else None
            description = _s(row[4]) if len(row) > 4 else ""
            debit = _d(row[5]) if len(row) > 5 else None
            credit = _d(row[6]) if len(row) > 6 else None

            # Parse date
            tx_date: Optional[date] = None
            if isinstance(date_raw, datetime):
                tx_date = date_raw.date()
            elif isinstance(date_raw, date):
                tx_date = date_raw
            elif date_raw:
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
                    try:
                        tx_date = datetime.strptime(str(date_raw).strip()[:10], fmt).date()
                        break
                    except Exception:
                        pass

            if not bank and not account and not description:
                continue

            transactions.append(MovementTransaction(
                source=source or "Manual",
                bank=bank,
                account=account,
                date=tx_date or date.today(),
                description=description,
                debit=debit,
                credit=credit,
                nature="",
            ))

        wb_in.close()
        emit("progress", "classifying", f"Classifying {len(transactions)} transactions...", percentage=30)

        # ── Classify transactions ─────────────────────────────────────────────
        if transactions:
            transactions = await self._classify_transactions(transactions, progress_callback=None)

        emit("progress", "building", "Building export file...", percentage=80)

        # ── Build export workbook ─────────────────────────────────────────────
        wb_out = Workbook()
        ws_out = wb_out.active
        ws_out.title = "Sheet1"

        # Header row
        headers = [
            "Source", "Bank", "Account", "Date", "Bank description",
            "Nợ", "Có", "Net", "Nature", "Entity",
            "Grouping", "Key payment", "Currcency", "Account type",
            "Period", "Text",
        ]
        ws_out.append(headers)

        for tx in transactions:
            debit_val = float(tx.debit) if tx.debit else 0
            credit_val = float(tx.credit) if tx.credit else 0
            net_val = debit_val - credit_val

            # "Text" column = last N digits of account number (strip leading zeros)
            text_val = tx.account.lstrip("0") if tx.account else ""

            ws_out.append([
                tx.source,
                tx.bank,
                tx.account,
                tx.date,
                tx.description,
                debit_val if debit_val else None,
                credit_val if credit_val else None,
                net_val if net_val != 0 else None,
                tx.nature or "",
                tx.entity or "",
                tx.grouping or "",
                tx.key_payment or tx.nature or "",
                tx.currency or "VND",
                tx.account_type or "Current Account",
                period_name,
                text_val,
            ])

        emit("complete", "done", f"Generated export file with {len(transactions)} rows", percentage=100)

        out = BytesIO()
        wb_out.save(out)
        return out.getvalue()

    def _read_saving_accounts(self, working_file: str) -> List[Dict[str, Any]]:
        """
        Read Saving Account sheet from working file.

        Returns:
            List of dicts with keys: account, bank_1, bank, entity, branch,
            closing_balance_vnd, maturity_date
        """
        import openpyxl
        from datetime import datetime
        wb = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
        saving_accounts = []
        try:
            if "Saving Account" not in wb.sheetnames:
                logger.warning("Saving Account sheet not found")
                return []

            ws = wb["Saving Account"]
            # Row 3 = headers, Row 4+ = data
            # Col A=Entity, B=Branch, C=Account Number, D=Type, E=Currency,
            # Col F=CLOSING BALANCE (VND), Col K=Maturity date,
            # Col N(14)=Bank_1 (short code like TCB, BIDV), Col O(15)=Bank (full name)
            for row_data in ws.iter_rows(min_row=4, values_only=False):
                account = row_data[2].value if len(row_data) > 2 else None
                if not account:
                    continue

                # Col F (index 5): CLOSING BALANCE (VND)
                closing_raw = row_data[5].value if len(row_data) > 5 else None
                closing_balance = Decimal("0")
                if closing_raw is not None:
                    try:
                        closing_balance = Decimal(str(closing_raw))
                    except Exception:
                        pass

                # Col K (index 10): Maturity date
                maturity_raw = row_data[10].value if len(row_data) > 10 else None
                maturity_date = None
                if isinstance(maturity_raw, datetime):
                    maturity_date = maturity_raw.date()
                elif maturity_raw:
                    try:
                        maturity_date = datetime.strptime(str(maturity_raw).strip(), "%Y-%m-%d").date()
                    except Exception:
                        try:
                            maturity_date = datetime.strptime(str(maturity_raw).strip(), "%d/%m/%Y").date()
                        except Exception:
                            pass

                # Col L (index 11): Interest rate
                rate_raw = row_data[11].value if len(row_data) > 11 else None
                interest_rate = None
                if rate_raw is not None:
                    try:
                        rv = float(rate_raw)
                        # Stored as decimal (e.g. 0.0475) or percentage (4.75)
                        interest_rate = rv if rv < 1 else rv / 100
                    except (ValueError, TypeError):
                        pass

                saving_accounts.append({
                    "entity": str(row_data[0].value or "").strip(),
                    "branch": str(row_data[1].value or "").strip(),
                    "account": str(account).strip(),
                    "bank_1": str(row_data[13].value or "").strip() if len(row_data) > 13 else "",
                    "bank": str(row_data[14].value or "").strip() if len(row_data) > 14 else "",
                    "closing_balance_vnd": closing_balance,
                    "maturity_date": maturity_date,
                    "interest_rate": interest_rate,
                })
        finally:
            wb.close()

        logger.info(f"Loaded {len(saving_accounts)} saving accounts")
        return saving_accounts

    @staticmethod
    def _read_acc_char_account_to_code(working_file) -> Dict[str, str]:
        """
        Read accountâ†’code mapping from Acc_Char sheet columns B (Account No.) and C (CODE).
        Both are value columns (not formulas), so always reliable.

        Returns:
            {account_no: code} e.g. {"8680028808": "TCS", "1037150624": "T3B"}
        """
        import openpyxl
        account_to_code: Dict[str, str] = {}
        try:
            wb = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
            if "Acc_Char" not in wb.sheetnames:
                wb.close()
                return {}
            ws = wb["Acc_Char"]
            for row in ws.iter_rows(min_row=2, max_col=3):
                acc = row[1].value if len(row) > 1 else None   # B: Account No.
                code = row[2].value if len(row) > 2 else None  # C: CODE
                if acc and code:
                    acc_str = str(acc).strip()
                    try:
                        if "." in acc_str and float(acc_str) == int(float(acc_str)):
                            acc_str = str(int(float(acc_str)))
                    except (ValueError, OverflowError):
                        pass
                    account_to_code[acc_str] = str(code).strip()
            wb.close()
        except Exception as e:
            logger.warning(f"Failed to read Acc_Char account map: {e}")
        return account_to_code

    @staticmethod
    def _read_acc_char_full_data(working_file) -> Dict[str, Dict[str, str]]:
        """
        Read full account data from Acc_Char sheet (columns B-H).

        Returns:
            {account_no: {
                "code": str, "entity": str, "branch": str,
                "currency": str, "account_type": str, "grouping": str
            }}
        """
        import openpyxl
        acc_data: Dict[str, Dict[str, str]] = {}
        try:
            wb = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
            if "Acc_Char" not in wb.sheetnames:
                wb.close()
                return {}
            ws = wb["Acc_Char"]
            for row in ws.iter_rows(min_row=2, max_col=8):  # B..H = indices 1..7
                acc = row[1].value if len(row) > 1 else None   # B: Account No.
                if not acc:
                    continue
                acc_str = str(acc).strip()
                # Normalize float account numbers
                try:
                    if "." in acc_str and float(acc_str) == int(float(acc_str)):
                        acc_str = str(int(float(acc_str)))
                except (ValueError, OverflowError):
                    pass
                if not acc_str or acc_str.lower() == "x":
                    continue

                code_val = row[2].value if len(row) > 2 else None       # C: CODE
                entity_val = row[3].value if len(row) > 3 else None     # D: ENTITY (formula)
                branch_val = row[4].value if len(row) > 4 else None     # E: BRANCH
                currency_val = row[5].value if len(row) > 5 else None   # F: CURRENCY
                acc_type_val = row[6].value if len(row) > 6 else None   # G: ACCOUNT TYPE
                grouping_val = row[7].value if len(row) > 7 else None   # H: NAME/Grouping (formula)

                acc_data[acc_str] = {
                    "code": str(code_val).strip() if code_val else "",
                    "entity": str(entity_val).strip() if entity_val else "",
                    "branch": str(branch_val).strip() if branch_val else "",
                    "currency": str(currency_val).strip() if currency_val else "",
                    "account_type": str(acc_type_val).strip() if acc_type_val else "",
                    "grouping": str(grouping_val).strip() if grouping_val else "",
                }
            wb.close()
        except Exception as e:
            logger.warning(f"Failed to read Acc_Char full data: {e}")
        logger.info(f"Read Acc_Char full data: {len(acc_data)} accounts")
        return acc_data

    @staticmethod
    def _resolve_acc_char_for_saving_account(
        saving_acc: str,
        acc_char_data: Dict[str, Dict[str, str]],
        max_suffix: int = 20,
    ) -> Optional[Dict[str, str]]:
        """
        Look up saving account in Acc_Char data, trying suffix variations.

        Search order:
        1. Exact match: saving_acc
        2. Suffixed: saving_acc_1, saving_acc_2, ..., saving_acc_{max_suffix}

        Returns:
            The Acc_Char data dict if found, or None.
        """
        # 1) Exact match
        if saving_acc in acc_char_data:
            return acc_char_data[saving_acc]
        # 2) Suffixed variations
        base_acc = saving_acc.split("_")[0]
        for suffix in range(1, max_suffix + 1):
            suffixed = f"{base_acc}_{suffix}"
            if suffixed in acc_char_data:
                return acc_char_data[suffixed]
        return None

    @staticmethod
    def _build_account_entity_map(working_file: str) -> Dict[str, str]:
        """
        Build current_account â†’ entity_name mapping from Cash Balance sheet.

        Cash Balance columns: A=Entity, C=Account, X=Bank_1.
        These are value columns (not formulas), so always available.

        This replaces reliance on the Movement sheet's Entity formula column
        (col J), which has no cached values when the file hasn't been opened
        in Excel.

        Returns:
            {account_number: entity_name} e.g. {"110002915294": "BW BAU BANG 01"}
        """
        import openpyxl
        account_entity: Dict[str, str] = {}
        try:
            wb = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
            sheet_name = None
            for sn in wb.sheetnames:
                if sn == "Cash Balance":
                    sheet_name = sn
                    break
            if not sheet_name:
                wb.close()
                return {}

            ws = wb[sheet_name]
            for row in ws.iter_rows(min_row=4, max_col=25):
                account = row[2].value if len(row) > 2 else None  # C: Account
                entity = row[0].value if len(row) > 0 else None   # A: Entity
                if account and entity:
                    acc_str = str(account).strip()
                    # Normalize float account numbers (Excel sometimes stores as float)
                    try:
                        if "." in acc_str and float(acc_str) == int(float(acc_str)):
                            acc_str = str(int(float(acc_str)))
                    except (ValueError, OverflowError):
                        pass
                    account_entity[acc_str] = str(entity).strip()
            wb.close()
        except Exception as e:
            logger.warning(f"Failed to build accountâ†’entity map from Cash Balance: {e}")
        logger.info(f"Built accountâ†’entity map: {len(account_entity)} entries")
        return account_entity

    @staticmethod
    def _read_def_entity_to_code(working_file) -> Dict[str, str]:
        """
        Read entityâ†’code mapping from the Def sheet.
        Def sheet: Row 3 = headers, Row 4+ = data.
        Column A (0) = Abb. (CODE), Column B (1) = Entities (ENTITY).

        Returns:
            {entity_upper: code} e.g. {"TCS JSC": "TCS", "THUAN THANH 3B": "T3B"}
        """
        import openpyxl
        entity_to_code: Dict[str, str] = {}
        try:
            wb = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
            if "Def" not in wb.sheetnames:
                wb.close()
                return {}
            ws = wb["Def"]
            for row in ws.iter_rows(min_row=4, max_col=2):
                code = row[0].value if len(row) > 0 else None   # A: Abb. (CODE)
                entity = row[1].value if len(row) > 1 else None  # B: Entities
                if code and entity:
                    entity_to_code[str(entity).strip().upper()] = str(code).strip()
            wb.close()
        except Exception as e:
            logger.warning(f"Failed to read Def entity map: {e}")
        return entity_to_code

    @staticmethod
    def _read_prior_period_branches(working_file) -> Dict[str, List[str]]:
        """
        Read entityâ†’branches mapping from 'Cash balance (Prior period)' sheet.
        Row 3 = headers, Row 4+ = data.
        Column A (0) = ENTITY, Column B (1) = BRANCH.

        Returns:
            {entity_upper: [branch1, branch2, ...]} e.g. {"TCS JSC": ["BIDV - THANH XUAN", "VIETINBANK - QUANG MINH"]}
        """
        import openpyxl
        entity_branches: Dict[str, List[str]] = {}
        try:
            wb = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
            sheet_name = None
            for sn in wb.sheetnames:
                if "prior" in sn.lower() and "cash" in sn.lower():
                    sheet_name = sn
                    break
            if not sheet_name:
                wb.close()
                return {}
            ws = wb[sheet_name]
            for row in ws.iter_rows(min_row=4, max_col=2):
                entity = row[0].value if len(row) > 0 else None
                branch = row[1].value if len(row) > 1 else None
                if entity and branch:
                    ent_key = str(entity).strip().upper()
                    br_val = str(branch).strip()
                    if ent_key not in entity_branches:
                        entity_branches[ent_key] = []
                    if br_val not in entity_branches[ent_key]:
                        entity_branches[ent_key].append(br_val)
            wb.close()
        except Exception as e:
            logger.warning(f"Failed to read prior period branches: {e}")
        return entity_branches

    @staticmethod
    def _determine_saving_branch(
        saving_acc: str,
        entity: str,
        entity_branches: Dict[str, List[str]],
        original_bank: str,
    ) -> str:
        """
        Determine BRANCH for a new saving account by looking up the entity's
        branches in Cash Balance (Prior period), then matching with the saving
        account's bank prefix.

        Logic:
        1. Get all branches for entity from prior period
        2. Determine bank from saving account prefix (8xxâ†’BIDV, 1xx/10â†’VCB, 2xx/12â†’VTB)
        3. Pick the branch that matches the bank
        4. Fallback: first branch for entity, or original_bank
        """
        acc = str(saving_acc).strip()

        # Determine bank keyword from account prefix
        bank_keyword = ""
        if len(acc) >= 10:
            if acc.startswith("8"):
                bank_keyword = "BIDV"
            elif acc.startswith("1") and len(acc) == 10:
                bank_keyword = "VIETCOMBANK"
            elif acc.startswith("1") and len(acc) == 13:
                bank_keyword = "VIETCOMBANK"
            elif acc.startswith("2") and len(acc) == 12:
                bank_keyword = "VIETINBANK"

        # Look up entity's branches
        branches = entity_branches.get(entity.upper(), [])
        if branches and bank_keyword:
            for br in branches:
                if bank_keyword in br.upper():
                    return br

        # Fallback: first branch for this entity
        if branches:
            return branches[0]

        return original_bank

    @staticmethod
    def _extract_saving_account_from_description(description: str) -> Optional[str]:
        """
        Extract saving account number from description for settlement transactions.

        Patterns:
        - "TAT TOAN TIEN GUI SO 14501110378000" â†’ 14501110378000
        - "TAT TOAN HDTG RGLH SO... (TK 1059960714)" â†’ 1059960714
        - "TAT TOAN TAI KHOAN TIEN GUI CO KY HAN SO 218000472157" â†’ 218000472157
        - "Tra goc TK tien gui 217000486074" â†’ 217000486074
        - "07003600017772" (bare account number) â†’ 07003600017772

        Returns:
            Account number string or None
        """
        if not description:
            return None

        desc_stripped = description.strip()

        # Pattern 0: Bare account number (SINOPAC-style, description = only digits)
        # Accept optional ".0" suffix from Excel numeric string conversion.
        bare_acc_match = re.match(r'^(\d{10,})(?:\.0+)?$', desc_stripped)
        if bare_acc_match:
            return bare_acc_match.group(1)

        # Pattern 1: "(TK xxxxxx)" - account in parentheses
        match = re.search(r'\(TK\s*(\d{6,20})\)', description, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: "SO xxxxxx" at end - "TAT TOAN ... SO 14501110378000"
        # Must have "TAT TOAN" or "TIEN GUI" or "HDTG" context
        if re.search(r'TAT\s*TOAN|TIEN\s*GUI|HDTG', description, re.IGNORECASE):
            match = re.search(r'SO\s*(\d{6,20})\s*$', description, re.IGNORECASE)
            if match:
                return match.group(1)
            # Also try without $ anchor if there's trailing text
            match = re.search(r'SO\s*(\d{10,20})(?:\s|$)', description, re.IGNORECASE)
            if match:
                return match.group(1)

        # Pattern 3: "tien gui XXXXXXXX" - account after "tien gui" (VTB "Tra goc" pattern)
        # E.g., "Tra goc TK tien gui 217000486074" â†’ 217000486074
        match = re.search(r'tien\s*gui\s+(\d{10,20})', description, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def _split_settlement_amount(amount: Decimal) -> tuple:
        """
        Split a settlement amount into principal (round) and interest.

        >= 1B  → always split at 1B boundary (principal = floor(amount/1B)*1B)
        < 1B & divisible by 100M → no split (round, full amount is principal)
        < 1B & not divisible by 100M → principal=0, pure interest

        Returns:
            (principal, interest) — both Decimal.
        """
        UNIT_100M = Decimal("100000000")   # 100M VND
        UNIT_1B = Decimal("1000000000")    # 1B VND
        # Always split at 1B boundary first
        principal = (amount // UNIT_1B) * UNIT_1B
        interest = amount - principal
        # Small round amount (< 1B, divisible by 100M) → no split
        if principal == 0 and amount % UNIT_100M == 0:
            return amount, Decimal("0")
        return principal, interest

    @staticmethod
    def _bank_matches(tx_bank: str, sa_bank_1: str) -> bool:
        """
        Check if a Movement bank code matches a Saving Account bank_1.

        Handles mismatches like:
        - "WOORI" vs "Woori Bank" (abbreviation vs full name)
        - "SINOPAC" vs "SNP" (different abbreviations)
        """
        a = tx_bank.upper().strip()
        b = sa_bank_1.upper().strip()
        if not a or not b:
            return False
        if a == b:
            return True
        # Partial match: one contains the other
        if a in b or b in a:
            return True
        # Known aliases
        _BANK_ALIASES = {
            "SINOPAC": "SNP",
            "SNP": "SINOPAC",
        }
        alias = _BANK_ALIASES.get(a)
        if alias and (alias == b or alias in b or b in alias):
            return True
        return False

    def _find_saving_account(
        self,
        bank: str,
        entity: str,
        saving_accounts: List[Dict[str, Any]],
        description: str = "",
        principal: Decimal = None,
        tx_date=None,
    ) -> Optional[str]:
        """
        Find saving account for settlement transaction.

        Priority:
        1. Extract account number from description (for TAT TOAN transactions)
        2. Match by CLOSING BALANCE (VND) = principal AND Maturity date ±3 days
           (filtered by bank + entity first)
        3. Fallback: bank + entity only (single match)

        Returns:
            Saving account number or None
        """
        from datetime import timedelta

        # First try to extract from description
        extracted = self._extract_saving_account_from_description(description)
        if extracted:
            logger.info(f"Settlement: extracted account {extracted} from description")
            return extracted

        entity_upper = entity.upper().strip() if entity else ""

        # Filter by bank + entity
        bank_entity_matches = [
            sa for sa in saving_accounts
            if self._bank_matches(bank, sa["bank_1"])
            and sa["entity"].upper() == entity_upper
        ]

        # Try matching by CLOSING BALANCE + Maturity date ±3 days
        if principal and tx_date and bank_entity_matches:
            amount_date_matches = []
            for sa in bank_entity_matches:
                # Check closing balance matches principal
                if sa.get("closing_balance_vnd") != principal:
                    continue
                # Check maturity date within ±3 days
                mat_date = sa.get("maturity_date")
                if mat_date is None:
                    continue
                delta = abs((mat_date - tx_date).days)
                if delta <= 3:
                    amount_date_matches.append((sa, delta))

            if amount_date_matches:
                # Pick the closest maturity date
                amount_date_matches.sort(key=lambda x: x[1])
                best = amount_date_matches[0][0]
                logger.info(
                    f"Settlement: found saving account {best['account']} by "
                    f"closing_balance={principal}, maturity_date={best['maturity_date']}, "
                    f"tx_date={tx_date}, bank={bank}, entity={entity}"
                )
                return best["account"]

            # Partial withdrawal fallback: principal < closing_balance + maturity ±3 days
            # For "TAT TOAN MOT PHAN" / partial withdrawals, the principal is less than
            # the full deposit balance, so exact match fails.
            partial_matches = []
            for sa in bank_entity_matches:
                bal = sa.get("closing_balance_vnd") or Decimal("0")
                if bal <= 0 or principal >= bal:
                    continue
                mat_date = sa.get("maturity_date")
                if mat_date is None:
                    continue
                delta = abs((mat_date - tx_date).days)
                if delta <= 3:
                    partial_matches.append((sa, delta))

            if len(partial_matches) == 1:
                best = partial_matches[0][0]
                logger.info(
                    f"Settlement: found saving account {best['account']} by "
                    f"partial withdrawal (principal={principal} < balance={best['closing_balance_vnd']}), "
                    f"maturity_date={best['maturity_date']}, tx_date={tx_date}, "
                    f"bank={bank}, entity={entity}"
                )
                return best["account"]
            elif len(partial_matches) > 1:
                logger.info(
                    f"Settlement: {len(partial_matches)} partial-withdrawal matches for "
                    f"bank={bank}, entity={entity}, principal={principal}, tx_date={tx_date}: "
                    f"{[(m[0]['account'], m[0]['closing_balance_vnd']) for m in partial_matches]}"
                )

        # Fallback: single match by bank + entity
        if len(bank_entity_matches) == 1:
            logger.info(f"Settlement: found saving account {bank_entity_matches[0]['account']} by bank={bank} + entity={entity}")
            return bank_entity_matches[0]["account"]
        elif len(bank_entity_matches) > 1:
            logger.warning(
                f"Settlement: multiple saving accounts for bank={bank}, entity={entity}, "
                f"principal={principal}, tx_date={tx_date}: {[m['account'] for m in bank_entity_matches]} — no amount/date match"
            )

        # Fallback: match by bank only
        matches_bank = [sa for sa in saving_accounts if self._bank_matches(bank, sa["bank_1"])]
        if len(matches_bank) == 1:
            logger.info(f"Settlement fallback: found saving account {matches_bank[0]['account']} by bank={bank}")
            return matches_bank[0]["account"]

        logger.warning(f"Settlement: no saving account found for bank={bank}, entity={entity}")
        return None

    @staticmethod
    def _has_dual_entity_transfer(desc_lower: str) -> bool:
        """
        Check if description mentions 2+ different entity codes â†’ internal transfer.
        Example: "VC3_DC2_TRANSFER MONEY" â†’ VC3 + DC2 = 2 entities.

        Handles substring overlap: if both "th2" and "th2hc" match,
        "th2" is removed (it's a substring of the longer match).
        """
        matched = {code for code in ENTITY_CODES if code in desc_lower}
        if len(matched) < 2:
            return False
        # Remove codes that are substrings of longer matched codes
        # e.g., "th2" inside "th2hc" â†’ only keep "th2hc"
        filtered = {
            code for code in matched
            if not any(code != other and code in other for other in matched)
        }
        return len(filtered) >= 2

    def _is_settlement_candidate(self, tx) -> bool:
        """
        Detect if a transaction is a settlement candidate.
        Requires BOTH conditions:
        1. Nature == "Internal transfer in"  (mandatory)
        2. Description matches settlement patterns/keywords OR contains dual entity codes

        A transaction is a settlement candidate if:
        - debit > 0 (cash in)
        - NOT an existing counter entry
        - Nature is "Internal transfer in"
        - AND (regex pattern match OR keyword match OR dual-entity transfer)
        """
        # Must have debit > 0 (cash in)
        if not tx.debit or tx.debit <= 0:
            return False

        nature = (tx.nature or "").strip().lower()

        # Hard-block counter entries and interest rows — these are NEVER settlement
        # candidates regardless of description content.
        if nature in SETTLEMENT_BLOCKED_NATURES:
            return False

        description = (tx.description or "").strip()
        desc_lower = description.lower()
        desc_norm = self._normalize_bank_description(description)

        # Description-level overrides to prevent false settlement detection.
        if self._is_numeric_account_like_description(desc_norm):
            _, interest = self._split_settlement_amount(tx.debit)
            if interest > 0:
                return False

        # Check settlement regex patterns (Group A patterns)
        for pattern in self._settlement_patterns_compiled:
            if pattern.search(description):
                return True

        # Check simple settlement keywords
        for keyword in SETTLEMENT_KEYWORDS:
            if keyword in desc_lower:
                return True

        # NOTE: Dual-entity check removed â€” it caused false positives on
        # intercompany transfers (e.g., "INTERNAL TRANSFER FROM VC3 TO DC2").
        # All legitimate settlements are caught by patterns above.

        return False

    def _is_settlement_interest_row(self, tx) -> bool:
        """
        Detect if a transaction is an interest row accompanying a settlement.
        These have Nature="Other receipts" (interest payment) but their description
        matches settlement patterns (e.g., "TAT TOAN HDTG SO ...").
        They should be highlighted but NOT generate counter entries.
        """
        if not tx.debit or tx.debit <= 0:
            return False
        nature = (tx.nature or "").strip().lower()
        if nature != "other receipts":
            return False
        description = (tx.description or "").strip()
        desc_lower = description.lower()
        for pattern in self._settlement_patterns_compiled:
            if pattern.search(description):
                return True
        for keyword in SETTLEMENT_KEYWORDS:
            if keyword in desc_lower:
                return True
        return False

    def _is_open_new_candidate(self, tx) -> bool:
        """
        Detect if a transaction is an "open new saving account" candidate.
        Requires BOTH conditions:
        1. Nature == "Internal transfer out" (mandatory)
        2. Description matches GROUP B patterns (Gá»­i tiá»n / Má»Ÿ HDTG)

        A transaction is an open-new candidate if:
        - credit > 0 (cash out from current account)
        - NOT an existing counter entry (created by previous automation)
        - Nature is "Internal transfer out"
        - AND regex pattern match (GROUP B)
        """
        # Must have credit > 0 (cash out)
        if not tx.credit or tx.credit <= 0:
            return False

        # Nature must be "Internal transfer out" (mandatory)
        # This implicitly skips counter entries ("Internal transfer in")
        # regardless of source (Automation, NS, Manual)
        nature = (tx.nature or "").strip().lower()
        if nature != "internal transfer out":
            return False

        description = tx.description or ""

        # Check open-new regex patterns (Group B patterns)
        for pattern in self._open_new_patterns_compiled:
            if pattern.search(description):
                return True

        return False

    @staticmethod
    def _extract_saving_account_for_open_new(description: str) -> Optional[str]:
        """
        Extract saving account number from description for open-new transactions.

        Patterns for HDTG/saving account:
        - "CK den tai khoan 813015095347" â†’ 813015095347
        - "HDTG SO 123456789" â†’ 123456789
        - "(TK 123456789)" â†’ 123456789
        - "GUI HDTG ... SO 123456789" â†’ 123456789

        Returns:
            Account number string or None
        """
        if not description:
            return None

        # Pattern 1: "(TK xxxxxx)" - account in parentheses
        match = re.search(r'\(TK\s*(\d{6,20})\)', description, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: "tai khoan XXXXXX" or "account XXXXXX" - BIDV/VCB style
        match = re.search(r'(?:tai\s*khoan|account)\s*(\d{6,20})', description, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 3: "HDTG SO xxxxxx" or "SO xxxxxx" in HDTG context
        if re.search(r'HDTG|GUI\s*TIEN|MO\s*HD', description, re.IGNORECASE):
            match = re.search(r'SO\s*(\d{8,20})', description, re.IGNORECASE)
            if match:
                return match.group(1)

        # Pattern 4: Long number sequence that looks like account (10+ digits)
        # Used when description is mostly numeric (SNP/SINOPAC case)
        if re.match(r'^[\d,.\sE+]+$', description.strip()):
            match = re.search(r'(\d{10,20})', description)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def _extract_term_info(description: str) -> dict:
        """
        Extract term, interest rate, and maturity date from description.

        Examples:
        - "HDTG 900/2026/47248, KH 01 THANG, LS 4.75%/NAM" â†’ term="1 months", rate=0.0475
        - "KY HAN 175 NGAY, DEN HAN 24/07/2026" â†’ term="175 days", maturity="2026-07-24"
        - "TKKH 46 NGAY THEO HD SO 0034.2026" â†’ term="46 days"
        - "1 THANG THEO HD" â†’ term="1 months"
        - "LS 4.75%" or "LAI SUAT 4.75%" â†’ rate=0.0475

        Returns:
            dict with keys: term_months, term_days, interest_rate, maturity_date (all optional)
        """
        result = {
            "term_months": None,
            "term_days": None,
            "interest_rate": None,
            "maturity_date": None,
        }

        if not description:
            return result

        desc_upper = description.upper()

        # Extract term in months: "KH 01 THANG", "1 THANG", "KY HAN 3 THANG", "3M", "01M"
        month_patterns = [
            r'(?:KH|KY\s*HAN|TKKH)?\s*(\d{1,2})\s*(?:THANG|M(?:ONTH)?S?)\b',
            r'\b(\d{1,2})\s*THANG\b',
        ]
        for pattern in month_patterns:
            match = re.search(pattern, desc_upper)
            if match:
                result["term_months"] = int(match.group(1))
                break

        # Extract term in days: "175 NGAY", "KY HAN 84 NGAY"
        day_patterns = [
            r'(?:KY\s*HAN|TKKH)?\s*(\d{1,3})\s*NGAY\b',
            r'\b(\d{2,3})\s*(?:NGAY|DAY)\b',
        ]
        for pattern in day_patterns:
            match = re.search(pattern, desc_upper)
            if match:
                days = int(match.group(1))
                if days > 1:  # Avoid matching single digit dates
                    result["term_days"] = days
                    break

        # Extract interest rate: "LS 4.75%", "LAI SUAT 4.75%/NAM", "4.75%"
        rate_patterns = [
            r'(?:LS|LAI\s*SUAT)\s*(\d+[.,]\d+)\s*%',
            r'\b(\d+[.,]\d{1,2})\s*%\s*(?:/\s*NAM)?',
        ]
        for pattern in rate_patterns:
            match = re.search(pattern, desc_upper)
            if match:
                rate_str = match.group(1).replace(',', '.')
                result["interest_rate"] = float(rate_str) / 100  # Convert to decimal
                break

        # Extract maturity date: "DEN HAN 24/07/2026", "24.07.2026"
        date_patterns = [
            r'(?:DEN\s*HAN|DAO\s*HAN)\s*(\d{1,2})[/.](\d{1,2})[/.](\d{4})',
            r'\bNGAY\s*(\d{1,2})[/.](\d{1,2})[/.](\d{4})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, desc_upper)
            if match:
                try:
                    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    from datetime import date as dt_date
                    result["maturity_date"] = dt_date(year, month, day)
                except (ValueError, TypeError):
                    pass
                break

        return result

    async def run_settlement_automation(
        self,
        session_id: str,
        user_id: Optional[int] = None,
        progress_callback=None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run settlement (táº¥t toÃ¡n) automation on Movement data.

        Logic:
        1. Detect settlement transactions by keyword patterns
        2. Lookup saving account from Saving Account sheet (by account number in description + bank match)
        3. Create counter entry with saving account, reversed debit/credit, nature = Internal transfer out

        Args:
            session_id: The session ID
            user_id: Owner user ID for access control
            progress_callback: Optional callable for SSE progress events
            dry_run: If True, run detect+lookup but do NOT write to Excel.
                     Returns candidate preview data instead.

        Returns:
            Dict with results summary (or preview data when dry_run=True)
        """
        from .progress_store import ProgressEvent

        def emit(event_type, step, message, detail="", percentage=0, data=None):
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type=event_type, step=step,
                    message=message, detail=detail,
                    percentage=percentage, data=data or {},
                ))

        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            raise ValueError(f"Session {session_id} not found")

        writer = MovementDataWriter(working_file)
        self._refresh_detection_pattern_indexes_if_needed()

        # â”€â”€ Step 1: Scan Movement data â”€â”€
        emit("step_start", "scanning", "Scanning Movement data...", percentage=0)
        await asyncio.sleep(0.3)  # Allow SSE to deliver step_start

        transactions = writer.get_all_transactions()
        if not transactions:
            emit("complete", "done", "No transactions found", percentage=100,
                 data={"status": "no_transactions"})
            return {
                "session_id": session_id,
                "status": "no_transactions",
                "message": "No transactions found in Movement sheet",
                "counter_entries_created": 0,
            }

        # Build current_account â†’ entity mapping from Cash Balance sheet.
        # The Movement sheet's Entity column (J) is formula-based and has no
        # cached values when the file hasn't been opened in Excel, so we
        # resolve entity from Cash Balance (which has value columns).
        account_entity_map = self._build_account_entity_map(working_file)

        # Build comprehensive Acc_Char lookup for populating cached VLOOKUP values
        # on settlement counter entries (prevents #N/A in Entity/Currency/etc.)
        acc_char_data = self._read_acc_char_full_data(working_file)

        # Load entity→code and branch mappings for inserting new Acc_Char rows
        def_entity_to_code = self._read_def_entity_to_code(working_file)
        entity_branches = self._read_prior_period_branches(working_file)

        emit("step_complete", "scanning",
             f"Found {len(transactions)} transactions",
             percentage=20,
             data={"total_transactions": len(transactions)})
        await asyncio.sleep(0.4)  # Allow SSE to deliver step transition

        # â"€â"€ Step 2: Detect settlement transactions â"€â"€
        emit("step_start", "detecting", "Detecting settlement transactions...", percentage=20)
        await asyncio.sleep(0.3)

        saving_accounts = self._read_saving_accounts(working_file)

        # Detect by keyword patterns + amount splitting upfront.
        # Amount splitting BEFORE detection ensures pure-interest rows
        # (< 100M, no round principal) are classified correctly from the start.
        settlement_transactions = []
        interest_row_indices = []  # Interest rows — nature correction only, NO counter
        # Track cells that need modification (amount splitting + nature correction)
        cell_modifications = {}  # {row_num: {"F": new_debit_str, "I": new_nature}}
        # Interest rows to INSERT right below their settlement row
        interest_inserts = {}  # {row_idx: tx_data_dict}
        _skip_reasons = {"no_debit": 0, "wrong_nature": 0, "no_pattern": 0}

        for idx, tx in enumerate(transactions):
            row_idx = idx + 4

            if self._is_settlement_interest_row(tx):
                interest_row_indices.append(row_idx)
                continue

            if not self._is_settlement_candidate(tx):
                if not tx.debit or tx.debit <= 0:
                    _skip_reasons["no_debit"] += 1
                elif (tx.nature or "").strip().lower() in SETTLEMENT_BLOCKED_NATURES:
                    _skip_reasons["wrong_nature"] += 1
                else:
                    _skip_reasons["no_pattern"] += 1
                continue

            # — Amount splitting: separate principal from interest —
            # Bank deposits have round principal amounts (divisible by 100M).
            # If the total debit is NOT round, the bank combined principal + interest.
            # Settlement prerequisite: principal must be round AND >= 100M.
            principal, interest = self._split_settlement_amount(tx.debit)

            if principal <= 0:
                # Pure interest (no round principal, amount < 1B) → NOT a settlement.
                # Always "Other receipts".
                interest_row_indices.append(row_idx)
                nature_now = (tx.nature or "").strip().lower()
                if nature_now != "other receipts":
                    cell_modifications[row_idx] = {"I": "Other receipts"}
                logger.info(f"Settlement: pure interest row {row_idx}, amount={tx.debit}, nature fixed to Other receipts")
                continue

            if interest > 0:
                # Combined amount → split: modify original row debit to principal,
                # insert interest row DIRECTLY BELOW the settlement row.
                # Interest part is always "Other receipts".
                cell_modifications[row_idx] = {"F": str(principal)}
                interest_inserts[row_idx] = {
                    'source': tx.source or "Automation",
                    'bank': tx.bank or "",
                    'account': tx.account or "",
                    'date': tx.date,
                    'description': tx.description or "",
                    'debit': interest,
                    'credit': 0,
                    'nature': "Other receipts",
                }
                logger.info(
                    f"Settlement split row {row_idx}: "
                    f"principal={principal}, interest={interest} (total={tx.debit})"
                )

            # Store principal for counter entry creation in Step 3
            tx._principal = principal
            settlement_transactions.append((tx, row_idx))

        logger.info(f"Settlement detect: {len(settlement_transactions)} candidates, {len(interest_row_indices)} interest rows, skip reasons: {_skip_reasons}")

        if not settlement_transactions:
            emit("complete", "done", "No settlement transactions found", percentage=100,
                 data={"status": "no_settlements", "total_transactions_scanned": len(transactions)})
            return {
                "session_id": session_id,
                "status": "no_settlements",
                "message": "No settlement transactions found",
                "counter_entries_created": 0,
                "total_transactions_scanned": len(transactions),
            }

        emit("step_complete", "detecting",
             f"Found {len(settlement_transactions)} settlement transactions",
             percentage=40,
             data={"settlement_found": len(settlement_transactions)})
        await asyncio.sleep(0.4)

        # â”€â”€ Step 3: Lookup saving accounts â”€â”€
        emit("step_start", "lookup", "Looking up saving accounts...", percentage=40)
        await asyncio.sleep(0.3)

        existing_descs = set()
        for tx in transactions:
            if tx.nature and "transfer out" in tx.nature.lower():
                existing_descs.add(tx.description.strip() if tx.description else "")

        counter_entries = []
        original_row_indices = []  # Rows to highlight (exclude nature="Other receipts")
        skipped_no_account = []
        skipped_duplicate = []
        acc_char_inserts = []  # Saving accounts to add to Acc_Char (not found anywhere)

        for tx, row_idx in settlement_transactions:
            desc = tx.description.strip() if tx.description else ""
            if desc in existing_descs:
                skipped_duplicate.append(desc)
                continue

            # Principal was computed in Step 2 (amount splitting)
            principal = tx._principal

            # Resolve entity from Cash Balance account→entity map
            tx_account = str(tx.account).strip() if tx.account else ""
            entity = account_entity_map.get(tx_account, "")

            # Normalize settlement rows to Internal transfer in even if classifier mislabeled.
            nature_now = (tx.nature or "").strip().lower()
            if nature_now != "internal transfer in":
                cell_modifications.setdefault(row_idx, {})["I"] = "Internal transfer in"

            saving_acc = self._find_saving_account(
                bank=tx.bank or "",
                entity=entity,
                saving_accounts=saving_accounts,
                description=desc,
                principal=principal,
                tx_date=tx.date if isinstance(tx.date, date) else None,
            )

            if not saving_acc:
                skipped_no_account.append(tx.description)
                logger.warning(f"Settlement: no saving account found for '{tx.description}', bank={tx.bank}, entity={entity}")
                continue

            # --- Resolve Acc_Char data for the saving account ---
            # Populate cached VLOOKUP values to prevent #N/A in Movement formula columns
            acc_data = self._resolve_acc_char_for_saving_account(saving_acc, acc_char_data)

            # Determine the account number to use in the Movement counter entry
            counter_account = saving_acc  # Default: use saving account from description

            if acc_data:
                # Found in Acc_Char (exact or suffixed) — use saving_acc as-is
                counter_entity = acc_data.get("entity") or account_entity_map.get(saving_acc, "") or entity
                counter_grouping = acc_data.get("grouping") or "Subsidiaries"
                counter_currency = acc_data.get("currency") or "VND"
                counter_account_type = acc_data.get("account_type") or "Saving Account"
                logger.debug(f"Settlement: found Acc_Char data for {saving_acc}: entity={counter_entity}")
            else:
                # Not found by saving_acc — check current_account_1, _2, ... in Acc_Char
                # If _1 exists → reuse it (no insert needed)
                # If _1 NOT exist → insert _1. If _1 taken → insert _2, etc.
                found_existing_suffix = None
                for s in range(1, 21):
                    candidate = f"{tx_account}_{s}"
                    if candidate in acc_char_data:
                        found_existing_suffix = (candidate, acc_char_data[candidate])
                        break

                if found_existing_suffix:
                    # Reuse existing suffixed account — no Acc_Char insert needed
                    counter_account, existing_data = found_existing_suffix
                    counter_entity = existing_data.get("entity") or entity
                    counter_grouping = existing_data.get("grouping") or "Subsidiaries"
                    counter_currency = existing_data.get("currency") or "VND"
                    counter_account_type = existing_data.get("account_type") or "Saving Account"
                    logger.debug(
                        f"Settlement: {saving_acc} not in Acc_Char, "
                        f"reusing existing {counter_account} (entity={counter_entity})"
                    )
                else:
                    # No suffixed account exists — create _1 (or next available)
                    suffix = 1
                    while f"{tx_account}_{suffix}" in acc_char_data:
                        suffix += 1
                    counter_account = f"{tx_account}_{suffix}"

                    current_acc_data = acc_char_data.get(tx_account, {})
                    code = current_acc_data.get("code", "")
                    if not code and entity:
                        code = def_entity_to_code.get(entity.upper(), "")
                    branch = self._determine_saving_branch(
                        saving_acc, entity, entity_branches, tx.bank or ""
                    )
                    counter_currency = current_acc_data.get("currency") or "VND"

                    acc_char_inserts.append({
                        "saving_acc": counter_account,
                        "code": code,
                        "entity": entity,
                        "branch": branch,
                        "currency": counter_currency,
                        "account_type": "Saving Account",
                    })

                    # Register so next iteration sees it
                    acc_char_data[counter_account] = {
                        "code": code,
                        "entity": entity,
                        "branch": branch,
                        "currency": counter_currency,
                        "account_type": "Saving Account",
                        "grouping": "Subsidiaries",
                    }

                    counter_entity = entity
                    counter_grouping = "Subsidiaries"
                    counter_account_type = "Saving Account"
                    logger.info(
                        f"Settlement: {saving_acc} not in Acc_Char, "
                        f"will insert as {counter_account} (code={code}, entity={entity})"
                    )

            counter = MovementTransaction(
                source=tx.source or "Automation",
                bank=tx.bank,
                account=counter_account,  # May be saving_acc or current_account_N
                date=tx.date,
                description=tx.description or "",
                debit=Decimal("0"),
                credit=principal,  # Use principal (round part), NOT full amount
                nature="Internal transfer out",
                entity=counter_entity,
                grouping=counter_grouping,
                currency=counter_currency,
                account_type=counter_account_type,
            )

            counter_entries.append(counter)
            final_nature = cell_modifications.get(row_idx, {}).get("I", tx.nature or "")
            if final_nature.strip().lower() != "other receipts":
                original_row_indices.append(row_idx)

        emit("step_complete", "lookup",
             f"Matched {len(counter_entries)} accounts, {len(skipped_no_account)} not found, {len(skipped_duplicate)} duplicates",
             percentage=60,
             data={
                 "matched": len(counter_entries),
                 "skipped_no_account": len(skipped_no_account),
                 "skipped_duplicate": len(skipped_duplicate),
             })
        await asyncio.sleep(0.4)

        # ── Dry-run: return preview without writing ──
        if dry_run:
            nature_corrections_count = len([m for m in cell_modifications.values() if "I" in m])
            candidates_preview = []
            for tx, row_idx in settlement_transactions:
                principal = getattr(tx, '_principal', tx.debit)
                # Find the matched counter entry for this tx (by description)
                matched_counter = next(
                    (c for c in counter_entries if c.description == (tx.description or "").strip()),
                    None
                )
                desc = (tx.description or "").strip()
                candidates_preview.append({
                    "row": row_idx,
                    "date": tx.date.isoformat() if tx.date else None,
                    "bank": tx.bank or "",
                    "account": str(tx.account or ""),
                    "description": desc,
                    "debit": float(principal),
                    "nature": tx.nature or "",
                    "status": (
                        "duplicate" if desc in [d for d in skipped_duplicate]
                        else "no_account" if desc in skipped_no_account
                        else "matched"
                    ),
                    "counter_account": str(matched_counter.account) if matched_counter else None,
                    "has_interest_split": row_idx in interest_inserts,
                    "interest_amount": float(interest_inserts[row_idx]["debit"]) if row_idx in interest_inserts else None,
                })
            return {
                "session_id": session_id,
                "dry_run": True,
                "status": "preview",
                "total_transactions_scanned": len(transactions),
                "settlement_candidates": len(settlement_transactions),
                "counter_entries_would_create": len(counter_entries),
                "interest_splits_would_create": len(interest_inserts),
                "nature_corrections_would_apply": nature_corrections_count,
                "skipped_no_account": skipped_no_account,
                "skipped_duplicate": skipped_duplicate,
                "acc_char_inserts_would_add": [i["saving_acc"] for i in acc_char_inserts],
                "candidates": candidates_preview,
            }

        if not counter_entries:
            # Even without counter entries, apply modifications and insertions
            if cell_modifications:
                try:
                    writer.modify_cell_values(cell_modifications)
                    logger.info(f"Applied {len(cell_modifications)} cell modifications (no counter entries)")
                except Exception as e:
                    logger.warning(f"Failed to apply cell modifications: {e}")
            if interest_inserts:
                try:
                    writer.insert_rows_after(interest_inserts)
                except Exception as e:
                    logger.warning(f"Failed to insert interest rows: {e}")
            self._save_step_snapshot(session_id, "settlement")

            msg = "No counter entries created"
            if skipped_no_account:
                msg += f". {len(skipped_no_account)} skipped (no matching saving account)"
            if skipped_duplicate:
                msg += f". {len(skipped_duplicate)} skipped (already exists)"
            emit("complete", "done", msg, percentage=100,
                 data={"status": "no_counter_entries"})
            return {
                "session_id": session_id,
                "status": "no_counter_entries",
                "message": msg,
                "counter_entries_created": 0,
                "interest_splits_created": len(interest_inserts),
                "nature_corrections": len(cell_modifications),
                "skipped_no_account": len(skipped_no_account),
                "skipped_duplicate": len(skipped_duplicate),
                "total_transactions_scanned": len(transactions),
            }

        # â"€â"€ Step 4: Write changes â"€â"€
        emit("step_start", "writing",
             f"Creating {len(counter_entries)} counter entries"
             + (f", {len(interest_inserts)} interest splits" if interest_inserts else "")
             + "...", percentage=60)
        await asyncio.sleep(0.3)

        # Phase A: Apply cell modifications (debit splits + nature corrections)
        # Uses original row numbers (before any insertion shifts rows)
        if cell_modifications:
            try:
                writer.modify_cell_values(cell_modifications)
                logger.info(f"Modified {len(cell_modifications)} rows (debit splits + nature fixes)")
            except Exception as e:
                logger.warning(f"Failed to modify cells for splits/nature: {e}")

        # Phase B: Insert interest rows directly below their settlement rows
        # Returns mapping {original_row: new_row} for all data rows
        row_mapping = {}
        if interest_inserts:
            try:
                row_mapping = writer.insert_rows_after(interest_inserts)
                logger.info(f"Inserted {len(interest_inserts)} interest rows, mapping has {len(row_mapping)} entries")
            except Exception as e:
                logger.warning(f"Failed to insert interest rows: {e}")

        # Adjust tracked row indices using the mapping (insertions shifted rows)
        if row_mapping:
            original_row_indices = [row_mapping.get(r, r) for r in original_row_indices]

        # Phase C: Append ONLY counter entries at the end
        rows_added, total_rows = writer.append_transactions(counter_entries)

        # Calculate counter entry row indices for highlighting
        append_start_row = total_rows - rows_added + 4
        counter_row_indices = list(range(
            append_start_row, append_start_row + len(counter_entries)
        ))

        # Phase D: Add missing saving accounts to Acc_Char
        acc_char_added = 0
        if acc_char_inserts:
            from .openpyxl_handler import get_openpyxl_handler
            handler_early = get_openpyxl_handler()
            for insert_info in acc_char_inserts:
                try:
                    row_added = handler_early.add_row_to_acc_char(
                        Path(working_file),
                        account_no=insert_info["saving_acc"],
                        code=insert_info["code"],
                        entity=insert_info["entity"],
                        branch=insert_info["branch"],
                        currency=insert_info["currency"],
                        account_type=insert_info["account_type"],
                    )
                    if row_added > 0:
                        acc_char_added += 1
                except Exception as e:
                    logger.warning(f"Failed to add {insert_info['saving_acc']} to Acc_Char: {e}")
            logger.info(f"Settlement: added {acc_char_added}/{len(acc_char_inserts)} accounts to Acc_Char")

        # Highlight: original settlement rows (excluding nature="Other receipts")
        # + counter entries ONLY. "Other receipts" rows are NEVER highlighted.
        all_highlight_rows = original_row_indices + counter_row_indices
        try:
            writer.highlight_settlement_rows(all_highlight_rows)
        except Exception as e:
            logger.warning(f"Failed to highlight settlement rows: {e}")

        emit("step_complete", "writing",
             f"Created {len(counter_entries)} counter entries"
             + (f", {len(interest_inserts)} interest splits" if interest_inserts else "")
             + (f", added {acc_char_added} accounts to Acc_Char" if acc_char_added else "")
             + f", highlighted {len(all_highlight_rows)} rows",
             percentage=80,
             data={
                 "counter_entries": len(counter_entries),
                 "interest_splits": len(interest_inserts),
                 "rows_appended": rows_added,
                 "rows_inserted": len(interest_inserts),
                 "acc_char_accounts_added": acc_char_added,
             })
        await asyncio.sleep(0.4)

        # â”€â”€ Step 5: Cleanup & Finalize â”€â”€
        emit("step_start", "cleanup", "Cleaning up zero-balance saving accounts...", percentage=80)
        await asyncio.sleep(0.3)

        from .openpyxl_handler import get_openpyxl_handler
        handler = get_openpyxl_handler()
        saving_rows_removed = 0
        # Collect settled saving account numbers from counter entries
        settled_saving_accounts = {
            str(c.account).strip() for c in counter_entries if c.account
        }
        try:
            saving_rows_removed = handler.remove_settled_saving_account_rows(
                Path(working_file), settled_saving_accounts,
            )
        except Exception as e:
            logger.warning(f"Failed to remove settled saving account rows: {e}")

        total_new_rows = rows_added + len(interest_inserts)
        if self.db_session:
            await self.db_session.execute(
                update(CashReportSessionModel)
                .where(CashReportSessionModel.session_id == session_id)
                .values(
                    total_transactions=CashReportSessionModel.total_transactions + total_new_rows,
                )
            )
            await self.db_session.commit()

        nature_corrections = len([m for m in cell_modifications.values() if "I" in m])
        result = {
            "session_id": session_id,
            "status": "success",
            "message": f"Created {len(counter_entries)} counter entries"
                       + (f", {len(interest_inserts)} interest splits" if interest_inserts else "")
                       + (f", added {acc_char_added} accounts to Acc_Char" if acc_char_added else ""),
            "counter_entries_created": len(counter_entries),
            "interest_splits_created": len(interest_inserts),
            "nature_corrections": nature_corrections,
            "rows_appended": rows_added,
            "rows_inserted": len(interest_inserts),
            "total_rows_in_movement": total_rows,
            "settlement_transactions_found": len(settlement_transactions),
            "skipped_no_account": len(skipped_no_account),
            "skipped_duplicate": len(skipped_duplicate),
            "total_transactions_scanned": len(transactions),
            "saving_rows_removed": saving_rows_removed,
            "acc_char_accounts_added": acc_char_added,
        }

        emit("step_complete", "cleanup",
             f"Removed {saving_rows_removed} zero-balance rows",
             percentage=95,
             data={"saving_rows_removed": saving_rows_removed})
        await asyncio.sleep(0.4)

        # Save snapshot so this step's result can be downloaded independently
        self._save_step_snapshot(session_id, "settlement")

        emit("complete", "done",
             f"Settlement complete â€” {len(counter_entries)} counter entries"
             + (f", {len(interest_inserts)} interest splits" if interest_inserts else ""),
             percentage=100, data=result)

        return result

    async def run_open_new_automation(
        self,
        session_id: str,
        lookup_file_contents: Optional[List[bytes]] = None,
        user_id: Optional[int] = None,
        progress_callback=None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run "má»Ÿ má»›i" (open new saving account) automation on Movement data.

        Logic:
        1. Detect open-new transactions by GROUP B patterns (Nature = Internal transfer out)
        2. Extract or lookup saving account number from:
           - Description (HDTG SO xxx, TK xxx, etc.)
           - Lookup file(s) (VTB Saving style) if provided
        3. Create counter entry with saving account, reversed credit/debit, nature = Internal transfer in

        Args:
            session_id: The session ID
            lookup_file_contents: Optional list of lookup file bytes (.xls/.xlsx)
            user_id: Owner user ID for access control
            progress_callback: Optional callable for SSE progress events
            dry_run: If True, run detect+lookup but do NOT write to Excel.
                     Returns candidate preview data instead.

        Returns:
            Dict with results summary (or preview data when dry_run=True)
        """
        from .progress_store import ProgressEvent

        def emit(event_type, step, message, detail="", percentage=0, data=None):
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type=event_type, step=step,
                    message=message, detail=detail,
                    percentage=percentage, data=data or {},
                ))

        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)

        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            raise ValueError(f"Session {session_id} not found")

        writer = MovementDataWriter(working_file)
        self._refresh_detection_pattern_indexes_if_needed()

        # â”€â”€ Step 1: Scan Movement data â”€â”€
        emit("step_start", "scanning", "Scanning Movement data for open-new transactions...", percentage=0)
        await asyncio.sleep(0.3)

        transactions = writer.get_all_transactions()
        if not transactions:
            emit("complete", "done", "No transactions found", percentage=100,
                 data={"status": "no_transactions"})
            return {
                "session_id": session_id,
                "status": "no_transactions",
                "message": "No transactions found in Movement sheet",
                "counter_entries_created": 0,
            }

        # Build current_account â†’ entity mapping from Cash Balance sheet
        # (same approach as settlement â€” avoids reliance on formula columns)
        account_entity_map = self._build_account_entity_map(working_file)

        emit("step_complete", "scanning",
             f"Found {len(transactions)} transactions",
             percentage=20,
             data={"total_transactions": len(transactions)})
        await asyncio.sleep(0.4)

        # â”€â”€ Step 2: Detect open-new transactions â”€â”€
        emit("step_start", "detecting", "Detecting open-new transactions (GROUP B patterns)...", percentage=20)
        await asyncio.sleep(0.3)

        open_new_transactions = []
        _on_skip = {"no_credit": 0, "wrong_nature": 0, "no_pattern": 0}
        for idx, tx in enumerate(transactions):
            row_idx = idx + 4
            if self._is_open_new_candidate(tx):
                open_new_transactions.append((tx, row_idx))
            else:
                if not tx.credit or tx.credit <= 0:
                    _on_skip["no_credit"] += 1
                elif (tx.nature or "").strip().lower() != "internal transfer out":
                    _on_skip["wrong_nature"] += 1
                else:
                    _on_skip["no_pattern"] += 1
        logger.info(f"Open-new detect: {len(open_new_transactions)} candidates, skip reasons: {_on_skip}")

        if not open_new_transactions:
            emit("complete", "done", "No open-new transactions found", percentage=100,
                 data={"status": "no_open_new", "total_transactions_scanned": len(transactions)})
            return {
                "session_id": session_id,
                "status": "no_open_new",
                "message": "No open-new transactions found (GROUP B patterns)",
                "counter_entries_created": 0,
                "total_transactions_scanned": len(transactions),
            }

        emit("step_complete", "detecting",
             f"Found {len(open_new_transactions)} open-new transactions",
             percentage=40,
             data={"open_new_found": len(open_new_transactions)})
        await asyncio.sleep(0.4)

        # â”€â”€ Step 3: Load lookup data if provided â”€â”€
        emit("step_start", "lookup", "Looking up saving accounts...", percentage=40)
        await asyncio.sleep(0.3)

        lookup_accounts: Dict[Tuple[str, float], List[str]] = {}
        lookup_account_details: Dict[str, Dict[str, Any]] = {}
        if lookup_file_contents:
            for i, file_content in enumerate(lookup_file_contents):
                parsed, parsed_details = self._parse_saving_lookup_file_with_metadata(file_content)
                loaded_count = 0
                for key, accounts in parsed.items():
                    if not accounts:
                        continue
                    bucket = lookup_accounts.setdefault(key, [])
                    for account in accounts:
                        if account not in bucket:
                            bucket.append(account)
                            loaded_count += 1
                for acc, meta in parsed_details.items():
                    if not acc:
                        continue
                    dst = lookup_account_details.setdefault(acc, {})
                    for mk, mv in meta.items():
                        if mv in (None, ""):
                            continue
                        if dst.get(mk) in (None, ""):
                            dst[mk] = mv
                logger.info(f"Lookup file {i+1}: loaded {loaded_count} accounts")
            total_lookup_accounts = sum(len(v) for v in lookup_accounts.values())
            logger.info(
                f"Total lookup accounts from {len(lookup_file_contents)} file(s): "
                f"{total_lookup_accounts} across {len(lookup_accounts)} (entity,amount) keys"
            )

        # Load existing saving accounts to filter out non-new accounts
        saving_accounts = self._read_saving_accounts(working_file)
        existing_saving_acc_set = {sa["account"] for sa in saving_accounts}
        # Map account → old interest rate (for rate-decrease detection)
        existing_acc_rates: Dict[str, float] = {
            sa["account"]: sa["interest_rate"]
            for sa in saving_accounts
            if sa.get("interest_rate") is not None
        }
        logger.info(f"Existing saving accounts in template: {len(existing_saving_acc_set)}")

        # Load full Acc_Char data for suffix search (current_account_1, _2...)
        acc_char_data = self._read_acc_char_full_data(working_file)

        # Load lookup maps for CODE and BRANCH determination
        # 1) accountâ†'code from Acc_Char Bâ†'C (reliable fallback)
        account_to_code = self._read_acc_char_account_to_code(working_file)
        # 2) entityâ†’code from Def sheet (primary CODE source per user spec)
        def_entity_to_code = self._read_def_entity_to_code(working_file)
        # 3) entityâ†’branches from Cash balance (Prior period) for BRANCH
        entity_branches = self._read_prior_period_branches(working_file)
        logger.info(f"Lookups: {len(account_to_code)} accâ†’code, {len(def_entity_to_code)} entâ†’code (Def), {len(entity_branches)} entâ†’branches (Prior)")

        # Check for existing counter entries to avoid duplicates
        # Counter entries have nature "Internal transfer in" (from any source)
        existing_descs = set()
        for tx in transactions:
            if tx.nature and "transfer in" in tx.nature.lower():
                existing_descs.add(tx.description.strip() if tx.description else "")

        counter_entries = []
        counter_entry_info = []  # Additional info for B1-B3 steps
        original_row_indices = []
        skipped_no_account = []
        skipped_duplicate = []
        skipped_existing = []  # Accounts that already exist (not genuinely new)
        used_lookup_accounts = set()

        def _normalize_account_text(value: str) -> str:
            text = str(value or "").strip().replace("'", "")
            if not text:
                return ""
            try:
                if "." in text and float(text) == int(float(text)):
                    return str(int(float(text)))
            except (ValueError, OverflowError):
                pass
            return text

        # Normalize existing account set once to prevent same-run duplicates.
        existing_saving_acc_set = {_normalize_account_text(acc) for acc in existing_saving_acc_set if acc}

        # Index lookup accounts by amount for fallback matching (entity may be abbreviated in template).
        def _amount_key(value: Any) -> int:
            try:
                return int(round(float(value or 0)))
            except (ValueError, TypeError):
                return 0

        lookup_accounts_by_amount: Dict[int, List[str]] = {}
        for (_, amt), accounts in lookup_accounts.items():
            amt_key = _amount_key(amt)
            if amt_key <= 0:
                continue
            bucket = lookup_accounts_by_amount.setdefault(amt_key, [])
            for account in accounts:
                acc_norm = _normalize_account_text(account)
                if acc_norm and acc_norm not in bucket:
                    bucket.append(acc_norm)

        entity_noise_tokens = {
            "CT", "CTY", "CONG", "TY", "TNHH", "MTV", "CP", "CO", "PHAN", "VA",
            "PHAT", "TRIEN", "NGHIEP", "DAU", "TU", "DU", "AN", "MOT", "THANH",
            "VIEN", "HUU", "HAN", "JSC", "PTCN", "BW",
        }

        def _normalize_entity_text(value: str) -> str:
            text = str(value or "").upper()
            text = text.replace("BWID", "BW")
            text = re.sub(r'[^A-Z0-9]+', ' ', text)
            tokens = [tok for tok in text.split() if tok and tok not in entity_noise_tokens]
            return "".join(tokens)

        def _entity_similarity_score(left: str, right: str) -> float:
            left_norm = _normalize_entity_text(left)
            right_norm = _normalize_entity_text(right)
            if not left_norm or not right_norm:
                return 0.0
            if left_norm == right_norm:
                return 1.0
            base = SequenceMatcher(None, left_norm, right_norm).ratio()
            if left_norm in right_norm or right_norm in left_norm:
                base = max(base, 0.85)
            return base

        def _available_account(candidate: str) -> bool:
            acc_norm = _normalize_account_text(candidate)
            if not acc_norm:
                return False
            if acc_norm in existing_saving_acc_set:
                return False
            if acc_norm in used_lookup_accounts:
                return False
            return True

        def _select_lookup_account(candidates: List[str]) -> Optional[str]:
            """Pick first available lookup account not already used/existing."""
            for candidate in candidates:
                acc_norm = _normalize_account_text(candidate)
                if not _available_account(acc_norm):
                    continue
                return acc_norm
            return None

        def _match_lookup_by_metadata(
            *,
            tx_bank: str,
            tx_date: Optional[date],
            amount_value: float,
            entity_name: str,
            term_info: Dict[str, Any],
        ) -> Optional[str]:
            """
            Fallback matcher when description has no explicit account and entity text is abbreviated.
            Uses amount + bank + opening date + term/maturity + fuzzy entity score.
            """
            amount_candidates = lookup_accounts_by_amount.get(_amount_key(amount_value), [])
            candidates = [
                _normalize_account_text(acc)
                for acc in amount_candidates
                if _available_account(acc)
            ]
            # Keep order stable and unique.
            candidates = list(dict.fromkeys([acc for acc in candidates if acc]))
            if not candidates:
                return None

            bank_norm = str(tx_bank or "").strip().upper()
            if bank_norm:
                bank_candidates = [
                    acc for acc in candidates
                    if str(lookup_account_details.get(acc, {}).get("provider") or "").strip().upper() == bank_norm
                ]
                if bank_candidates:
                    candidates = bank_candidates

            if tx_date:
                date_candidates = [
                    acc for acc in candidates
                    if lookup_account_details.get(acc, {}).get("opening_date") == tx_date
                ]
                if date_candidates:
                    candidates = date_candidates

            maturity = term_info.get("maturity_date")
            if maturity:
                maturity_candidates = [
                    acc for acc in candidates
                    if lookup_account_details.get(acc, {}).get("maturity_date") == maturity
                ]
                if maturity_candidates:
                    candidates = maturity_candidates

            term_days = term_info.get("term_days")
            if term_days:
                day_candidates = [
                    acc for acc in candidates
                    if lookup_account_details.get(acc, {}).get("term_days") == term_days
                ]
                if day_candidates:
                    candidates = day_candidates

            term_months = term_info.get("term_months")
            if term_months:
                month_candidates = [
                    acc for acc in candidates
                    if lookup_account_details.get(acc, {}).get("term_months") == term_months
                ]
                if month_candidates:
                    candidates = month_candidates

            if len(candidates) == 1:
                return candidates[0]

            if entity_name and candidates:
                scored = []
                for acc in candidates:
                    lookup_entity = str(lookup_account_details.get(acc, {}).get("entity") or "")
                    score = _entity_similarity_score(entity_name, lookup_entity)
                    scored.append((score, acc))
                scored.sort(key=lambda item: item[0], reverse=True)
                top_score, top_acc = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else 0.0
                if top_score >= 0.55 and (top_score - second_score) >= 0.12:
                    return top_acc

            return None

        for tx, row_idx in open_new_transactions:
            desc = tx.description.strip() if tx.description else ""
            if desc in existing_descs:
                skipped_duplicate.append(desc)
                continue

            # Resolve entity from Cash Balance accountâ†’entity map
            tx_account = str(tx.account).strip() if tx.account else ""
            entity = account_entity_map.get(tx_account, "")

            # If still empty, try Acc_Char â†’ Def reverse lookup
            if not entity:
                code_for_entity = account_to_code.get(tx_account, "")
                if code_for_entity:
                    for ent_name, c in def_entity_to_code.items():
                        if c == code_for_entity:
                            entity = ent_name
                            break
                    if not entity:
                        entity = code_for_entity
                logger.debug(f"Derived entity '{entity}' from current account {tx_account}")
            amount = float(tx.credit) if tx.credit else 0
            term_info = self._extract_term_info(desc)

            # Try to find saving account:
            # 1. Extract from description
            saving_acc = _normalize_account_text(self._extract_saving_account_for_open_new(desc))

            # 2. Lookup from uploaded file by entity + amount
            if not saving_acc and lookup_accounts:
                # Try exact match first
                key = (entity.upper(), amount)
                saving_acc = _select_lookup_account(lookup_accounts.get(key, []))
                # Try partial entity match (normalize spaces for shared-string artifacts)
                if not saving_acc:
                    entity_norm = re.sub(r'\s+', '', entity.upper())
                    for (ent, amt), accounts in lookup_accounts.items():
                        ent_norm = re.sub(r'\s+', '', ent)
                        if ent_norm in entity_norm or entity_norm in ent_norm:
                            if abs(amt - amount) < 1:  # Allow small difference
                                saving_acc = _select_lookup_account(accounts)
                                if saving_acc:
                                    break
                # Fallback: infer from amount/bank/date/term metadata.
                if not saving_acc:
                    saving_acc = _match_lookup_by_metadata(
                        tx_bank=str(tx.bank or ""),
                        tx_date=tx.date if isinstance(tx.date, date) else None,
                        amount_value=amount,
                        entity_name=entity,
                        term_info=term_info,
                    )
                    if saving_acc:
                        logger.info(
                            f"Open-new: matched by metadata: {saving_acc} "
                            f"(bank={tx.bank}, amount={amount}, date={tx.date})"
                        )

                # Last fallback: if entity/metadata did not help, match by amount only
                # when exactly one available account candidate remains for this amount.
                if not saving_acc:
                    amount_candidates = lookup_accounts_by_amount.get(_amount_key(amount), [])
                    available = []
                    for candidate in amount_candidates:
                        acc_norm = _normalize_account_text(candidate)
                        if not _available_account(acc_norm):
                            continue
                        available.append(acc_norm)
                    available_unique = list(dict.fromkeys(available))
                    if len(available_unique) == 1:
                        saving_acc = available_unique[0]
                        logger.info(f"Open-new: matched by amount only: {saving_acc}")

            # NOTE: Do NOT lookup from existing Saving Account sheet for mở mới.
            # Step 3 from settlement flow is intentionally omitted here because
            # mở mới only creates counter entries for genuinely NEW accounts.

            # Fallback: if no saving account found, create current_account_{next}.
            # Find the next available suffix: if _1 exists → _2, if _1,_2 exist → _3.
            is_current_account_fallback = False
            if not saving_acc:
                if not tx_account:
                    skipped_no_account.append(tx.description)
                    logger.warning(f"Open-new: no saving account and no current account for '{tx.description}'")
                    continue

                # Find next available suffix (_1, _2, _3, ...)
                # Check both Acc_Char and already-reserved accounts in this run
                suffix = 1
                while (f"{tx_account}_{suffix}" in acc_char_data
                       or f"{tx_account}_{suffix}" in existing_saving_acc_set):
                    suffix += 1
                saving_acc = f"{tx_account}_{suffix}"
                logger.info(f"Open-new: created suffixed account {saving_acc} (no lookup match)")
                is_current_account_fallback = True

            # If saving account already exists, create a suffixed version (_1, _2, _3, ...)
            # This handles multiple deposits to the same base account in the same period.
            if saving_acc in existing_saving_acc_set:
                base_acc = saving_acc.split('_')[0]
                suffix = 1
                while f"{base_acc}_{suffix}" in existing_saving_acc_set:
                    suffix += 1
                saving_acc = f"{base_acc}_{suffix}"
                logger.info(f"Open-new: created suffixed account {saving_acc} (base already exists)")

            # Reserve this account immediately to prevent duplicate inserts in same run.
            existing_saving_acc_set.add(saving_acc)
            used_lookup_accounts.add(saving_acc)

            lookup_meta = lookup_account_details.get(saving_acc, {})
            # Flag: no lookup data → insert row but leave term/rate empty, highlight yellow
            # Always True for current_account fallback (suffixed accounts never in lookup)
            no_lookup_data = is_current_account_fallback or not lookup_meta
            if no_lookup_data:
                # Clear all term info — cells will be left empty in Saving Account
                term_info = {"term_months": None, "term_days": None, "interest_rate": None, "maturity_date": None}
                logger.info(f"Open-new: no lookup data for {saving_acc}, will insert with empty term/rate and highlight yellow")
            else:
                # Lookup file is authoritative for dates and term — override
                # description-extracted values when lookup provides them.
                if lookup_meta.get("maturity_date"):
                    term_info["maturity_date"] = lookup_meta["maturity_date"]
                if lookup_meta.get("term_months"):
                    term_info["term_months"] = lookup_meta["term_months"]
                elif lookup_meta.get("term_days"):
                    term_info["term_days"] = lookup_meta["term_days"]
                # Interest rate: ONLY use lookup value. Description-extracted rates
                # are unreliable — clear them so the cell shows "check web" instead.
                if lookup_meta.get("interest_rate") is not None:
                    term_info["interest_rate"] = lookup_meta["interest_rate"]
                else:
                    term_info["interest_rate"] = None

            opening_date_for_insert = tx.date
            if lookup_meta.get("opening_date"):
                opening_date_for_insert = lookup_meta["opening_date"]

            currency_for_insert = str(lookup_meta.get("currency") or "VND").strip().upper() or "VND"

            # Determine BRANCH from Cash balance (Prior period) by entity + bank prefix
            saving_bank = self._determine_saving_branch(saving_acc, entity, entity_branches, tx.bank or "")

            # Create counter entry: swap credit to debit, nature = Internal transfer in
            counter = MovementTransaction(
                source=tx.source or "Automation",
                bank=tx.bank,
                account=saving_acc,
                date=tx.date,
                description=tx.description or "",
                debit=tx.credit,  # Swap: original credit â†' counter debit
                credit=Decimal("0"),
                nature="Internal transfer in",
            )
            counter_entries.append(counter)
            original_row_indices.append(row_idx)

            # Look up CODE from Def sheet (primary): entity â†’ code
            # Fallback: Acc_Char accountâ†’code using CURRENT account (tx.account)
            code = ""
            if entity:
                code = def_entity_to_code.get(entity.upper(), "")
                # Partial match with whitespace normalization (for shared-string artifacts)
                if not code:
                    entity_norm = re.sub(r'\s+', '', entity.upper())
                    for ent_name, c in def_entity_to_code.items():
                        ent_norm = re.sub(r'\s+', '', ent_name)
                        if ent_norm == entity_norm or ent_norm in entity_norm or entity_norm in ent_norm:
                            code = c
                            break
            if not code:
                current_acc = str(tx.account).strip() if tx.account else ""
                code = account_to_code.get(current_acc, "")
            if not code:
                logger.warning(f"Open-new: no CODE found for entity={entity}, current_acc={str(tx.account).strip() if tx.account else ''}")

            # Detect interest rate decrease (1-2%) vs same account's old rate
            rate_decreased = False
            new_rate = term_info.get("interest_rate")
            if new_rate is not None:
                # For suffixed accounts (e.g. 1064099857_1), check the base account
                base_acc = saving_acc.split("_")[0]
                old_rate = existing_acc_rates.get(base_acc)
                if old_rate is not None:
                    drop = old_rate - new_rate
                    if 0.01 <= drop <= 0.02:  # 1-2% decrease (stored as decimal)
                        rate_decreased = True
                        logger.info(
                            f"Open-new: rate decrease detected for {saving_acc}: "
                            f"{old_rate:.4f} → {new_rate:.4f} (drop {drop:.4f})"
                        )

            # Store additional info for B1-B3 steps
            # Even for current-account fallback, the suffixed account (e.g. 116632406789_1)
            # is treated as a Saving Account entry — it goes into all 3 sheets.
            account_type = "Saving Account"
            counter_entry_info.append({
                "saving_acc": saving_acc,
                "entity": entity,
                "bank": saving_bank,
                "code": code,
                "amount": amount,
                "currency": currency_for_insert,
                "term_info": term_info,
                "opening_date": opening_date_for_insert,
                "rate_decreased": rate_decreased,
                "account_type": account_type,
                "no_lookup_data": no_lookup_data,
            })

        emit("step_complete", "lookup",
             f"Matched {len(counter_entries)} new accounts, {len(skipped_existing)} existing, {len(skipped_no_account)} not found",
             percentage=60,
             data={
                 "matched": len(counter_entries),
                 "skipped_existing": len(skipped_existing),
                 "skipped_no_account": len(skipped_no_account),
                 "skipped_duplicate": len(skipped_duplicate),
             })
        await asyncio.sleep(0.4)

        # ── Dry-run: return preview without writing ──
        if dry_run:
            candidates_preview = []
            for (tx, row_idx), info in zip(open_new_transactions, counter_entry_info):
                desc = (tx.description or "").strip()
                candidates_preview.append({
                    "row": row_idx,
                    "date": tx.date.isoformat() if tx.date else None,
                    "bank": tx.bank or "",
                    "account": str(tx.account or ""),
                    "description": desc,
                    "credit": float(tx.credit) if tx.credit else 0,
                    "nature": tx.nature or "",
                    "status": (
                        "duplicate" if desc in [d for d in skipped_duplicate]
                        else "no_account" if desc in skipped_no_account
                        else "matched"
                    ),
                    "saving_account": info["saving_acc"],
                    "entity": info.get("entity", ""),
                    "code": info.get("code", ""),
                    "currency": info.get("currency", "VND"),
                    "no_lookup_data": info.get("no_lookup_data", False),
                    "term_months": info.get("term_info", {}).get("term_months"),
                    "interest_rate": info.get("term_info", {}).get("interest_rate"),
                    "maturity_date": (
                        info.get("term_info", {}).get("maturity_date").isoformat()
                        if info.get("term_info", {}).get("maturity_date") else None
                    ),
                })
            # Also add skipped rows
            for desc in skipped_no_account:
                candidates_preview.append({
                    "description": desc,
                    "status": "no_account",
                    "saving_account": None,
                })
            for desc in skipped_duplicate:
                candidates_preview.append({
                    "description": desc,
                    "status": "duplicate",
                    "saving_account": None,
                })
            return {
                "session_id": session_id,
                "dry_run": True,
                "status": "preview",
                "total_transactions_scanned": len(transactions),
                "open_new_candidates": len(open_new_transactions),
                "counter_entries_would_create": len(counter_entries),
                "skipped_no_account": skipped_no_account,
                "skipped_duplicate": skipped_duplicate,
                "skipped_existing": skipped_existing,
                "candidates": candidates_preview,
            }

        if not counter_entries:
            msg = "No counter entries created"
            if skipped_no_account:
                msg += f". {len(skipped_no_account)} skipped (no matching saving account)"
            if skipped_duplicate:
                msg += f". {len(skipped_duplicate)} skipped (already exists)"
            emit("complete", "done", msg, percentage=100,
                 data={"status": "no_counter_entries"})
            return {
                "session_id": session_id,
                "status": "no_counter_entries",
                "message": msg,
                "counter_entries_created": 0,
                "skipped_no_account": len(skipped_no_account),
                "skipped_duplicate": len(skipped_duplicate),
                "total_transactions_scanned": len(transactions),
            }

        # â”€â”€ Step 4: Create counter entries â”€â”€
        emit("step_start", "writing", f"Creating {len(counter_entries)} counter entries...", percentage=60)
        await asyncio.sleep(0.3)

        rows_added, total_rows = writer.append_transactions(counter_entries)

        # Highlight both original and counter rows
        counter_start_row = total_rows - rows_added + 4
        counter_row_indices = list(range(counter_start_row, counter_start_row + rows_added))
        all_highlight_rows = original_row_indices + counter_row_indices
        try:
            writer.highlight_open_new_rows(all_highlight_rows)
        except Exception as e:
            logger.warning(f"Failed to highlight open-new rows: {e}")

        emit("step_complete", "writing",
             f"Created {rows_added} counter entries, highlighted {len(all_highlight_rows)} rows",
             percentage=70,
             data={"rows_added": rows_added})
        await asyncio.sleep(0.4)

        # â”€â”€ Step 4.5: Add rows to Acc_Char, Saving Account, Cash Balance (B1-B3) â”€â”€
        # Batch: ONE ZIP read/write for all 3 sheets
        emit("step_start", "catalog",
             "Adding new saving accounts to catalog sheets (Acc_Char, Saving Account, Cash Balance)...",
             percentage=70)
        await asyncio.sleep(0.2)

        from .openpyxl_handler import get_openpyxl_handler
        handler = get_openpyxl_handler()

        sa_acc_to_row: Dict[str, int] = {}
        try:
            batch_result = await asyncio.to_thread(
                handler.add_rows_to_sheets_batch,
                working_file,
                counter_entry_info,
            )
            acc_char_added = batch_result["acc_char_added"]
            saving_account_added = batch_result["saving_added"]
            cash_balance_added = batch_result["cash_balance_added"]
            sa_acc_to_row = batch_result.get("sa_acc_to_row", {})
        except Exception as e:
            logger.error(f"Batch add to catalog sheets failed: {e}", exc_info=True)
            acc_char_added = saving_account_added = cash_balance_added = 0

        logger.info(f"B1-B3 completed: Acc_Char +{acc_char_added}, Saving Account +{saving_account_added}, Cash Balance +{cash_balance_added}")

        # Highlight interest-rate cells in red for accounts with rate decrease
        rate_decrease_cells: List[Tuple[int, str]] = []
        for info in counter_entry_info:
            if info.get("rate_decreased") and info["saving_acc"] in sa_acc_to_row:
                rate_decrease_cells.append((sa_acc_to_row[info["saving_acc"]], "L"))
        if rate_decrease_cells:
            try:
                await asyncio.to_thread(
                    handler.highlight_cells,
                    working_file,
                    "Saving Account",
                    rate_decrease_cells,
                )
                logger.info(f"Highlighted {len(rate_decrease_cells)} rate-decrease cells in red")
            except Exception as e:
                logger.warning(f"Failed to highlight rate-decrease cells: {e}")

        # Highlight yellow: rows with no lookup data (missing term/rate info)
        no_lookup_rows: List[int] = []
        for info in counter_entry_info:
            if info.get("no_lookup_data") and info["saving_acc"] in sa_acc_to_row:
                no_lookup_rows.append(sa_acc_to_row[info["saving_acc"]])
        if no_lookup_rows:
            yellow_fill = (
                b'<fill><patternFill patternType="solid">'
                b'<fgColor rgb="FFFFFF00"/>'
                b'</patternFill></fill>'
            )
            try:
                await asyncio.to_thread(
                    handler.highlight_rows,
                    Path(working_file),
                    "Saving Account",
                    no_lookup_rows,
                    yellow_fill,
                )
                logger.info(f"Highlighted {len(no_lookup_rows)} no-lookup-data rows yellow in Saving Account")
            except Exception as e:
                logger.warning(f"Failed to highlight no-lookup-data rows yellow: {e}")

        emit("step_complete", "catalog",
             f"Added to catalogs: Acc_Char +{acc_char_added}, Saving Account +{saving_account_added}, Cash Balance +{cash_balance_added}",
             percentage=85,
             data={
                 "acc_char_added": acc_char_added,
                 "saving_account_added": saving_account_added,
                 "cash_balance_added": cash_balance_added,
             })
        await asyncio.sleep(0.3)

        # â”€â”€ Step 5: Finalize â”€â”€
        if self.db_session:
            await self.db_session.execute(
                update(CashReportSessionModel)
                .where(CashReportSessionModel.session_id == session_id)
                .values(
                    total_transactions=CashReportSessionModel.total_transactions + rows_added,
                )
            )
            await self.db_session.commit()

        result = {
            "session_id": session_id,
            "status": "success",
            "message": f"Created {rows_added} counter entries for open-new transactions",
            "counter_entries_created": rows_added,
            "total_rows_in_movement": total_rows,
            "open_new_transactions_found": len(open_new_transactions),
            "candidates_found": len(open_new_transactions),
            "skipped_existing": len(skipped_existing),
            "skipped_no_account": len(skipped_no_account),
            "skipped_duplicate": len(skipped_duplicate),
            "total_transactions_scanned": len(transactions),
            # B1-B3 stats
            "acc_char_added": acc_char_added,
            "saving_account_added": saving_account_added,
            "cash_balance_added": cash_balance_added,
        }

        # Save snapshot so this step's result can be downloaded independently
        self._save_step_snapshot(session_id, "open_new")

        emit("complete", "done",
             f"Open-new complete â€” {rows_added} entries, {acc_char_added} new accounts added",
             percentage=100, data=result)

        return result

    def _parse_saving_lookup_file(self, file_content: bytes) -> Dict[Tuple[str, float], List[str]]:
        """Backward-compatible wrapper returning only entity+amount -> accounts mapping."""
        lookup, _ = self._parse_saving_lookup_file_with_metadata(file_content)
        return lookup

    def _parse_saving_lookup_file_with_metadata(
        self,
        file_content: bytes,
    ) -> Tuple[Dict[Tuple[str, float], List[str]], Dict[str, Dict[str, Any]]]:
        """
        Parse lookup file and return:
        - lookup map: ``(ENTITY_UPPER, AMOUNT) -> [ACCOUNT_NO, ...]``
        - account metadata: ``ACCOUNT_NO -> {provider, maturity_date, opening_date, term, rate, ...}``
        """
        import io
        import zipfile as _zf
        import xml.etree.ElementTree as _ET

        lookup: Dict[Tuple[str, float], List[str]] = {}
        account_meta: Dict[str, Dict[str, Any]] = {}

        def _normalize_lookup_account(raw: Any) -> str:
            acc_str = str(raw or "").strip().replace("'", "")
            if not acc_str:
                return ""
            try:
                if "." in acc_str and float(acc_str) == int(float(acc_str)):
                    acc_str = str(int(float(acc_str)))
            except (ValueError, OverflowError):
                pass
            return acc_str

        def _parse_lookup_amount(raw: Any) -> Optional[float]:
            if raw is None:
                return None
            if isinstance(raw, (int, float)):
                return float(raw)
            s = str(raw).strip().replace(" ", "")
            if not s:
                return None
            if re.match(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$", s):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        def _parse_lookup_rate(raw: Any) -> Optional[float]:
            if raw is None or raw == "":
                return None
            if isinstance(raw, (int, float)):
                v = float(raw)
                if v <= 0:
                    return None
                return v / 100 if v > 1 else v
            s = str(raw).strip().replace("%", "").replace(",", ".")
            if not s:
                return None
            try:
                v = float(s)
            except (ValueError, TypeError):
                return None
            if v <= 0:
                return None
            return v / 100 if v > 1 else v

        def _parse_lookup_date(raw: Any) -> Optional[date]:
            if raw is None or raw == "":
                return None
            if isinstance(raw, datetime):
                return raw.date()
            if isinstance(raw, date):
                return raw
            if isinstance(raw, (int, float)):
                try:
                    serial = int(float(raw))
                    if 20000 <= serial <= 80000:
                        base_ord = datetime(1899, 12, 30).toordinal()
                        return datetime.fromordinal(base_ord + serial).date()
                except Exception:
                    return None
                return None

            s = str(raw).strip()
            if not s:
                return None
            s = s.split()[0]
            # Handle string-encoded Excel serial numbers (e.g., "46051" from XML)
            if s.replace('.', '', 1).isdigit():
                try:
                    serial = int(float(s))
                    if 20000 <= serial <= 80000:
                        base_ord = datetime(1899, 12, 30).toordinal()
                        return datetime.fromordinal(base_ord + serial).date()
                except Exception:
                    pass
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
                try:
                    return datetime.strptime(s, fmt).date()
                except ValueError:
                    continue
            return None

        def _parse_term_info(term_raw: Any, product_raw: Any = None) -> Tuple[Optional[int], Optional[int]]:
            term_months = None
            term_days = None

            def _consume(text: str) -> None:
                nonlocal term_months, term_days
                if not text:
                    return
                up = text.upper()
                m_month = re.search(r'(\d{1,2})\s*(THANG|THÁNG|M)\b', up)
                if m_month:
                    term_months = int(m_month.group(1))
                    return
                m_day = re.search(r'(\d{1,3})\s*(NGAY|NGÀY|D)\b', up)
                if m_day:
                    term_days = int(m_day.group(1))

            _consume(str(term_raw or ""))
            if term_months is None and term_days is None:
                _consume(str(product_raw or ""))
            return term_months, term_days

        def _merge_meta(
            account: str,
            *,
            entity: str = "",
            currency: str = "",
            provider: str = "",
            opening_date: Optional[date] = None,
            maturity_date: Optional[date] = None,
            interest_rate: Optional[float] = None,
            term_months: Optional[int] = None,
            term_days: Optional[int] = None,
        ) -> None:
            if not account:
                return
            meta = account_meta.setdefault(account, {
                "entity": "",
                "currency": "",
                "provider": "",
                "opening_date": None,
                "maturity_date": None,
                "interest_rate": None,
                "term_months": None,
                "term_days": None,
            })
            if entity and not meta["entity"]:
                meta["entity"] = entity
            if currency and not meta["currency"]:
                meta["currency"] = currency
            if provider and not meta["provider"]:
                meta["provider"] = provider
            if opening_date and not meta["opening_date"]:
                meta["opening_date"] = opening_date
            if maturity_date and not meta["maturity_date"]:
                meta["maturity_date"] = maturity_date
            if interest_rate is not None and meta["interest_rate"] is None:
                meta["interest_rate"] = interest_rate
            if term_months and not meta["term_months"]:
                meta["term_months"] = term_months
            if term_days and not meta["term_days"]:
                meta["term_days"] = term_days

        def _add_lookup_account(
            entity_raw: Any,
            amount_raw: Any,
            account_raw: Any,
            *,
            term_raw: Any = None,
            rate_raw: Any = None,
            opening_date_raw: Any = None,
            maturity_date_raw: Any = None,
            currency_raw: Any = None,
            product_raw: Any = None,
            provider_raw: Any = None,
        ) -> None:
            acc_str = _normalize_lookup_account(account_raw)
            ent_str = str(entity_raw or "").strip().upper()
            amount_val = _parse_lookup_amount(amount_raw)
            currency_str = str(currency_raw or "").strip().upper()
            provider_str = str(provider_raw or "").strip().upper()
            opening_date_val = _parse_lookup_date(opening_date_raw)
            maturity_date_val = _parse_lookup_date(maturity_date_raw)
            interest_rate_val = _parse_lookup_rate(rate_raw)
            term_months, term_days = _parse_term_info(term_raw, product_raw)

            # Compute term_months from opening/maturity dates when not
            # explicitly provided (VCB lookup files have dates but no term column).
            if term_months is None and term_days is None and opening_date_val and maturity_date_val:
                # Count complete calendar months between the two dates.
                m_diff = (maturity_date_val.year - opening_date_val.year) * 12 + (maturity_date_val.month - opening_date_val.month)
                if maturity_date_val.day < opening_date_val.day:
                    m_diff -= 1  # not a full month yet
                if m_diff > 0:
                    term_months = m_diff
                elif m_diff == 0 and maturity_date_val > opening_date_val:
                    # Less than a month apart → compute days instead
                    term_days = (maturity_date_val - opening_date_val).days

            _merge_meta(
                acc_str,
                entity=ent_str,
                currency=currency_str,
                provider=provider_str,
                opening_date=opening_date_val,
                maturity_date=maturity_date_val,
                interest_rate=interest_rate_val,
                term_months=term_months,
                term_days=term_days,
            )

            if not ent_str or not acc_str or amount_val is None or amount_val <= 0:
                return

            key = (ent_str, amount_val)
            bucket = lookup.setdefault(key, [])
            if acc_str not in bucket:
                bucket.append(acc_str)

        # Try .xlsx
        try:
            zdata = io.BytesIO(file_content)
            with _zf.ZipFile(zdata) as z:
                if "xl/workbook.xml" not in z.namelist():
                    raise ValueError("Not xlsx")

                shared_strings = []
                if "xl/sharedStrings.xml" in z.namelist():
                    ss_xml = z.read("xl/sharedStrings.xml")
                    for m in re.finditer(rb'<si[^>]*>.*?</si>', ss_xml, re.DOTALL):
                        parts = re.findall(rb'<t[^>]*>([^<]*)</t>', m.group(0))
                        shared_strings.append("".join(
                            p.decode("utf-8", errors="replace") for p in parts
                        ))

                worksheet_paths = sorted(
                    n for n in z.namelist()
                    if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
                )
                if not worksheet_paths:
                    raise ValueError("No worksheet XML found")
                sheet_xml = z.read(worksheet_paths[0])

            ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            root = _ET.fromstring(sheet_xml)
            sheet_data = root.find(f"{{{ns}}}sheetData")
            if sheet_data is None:
                raise ValueError("No sheetData")

            for row_el in sheet_data:
                row: Dict[str, Any] = {}
                for cell in row_el:
                    ref = cell.get("r", "")
                    col = re.match(r"([A-Z]+)", ref)
                    if not col:
                        continue
                    col_letter = col.group(1)
                    t = cell.get("t", "")
                    v_el = cell.find(f"{{{ns}}}v")
                    val = v_el.text if v_el is not None else None
                    is_el = cell.find(f"{{{ns}}}is")
                    if is_el is not None:
                        t_el = is_el.find(f"{{{ns}}}t")
                        val = t_el.text if t_el is not None else ""
                    elif t == "s" and val and shared_strings:
                        try:
                            idx = int(val)
                            val = shared_strings[idx] if idx < len(shared_strings) else val
                        except (ValueError, TypeError):
                            pass
                    row[col_letter] = val

                # VCB style
                _add_lookup_account(
                    row.get("C"), row.get("E"), row.get("D"),
                    currency_raw=row.get("F"),
                    opening_date_raw=row.get("G"),
                    maturity_date_raw=row.get("H"),
                    product_raw=row.get("I"),
                    provider_raw="VCB",
                )
                # VTB detail style
                _add_lookup_account(
                    row.get("E"), row.get("L"), row.get("B"),
                    term_raw=row.get("G"),
                    rate_raw=row.get("H"),
                    currency_raw=row.get("I"),
                    opening_date_raw=row.get("J"),
                    maturity_date_raw=row.get("K"),
                    product_raw=row.get("F"),
                    provider_raw="VTB",
                )
                # BIDV-like style
                _add_lookup_account(
                    row.get("F") or row.get("E"), row.get("G"), row.get("C"),
                    term_raw=row.get("N"),
                    rate_raw=row.get("M"),
                    currency_raw=row.get("L"),
                    opening_date_raw=row.get("O"),
                    maturity_date_raw=row.get("P"),
                    provider_raw="BIDV",
                )

            if lookup or account_meta:
                return lookup, account_meta

        except Exception as xlsx_err:
            logger.debug(f"xlsx XML parse failed: {xlsx_err}")

        # Try .xls
        try:
            import xlrd
            wb = xlrd.open_workbook(file_contents=file_content)
            ws = wb.sheet_by_index(0)

            for row_idx in range(ws.nrows):
                row = [ws.cell_value(row_idx, c) for c in range(ws.ncols)]

                def _v(i: int) -> Any:
                    return row[i] if i < len(row) else None

                # VTB detail block
                _add_lookup_account(
                    _v(4), _v(11), _v(1),
                    term_raw=_v(6),
                    rate_raw=_v(7),
                    currency_raw=_v(8),
                    opening_date_raw=_v(9),
                    maturity_date_raw=_v(10),
                    product_raw=_v(5),
                    provider_raw="VTB",
                )
                # BIDV style
                _add_lookup_account(
                    _v(5), _v(6), _v(2),
                    term_raw=_v(13),
                    rate_raw=_v(12),
                    currency_raw=_v(11),
                    opening_date_raw=_v(14),
                    maturity_date_raw=_v(15),
                    provider_raw="BIDV",
                )
                # VCB-like fallback (when saved to xls)
                _add_lookup_account(
                    _v(2), _v(4), _v(3),
                    currency_raw=_v(5),
                    opening_date_raw=_v(6),
                    maturity_date_raw=_v(7),
                    product_raw=_v(8),
                    provider_raw="VCB",
                )

            return lookup, account_meta

        except Exception as xls_err:
            logger.warning(f"Failed to parse lookup file (tried xlsx and xls): {xls_err}")
            return {}, {}

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Test Automation (settlement + open-new using test template)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    async def run_test_automation(
        self,
        lookup_file_contents: Optional[List[bytes]] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Run settlement + open-new automation using the test template
        (cash_report_for_test.xlsx). No AI calls needed â€” natures are pre-classified.

        Creates a temporary session, runs both automations, returns combined results.
        The test session stays available for download until manually deleted.

        Args:
            lookup_file_contents: Optional list of lookup file bytes for open-new account matching.
            progress_callback: Optional callback for progress events.

        Returns:
            Dict with combined settlement + open-new results and session_id for download.
        """
        from .progress_store import ProgressEvent
        from .master_template_manager import TEMPLATES_DIR, WORKING_DIR

        def emit(event_type, step, message, detail="", percentage=0, data=None):
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type=event_type, step=step,
                    message=message, detail=detail,
                    percentage=percentage, data=data or {},
                ))

        test_template = TEMPLATES_DIR / "cash_report_for_test.xlsx"
        if not test_template.exists():
            raise FileNotFoundError(f"Test template not found: {test_template}")

        # Create temporary test session
        test_session_id = f"test-{uuid.uuid4().hex[:8]}"
        working_dir = WORKING_DIR / test_session_id
        working_dir.mkdir(parents=True, exist_ok=True)
        working_file = working_dir / "working_master.xlsx"
        shutil.copy2(test_template, working_file)

        emit("progress", "setup", "Test session created", percentage=5)

        try:
            # Step 1: Settlement
            emit("progress", "settlement", "Running settlement automation...", percentage=10)
            settlement_result = await self.run_settlement_automation(
                session_id=test_session_id,
                user_id=None,
                progress_callback=None,
            )

            emit("progress", "settlement_done",
                 f"Settlement: {settlement_result.get('counter_entries_created', 0)} entries created",
                 percentage=50)

            # Step 2: Open-new
            emit("progress", "open_new", "Running open-new automation...", percentage=55)
            open_new_result = await self.run_open_new_automation(
                session_id=test_session_id,
                lookup_file_contents=lookup_file_contents,
                user_id=None,
                progress_callback=None,
            )

            emit("progress", "open_new_done",
                 f"Open-new: {open_new_result.get('counter_entries_created', 0)} entries created",
                 percentage=95)

            combined = {
                "status": "success",
                "test_session_id": test_session_id,
                "settlement": {
                    "status": settlement_result.get("status"),
                    "counter_entries_created": settlement_result.get("counter_entries_created", 0),
                    "settlement_transactions_found": settlement_result.get("settlement_transactions_found", 0),
                    "skipped_no_account": settlement_result.get("skipped_no_account", 0),
                    "skipped_duplicate": settlement_result.get("skipped_duplicate", 0),
                    "saving_rows_removed": settlement_result.get("saving_rows_removed", 0),
                },
                "open_new": {
                    "status": open_new_result.get("status"),
                    "counter_entries_created": open_new_result.get("counter_entries_created", 0),
                    "candidates_found": open_new_result.get("candidates_found", 0),
                    "skipped_no_account": open_new_result.get("skipped_no_account", 0),
                    "skipped_duplicate": open_new_result.get("skipped_duplicate", 0),
                    "acc_char_added": open_new_result.get("acc_char_added", 0),
                    "saving_account_added": open_new_result.get("saving_account_added", 0),
                    "cash_balance_added": open_new_result.get("cash_balance_added", 0),
                },
            }

            emit("complete", "done", "Test automation completed", percentage=100, data=combined)
            return combined

        except Exception as e:
            # Clean up on failure
            if working_dir.exists():
                shutil.rmtree(working_dir, ignore_errors=True)
            emit("error", "error", str(e))
            raise

