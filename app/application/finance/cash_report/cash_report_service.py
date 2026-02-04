"""
Cash Report Service - Main service for cash report automation.
Integrates all components: MasterTemplateManager, MovementDataWriter, BankStatementReader, AI Classifier.
"""
import asyncio
from datetime import date, datetime
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
from app.domain.finance.cash_report.services.ai_classifier import AITransactionClassifier

from .master_template_manager import MasterTemplateManager
from .movement_data_writer import MovementDataWriter, MovementTransaction
from .bank_statement_reader import BankStatementReader
from .progress_store import ProgressEvent
from app.infrastructure.cache.redis_cache import get_cache_service

logger = get_logger(__name__)


class CashReportService:
    """
    Main service for cash report automation.

    Orchestrates the flow:
    1. Create session with config
    2. Upload parsed bank statements
    3. Classify transactions
    4. Write to Movement sheet
    5. Download result
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

    async def get_or_create_session(
        self,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal = Decimal("26175"),
        period_name: str = "",
        user_id: Optional[int] = None,
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
        return await self._create_new_session(opening_date, ending_date, fx_rate, period_name, user_id=user_id)

    async def _create_new_session(
        self,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal,
        period_name: str,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Internal method to create a new session."""
        # Create session with template manager
        session_info = self.template_manager.create_session(
            opening_date=opening_date,
            ending_date=ending_date,
            fx_rate=fx_rate,
            period_name=period_name,
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
    ) -> Dict[str, Any]:
        """
        Upload and process parsed bank statement files.

        Args:
            session_id: The session ID
            files: List of (filename, file_content) tuples
            filter_by_date: Whether to filter transactions by session date range

        Returns:
            Processing result summary
        """
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

        # Classify transactions using AI
        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_start",
                step="classifying",
                message=f"Classifying {len(all_transactions)} transactions with AI...",
                detail="Using Gemini AI to determine transaction categories",
                percentage=35,
            ))
            await asyncio.sleep(0)  # Flush SSE

        logger.info(f"Classifying {len(all_transactions)} transactions...")
        classified_transactions = await self._classify_transactions(
            all_transactions,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(ProgressEvent(
                event_type="step_complete",
                step="classifying",
                message=f"Classified {len(classified_transactions)} transactions",
                percentage=80,
            ))
            await asyncio.sleep(0)  # Flush SSE

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
                    "total_rows_in_movement": total_rows,
                },
            ))

        return {
            "session_id": session_id,
            "files_processed": len(files),
            "total_transactions_added": rows_added,
            "total_transactions_skipped": total_skipped,
            "total_rows_in_movement": total_rows,
            "file_results": file_results,
        }

    async def _classify_transactions(
        self,
        transactions: List[MovementTransaction],
        progress_callback: Optional[callable] = None,
    ) -> List[MovementTransaction]:
        """
        Classify transactions using AI.

        Args:
            transactions: List of transactions without Nature
            progress_callback: Optional callback for progress updates

        Returns:
            List of transactions with Nature classified
        """
        if not transactions:
            return []

        # Prepare batch for classification
        batch = []
        for tx in transactions:
            # Determine if receipt or payment based on debit/credit
            is_receipt = bool(tx.debit)  # Debit = money in = Receipt
            batch.append((tx.description, is_receipt))

        # Pre-load Redis cache into in-memory cache (production only)
        cache_keys_before = set(self.ai_classifier._cache.keys())
        try:
            cache_service = await get_cache_service()
            if cache_service.is_connected:
                key_map = self.ai_classifier.get_cache_keys_for_batch(batch)
                redis_results = await cache_service.get_classifications_bulk(list(key_map.keys()))
                if redis_results:
                    self.ai_classifier.preload_cache(redis_results)
        except Exception as e:
            logger.warning(f"Redis cache pre-load failed (continuing without): {e}")

        # Classify with per-batch progress updates
        # Uses asyncio.to_thread() to avoid blocking the event loop,
        # which would prevent SSE progress events from being flushed to the client.
        try:
            batch_size = 50
            all_natures = []
            total_batches = (len(batch) + batch_size - 1) // batch_size

            for i in range(0, len(batch), batch_size):
                chunk = batch[i:i + batch_size]
                batch_num = i // batch_size + 1

                if progress_callback:
                    pct = 35 + int((batch_num / total_batches) * 45)  # 35-80%
                    progress_callback(ProgressEvent(
                        event_type="step_update",
                        step="classifying",
                        message=f"AI batch {batch_num}/{total_batches} ({len(chunk)} transactions)...",
                        percentage=min(pct, 79),
                    ))
                    # Yield control so event loop can flush SSE events
                    await asyncio.sleep(0)

                # Run blocking AI call in thread to not block event loop
                batch_natures = await asyncio.to_thread(
                    self.ai_classifier._classify_batch_internal, chunk
                )
                all_natures.extend(batch_natures)
                logger.info(f"AI classified batch {batch_num}/{total_batches}: {len(chunk)} transactions")

            # Apply classifications
            for i, nature in enumerate(all_natures):
                if i < len(transactions):
                    transactions[i].nature = nature

        except Exception as e:
            logger.error(f"AI classification error: {e}")
            # Leave nature empty if classification fails

        # Save new classification results to Redis cache
        try:
            cache_service = await get_cache_service()
            if cache_service.is_connected:
                new_entries = self.ai_classifier.get_new_cache_entries(cache_keys_before)
                if new_entries:
                    await cache_service.set_classifications_bulk(new_entries)
        except Exception as e:
            logger.warning(f"Redis cache save failed (results still applied): {e}")

        return transactions

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

    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session status and statistics (fast - uses database).

        Args:
            session_id: The session ID

        Returns:
            Session info dict or None if not found
        """
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

    async def reset_session(self, session_id: str) -> Dict[str, Any]:
        """
        Reset session to clean state.

        Args:
            session_id: The session ID

        Returns:
            Reset result
        """
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

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session ID

        Returns:
            True if deleted, False if not found
        """
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

    def get_working_file_path(self, session_id: str) -> Optional[Path]:
        """Get the working file path for download."""
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

    def get_data_preview(self, session_id: str, limit: int = 20) -> List[dict]:
        """
        Get preview of Movement data.

        Args:
            session_id: The session ID
            limit: Maximum number of rows

        Returns:
            List of row dicts
        """
        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            return []

        writer = MovementDataWriter(working_file)
        return writer.get_data_preview(limit)

    async def run_settlement_automation(self, session_id: str) -> Dict[str, Any]:
        """
        Run settlement (tất toán) automation on Movement data.

        This detects saving account transactions and creates counter entries:
        - Detect "tất toán" (close/settle) transactions
        - Create counter entries for internal transfers
        - Append counter entries to Movement sheet

        Args:
            session_id: The session ID

        Returns:
            Dict with results summary
        """
        import re

        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            raise ValueError(f"Session {session_id} not found")

        writer = MovementDataWriter(working_file)

        # Read all current transactions
        transactions = writer.get_all_transactions()
        if not transactions:
            return {
                "session_id": session_id,
                "status": "no_transactions",
                "message": "No transactions found in Movement sheet",
                "counter_entries_created": 0,
            }

        # Patterns for detecting tất toán (settlement) transactions
        settlement_patterns = [
            r'tất\s*toán',
            r'tat\s*toan',
            r'close\s*(?:account|saving|term)',
            r'rút\s*(?:tiền|gốc|tiết kiệm)',
            r'rut\s*(?:tien|goc|tiet kiem)',
            r'withdraw',
            r'maturity',
            r'đáo\s*hạn',
            r'dao\s*han',
        ]
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in settlement_patterns]

        # Detect settlement transactions
        settlement_transactions = []
        for tx in transactions:
            # Skip if already a counter entry
            if tx.source and "Counter" in tx.source:
                continue

            # Check description for settlement patterns
            desc = tx.description.lower() if tx.description else ""
            is_settlement = any(p.search(desc) for p in compiled_patterns)

            if is_settlement:
                settlement_transactions.append(tx)

        if not settlement_transactions:
            return {
                "session_id": session_id,
                "status": "no_settlements",
                "message": "No settlement transactions found",
                "counter_entries_created": 0,
                "total_transactions_scanned": len(transactions),
            }

        # Create counter entries
        counter_entries = []
        for tx in settlement_transactions:
            # Counter entry has reversed debit/credit
            counter = MovementTransaction(
                source="Counter Entry",
                bank=tx.bank,
                account=tx.account,  # Same account for now
                date=tx.date,
                description=f"[Counter] {tx.description[:80] if tx.description else ''}",
                debit=tx.credit,  # Reverse: original credit -> counter debit
                credit=tx.debit,  # Reverse: original debit -> counter credit
                nature="Internal transfer" if tx.debit else "Internal transfer",
            )
            counter_entries.append(counter)

        # Append counter entries to Movement sheet
        if counter_entries:
            rows_added, total_rows = writer.append_transactions(counter_entries)

            # Update session stats
            if self.db_session:
                await self.db_session.execute(
                    update(CashReportSessionModel)
                    .where(CashReportSessionModel.session_id == session_id)
                    .values(
                        total_transactions=CashReportSessionModel.total_transactions + rows_added,
                    )
                )
                await self.db_session.commit()

            return {
                "session_id": session_id,
                "status": "success",
                "message": f"Created {rows_added} counter entries",
                "counter_entries_created": rows_added,
                "total_rows_in_movement": total_rows,
                "settlement_transactions_found": len(settlement_transactions),
                "total_transactions_scanned": len(transactions),
            }

        return {
            "session_id": session_id,
            "status": "no_counter_entries",
            "message": "No counter entries were created",
            "counter_entries_created": 0,
        }
