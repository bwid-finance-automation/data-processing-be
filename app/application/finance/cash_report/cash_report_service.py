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
        import time as _time
        _classify_start = _time.monotonic()
        classified_transactions = await self._classify_transactions(
            all_transactions,
            progress_callback=progress_callback,
        )
        _classify_elapsed_ms = (_time.monotonic() - _classify_start) * 1000

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

    async def get_working_file_path(self, session_id: str, user_id: Optional[int] = None) -> Optional[Path]:
        """Get the working file path for download."""
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)
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

    def _read_saving_accounts(self, working_file: str) -> List[Dict[str, Any]]:
        """
        Read Saving Account sheet from working file.

        Returns:
            List of dicts with keys: account, bank_1, bank, entity, branch
        """
        import openpyxl
        wb = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
        saving_accounts = []
        try:
            if "Saving Account" not in wb.sheetnames:
                logger.warning("Saving Account sheet not found")
                return []

            ws = wb["Saving Account"]
            # Row 3 = headers, Row 4+ = data
            # Col A=Entity, B=Branch, C=Account Number, D=Type, E=Currency,
            # Col N(14)=Bank_1 (short code like TCB, BIDV), Col O(15)=Bank (full name)
            for row_data in ws.iter_rows(min_row=4, values_only=False):
                account = row_data[2].value if len(row_data) > 2 else None
                if not account:
                    continue
                saving_accounts.append({
                    "entity": str(row_data[0].value or "").strip(),
                    "branch": str(row_data[1].value or "").strip(),
                    "account": str(account).strip(),
                    "bank_1": str(row_data[13].value or "").strip() if len(row_data) > 13 else "",
                    "bank": str(row_data[14].value or "").strip() if len(row_data) > 14 else "",
                })
        finally:
            wb.close()

        logger.info(f"Loaded {len(saving_accounts)} saving accounts")
        return saving_accounts

    def _find_saving_account(
        self,
        description: str,
        bank: str,
        entity: str,
        saving_accounts: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Find saving account number with 2 separate logics:

        Logic 1: Description contains account number (e.g. 'TAT TOAN TIEN GUI SO 14501110378000')
                 → Extract and use directly as saving account
        Logic 2: Description does NOT contain account number
                 → Lookup in Saving Account sheet by bank + entity match

        Returns:
            Saving account number or None
        """
        import re

        # Logic 1: Extract account number from description
        # Try patterns: "SO 14501110378000", "STK ...", "TK ..."
        acc_match = re.search(r'(?:SO|STK|TK|S/N|SN)\s*(\d{6,20})', description, re.IGNORECASE)
        if acc_match:
            logger.info(f"Settlement Logic 1: found account {acc_match.group(1)} in description")
            return acc_match.group(1)

        # Also try any long number sequence in description
        all_numbers = re.findall(r'\d{8,20}', description)
        if all_numbers:
            logger.info(f"Settlement Logic 1: found number {all_numbers[0]} in description")
            return all_numbers[0]

        # Logic 2: No account in description → lookup Saving Account sheet
        bank_upper = bank.upper().strip() if bank else ""
        entity_upper = entity.upper().strip() if entity else ""

        # Match by bank AND entity
        matches = [
            sa for sa in saving_accounts
            if sa["bank_1"].upper() == bank_upper
            and sa["entity"].upper() == entity_upper
        ]
        if len(matches) == 1:
            logger.info(f"Settlement Logic 2: found saving account {matches[0]['account']} by bank={bank} + entity={entity}")
            return matches[0]["account"]
        elif len(matches) > 1:
            logger.warning(f"Settlement Logic 2: multiple saving accounts found for bank={bank}, entity={entity}: {[m['account'] for m in matches]}")
            return matches[0]["account"]

        # Fallback: match by bank only
        matches_bank = [sa for sa in saving_accounts if sa["bank_1"].upper() == bank_upper]
        if len(matches_bank) == 1:
            logger.info(f"Settlement Logic 2 fallback: found saving account {matches_bank[0]['account']} by bank={bank}")
            return matches_bank[0]["account"]

        logger.warning(f"Settlement: no saving account found for description='{description}', bank={bank}, entity={entity}")
        return None

    async def run_settlement_automation(self, session_id: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run settlement (tất toán) automation on Movement data.

        Logic:
        1. Detect settlement transactions by keyword patterns
        2. Lookup saving account from Saving Account sheet (by account number in description + bank match)
        3. Create counter entry with saving account, reversed debit/credit, nature = Internal transfer out

        Args:
            session_id: The session ID
            user_id: Owner user ID for access control

        Returns:
            Dict with results summary
        """
        if user_id is not None:
            await self._verify_session_owner(session_id, user_id)
        import re

        working_file = self.template_manager.get_working_file_path(session_id)
        if not working_file:
            raise ValueError(f"Session {session_id} not found")

        writer = MovementDataWriter(working_file)

        # Read all current transactions + entity (col J) for settlement lookup
        transactions = writer.get_all_transactions()
        if not transactions:
            return {
                "session_id": session_id,
                "status": "no_transactions",
                "message": "No transactions found in Movement sheet",
                "counter_entries_created": 0,
            }

        # Read entity (col J = col 10) from Movement for each transaction
        import openpyxl
        wb_tmp = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
        ws_tmp = wb_tmp["Movement"]
        tx_entities = {}
        for row_idx, row_data in enumerate(ws_tmp.iter_rows(min_row=4, max_col=10), start=4):
            acc = row_data[2].value if len(row_data) > 2 else None
            if acc:
                entity_val = str(row_data[9].value or "").strip() if len(row_data) > 9 else ""
                tx_entities[row_idx] = entity_val
        wb_tmp.close()

        # Read Saving Account sheet for lookup
        saving_accounts = self._read_saving_accounts(working_file)

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

        # Detect settlement transactions (only Cash In / debit > 0)
        # Store as (tx, row_index) tuples to access entity later
        settlement_transactions = []
        for idx, tx in enumerate(transactions):
            row_idx = idx + 4  # transactions start at row 4

            # Skip if source is already a settlement counter
            if tx.source and tx.source.strip() == "Automation" and tx.nature and "transfer out" in tx.nature.lower():
                continue

            # Only detect on Cash In transactions (debit > 0)
            if not tx.debit or tx.debit <= 0:
                continue

            # Check description for settlement patterns
            desc = tx.description if tx.description else ""
            is_settlement = any(p.search(desc) for p in compiled_patterns)

            if is_settlement:
                settlement_transactions.append((tx, row_idx))

        if not settlement_transactions:
            return {
                "session_id": session_id,
                "status": "no_settlements",
                "message": "No settlement transactions found",
                "counter_entries_created": 0,
                "total_transactions_scanned": len(transactions),
            }

        # Check if counter entries already exist (prevent duplicates)
        existing_descs = set()
        for tx in transactions:
            if tx.nature and "transfer out" in tx.nature.lower():
                existing_descs.add(tx.description.strip() if tx.description else "")

        # Create counter entries
        counter_entries = []
        original_row_indices = []  # Track original rows for highlighting
        skipped_no_account = []
        skipped_duplicate = []
        for tx, row_idx in settlement_transactions:
            # Skip if counter entry already exists for this description
            desc = tx.description.strip() if tx.description else ""
            if desc in existing_descs:
                skipped_duplicate.append(desc)
                continue

            # Get entity from Movement col J for this row
            entity = tx_entities.get(row_idx, "")

            # Lookup saving account (2 logics: from description or from Saving Account sheet)
            saving_acc = self._find_saving_account(
                description=tx.description or "",
                bank=tx.bank or "",
                entity=entity,
                saving_accounts=saving_accounts,
            )

            if not saving_acc:
                skipped_no_account.append(tx.description)
                logger.warning(f"Settlement: no saving account found for '{tx.description}', bank={tx.bank}, entity={entity}")
                continue

            # Counter entry: same description, saving account, reversed amounts
            counter = MovementTransaction(
                source="Automation",
                bank=tx.bank,
                account=saving_acc,
                date=tx.date,
                description=tx.description or "",
                debit=Decimal("0"),
                credit=tx.debit,  # Original debit (cash in) becomes credit (cash out) on saving account
                nature="Internal transfer out",
            )
            counter_entries.append(counter)
            original_row_indices.append(row_idx)

        if not counter_entries:
            msg = "No counter entries created"
            if skipped_no_account:
                msg += f". {len(skipped_no_account)} skipped (no matching saving account)"
            if skipped_duplicate:
                msg += f". {len(skipped_duplicate)} skipped (already exists)"
            return {
                "session_id": session_id,
                "status": "no_counter_entries",
                "message": msg,
                "counter_entries_created": 0,
                "skipped_no_account": len(skipped_no_account),
                "skipped_duplicate": len(skipped_duplicate),
                "total_transactions_scanned": len(transactions),
            }

        # Append counter entries to Movement sheet
        rows_added, total_rows = writer.append_transactions(counter_entries)

        # Highlight both original settlement rows and counter entry rows
        # Counter entries start at (total_rows - rows_added + FIRST_DATA_ROW)
        counter_start_row = total_rows - rows_added + 4  # 4 = FIRST_DATA_ROW
        counter_row_indices = list(range(counter_start_row, counter_start_row + rows_added))
        all_highlight_rows = original_row_indices + counter_row_indices
        try:
            writer.highlight_settlement_rows(all_highlight_rows)
        except Exception as e:
            logger.warning(f"Failed to highlight settlement rows: {e}")

        # Remove rows from Saving Account sheet where CLOSING BALANCE (VND) = 0
        from .openpyxl_handler import get_openpyxl_handler
        handler = get_openpyxl_handler()
        saving_rows_removed = 0
        try:
            saving_rows_removed = handler.remove_zero_closing_balance_saving_rows(Path(working_file))
        except Exception as e:
            logger.warning(f"Failed to remove zero-closing-balance saving rows: {e}")

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
            "skipped_no_account": len(skipped_no_account),
            "skipped_duplicate": len(skipped_duplicate),
            "total_transactions_scanned": len(transactions),
            "saving_rows_removed": saving_rows_removed,
        }
