"""
GLA Variance Analysis Use Case
Orchestrates the workflow for comparing GLA data between two periods.
"""
from fastapi import UploadFile
from pathlib import Path
from datetime import datetime
import shutil
import os
from typing import Dict, Any, Optional

from app.domain.fpa.gla_variance.services.gla_processor import GLAProcessor
from app.domain.fpa.gla_variance.services.gla_variance_calculator import GLAVarianceCalculator
from app.domain.fpa.gla_variance.services.gla_ai_analyzer import GLAAIAnalyzer
from app.domain.fpa.gla_variance.services.gla_pdf_generator import GLAPDFGenerator
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class GLAVarianceUseCase:
    """
    Use case for GLA variance analysis between previous and current periods.
    Supports both basic Python analysis and AI-powered analysis with explanations.
    """

    def __init__(self):
        self.upload_dir = Path("uploads/gla")
        self.output_dir = Path("outputs/gla")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processor = GLAProcessor()
        self.calculator = GLAVarianceCalculator()
        self.ai_analyzer = None  # Lazy initialization
        self.pdf_generator = GLAPDFGenerator()

    def _get_ai_analyzer(self) -> GLAAIAnalyzer:
        """Lazy initialization of AI analyzer."""
        if self.ai_analyzer is None:
            self.ai_analyzer = GLAAIAnalyzer()
        return self.ai_analyzer

    def _process_file(
        self,
        file_path: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Process GLA file using standard 4-sheet format.

        Expected sheets:
        - Handover GLA - Previous
        - Handover GLA - Current
        - Committed GLA - Previous
        - Committed GLA - Current

        Args:
            file_path: Path to the Excel file
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with structure: {
                'handover': {'previous': {...}, 'current': {...}},
                'committed': {'previous': {...}, 'current': {...}}
            }
        """
        logger.info("Processing as standard 4-sheet format...")
        if progress_callback:
            progress_callback(20, "Processing GLA data...")

        return self.processor.process_single_file_with_periods(file_path)

    async def execute(
        self,
        file: UploadFile,
        previous_label: Optional[str] = None,
        current_label: Optional[str] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Execute the GLA variance analysis workflow with a single file.
        Always uses AI for generating variance explanations and notes.

        Args:
            file: Excel file containing 4 sheets (2 previous + 2 current periods)
            previous_label: Optional label for previous period (e.g., "Oct 2025")
            current_label: Optional label for current period (e.g., "Nov 2025")
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with analysis results, statistics, and file paths
        """
        file_path = None

        try:
            # Save uploaded file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.upload_dir / f"gla_{timestamp}_{file.filename}"

            logger.info(f"Saving uploaded file: {file.filename}")

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Process file using standard 4-sheet format
            processed_data = self._process_file(
                str(file_path),
                progress_callback
            )

            # Initialize AI analyzer for notes generation
            ai_analyzer = self._get_ai_analyzer()

            # Extract data for variance calculation
            handover_previous = processed_data['handover'].get('previous', {})
            handover_current = processed_data['handover'].get('current', {})
            committed_previous = processed_data['committed'].get('previous', {})
            committed_current = processed_data['committed'].get('current', {})

            # Extract Accounting Net Rent data (optional Amor sheets)
            amor_previous = processed_data.get('amor', {}).get('previous', {})
            amor_current = processed_data.get('amor', {}).get('current', {})

            logger.info(f"Handover previous: {len(handover_previous)} projects")
            logger.info(f"Handover current: {len(handover_current)} projects")
            logger.info(f"Committed previous: {len(committed_previous)} projects")
            logger.info(f"Committed current: {len(committed_current)} projects")
            logger.info(f"Amor previous: {len(amor_previous)} projects")
            logger.info(f"Amor current: {len(amor_current)} projects")

            # Calculate variances
            logger.info("Calculating variances...")
            summary = self.calculator.calculate_variance(
                handover_previous=handover_previous,
                handover_current=handover_current,
                committed_previous=committed_previous,
                committed_current=committed_current,
                amor_previous=amor_previous,
                amor_current=amor_current
            )

            # Use provided labels or extract from filename
            prev_label = previous_label or self._extract_period_label(file.filename, "previous", "Previous")
            curr_label = current_label or self._extract_period_label(file.filename, "current", "Current")

            # Set period labels on summary
            summary.previous_period = prev_label
            summary.current_period = curr_label

            # Generate AI notes (always enabled - before Excel so notes appear in Excel)
            ai_result = None
            logger.info("Running AI analysis...")
            if progress_callback:
                progress_callback(40, "Generating tenant-based explanations...")

            try:
                # Generate project notes based on tenant changes
                logger.info("Generating project notes from tenant data...")
                summary.results = ai_analyzer.generate_project_notes(
                    summary.results,
                    progress_callback
                )

                if progress_callback:
                    progress_callback(60, "Running AI analysis...")

                # Generate overall analysis
                ai_result = ai_analyzer.analyze_variance(summary, progress_callback)

            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")
                ai_result = {"status": "error", "error": str(e)}

            # Collect AI usage
            ai_usage = ai_analyzer.get_and_reset_usage()

            # Generate output Excel file (now includes notes)
            excel_filename = f"gla_variance_{timestamp}.xlsx"
            excel_path = self.output_dir / excel_filename

            self.calculator.generate_output_excel(
                summary=summary,
                output_path=str(excel_path),
                previous_period=prev_label,
                current_period=curr_label
            )

            # Generate statistics
            statistics = self.calculator.generate_summary_statistics(summary)

            # Build response
            result = {
                "status": "success",
                "timestamp": timestamp,
                "filename": file.filename,
                "previous_period": prev_label,
                "current_period": curr_label,
                "output_file": excel_filename,
                "message": "GLA variance analysis completed successfully",
                "statistics": statistics,
                "results": [r.to_dict() for r in summary.results],
                "ai_usage": ai_usage,
            }

            # Generate PDF with AI analysis
            if ai_result:
                try:
                    if progress_callback:
                        progress_callback(80, "Generating PDF report...")

                    pdf_filename = f"gla_analysis_{timestamp}.pdf"
                    pdf_path = self.output_dir / pdf_filename

                    self.pdf_generator.generate_pdf(
                        summary=summary,
                        statistics=statistics,
                        ai_analysis=ai_result.get('analysis', ''),
                        output_path=str(pdf_path),
                        previous_period=prev_label,
                        current_period=curr_label
                    )

                    result["pdf_file"] = pdf_filename
                    result["ai_analysis"] = {
                        "status": ai_result.get("status", "success"),
                        "model": ai_result.get('model', 'AI')
                    }
                    result["message"] = "GLA variance analysis with AI insights completed successfully"

                except Exception as e:
                    logger.warning(f"PDF generation failed: {e}")
                    result["ai_analysis"] = {
                        "status": "error",
                        "error": str(e)
                    }

            # Cleanup uploaded file (keep outputs)
            logger.info("Cleaning up temporary files...")
            file_path.unlink()

            logger.info(f"GLA variance analysis completed: {statistics['total_projects']} projects analyzed")
            return result

        except Exception as e:
            logger.error(f"GLA variance analysis failed: {str(e)}")
            # Cleanup file on error
            if file_path and file_path.exists():
                file_path.unlink()
            raise

    def _extract_period_label(self, filename: str, period_type: str, default: str) -> str:
        """
        Try to extract period label from filename based on sheet naming.
        E.g., filename might contain T10/T11 references.
        """
        import re

        # Try to find T## patterns for previous (T10) and current (T11)
        if period_type == "previous":
            t_pattern = r'T10'
            match = re.search(t_pattern, filename, re.IGNORECASE)
            if match:
                return "T10"
        elif period_type == "current":
            t_pattern = r'T11'
            match = re.search(t_pattern, filename, re.IGNORECASE)
            if match:
                return "T11"

        # Try to find month patterns
        month_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[_\s-]?(\d{4})?'
        match = re.search(month_pattern, filename, re.IGNORECASE)
        if match:
            month = match.group(1)
            year = match.group(2) or ""
            return f"{month} {year}".strip()

        return default

    def get_output_file_path(self, filename: str) -> Path:
        """Get path to an output file"""
        return self.output_dir / filename

    def list_output_files(self) -> list:
        """List all available GLA output files"""
        files = []
        for file_path in self.output_dir.glob("*.*"):
            if file_path.suffix in ['.xlsx', '.pdf']:
                files.append({
                    "filename": file_path.name,
                    "type": "excel" if file_path.suffix == '.xlsx' else "pdf",
                    "size": file_path.stat().st_size,
                    "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
                })
        return sorted(files, key=lambda x: x['created'], reverse=True)

    def delete_output_file(self, filename: str):
        """Delete an output file"""
        file_path = self.output_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        file_path.unlink()
        logger.info(f"Deleted output file: {filename}")

    def cleanup_old_files(self):
        """Remove files older than 24 hours"""
        import time
        current_time = time.time()

        for directory in [self.upload_dir, self.output_dir]:
            if not directory.exists():
                continue
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > 86400:  # 24 hours
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path.name}")
