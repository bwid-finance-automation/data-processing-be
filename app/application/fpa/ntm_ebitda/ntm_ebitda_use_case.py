"""
NTM EBITDA Variance Analysis Use Case
Orchestrates the workflow for comparing NTM EBITDA data between two periods.
"""
from fastapi import UploadFile
from pathlib import Path
from datetime import datetime
import shutil
import os
from typing import Dict, Any, Optional, List

from app.domain.fpa.ntm_ebitda.services.ntm_processor import NTMProcessor
from app.domain.fpa.ntm_ebitda.services.ntm_variance_calculator import NTMVarianceCalculator
from app.domain.fpa.ntm_ebitda.services.ntm_ai_analyzer import NTMAIAnalyzer
from app.domain.fpa.ntm_ebitda.models.ntm_ebitda_models import AnalysisConfig
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class NTMEBITDAUseCase:
    """
    Use case for NTM EBITDA variance analysis between two periods.
    Orchestrates file processing, variance calculation, and AI commentary.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the use case.

        Args:
            config: Optional analysis configuration
        """
        self.config = config or AnalysisConfig()

        self.upload_dir = Path("uploads/ntm_ebitda")
        self.output_dir = Path("outputs/ntm_ebitda")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processor = NTMProcessor(self.config)
        self.calculator = NTMVarianceCalculator(self.config)
        self.ai_analyzer = None  # Lazy initialization

    def _get_ai_analyzer(self) -> NTMAIAnalyzer:
        """Lazy initialization of AI analyzer."""
        if self.ai_analyzer is None:
            self.ai_analyzer = NTMAIAnalyzer(self.config)
        return self.ai_analyzer

    async def execute(
        self,
        file: UploadFile,
        prev_sheet: Optional[str] = None,
        curr_sheet: Optional[str] = None,
        previous_label: Optional[str] = None,
        current_label: Optional[str] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Execute the NTM EBITDA variance analysis workflow.

        Args:
            file: Excel file containing leasing model data
            prev_sheet: Sheet name for previous period (auto-detected if None)
            curr_sheet: Sheet name for current period (auto-detected if None)
            previous_label: Optional label for previous period
            current_label: Optional label for current period
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with analysis results, statistics, and file paths
        """
        file_path = None

        try:
            # Save uploaded file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.upload_dir / f"ntm_{timestamp}_{file.filename}"

            logger.info(f"Saving uploaded file: {file.filename}")

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            if progress_callback:
                progress_callback(10, "File uploaded, detecting sheets...")

            # Auto-detect sheets if not provided
            if not prev_sheet or not curr_sheet:
                detected_prev, detected_curr = self.processor.detect_period_sheets(str(file_path))
                prev_sheet = prev_sheet or detected_prev
                curr_sheet = curr_sheet or detected_curr

                if not prev_sheet or not curr_sheet:
                    raise ValueError(
                        "Could not auto-detect period sheets. Please specify prev_sheet and curr_sheet."
                    )

                logger.info(f"Auto-detected sheets: {prev_sheet} (prev), {curr_sheet} (curr)")

            if progress_callback:
                progress_callback(20, f"Processing sheets: {prev_sheet} -> {curr_sheet}")

            # Extract period labels
            prev_label = previous_label or self.processor.extract_period_label(prev_sheet)
            curr_label = current_label or self.processor.extract_period_label(curr_sheet)

            # Process file
            processed_data = self.processor.process_file(
                str(file_path),
                prev_sheet,
                curr_sheet
            )

            if progress_callback:
                progress_callback(40, "Calculating variances...")

            # Calculate variances
            summary = self.calculator.calculate_variance(
                previous=processed_data["previous"],
                current=processed_data["current"]
            )

            # Set period labels
            summary.previous_period = prev_label
            summary.current_period = curr_label

            # Generate AI commentary
            ai_analyzer = self._get_ai_analyzer()

            if progress_callback:
                progress_callback(50, "Generating AI commentary...")

            try:
                summary.results = ai_analyzer.generate_commentary(
                    summary.results,
                    progress_callback
                )
            except Exception as e:
                logger.warning(f"AI commentary failed: {e}")

            if progress_callback:
                progress_callback(70, "Generating output Excel...")

            # Generate output Excel
            excel_filename = f"ntm_ebitda_variance_{timestamp}.xlsx"
            excel_path = self.output_dir / excel_filename

            self.calculator.generate_output_excel(
                summary=summary,
                output_path=str(excel_path),
                previous_period=prev_label,
                current_period=curr_label
            )

            # Generate statistics
            statistics = self.calculator.generate_summary_statistics(summary)

            if progress_callback:
                progress_callback(85, "Running portfolio analysis...")

            # Generate portfolio analysis
            ai_result = None
            try:
                ai_result = ai_analyzer.analyze_portfolio(summary, progress_callback)
            except Exception as e:
                logger.warning(f"Portfolio analysis failed: {e}")
                ai_result = {"status": "error", "error": str(e)}

            # Build response
            result = {
                "status": "success",
                "timestamp": timestamp,
                "filename": file.filename,
                "previous_period": prev_label,
                "current_period": curr_label,
                "prev_sheet": prev_sheet,
                "curr_sheet": curr_sheet,
                "output_file": excel_filename,
                "message": "NTM EBITDA variance analysis completed successfully",
                "statistics": statistics,
                "results": [r.to_dict() for r in summary.results],
            }

            if ai_result:
                result["ai_analysis"] = {
                    "status": ai_result.get("status", "success"),
                    "model": ai_result.get("model", "AI"),
                    "analysis": ai_result.get("analysis", ""),
                }

            # Cleanup uploaded file
            logger.info("Cleaning up temporary files...")
            file_path.unlink()

            if progress_callback:
                progress_callback(100, "Analysis complete!")

            logger.info(f"NTM EBITDA analysis completed: {statistics['total_projects']} projects analyzed")
            return result

        except Exception as e:
            logger.error(f"NTM EBITDA analysis failed: {str(e)}")
            # Cleanup file on error
            if file_path and file_path.exists():
                file_path.unlink()
            raise

    def get_available_sheets(self, file: UploadFile) -> List[str]:
        """
        Get available sheet names from an uploaded file.

        Args:
            file: Excel file to analyze

        Returns:
            List of sheet names
        """
        file_path = None
        try:
            # Save temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.upload_dir / f"temp_{timestamp}_{file.filename}"

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            sheets = self.processor.detect_sheets(str(file_path))

            # Cleanup
            file_path.unlink()

            return sheets

        except Exception as e:
            logger.error(f"Failed to detect sheets: {e}")
            if file_path and file_path.exists():
                file_path.unlink()
            raise

    def get_output_file_path(self, filename: str) -> Path:
        """Get path to an output file"""
        return self.output_dir / filename

    def list_output_files(self) -> list:
        """List all available NTM EBITDA output files"""
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
