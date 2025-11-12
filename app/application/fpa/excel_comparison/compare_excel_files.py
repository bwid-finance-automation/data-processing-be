# app/application/fpa/use_cases/compare_excel_files.py
"""
FPA Excel Comparison Use Case
Orchestrates the workflow for comparing Excel summary files
"""
from fastapi import UploadFile
from pathlib import Path
from datetime import datetime
import shutil
import os
from typing import Dict, Any

from app.infrastructure.persistence.excel.summary_comparator import SummaryComparator
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class CompareExcelFilesUseCase:
    """Use case for comparing Excel summary files"""

    def __init__(self):
        self.upload_dir = Path("uploads")
        self.output_dir = Path("outputs")
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.comparator = SummaryComparator()

    async def execute(
        self,
        old_file: UploadFile,
        new_file: UploadFile
    ) -> Dict[str, Any]:
        """
        Execute the Excel comparison workflow.

        Args:
            old_file: Previous month Excel file
            new_file: Current month Excel file

        Returns:
            Dict with comparison results and file paths
        """
        old_path = None
        new_path = None

        try:
            # Save uploaded files with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            old_path = self.upload_dir / f"old_{timestamp}_{old_file.filename}"
            new_path = self.upload_dir / f"new_{timestamp}_{new_file.filename}"

            logger.info(f"Saving uploaded files: {old_file.filename}, {new_file.filename}")

            with open(old_path, "wb") as f:
                shutil.copyfileobj(old_file.file, f)
            with open(new_path, "wb") as f:
                shutil.copyfileobj(new_file.file, f)

            # Generate output file with new_rows and update_rows sheets
            logger.info("Generating comparison output file...")
            output_path = self.comparator.generate_excel_output_files(
                str(old_path),
                str(new_path),
                str(self.output_dir / f"comparison_{timestamp}.xlsx")
            )

            # Create highlighted copy (does NOT modify original uploaded files)
            logger.info("Generating highlighted file...")
            highlighted_path = self.output_dir / f"highlighted_{timestamp}_{new_file.filename}"
            highlighted_output = self.comparator.apply_highlighting_to_summary_with_90_day_rule(
                str(old_path),
                str(new_path),
                str(highlighted_path)
            )

            # Get comparison statistics
            logger.info("Calculating comparison statistics...")
            comparison_results = self.comparator.compare_summary_files_by_document_item_key(
                str(old_path),
                str(new_path)
            )

            # Build response
            result = {
                "status": "success",
                "timestamp": timestamp,
                "old_filename": old_file.filename,
                "new_filename": new_file.filename,
                "output_file": os.path.basename(output_path),
                "highlighted_file": os.path.basename(highlighted_path),
                "message": "Comparison and highlighting completed successfully",
                "statistics": {
                    "new_rows": len(comparison_results['new_rows_indices']),
                    "updated_rows": len(comparison_results['update_rows_indices']),
                    "unchanged_rows": len(comparison_results['unchanged_rows_indices'])
                }
            }

            # Cleanup uploaded files (keep outputs)
            logger.info("Cleaning up temporary files...")
            old_path.unlink()
            new_path.unlink()

            logger.info(f"Comparison completed successfully: {result['statistics']}")
            return result

        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            # Cleanup files on error
            if old_path and old_path.exists():
                old_path.unlink()
            if new_path and new_path.exists():
                new_path.unlink()
            raise

    def get_output_file_path(self, filename: str) -> Path:
        """Get path to an output file"""
        return self.output_dir / filename

    def list_output_files(self) -> list:
        """List all available output files"""
        files = []
        for file_path in self.output_dir.glob("*.xlsx"):
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
            })
        return files

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
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > 86400:  # 24 hours
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path.name}")
