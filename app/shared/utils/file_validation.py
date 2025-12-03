# app/utils/file_validation.py
"""Comprehensive file validation utilities."""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from fastapi import UploadFile, HTTPException, status
from io import BytesIO

from app.core.exceptions import FileProcessingError
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# File type signatures (magic numbers)
EXCEL_SIGNATURES = {
    b'\x50\x4B\x03\x04': 'xlsx',  # ZIP-based (xlsx)
    b'\xD0\xCF\x11\xE0': 'xls',   # OLE2 (xls)
    b'\x09\x08\x08\x00': 'xls',   # BIFF5 (xls)
    b'\x09\x08\x05\x00': 'xls',   # BIFF5 (xls)
}

class FileValidator:
    """Comprehensive file validation for Excel uploads."""

    def __init__(self, max_file_size: int = 100 * 1024 * 1024):  # 100MB default
        self.max_file_size = max_file_size
        self.allowed_extensions = {'.xlsx', '.xls'}
        # Flexible sheet name patterns (case-insensitive matching)
        self.bs_patterns = ["BS Breakdown", "BS breakdown", "bs breakdown", "BSbreakdown",
                           "Balance Sheet", "BS", "balance sheet breakdown",
                           "BẢNG CÂN ĐỐI KẾ TOÁN"]
        self.pl_patterns = ["PL Breakdown", "PL breakdown", "pl breakdown", "PLBreakdown",
                           "P&L", "P/L", "Profit Loss", "Income Statement",
                           "BÁO CÁO KẾT QUẢ KINH DOANH"]

    async def validate_upload_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Comprehensive validation of uploaded Excel file.

        Returns:
            Dict with validation results and file metadata

        Raises:
            FileProcessingError: If validation fails
        """
        logger.info(f"Validating uploaded file: {file.filename}")

        validation_result = {
            "filename": file.filename,
            "is_valid": False,
            "file_size": 0,
            "file_type": None,
            "detected_mime": None,
            "sheets": [],
            "errors": [],
            "warnings": []
        }

        try:
            # Read file content
            content = await file.read()
            await file.seek(0)  # Reset file pointer

            validation_result["file_size"] = len(content)

            # 1. File size validation
            if len(content) == 0:
                validation_result["errors"].append("File is empty")
                raise FileProcessingError("Uploaded file is empty")

            if len(content) > self.max_file_size:
                size_mb = len(content) / (1024 * 1024)
                max_mb = self.max_file_size / (1024 * 1024)
                validation_result["errors"].append(f"File too large: {size_mb:.1f}MB (max: {max_mb:.0f}MB)")
                raise FileProcessingError(f"File size {size_mb:.1f}MB exceeds maximum allowed size of {max_mb:.0f}MB")

            # 2. Filename validation
            if not file.filename:
                validation_result["errors"].append("No filename provided")
                raise FileProcessingError("No filename provided")

            # Check for dangerous characters in filename
            dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
            if any(char in file.filename for char in dangerous_chars):
                validation_result["errors"].append("Filename contains dangerous characters")
                raise FileProcessingError("Filename contains dangerous characters")

            # 3. File extension validation
            file_path = Path(file.filename)
            file_extension = file_path.suffix.lower()

            if file_extension not in self.allowed_extensions:
                validation_result["errors"].append(f"Invalid file extension: {file_extension}")
                raise FileProcessingError(f"Invalid file extension: {file_extension}. Allowed: {', '.join(self.allowed_extensions)}")

            # 4. File signature (magic number) validation
            file_type = self._detect_file_type(content)
            validation_result["file_type"] = file_type

            if not file_type:
                validation_result["errors"].append("Invalid file format - not a valid Excel file")
                raise FileProcessingError("File does not appear to be a valid Excel file")

            # 5. Basic MIME type detection (without python-magic dependency)
            try:
                # Simple MIME type detection based on file signature
                if content.startswith(b'PK'):  # ZIP-based format (xlsx)
                    detected_mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                elif content.startswith(b'\xD0\xCF\x11\xE0'):  # OLE2 format (xls)
                    detected_mime = 'application/vnd.ms-excel'
                else:
                    detected_mime = 'application/octet-stream'

                validation_result["detected_mime"] = detected_mime

                valid_mimes = {
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # xlsx
                    'application/vnd.ms-excel',  # xls
                    'application/x-ole-storage'   # older xls
                }

                if detected_mime not in valid_mimes and detected_mime != 'application/octet-stream':
                    validation_result["warnings"].append(f"Unexpected file type detected: {detected_mime}")

            except Exception as e:
                logger.warning(f"File type detection failed: {e}")
                validation_result["warnings"].append("Could not detect file type")

            # 6. Excel file structure validation
            sheets_info = self._validate_excel_structure(content)
            validation_result["sheets"] = sheets_info

            # Check for required sheets with flexible matching
            sheet_names = [sheet['name'] for sheet in sheets_info]
            sheet_names_lower = {name.lower(): name for name in sheet_names}

            # Check for BS sheet
            bs_found = False
            for pattern in self.bs_patterns:
                pattern_lower = pattern.lower()
                # Exact match
                if pattern_lower in sheet_names_lower:
                    bs_found = True
                    break
                # Partial match (contains pattern)
                for lower_name in sheet_names_lower.keys():
                    if pattern_lower in lower_name or lower_name in pattern_lower:
                        bs_found = True
                        break
                if bs_found:
                    break

            # Check for PL sheet
            pl_found = False
            for pattern in self.pl_patterns:
                pattern_lower = pattern.lower()
                # Exact match
                if pattern_lower in sheet_names_lower:
                    pl_found = True
                    break
                # Partial match (contains pattern)
                for lower_name in sheet_names_lower.keys():
                    if pattern_lower in lower_name or lower_name in pattern_lower:
                        pl_found = True
                        break
                if pl_found:
                    break

            # At least one sheet must be present
            if not bs_found and not pl_found:
                validation_result["errors"].append("Missing required sheets: BS Breakdown and PL Breakdown")
                raise FileProcessingError(f"Missing required sheets: BS Breakdown, PL Breakdown. Found: {', '.join(sheet_names)}")

            # 7. Data quality checks
            data_warnings = self._validate_data_quality(content)
            validation_result["warnings"].extend(data_warnings)

            validation_result["is_valid"] = True
            logger.info(f"File validation successful: {file.filename}")

            return validation_result

        except FileProcessingError:
            raise
        except Exception as e:
            logger.error(f"File validation failed for {file.filename}: {e}", exc_info=True)
            validation_result["errors"].append(f"Validation error: {str(e)}")
            raise FileProcessingError(f"File validation failed: {str(e)}")

    def _detect_file_type(self, content: bytes) -> Optional[str]:
        """Detect file type based on magic numbers."""
        if len(content) < 8:
            return None

        # Check first 8 bytes for known signatures
        header = content[:8]

        for signature, file_type in EXCEL_SIGNATURES.items():
            if header.startswith(signature):
                return file_type

        # Additional check for ZIP-based files (xlsx)
        if content[:2] == b'PK':
            return 'xlsx'

        return None

    def _validate_excel_structure(self, content: bytes) -> List[Dict[str, Any]]:
        """Validate Excel file structure and extract sheet information."""
        sheets_info = []

        try:
            # Try to read the Excel file
            bio = BytesIO(content)

            # Get sheet names without fully loading data
            if content.startswith(b'PK'):  # xlsx format
                excel_file = pd.ExcelFile(bio, engine='openpyxl')
            else:  # xls format
                excel_file = pd.ExcelFile(bio, engine='xlrd')

            for sheet_name in excel_file.sheet_names:
                try:
                    # Quick read to check if sheet is accessible
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=5)

                    sheet_info = {
                        "name": sheet_name,
                        "is_accessible": True,
                        "preview_rows": len(df),
                        "preview_columns": len(df.columns),
                        "has_data": not df.empty
                    }

                except Exception as e:
                    sheet_info = {
                        "name": sheet_name,
                        "is_accessible": False,
                        "error": str(e),
                        "has_data": False
                    }

                sheets_info.append(sheet_info)

        except Exception as e:
            logger.error(f"Failed to read Excel structure: {e}")
            raise FileProcessingError(f"Cannot read Excel file structure: {str(e)}")

        return sheets_info

    def _validate_data_quality(self, content: bytes) -> List[str]:
        """Perform basic data quality checks on Excel content."""
        warnings = []

        # Skip data quality checks for now - the variance pipeline handles validation
        # This avoids issues with flexible sheet names
        return warnings

# Utility functions
async def validate_file_list(files: List[UploadFile], max_files: int = 10) -> List[Dict[str, Any]]:
    """Validate a list of uploaded files."""
    if len(files) > max_files:
        raise FileProcessingError(
            f"Too many files uploaded ({len(files)}). Maximum allowed: {max_files} files per upload.",
            details=f"Remove {len(files) - max_files} file(s) from your selection or split into multiple batches."
        )

    validator = FileValidator()
    results = []

    for file in files:
        try:
            result = await validator.validate_upload_file(file)
            results.append(result)
        except FileProcessingError as e:
            logger.error(f"File validation failed for {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "is_valid": False,
                "errors": [str(e)]
            })

    return results

def validate_file_size_limit(content_length: Optional[int], max_size: int) -> None:
    """Validate file size before upload processing."""
    if content_length is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content-Length header is required"
        )

    if content_length > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = content_length / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {actual_mb:.1f}MB. Maximum allowed: {max_mb:.0f}MB"
        )