# app/services/file_processing_service.py
"""
Enhanced file processing service with dependency injection and interface implementation.

This service handles all file-related operations including validation, processing,
and storage with comprehensive error handling and security validation.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import UploadFile

from .interfaces import FileProcessingServiceBase
from ..models.analysis import FileInfo
from ..core.exceptions import FileProcessingError, ValidationError
from ..utils.file_validation import FileValidator
from ..utils.data_recovery import DataRecoveryEngine
from ..core.unified_config import get_unified_config

class FileProcessingService(FileProcessingServiceBase):
    """
    Enhanced file processing service with comprehensive validation and recovery.

    Features:
    - Security validation with magic number detection
    - Automatic data recovery for malformed files
    - Structured error handling with user-friendly messages
    - Configuration-driven processing parameters
    """

    def __init__(
        self,
        file_validator: Optional[FileValidator] = None,
        data_recovery: Optional[DataRecoveryEngine] = None
    ):
        super().__init__()
        self.config = get_unified_config()
        self.file_validator = file_validator or FileValidator()
        self.data_recovery = data_recovery or DataRecoveryEngine()

        # Create upload directory if it doesn't exist
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)

        self.log_operation("FileProcessingService initialized")

    async def validate_file(self, file: UploadFile) -> FileInfo:
        """
        Validate uploaded file with comprehensive security and structure checks.

        Args:
            file: The uploaded file to validate

        Returns:
            FileInfo with validation results and metadata

        Raises:
            FileProcessingError: If file validation fails
            ValidationError: If file structure is invalid
        """
        try:
            self.log_operation("validate_file_start", filename=file.filename, size=file.size)

            # Read file content for validation
            content = await file.read()
            await file.seek(0)  # Reset file pointer

            # Basic validation
            if not file.filename:
                raise FileProcessingError(
                    "File name is required",
                    user_message="Please provide a valid file name",
                    error_code="MISSING_FILENAME"
                )

            # Size validation
            if len(content) > self.config.file_processing.max_file_size:
                max_size_mb = self.config.file_processing.max_file_size / (1024 * 1024)
                raise FileProcessingError(
                    f"File size {len(content)} exceeds maximum {self.config.file_processing.max_file_size}",
                    user_message=f"File size exceeds the maximum allowed size of {max_size_mb:.0f}MB",
                    error_code="FILE_TOO_LARGE",
                    suggestions=[
                        "Reduce the file size by removing unnecessary data",
                        "Split the data into multiple smaller files",
                        "Contact support if you need to process larger files"
                    ]
                )

            # File extension validation
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in self.config.file_processing.allowed_file_extensions:
                allowed = ", ".join(self.config.file_processing.allowed_file_extensions)
                raise FileProcessingError(
                    f"File extension {file_extension} not allowed",
                    user_message=f"Only {allowed} files are supported",
                    error_code="INVALID_FILE_TYPE",
                    suggestions=[
                        f"Convert your file to one of the supported formats: {allowed}",
                        "Ensure the file has the correct extension"
                    ]
                )

            # Security validation
            validation_result = self.file_validator.validate_file_content(content, file.filename)
            if not validation_result["is_valid"]:
                raise FileProcessingError(
                    f"File validation failed: {validation_result['error']}",
                    user_message="The file appears to be corrupted or in an unsupported format",
                    error_code="FILE_VALIDATION_FAILED",
                    suggestions=[
                        "Try re-saving the file in Excel",
                        "Check if the file opens correctly in Excel",
                        "Ensure the file is not password-protected"
                    ]
                )

            # Create file info
            file_info = FileInfo(
                filename=file.filename,
                size=len(content),
                content_type=file.content_type,
                extension=file_extension,
                validation_status="valid",
                security_scan_passed=True,
                estimated_sheets=validation_result.get("estimated_sheets", 0),
                metadata=validation_result.get("metadata", {})
            )

            self.log_operation(
                "validate_file_success",
                filename=file.filename,
                size=len(content),
                extension=file_extension
            )

            return file_info

        except (FileProcessingError, ValidationError):
            raise
        except Exception as e:
            self.log_error("validate_file", e, filename=file.filename)
            raise FileProcessingError(
                f"Unexpected error during file validation: {str(e)}",
                user_message="An unexpected error occurred while validating the file",
                error_code="VALIDATION_ERROR",
                suggestions=[
                    "Try uploading the file again",
                    "Check if the file is corrupted",
                    "Contact support if the problem persists"
                ]
            ) from e

    async def process_excel_file(self, file: UploadFile) -> Dict[str, pd.DataFrame]:
        """
        Process Excel file with automatic data recovery and validation.

        Args:
            file: The Excel file to process

        Returns:
            Dictionary mapping sheet names to DataFrames

        Raises:
            FileProcessingError: If file processing fails
        """
        try:
            self.log_operation("process_excel_file_start", filename=file.filename)

            # Save file temporarily
            temp_path = await self.save_uploaded_file(file, "temp")

            try:
                # Try to read Excel file
                excel_data = pd.read_excel(temp_path, sheet_name=None, engine='openpyxl')

                # Validate required sheets
                missing_sheets = []
                for required_sheet in self.config.file_processing.required_sheets:
                    if required_sheet not in excel_data:
                        missing_sheets.append(required_sheet)

                if missing_sheets:
                    # Try data recovery to find similar sheet names
                    recovered_data = self.data_recovery.recover_excel_structure(excel_data)
                    if recovered_data.get("recovered_sheets"):
                        excel_data = recovered_data["data"]
                        self.log_operation(
                            "data_recovery_applied",
                            filename=file.filename,
                            recovered_sheets=list(recovered_data["recovered_sheets"].keys())
                        )
                    else:
                        sheets_found = list(excel_data.keys())
                        raise FileProcessingError(
                            f"Required sheets not found: {missing_sheets}",
                            user_message=f"The Excel file is missing required sheets: {', '.join(missing_sheets)}",
                            error_code="MISSING_REQUIRED_SHEETS",
                            suggestions=[
                                f"Ensure your Excel file contains sheets named: {', '.join(self.config.file_processing.required_sheets)}",
                                f"Found sheets: {', '.join(sheets_found)}",
                                "Check the sheet names in your Excel file",
                                "Rename sheets to match the required names"
                            ]
                        )

                # Apply data recovery to each sheet
                processed_data = {}
                for sheet_name, df in excel_data.items():
                    try:
                        processed_df = self.data_recovery.recover_dataframe(df, sheet_name)
                        processed_data[sheet_name] = processed_df

                        self.log_operation(
                            "sheet_processed",
                            filename=file.filename,
                            sheet_name=sheet_name,
                            rows=len(processed_df),
                            columns=len(processed_df.columns)
                        )

                    except Exception as e:
                        self.log_error("process_sheet", e, sheet_name=sheet_name, filename=file.filename)
                        # Continue processing other sheets
                        processed_data[sheet_name] = df

                self.log_operation(
                    "process_excel_file_success",
                    filename=file.filename,
                    sheets_processed=len(processed_data),
                    total_rows=sum(len(df) for df in processed_data.values())
                )

                return processed_data

            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

        except FileProcessingError:
            raise
        except Exception as e:
            self.log_error("process_excel_file", e, filename=file.filename)
            raise FileProcessingError(
                f"Failed to process Excel file: {str(e)}",
                user_message="Unable to process the Excel file. Please check the file format and try again.",
                error_code="EXCEL_PROCESSING_ERROR",
                suggestions=[
                    "Ensure the file is a valid Excel file (.xlsx or .xls)",
                    "Try opening and re-saving the file in Excel",
                    "Check if the file is password-protected",
                    "Remove any special formatting or macros from the file"
                ]
            ) from e

    async def save_uploaded_file(self, file: UploadFile, session_id: str) -> Path:
        """
        Save uploaded file to session directory with proper naming and security.

        Args:
            file: The file to save
            session_id: Session identifier for organization

        Returns:
            Path to the saved file

        Raises:
            FileProcessingError: If file saving fails
        """
        try:
            self.log_operation("save_uploaded_file_start", filename=file.filename, session_id=session_id)

            # Create session directory
            session_dir = self.upload_dir / session_id
            session_dir.mkdir(exist_ok=True)

            # Create safe filename
            safe_filename = self._create_safe_filename(file.filename)
            file_path = session_dir / safe_filename

            # Save file content
            content = await file.read()
            await file.seek(0)  # Reset file pointer

            with open(file_path, "wb") as f:
                f.write(content)

            self.log_operation(
                "save_uploaded_file_success",
                filename=file.filename,
                session_id=session_id,
                saved_path=str(file_path),
                size=len(content)
            )

            return file_path

        except Exception as e:
            self.log_error("save_uploaded_file", e, filename=file.filename, session_id=session_id)
            raise FileProcessingError(
                f"Failed to save uploaded file: {str(e)}",
                user_message="Unable to save the uploaded file. Please try again.",
                error_code="FILE_SAVE_ERROR",
                suggestions=[
                    "Check available disk space",
                    "Try uploading the file again",
                    "Contact support if the problem persists"
                ]
            ) from e

    def get_file_info(self, file_path: Path) -> FileInfo:
        """
        Get comprehensive information about a file.

        Args:
            file_path: Path to the file

        Returns:
            FileInfo with file metadata
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            stat = file_path.stat()

            return FileInfo(
                filename=file_path.name,
                size=stat.st_size,
                extension=file_path.suffix.lower(),
                validation_status="unknown",
                created_at=stat.st_ctime,
                modified_at=stat.st_mtime,
                file_path=str(file_path)
            )

        except Exception as e:
            self.log_error("get_file_info", e, file_path=str(file_path))
            raise FileProcessingError(
                f"Failed to get file information: {str(e)}",
                error_code="FILE_INFO_ERROR"
            ) from e

    def _create_safe_filename(self, filename: str) -> str:
        """Create a safe filename by removing potentially dangerous characters."""
        # Remove path separators and other dangerous characters
        safe_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        safe_filename = "".join(c for c in filename if c in safe_chars)

        # Ensure it's not empty and has reasonable length
        if not safe_filename or len(safe_filename) < 3:
            safe_filename = "uploaded_file"

        # Limit length
        if len(safe_filename) > 100:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:95] + ext

        return safe_filename

    def cleanup_session_files(self, session_id: str) -> bool:
        """
        Clean up files for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            True if cleanup was successful
        """
        try:
            session_dir = self.upload_dir / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)
                self.log_operation("cleanup_session_files_success", session_id=session_id)
                return True
            return False

        except Exception as e:
            self.log_error("cleanup_session_files", e, session_id=session_id)
            return False