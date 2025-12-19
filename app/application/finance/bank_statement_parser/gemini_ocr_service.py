"""Gemini OCR Service for extracting text from PDF bank statements."""

import os
import tempfile
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from io import BytesIO

import google.generativeai as genai
from google.generativeai import types as genai_types
import pikepdf
from dotenv import load_dotenv

from app.shared.utils.logging_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


@dataclass
class OCRUsageMetrics:
    """Metrics for a single OCR operation."""
    file_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    processing_time_ms: float = 0.0
    model_name: str = ""
    success: bool = True
    error: Optional[str] = None


@dataclass
class OCRBatchMetrics:
    """Aggregated metrics for batch OCR operations."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_processing_time_ms: float = 0.0
    files_processed: int = 0
    files_successful: int = 0
    files_failed: int = 0
    model_name: str = ""
    file_metrics: List[OCRUsageMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "total_processing_time_seconds": round(self.total_processing_time_ms / 1000, 2),
            "files_processed": self.files_processed,
            "files_successful": self.files_successful,
            "files_failed": self.files_failed,
            "model_name": self.model_name,
            "file_metrics": [
                {
                    "file_name": m.file_name,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "total_tokens": m.total_tokens,
                    "processing_time_ms": round(m.processing_time_ms, 2),
                    "success": m.success,
                    "error": m.error,
                }
                for m in self.file_metrics
            ]
        }


class GeminiOCRService:
    """Service to OCR PDF files using Gemini Flash API."""

    def __init__(self):
        """Initialize Gemini OCR service."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API key not found! Please set GEMINI_API_KEY in your .env file.\n"
                "Get your API key from: https://aistudio.google.com/app/apikey"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        # Use deterministic generation to avoid OCR variability between requests
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config=genai_types.GenerationConfig(temperature=0),
        )

        logger.info(f"GeminiOCRService initialized with model: {self.model_name}")

    def _decrypt_pdf(self, pdf_bytes: bytes, password: str) -> bytes:
        """
        Decrypt a password-protected PDF.

        Args:
            pdf_bytes: Encrypted PDF content as bytes
            password: Password to decrypt the PDF

        Returns:
            Decrypted PDF as bytes

        Raises:
            ValueError: If password is incorrect or PDF cannot be decrypted
        """
        try:
            # Open the encrypted PDF with password
            input_stream = BytesIO(pdf_bytes)
            pdf = pikepdf.open(input_stream, password=password)

            # Save decrypted PDF to bytes
            # Use preserve_pdfa=True and avoid recompression to keep original structure
            # This helps ensure OCR produces consistent results
            output_stream = BytesIO()
            pdf.save(
                output_stream,
                linearize=False,  # Don't linearize - keeps structure closer to original
                object_stream_mode=pikepdf.ObjectStreamMode.preserve,  # Preserve object streams
                compress_streams=False,  # Don't recompress - keeps text streams intact
                stream_decode_level=pikepdf.StreamDecodeLevel.none,  # Don't decode streams
            )
            pdf.close()

            decrypted_bytes = output_stream.getvalue()
            logger.info(f"Successfully decrypted PDF ({len(decrypted_bytes)} bytes, original: {len(pdf_bytes)} bytes)")
            return decrypted_bytes

        except pikepdf.PasswordError:
            logger.error("Incorrect password for encrypted PDF")
            raise ValueError("Incorrect password for encrypted PDF")
        except Exception as e:
            logger.error(f"Error decrypting PDF: {e}")
            raise ValueError(f"Failed to decrypt PDF: {e}")

    def _is_pdf_encrypted(self, pdf_bytes: bytes) -> bool:
        """
        Check if a PDF is password-protected.

        Args:
            pdf_bytes: PDF content as bytes

        Returns:
            True if PDF is encrypted, False otherwise
        """
        try:
            input_stream = BytesIO(pdf_bytes)
            pdf = pikepdf.open(input_stream)
            pdf.close()
            return False  # Can open without password = not encrypted
        except pikepdf.PasswordError:
            return True  # Needs password = encrypted
        except Exception:
            return False  # Other error, assume not encrypted

    def extract_text_from_pdf(
        self, pdf_bytes: bytes, file_name: str, password: Optional[str] = None
    ) -> Tuple[str, OCRUsageMetrics]:
        """
        Extract text from a PDF file using Gemini Flash.

        Args:
            pdf_bytes: PDF file content as bytes
            file_name: Original file name (for logging)
            password: Optional password for encrypted PDFs

        Returns:
            Tuple of (extracted_text, usage_metrics)

        Raises:
            ValueError: If PDF is encrypted and no/wrong password provided
        """
        logger.info(f"Extracting text from PDF: {file_name}")
        start_time = time.time()

        metrics = OCRUsageMetrics(
            file_name=file_name,
            model_name=self.model_name,
        )

        try:
            # Check if PDF is encrypted and decrypt if needed
            if self._is_pdf_encrypted(pdf_bytes):
                if not password:
                    logger.error(f"PDF {file_name} is encrypted but no password provided")
                    raise ValueError(f"PDF file '{file_name}' is password-protected. Please provide the password.")
                logger.info(f"PDF {file_name} is encrypted, decrypting...")
                pdf_bytes = self._decrypt_pdf(pdf_bytes, password)

            # Write PDF bytes to a temporary file (required by Gemini API)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name

            try:
                # Upload file to Gemini
                uploaded_file = genai.upload_file(
                    path=tmp_path,
                    mime_type="application/pdf"
                )

                logger.info(f"File uploaded to Gemini: {uploaded_file.uri}")

                # Extract text using Gemini
                prompt = """Extract ALL text from this bank statement PDF document.

Instructions:
- Extract every single piece of text from all pages
- Preserve the table structure as much as possible
- Include all transaction details, dates, amounts, descriptions
- Include account numbers, balances, bank information
- Keep numbers exactly as they appear (with commas, dots, etc.)
- Do not summarize or interpret - just extract the raw text
- Separate different sections with blank lines

Output the extracted text:"""

                # Add timeout to prevent infinite waiting (120 seconds for PDF OCR)
                response = self.model.generate_content(
                    [prompt, uploaded_file],
                    request_options={"timeout": 120}
                )

                # Extract usage metadata from response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    metrics.input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    metrics.output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    metrics.total_tokens = getattr(usage, 'total_token_count', 0) or (metrics.input_tokens + metrics.output_tokens)

                # Clean up uploaded file from Gemini
                try:
                    genai.delete_file(uploaded_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete uploaded file from Gemini: {e}")

                extracted_text = response.text
                metrics.processing_time_ms = (time.time() - start_time) * 1000
                metrics.success = True

                logger.info(
                    f"Successfully extracted {len(extracted_text)} chars from {file_name} "
                    f"(tokens: {metrics.input_tokens}+{metrics.output_tokens}={metrics.total_tokens}, "
                    f"time: {metrics.processing_time_ms:.0f}ms)"
                )

                return extracted_text, metrics

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

        except Exception as e:
            metrics.processing_time_ms = (time.time() - start_time) * 1000
            metrics.success = False
            metrics.error = str(e)
            logger.error(f"Error extracting text from PDF {file_name}: {e}")
            raise

    def extract_text_from_pdf_batch(
        self,
        pdf_files: List[Tuple[str, bytes, Optional[str]]]
    ) -> Tuple[List[Tuple[str, str, Optional[str]]], OCRBatchMetrics]:
        """
        Extract text from multiple PDF files.

        Args:
            pdf_files: List of (file_name, pdf_bytes, password) tuples
                       password can be None for non-encrypted PDFs

        Returns:
            Tuple of:
            - List of (file_name, extracted_text, error) tuples
              - error is None if successful
              - extracted_text is empty string if failed
            - OCRBatchMetrics with aggregated usage statistics
        """
        results = []
        batch_metrics = OCRBatchMetrics(model_name=self.model_name)

        for file_name, pdf_bytes, password in pdf_files:
            batch_metrics.files_processed += 1
            try:
                text, metrics = self.extract_text_from_pdf(pdf_bytes, file_name, password)
                results.append((file_name, text, None))

                # Aggregate metrics
                batch_metrics.total_input_tokens += metrics.input_tokens
                batch_metrics.total_output_tokens += metrics.output_tokens
                batch_metrics.total_tokens += metrics.total_tokens
                batch_metrics.total_processing_time_ms += metrics.processing_time_ms
                batch_metrics.files_successful += 1
                batch_metrics.file_metrics.append(metrics)

            except Exception as e:
                logger.error(f"Failed to extract text from {file_name}: {e}")
                results.append((file_name, "", str(e)))

                # Record failed file metrics
                failed_metrics = OCRUsageMetrics(
                    file_name=file_name,
                    model_name=self.model_name,
                    success=False,
                    error=str(e)
                )
                batch_metrics.files_failed += 1
                batch_metrics.file_metrics.append(failed_metrics)

        return results, batch_metrics
