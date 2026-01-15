"""Gemini OCR Service for extracting text from PDF bank statements.

This module provides async OCR capabilities using Google's Gemini API.
Features:
- Async processing to avoid blocking the event loop
- Redis caching for OCR results (by file content hash)
- Concurrent batch processing with semaphore control
- Password-protected PDF support
"""

import asyncio
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
    cache_hit: bool = False


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
    cache_hits: int = 0
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
            "cache_hits": self.cache_hits,
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
                    "cache_hit": m.cache_hit,
                }
                for m in self.file_metrics
            ]
        }


class GeminiOCRService:
    """Service to OCR PDF files using Gemini Flash API.

    This service provides async methods for extracting text from PDF files
    using Google's Gemini API. It includes:
    - Redis caching to avoid redundant API calls
    - Async processing to prevent blocking the event loop
    - Concurrent batch processing with rate limiting
    """

    # Semaphore to limit concurrent Gemini API calls
    _semaphore: Optional[asyncio.Semaphore] = None
    MAX_CONCURRENT_REQUESTS = 3

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

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        """Get or create the semaphore for rate limiting."""
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(cls.MAX_CONCURRENT_REQUESTS)
        return cls._semaphore

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
            output_stream = BytesIO()
            pdf.save(
                output_stream,
                linearize=False,
                object_stream_mode=pikepdf.ObjectStreamMode.preserve,
                compress_streams=False,
                stream_decode_level=pikepdf.StreamDecodeLevel.none,
            )
            pdf.close()

            decrypted_bytes = output_stream.getvalue()
            logger.info(f"Successfully decrypted PDF ({len(decrypted_bytes)} bytes)")
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
            return False
        except pikepdf.PasswordError:
            return True
        except Exception:
            return False

    def _write_temp_file(self, pdf_bytes: bytes) -> str:
        """Write PDF bytes to a temporary file.

        This is a sync helper method for use with asyncio.to_thread().

        Args:
            pdf_bytes: PDF content as bytes

        Returns:
            Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            return tmp_file.name

    def _upload_to_gemini(self, tmp_path: str) -> Any:
        """Upload file to Gemini (sync, for use with to_thread).

        Args:
            tmp_path: Path to the temporary file

        Returns:
            Gemini uploaded file object
        """
        return genai.upload_file(path=tmp_path, mime_type="application/pdf")

    def _generate_content(self, prompt: str, uploaded_file: Any) -> Any:
        """Generate content from Gemini (sync, for use with to_thread).

        Args:
            prompt: The extraction prompt
            uploaded_file: Gemini uploaded file object

        Returns:
            Gemini response object
        """
        return self.model.generate_content(
            [prompt, uploaded_file],
            request_options={"timeout": 120}
        )

    def _delete_gemini_file(self, file_name: str) -> None:
        """Delete file from Gemini (sync, for use with to_thread).

        Args:
            file_name: Name of the file to delete from Gemini
        """
        try:
            genai.delete_file(file_name)
        except Exception as e:
            logger.warning(f"Failed to delete uploaded file from Gemini: {e}")

    async def extract_text_from_pdf(
        self, pdf_bytes: bytes, file_name: str, password: Optional[str] = None
    ) -> Tuple[str, OCRUsageMetrics]:
        """
        Extract text from a PDF file using Gemini Flash (async version).

        This method:
        1. Checks Redis cache for existing results
        2. Decrypts PDF if password-protected
        3. Uses asyncio.to_thread() for blocking Gemini SDK calls
        4. Caches successful results in Redis

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
            # Check cache first
            from app.infrastructure.cache.redis_cache import get_cache_service, RedisCacheService
            cache = await get_cache_service()
            file_hash = RedisCacheService.hash_content(pdf_bytes)

            cached_text = await cache.get_ocr_result(file_hash)
            if cached_text:
                logger.info(f"Cache hit for {file_name} (hash: {file_hash[:16]}...)")
                metrics.processing_time_ms = (time.time() - start_time) * 1000
                metrics.success = True
                metrics.cache_hit = True
                return cached_text, metrics

            # Check if PDF is encrypted and decrypt if needed
            if self._is_pdf_encrypted(pdf_bytes):
                if not password:
                    logger.error(f"PDF {file_name} is encrypted but no password provided")
                    raise ValueError(f"PDF file '{file_name}' is password-protected. Please provide the password.")
                logger.info(f"PDF {file_name} is encrypted, decrypting...")
                pdf_bytes = await asyncio.to_thread(self._decrypt_pdf, pdf_bytes, password)

            # Write PDF bytes to a temporary file (async via to_thread)
            tmp_path = await asyncio.to_thread(self._write_temp_file, pdf_bytes)

            try:
                # Upload file to Gemini (async via to_thread)
                uploaded_file = await asyncio.to_thread(self._upload_to_gemini, tmp_path)
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

                # Generate content (async via to_thread with semaphore for rate limiting)
                async with self._get_semaphore():
                    response = await asyncio.to_thread(
                        self._generate_content, prompt, uploaded_file
                    )

                # Extract usage metadata from response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    metrics.input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    metrics.output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    metrics.total_tokens = getattr(usage, 'total_token_count', 0) or (
                        metrics.input_tokens + metrics.output_tokens
                    )

                # Clean up uploaded file from Gemini (async via to_thread)
                await asyncio.to_thread(self._delete_gemini_file, uploaded_file.name)

                extracted_text = response.text
                metrics.processing_time_ms = (time.time() - start_time) * 1000
                metrics.success = True

                # Cache the result
                await cache.set_ocr_result(file_hash, extracted_text)

                logger.info(
                    f"Successfully extracted {len(extracted_text)} chars from {file_name} "
                    f"(tokens: {metrics.input_tokens}+{metrics.output_tokens}={metrics.total_tokens}, "
                    f"time: {metrics.processing_time_ms:.0f}ms)"
                )

                return extracted_text, metrics

            finally:
                # Clean up temporary file (async via to_thread)
                try:
                    await asyncio.to_thread(os.unlink, tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

        except Exception as e:
            metrics.processing_time_ms = (time.time() - start_time) * 1000
            metrics.success = False
            metrics.error = str(e)
            logger.error(f"Error extracting text from PDF {file_name}: {e}")
            raise

    async def extract_text_from_pdf_batch(
        self,
        pdf_files: List[Tuple[str, bytes, Optional[str]]]
    ) -> Tuple[List[Tuple[str, str, Optional[str]]], OCRBatchMetrics]:
        """
        Extract text from multiple PDF files concurrently.

        This method processes multiple PDFs in parallel using asyncio.gather(),
        with a semaphore to limit concurrent Gemini API calls.

        Args:
            pdf_files: List of (file_name, pdf_bytes, password) tuples
                       password can be None for non-encrypted PDFs

        Returns:
            Tuple of:
            - List of (file_name, extracted_text, error) tuples
            - OCRBatchMetrics with aggregated usage statistics
        """
        batch_metrics = OCRBatchMetrics(model_name=self.model_name)

        async def process_single(
            file_name: str, pdf_bytes: bytes, password: Optional[str]
        ) -> Tuple[str, str, Optional[str], OCRUsageMetrics]:
            """Process a single PDF file."""
            try:
                text, metrics = await self.extract_text_from_pdf(pdf_bytes, file_name, password)
                return (file_name, text, None, metrics)
            except Exception as e:
                logger.error(f"Failed to extract text from {file_name}: {e}")
                failed_metrics = OCRUsageMetrics(
                    file_name=file_name,
                    model_name=self.model_name,
                    success=False,
                    error=str(e)
                )
                return (file_name, "", str(e), failed_metrics)

        # Process all files concurrently
        tasks = [
            process_single(file_name, pdf_bytes, password)
            for file_name, pdf_bytes, password in pdf_files
        ]

        processed = await asyncio.gather(*tasks)

        # Aggregate results
        results = []
        for file_name, text, error, metrics in processed:
            batch_metrics.files_processed += 1
            results.append((file_name, text, error))
            batch_metrics.file_metrics.append(metrics)

            if error:
                batch_metrics.files_failed += 1
            else:
                batch_metrics.files_successful += 1
                batch_metrics.total_input_tokens += metrics.input_tokens
                batch_metrics.total_output_tokens += metrics.output_tokens
                batch_metrics.total_tokens += metrics.total_tokens
                batch_metrics.total_processing_time_ms += metrics.processing_time_ms

                if metrics.cache_hit:
                    batch_metrics.cache_hits += 1

        return results, batch_metrics
