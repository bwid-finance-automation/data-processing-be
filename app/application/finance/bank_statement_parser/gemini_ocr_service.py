"""Gemini OCR Service for extracting text from PDF bank statements."""

import os
import tempfile
from typing import List, Tuple, Optional

import google.generativeai as genai
from dotenv import load_dotenv

from app.shared.utils.logging_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class GeminiOCRService:
    """Service to OCR PDF files using Gemini Flash API."""

    def __init__(self):
        """Initialize Gemini OCR service."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API key not found! Please set GEMINI_API_KEY in your .env file.\n"
                "Get your API key from: https://aistudio.google.com/app/apikey"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        logger.info(f"GeminiOCRService initialized with model: {self.model_name}")

    def extract_text_from_pdf(self, pdf_bytes: bytes, file_name: str) -> str:
        """
        Extract text from a PDF file using Gemini Flash.

        Args:
            pdf_bytes: PDF file content as bytes
            file_name: Original file name (for logging)

        Returns:
            Extracted text from the PDF
        """
        logger.info(f"Extracting text from PDF: {file_name}")

        try:
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

                response = self.model.generate_content([prompt, uploaded_file])

                # Clean up uploaded file from Gemini
                try:
                    genai.delete_file(uploaded_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete uploaded file from Gemini: {e}")

                extracted_text = response.text
                logger.info(f"Successfully extracted {len(extracted_text)} characters from {file_name}")

                return extracted_text

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_name}: {e}")
            raise

    def extract_text_from_pdf_batch(
        self,
        pdf_files: List[Tuple[str, bytes]]
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        Extract text from multiple PDF files.

        Args:
            pdf_files: List of (file_name, pdf_bytes) tuples

        Returns:
            List of (file_name, extracted_text, error) tuples
            - error is None if successful
            - extracted_text is empty string if failed
        """
        results = []

        for file_name, pdf_bytes in pdf_files:
            try:
                text = self.extract_text_from_pdf(pdf_bytes, file_name)
                results.append((file_name, text, None))
            except Exception as e:
                logger.error(f"Failed to extract text from {file_name}: {e}")
                results.append((file_name, "", str(e)))

        return results
