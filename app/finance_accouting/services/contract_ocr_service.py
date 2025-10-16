"""Contract OCR service for extracting information from contract documents."""
import base64
import json
import time
import os
from pathlib import Path
from typing import Union, Optional, List

from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF
import pytesseract

from ..models.contract_schemas import ContractInfo, ContractExtractionResult, Party
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ContractOCRService:
    """Service for reading and extracting information from contract documents using Tesseract OCR and OpenAI text extraction."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OCR service with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.pdf_max_pages = 10  # Maximum pages to process from PDF

        # Configure Tesseract path if needed (for Windows or custom installations)
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Chunking configuration for long contracts
        self.chunk_size = 15000  # Characters per chunk (~3,750 words)
        self.chunk_overlap = 500  # Overlap between chunks to avoid missing context

        logger.info(f"Contract OCR Service initialized with Tesseract OCR + {self.model}")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for processing long contracts.

        Args:
            text: The full text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def _are_all_fields_extracted(self, data: dict) -> bool:
        """
        Check if all required fields have been extracted.

        Args:
            data: Extracted data dictionary

        Returns:
            True if all fields are present and non-null
        """
        required_fields = ['type', 'tenant', 'gla_for_lease', 'rate_periods']

        # Check basic fields
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == '':
                return False

        # Check rate_periods has at least one entry with required fields
        if not isinstance(data['rate_periods'], list) or len(data['rate_periods']) == 0:
            return False

        # Check that at least one rate period has all required fields
        for period in data['rate_periods']:
            if not isinstance(period, dict):
                continue
            period_fields = ['start_date', 'end_date', 'monthly_rate_per_sqm', 'total_monthly_rate']
            has_all = all(period.get(f) is not None and period.get(f) != '' for f in period_fields)
            if has_all:
                return True

        return False

    def _merge_extracted_data(self, existing: dict, new_data: dict) -> dict:
        """
        Merge newly extracted data with existing data, filling in missing fields.

        Args:
            existing: Previously extracted data
            new_data: Newly extracted data from current chunk

        Returns:
            Merged data dictionary
        """
        for key, value in new_data.items():
            if key == 'rate_periods':
                # Merge rate_periods arrays
                if 'rate_periods' not in existing or not existing['rate_periods']:
                    existing['rate_periods'] = value if isinstance(value, list) else []
                elif isinstance(value, list):
                    # Add new periods that don't already exist
                    existing_periods = existing.get('rate_periods', [])
                    for new_period in value:
                        # Check if this period already exists (by start_date)
                        if not any(p.get('start_date') == new_period.get('start_date')
                                  for p in existing_periods):
                            existing_periods.append(new_period)
                    existing['rate_periods'] = existing_periods
            else:
                # Only update if existing field is null/empty and new value is not
                if (key not in existing or existing[key] is None or existing[key] == '') and value:
                    existing[key] = value

        return existing

    def _extract_text_from_image(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from image using Tesseract OCR.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text string
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config='--psm 6')  # PSM 6: Assume uniform block of text
            logger.info(f"Extracted {len(text)} characters from image")

            # Print OCR output to terminal
            print("\n" + "="*80)
            print("📄 OCR EXTRACTED TEXT FROM IMAGE:")
            print("="*80)
            print(text)
            print("="*80 + "\n")

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""

    def _extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from PDF using PyMuPDF and Tesseract OCR.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Combined text from all pages
        """
        pdf_document = fitz.open(pdf_path)
        all_text = []

        total_pages = len(pdf_document)
        pages_to_process = min(total_pages, self.pdf_max_pages)

        logger.info(f"Processing {pages_to_process} page(s) from PDF with {total_pages} total pages")

        for page_num in range(pages_to_process):
            page = pdf_document[page_num]

            # First try to extract text directly (for text-based PDFs)
            page_text = page.get_text()

            # If no text found or very little text, use OCR on the page image
            if not page_text or len(page_text.strip()) < 50:
                logger.info(f"Page {page_num + 1}: Using OCR (insufficient text found)")
                # Render page to image with high DPI for better OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Extract text using Tesseract
                page_text = pytesseract.image_to_string(img, config='--psm 6')
                logger.info(f"Page {page_num + 1}: Extracted {len(page_text)} characters via OCR")

                # Print OCR output to terminal
                print("\n" + "="*80)
                print(f"📄 OCR EXTRACTED TEXT FROM PDF PAGE {page_num + 1}:")
                print("="*80)
                print(page_text)
                print("="*80 + "\n")
            else:
                logger.info(f"Page {page_num + 1}: Extracted {len(page_text)} characters (text-based PDF)")

                # Print direct text extraction to terminal
                print("\n" + "="*80)
                print(f"📄 DIRECT TEXT EXTRACTION FROM PDF PAGE {page_num + 1}:")
                print("="*80)
                print(page_text)
                print("="*80 + "\n")

            all_text.append(page_text)

        pdf_document.close()

        combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
        logger.info(f"Total extracted text: {len(combined_text)} characters")

        # Print complete combined text
        print("\n" + "="*80)
        print(f"📋 COMPLETE COMBINED TEXT ({len(combined_text)} characters):")
        print("="*80)
        print(combined_text)
        print("="*80 + "\n")

        return combined_text.strip()

    def _create_extraction_prompt(self, extracted_text: str) -> str:
        """
        Create the prompt for information extraction from OCR text.

        Args:
            extracted_text: The OCR-extracted text from the contract

        Returns:
            Formatted prompt string
        """
        return f"""You are an expert contract analyst specializing in lease contracts. Analyze the following contract text that was extracted via OCR and extract the following information in JSON format:

REQUIRED FIELDS:
1. type: Type of lease: Rent, Fit out, Service Charge in Rent Free, Rent Free, or Other
2. tenant: Tenant name (person or company name)
3. gla_for_lease: GLA (Gross Leasable Area) for lease in square meters (as string, numeric value only)
4. rate_periods: Array of rate periods, where each period has:
   - start_date: Period start date (YYYY-MM-DD as string)
   - end_date: Period end date (YYYY-MM-DD as string)
   - monthly_rate_per_sqm: Monthly rate per sqm for this period (as string, no currency symbol)
   - total_monthly_rate: Total monthly rate for this period (as string, no currency symbol)
     * If not stated, calculate: monthly_rate_per_sqm × gla_for_lease

IMPORTANT INSTRUCTIONS:
- Return the information as a valid JSON object
- ALL values must be strings (even numbers - wrap them in quotes: "250" not 250)
- If a field cannot be determined, use null
- Extract all monetary values as strings without currency symbols
- Extract all dates in YYYY-MM-DD format as strings
- For dates, convert any format to "YYYY-MM-DD"
- The text may contain OCR errors, so use context to interpret unclear words
- Look for keywords like "rent", "lease term", "lessee", "lessor", "tenant", "area", "sqm", "rate"
- IMPORTANT: Contracts often have different rates for different years/periods - extract ALL rate periods
- Each rate period should have its own start_date, end_date, and rates
- If only one rate period exists, the array will have one element

EXAMPLE OUTPUT FORMAT (Multiple Rate Periods):
{{
  "type": "Rent",
  "tenant": "ABC Corporation",
  "gla_for_lease": "200",
  "rate_periods": [
    {{
      "start_date": "2024-01-01",
      "end_date": "2024-12-31",
      "monthly_rate_per_sqm": "250",
      "total_monthly_rate": "50000"
    }},
    {{
      "start_date": "2025-01-01",
      "end_date": "2025-12-31",
      "monthly_rate_per_sqm": "275",
      "total_monthly_rate": "55000"
    }},
    {{
      "start_date": "2026-01-01",
      "end_date": "2026-12-31",
      "monthly_rate_per_sqm": "300",
      "total_monthly_rate": "60000"
    }}
  ]
}}

CONTRACT TEXT:
---
{extracted_text}
---

Based on the contract text above, extract ONLY the 7 required fields and return as valid JSON."""

    def _extract_from_text_chunk(self, text_chunk: str) -> dict:
        """
        Extract data from a single text chunk using LLM.

        Args:
            text_chunk: Text chunk to process

        Returns:
            Extracted data dictionary
        """
        prompt = self._create_extraction_prompt(text_chunk)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        content = response.choices[0].message.content

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def _process_text_with_chunking(self, extracted_text: str) -> dict:
        """
        Process text in chunks, stopping when all fields are found or all text is processed.

        Args:
            extracted_text: The full OCR-extracted text

        Returns:
            Extracted data dictionary with all found fields
        """
        chunks = self._chunk_text(extracted_text)
        merged_data = {}

        logger.info(f"Processing {len(chunks)} chunk(s) of text...")

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}...")

            try:
                chunk_data = self._extract_from_text_chunk(chunk)
                merged_data = self._merge_extracted_data(merged_data, chunk_data)

                # Check if we have all fields
                if self._are_all_fields_extracted(merged_data):
                    num_periods = len(merged_data.get('rate_periods', []))
                    logger.info(f"✓ All required fields found with {num_periods} rate period(s) after processing {i}/{len(chunks)} chunks. Stopping early.")
                    print("\n" + "="*80)
                    print(f"✓ ALL FIELDS EXTRACTED - {num_periods} RATE PERIOD(S) (stopped at chunk {i}/{len(chunks)})")
                    print("="*80 + "\n")
                    break
                else:
                    missing = []
                    for f in ['type', 'tenant', 'gla_for_lease']:
                        if f not in merged_data or not merged_data[f]:
                            missing.append(f)
                    if 'rate_periods' not in merged_data or not merged_data.get('rate_periods'):
                        missing.append('rate_periods')
                    else:
                        num_periods = len(merged_data.get('rate_periods', []))
                        logger.info(f"Found {num_periods} rate period(s) so far...")
                    logger.info(f"Missing fields: {missing}. Continuing to next chunk...")

            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {str(e)}. Continuing...")
                continue

        # Final check
        if self._are_all_fields_extracted(merged_data):
            num_periods = len(merged_data.get('rate_periods', []))
            logger.info(f"✓ Successfully extracted all required fields with {num_periods} rate period(s)")
        else:
            missing = []
            for f in ['type', 'tenant', 'gla_for_lease']:
                if f not in merged_data or not merged_data[f]:
                    missing.append(f)
            if 'rate_periods' not in merged_data or not merged_data.get('rate_periods'):
                missing.append('rate_periods (no valid periods found)')
            logger.warning(f"⚠ Processed all chunks but still missing fields: {missing}")

        return merged_data

    def process_contract(self, file_path: Union[str, Path]) -> ContractExtractionResult:
        """
        Read and extract information from a contract document using Tesseract OCR + OpenAI text extraction.

        Args:
            file_path: Path to the contract document (image or PDF)

        Returns:
            ContractExtractionResult with extracted contract information
        """
        start_time = time.time()
        file_path = Path(file_path)

        logger.info(f"Processing contract: {file_path.name}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return ContractExtractionResult(
                success=False,
                error=f"File not found: {file_path}",
                source_file=str(file_path.name)
            )

        try:
            # Step 1: Extract text using Tesseract OCR
            if file_path.suffix.lower() == '.pdf':
                logger.info(f"Extracting text from PDF using OCR...")
                extracted_text = self._extract_text_from_pdf(file_path)
            else:
                logger.info(f"Extracting text from image using OCR...")
                extracted_text = self._extract_text_from_image(file_path)

            if not extracted_text:
                logger.warning("No text extracted from document")
                return ContractExtractionResult(
                    success=False,
                    error="No text could be extracted from the document",
                    source_file=str(file_path.name),
                    processing_time=time.time() - start_time
                )

            logger.info(f"Successfully extracted {len(extracted_text)} characters of text")

            # Step 2: Process text with chunking (stops when all 7 fields found or all text processed)
            logger.info(f"Processing text with intelligent chunking...")
            extracted_data = self._process_text_with_chunking(extracted_text)

            # Add raw extracted text to the data
            extracted_data["raw_text"] = extracted_text

            # Convert gla_for_lease to string if numeric
            if 'gla_for_lease' in extracted_data and extracted_data['gla_for_lease'] is not None:
                if isinstance(extracted_data['gla_for_lease'], (int, float)):
                    extracted_data['gla_for_lease'] = str(extracted_data['gla_for_lease'])

            # Process rate_periods array
            if 'rate_periods' in extracted_data and isinstance(extracted_data['rate_periods'], list):
                gla = extracted_data.get('gla_for_lease')

                for period in extracted_data['rate_periods']:
                    if not isinstance(period, dict):
                        continue

                    # Convert numeric fields to strings
                    for field in ['monthly_rate_per_sqm', 'total_monthly_rate']:
                        if field in period and period[field] is not None:
                            if isinstance(period[field], (int, float)):
                                period[field] = str(period[field])

                    # Calculate total_monthly_rate if missing but we have the components
                    if (not period.get('total_monthly_rate') and
                        period.get('monthly_rate_per_sqm') and gla):
                        try:
                            rate_per_sqm = float(period['monthly_rate_per_sqm'])
                            gla_float = float(gla)
                            total = rate_per_sqm * gla_float
                            period['total_monthly_rate'] = str(round(total, 2))
                            logger.info(f"Calculated total_monthly_rate for period {period.get('start_date')}: {period['total_monthly_rate']}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not calculate total_monthly_rate for period: {e}")

            # Sanitize data - convert empty arrays to None for string fields
            string_fields = [
                'contract_title', 'contract_type', 'effective_date', 'expiration_date',
                'contract_value', 'payment_terms', 'termination_clauses', 'governing_law',
                'internal_id', 'id', 'master_record_id', 'plc_id', 'unit_for_lease',
                'type', 'start_date', 'end_date', 'tenant', 'monthly_rate_per_sqm',
                'gla_for_lease', 'total_monthly_rate', 'total_rate', 'status',
                'historical_journal_entry', 'amortization_journal_entry',
                'related_billing_schedule', 'subsidiary', 'ccs_product_type',
                'bwid_project', 'phase', 'raw_text'
            ]

            for field in string_fields:
                if field in extracted_data:
                    value = extracted_data[field]
                    # Convert empty arrays, empty strings, or invalid types to None
                    if isinstance(value, list) and len(value) == 0:
                        extracted_data[field] = None
                    elif isinstance(value, str) and value.strip() == '':
                        extracted_data[field] = None
                    elif value is not None and not isinstance(value, (str, int, float, bool)):
                        extracted_data[field] = None

            # Create ContractInfo object
            contract_info = ContractInfo(**extracted_data)

            processing_time = time.time() - start_time

            logger.info(f"Successfully processed contract: {file_path.name} in {processing_time:.2f}s")

            return ContractExtractionResult(
                success=True,
                data=contract_info,
                processing_time=processing_time,
                source_file=str(file_path.name)
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            logger.error(f"{error_msg}. Response preview: {content[:200] if 'content' in locals() else 'No content'}")
            return ContractExtractionResult(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time,
                source_file=str(file_path.name)
            )
        except Exception as e:
            error_msg = f"Error processing contract: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ContractExtractionResult(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time,
                source_file=str(file_path.name)
            )

    def process_contracts_batch(self, file_paths: List[Union[str, Path]]) -> List[ContractExtractionResult]:
        """
        Process multiple contract documents in batch.

        Args:
            file_paths: List of paths to contract documents

        Returns:
            List of ContractExtractionResult objects
        """
        logger.info(f"Processing batch of {len(file_paths)} contracts")
        results = []
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing contract {i}/{len(file_paths)}")
            result = self.process_contract(file_path)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch processing complete: {successful}/{len(results)} successful")
        return results
