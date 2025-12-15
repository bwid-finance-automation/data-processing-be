"""Contract OCR service for extracting information from contract documents."""
import json
import time
import os
from pathlib import Path
from typing import Union, Optional, List, Dict

from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF
import pytesseract

from app.presentation.schemas.contract_schemas import ContractInfo, ContractExtractionResult, RatePeriod, TokenUsage
from app.shared.utils.logging_config import get_logger
from .contract_validator import ContractValidator
from .contract_extraction_prompts import get_contract_extraction_prompt
from .unit_breakdown_reader import UnitBreakdownReader, UnitBreakdownResult, UnitBreakdown

logger = get_logger(__name__)


class ContractOCRService:
    """Service for reading and extracting information from contract documents using Tesseract OCR and OpenAI text extraction."""

    def __init__(self, api_key: Optional[str] = None, enable_validation: bool = True):
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

        # Validation
        self.enable_validation = enable_validation
        self.validator = ContractValidator() if enable_validation else None

        # Log and print model information prominently
        model_info = f"Contract OCR Service initialized with Tesseract OCR + {self.model} (validation: {enable_validation})"
        logger.info(model_info)
        print("\n" + "="*80)
        print(f"🤖 AI MODEL: {self.model}")
        print("="*80 + "\n")

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
        Includes both legacy fields and new multilingual fields.

        Args:
            data: Extracted data dictionary

        Returns:
            True if all fields are present and non-null
        """
        # Core required fields (original + new critical fields)
        required_fields = [
            'type',
            'tenant',
            'gla_for_lease',
            'rate_periods',
            'customer_name',
            'contract_number',
            'contract_date'
        ]

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

    def _calculate_service_charge(self, data: dict) -> str:
        """
        Calculate total service charge based on extracted data.

        Logic:
        - If service_charge_applies_to == "rent_free_only": Calculate for rent-free periods only
        - If service_charge_applies_to == "all_periods": Calculate for all periods
        - If service_charge_applies_to == "not_applicable": Return None

        Formula:
        service_charge_total = service_charge_rate × number_of_months × (gfa or gla_for_lease)

        Args:
            data: Extracted contract data with rate_periods, service_charge_rate, etc.

        Returns:
            Total service charge as string, or None if not applicable
        """
        try:
            # Check if service charge is applicable
            applies_to = data.get('service_charge_applies_to')
            if not applies_to or applies_to == 'not_applicable':
                return None

            # Get service charge rate
            service_charge_rate = data.get('service_charge_rate')
            if not service_charge_rate:
                return None

            # Get area (prefer gfa, fallback to gla_for_lease)
            area = data.get('gfa') or data.get('gla_for_lease')
            if not area:
                logger.warning("Cannot calculate service charge: no GFA or GLA found")
                return None

            # Convert to float for calculation
            rate = float(service_charge_rate)
            area_float = float(area)

            # Get rate periods
            rate_periods = data.get('rate_periods', [])
            if not rate_periods:
                return None

            total_months = 0

            if applies_to == 'rent_free_only':
                # Calculate months where foc_from and foc_to are set (rent-free periods)
                for period in rate_periods:
                    if not isinstance(period, dict):
                        continue

                    foc_from = period.get('foc_from')
                    foc_to = period.get('foc_to')

                    # Check if this period has FOC dates
                    if foc_from and foc_to:
                        try:
                            from datetime import datetime
                            start = datetime.strptime(foc_from, '%m-%d-%Y')
                            end = datetime.strptime(foc_to, '%m-%d-%Y')
                            months = ((end.year - start.year) * 12 + end.month - start.month) + 1
                            total_months += months
                            logger.info(f"Found FOC period: {foc_from} to {foc_to} ({months} month(s))")
                        except Exception as e:
                            logger.warning(f"Error calculating FOC months for period {foc_from} to {foc_to}: {e}")

            elif applies_to == 'all_periods':
                # Calculate total months across all periods
                for period in rate_periods:
                    if not isinstance(period, dict):
                        continue

                    start_date = period.get('start_date')
                    end_date = period.get('end_date')

                    if start_date and end_date:
                        try:
                            from datetime import datetime
                            start = datetime.strptime(start_date, '%m-%d-%Y')
                            end = datetime.strptime(end_date, '%m-%d-%Y')
                            months = ((end.year - start.year) * 12 + end.month - start.month) + 1
                            total_months += months
                        except Exception as e:
                            logger.warning(f"Error calculating months for period {start_date} to {end_date}: {e}")

            if total_months == 0:
                logger.info(f"Service charge applicable to '{applies_to}' but no matching periods found")
                return "0"

            # Calculate total: rate × months × area
            total_service_charge = rate * total_months * area_float

            logger.info(f"Calculated service charge: {rate}/sqm/month × {total_months} months × {area_float} sqm = {total_service_charge}")

            return str(round(total_service_charge, 2))

        except Exception as e:
            logger.error(f"Error calculating service charge: {e}", exc_info=True)
            return None

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
        Supports multilingual contracts (Vietnamese, English, Chinese).

        Args:
            extracted_text: The OCR-extracted text from the contract

        Returns:
            Formatted prompt string
        """
        return get_contract_extraction_prompt(extracted_text)

    def _extract_from_text_chunk(self, text_chunk: str) -> tuple[dict, dict]:
        """
        Extract data from a single text chunk using LLM.

        Args:
            text_chunk: Text chunk to process

        Returns:
            Tuple of (extracted data dictionary, usage dictionary with token counts)
        """
        prompt = self._create_extraction_prompt(text_chunk)

        # Print prompt for debugging
        print("\n" + "="*80)
        print("📝 PROMPT SENT TO AI (for debugging):")
        print("="*80)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print(f"\n[Full prompt length: {len(prompt)} characters]")
        print("="*80 + "\n")

        # Log AI call
        logger.info(f"Calling OpenAI API with model: {self.model}")
        print(f"🤖 Calling AI model: {self.model}")

        # Configure parameters based on model type
        try:
            if self.model.startswith("gpt-5"):
                # GPT-5 uses new parameters and extended reasoning
                print(f"   Using GPT-5 parameters: max_completion_tokens=2000, reasoning_effort=high")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=2000,
                    reasoning_effort="high",
                )
            else:
                # GPT-4 and earlier models use traditional parameters
                print(f"   Using GPT-4 parameters: max_tokens=2000, temperature=0.1")
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

            # Log response details
            print(f"   Response received. Finish reason: {response.choices[0].finish_reason}")
            content = response.choices[0].message.content

            # Extract token usage information
            usage_info = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }

            if hasattr(response, 'usage') and response.usage:
                usage_info['prompt_tokens'] = response.usage.prompt_tokens
                usage_info['completion_tokens'] = response.usage.completion_tokens
                usage_info['total_tokens'] = response.usage.total_tokens

                # Log token usage
                logger.info(f"Token usage - Prompt: {usage_info['prompt_tokens']}, "
                           f"Completion: {usage_info['completion_tokens']}, "
                           f"Total: {usage_info['total_tokens']}")
                print(f"   📊 Token usage: Prompt={usage_info['prompt_tokens']}, "
                      f"Completion={usage_info['completion_tokens']}, "
                      f"Total={usage_info['total_tokens']}")

            if not content:
                print(f"   ⚠️ WARNING: Empty response content from {self.model}")
                logger.warning(f"Empty response from {self.model}")
                return {}, usage_info

            print(f"   Response length: {len(content)} characters")

            # Print AI response for debugging
            print("\n" + "="*80)
            print("🤖 AI RESPONSE (for debugging):")
            print("="*80)
            print(content)
            print("="*80 + "\n")

        except Exception as e:
            print(f"   ❌ ERROR calling {self.model}: {str(e)}")
            logger.error(f"Error calling {self.model}: {str(e)}")
            raise

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content), usage_info

    def _process_text_with_chunking(self, extracted_text: str) -> tuple[dict, dict]:
        """
        Process text in chunks, stopping when all fields are found or all text is processed.

        Args:
            extracted_text: The full OCR-extracted text

        Returns:
            Tuple of (extracted data dictionary with all found fields, accumulated token usage)
        """
        chunks = self._chunk_text(extracted_text)
        merged_data = {}

        # Initialize accumulated token usage
        total_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

        logger.info(f"Processing {len(chunks)} chunk(s) of text...")

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}...")

            try:
                chunk_data, chunk_usage = self._extract_from_text_chunk(chunk)
                merged_data = self._merge_extracted_data(merged_data, chunk_data)

                # Accumulate token usage
                total_usage['prompt_tokens'] += chunk_usage.get('prompt_tokens', 0)
                total_usage['completion_tokens'] += chunk_usage.get('completion_tokens', 0)
                total_usage['total_tokens'] += chunk_usage.get('total_tokens', 0)

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

        # Log total token usage for this contract
        logger.info(f"📊 Total token usage for this contract - Prompt: {total_usage['prompt_tokens']}, "
                   f"Completion: {total_usage['completion_tokens']}, "
                   f"Total: {total_usage['total_tokens']}")

        print("\n" + "="*80)
        print(f"📊 TOKEN USAGE:")
        print(f"   • Input tokens:  {total_usage['prompt_tokens']:,}")
        print(f"   • Output tokens: {total_usage['completion_tokens']:,}")
        print(f"   • TOTAL TOKENS:  {total_usage['total_tokens']:,}")
        print(f"   • Model:         {self.model}")
        print("="*80 + "\n")

        return merged_data, total_usage

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
            extracted_data, token_usage = self._process_text_with_chunking(extracted_text)

            # Add raw extracted text to the data
            extracted_data["raw_text"] = extracted_text

            # Convert gla_for_lease to string if numeric
            if 'gla_for_lease' in extracted_data and extracted_data['gla_for_lease'] is not None:
                if isinstance(extracted_data['gla_for_lease'], (int, float)):
                    extracted_data['gla_for_lease'] = str(extracted_data['gla_for_lease'])

            # Process rate_periods array
            if 'rate_periods' in extracted_data and isinstance(extracted_data['rate_periods'], list):
                gla = extracted_data.get('gla_for_lease')
                processed_periods = []

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

                    # Convert dict to RatePeriod object
                    try:
                        rate_period_obj = RatePeriod(**period)
                        processed_periods.append(rate_period_obj)
                    except Exception as e:
                        logger.warning(f"Could not create RatePeriod object from {period}: {e}")

                # Replace the list of dicts with list of RatePeriod objects
                extracted_data['rate_periods'] = processed_periods
                logger.info(f"Successfully converted {len(processed_periods)} rate periods to RatePeriod objects")

            # Step 3: Calculate service charge (Python calculation layer)
            logger.info("Calculating service charge based on extracted data...")
            service_charge_total = self._calculate_service_charge(extracted_data)
            if service_charge_total:
                extracted_data['service_charge_total'] = service_charge_total
                logger.info(f"Service charge calculated: {service_charge_total}")
            else:
                extracted_data['service_charge_total'] = None
                logger.info("Service charge not calculated (not applicable or insufficient data)")

            # Convert new numeric fields to strings
            for field in ['gfa', 'deposit_amount', 'service_charge_rate']:
                if field in extracted_data and extracted_data[field] is not None:
                    if isinstance(extracted_data[field], (int, float)):
                        extracted_data[field] = str(extracted_data[field])

            # Sanitize data - convert empty arrays to None for string fields
            string_fields = [
                'contract_title', 'contract_type', 'effective_date', 'expiration_date',
                'contract_value', 'payment_terms', 'termination_clauses', 'governing_law',
                'internal_id', 'id', 'master_record_id', 'plc_id', 'unit_for_lease',
                'type', 'start_date', 'end_date', 'tenant', 'monthly_rate_per_sqm',
                'gla_for_lease', 'total_monthly_rate', 'total_rate', 'status',
                'historical_journal_entry', 'amortization_journal_entry',
                'related_billing_schedule', 'subsidiary', 'ccs_product_type',
                'bwid_project', 'phase', 'raw_text',
                # New fields
                'customer_name', 'contract_number', 'contract_date', 'payment_terms_details',
                'deposit_amount', 'handover_date', 'gfa', 'service_charge_rate',
                'service_charge_applies_to', 'service_charge_total'
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

            # Step 4: Validate extracted data (if enabled)
            if self.enable_validation and self.validator:
                logger.info("Running validation on extracted data...")
                validation_result = self.validator.validate_contract(contract_info)

                errors = validation_result.get('errors', [])
                warnings = validation_result.get('warnings', [])

                if errors:
                    logger.warning(f"Validation errors found: {len(errors)}")
                    for error in errors:
                        logger.warning(f"  - {error}")

                if warnings:
                    logger.info(f"Validation warnings: {len(warnings)}")
                    for warning in warnings:
                        logger.info(f"  - {warning}")

                # Log validation summary
                summary = self.validator.get_validation_summary(validation_result)
                print("\n" + "="*80)
                print("📋 VALIDATION SUMMARY:")
                print("="*80)
                print(summary)
                print("="*80 + "\n")

            processing_time = time.time() - start_time

            logger.info(f"Successfully processed contract: {file_path.name} in {processing_time:.2f}s")

            # Create TokenUsage object
            token_usage_obj = TokenUsage(**token_usage)

            return ContractExtractionResult(
                success=True,
                data=contract_info,
                processing_time=processing_time,
                source_file=str(file_path.name),
                token_usage=token_usage_obj
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            logger.error(error_msg)
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

    def create_unit_specific_contracts(
        self,
        base_contract: ContractInfo,
        unit_breakdown_result: UnitBreakdownResult,
        validate_gfa: bool = True,
        gfa_tolerance: float = 0.01
    ) -> List[ContractInfo]:
        """
        Create individual contract records for each unit from the breakdown file.

        Args:
            base_contract: The base contract extracted from PDF (without unit specified)
            unit_breakdown_result: Result from UnitBreakdownReader containing unit list
            validate_gfa: Whether to validate total GFA matches contract GLA (default: True)
            gfa_tolerance: Tolerance percentage for GFA validation (default: 0.01 = 1%)

        Returns:
            List of ContractInfo objects, one for each unit
        """
        logger.info("Creating unit-specific contracts from breakdown")

        if not unit_breakdown_result.success or not unit_breakdown_result.units:
            logger.error("Invalid unit breakdown result")
            return []

        # Validate GFA if requested
        if validate_gfa and base_contract.gla_for_lease:
            try:
                contract_gla = float(base_contract.gla_for_lease)
                reader = UnitBreakdownReader()
                validation = reader.validate_gfa_match(
                    unit_breakdown_result,
                    contract_gla,
                    tolerance=gfa_tolerance
                )

                if not validation['valid']:
                    logger.warning(
                        f"GFA validation failed: Contract GLA ({validation['contract_gla']}) "
                        f"does not match unit breakdown total ({validation['total_unit_gfa']}). "
                        f"Difference: {validation['difference']} ({validation['difference_pct']:.2f}%)"
                    )
                    # Continue anyway but log the warning
            except Exception as e:
                logger.warning(f"Could not validate GFA: {e}")

        # Create contracts for each unit
        unit_contracts = []

        for unit in unit_breakdown_result.units:
            # Create a copy of the base contract
            contract_dict = base_contract.model_dump()

            # Update unit-specific fields
            contract_dict['unit_for_lease'] = unit.unit
            contract_dict['gfa'] = str(unit.gfa)  # Use unit's specific GFA

            # Optionally update customer information if different in breakdown
            if unit.customer_code:
                # Could store this in a custom field or notes if needed
                pass

            # Update rate_periods to recalculate total_monthly_rate based on unit GFA
            if contract_dict.get('rate_periods'):
                updated_periods = []
                for period_data in contract_dict['rate_periods']:
                    # Convert RatePeriod object to dict if needed
                    if isinstance(period_data, RatePeriod):
                        period_dict = period_data.model_dump()
                    else:
                        period_dict = period_data

                    # Recalculate total_monthly_rate for this unit
                    if period_dict.get('monthly_rate_per_sqm'):
                        try:
                            rate_per_sqm = float(period_dict['monthly_rate_per_sqm'])
                            total_monthly = rate_per_sqm * unit.gfa
                            period_dict['total_monthly_rate'] = str(round(total_monthly, 2))
                            logger.debug(
                                f"Unit {unit.unit}: Recalculated total_monthly_rate = "
                                f"{rate_per_sqm} × {unit.gfa} = {period_dict['total_monthly_rate']}"
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not recalculate total_monthly_rate for unit {unit.unit}: {e}")

                    updated_periods.append(RatePeriod(**period_dict))

                contract_dict['rate_periods'] = updated_periods

            # Recalculate service charge if applicable
            if contract_dict.get('service_charge_rate') and contract_dict.get('service_charge_applies_to'):
                service_charge_total = self._calculate_service_charge(contract_dict)
                if service_charge_total:
                    contract_dict['service_charge_total'] = service_charge_total
                    logger.debug(f"Unit {unit.unit}: Recalculated service_charge_total = {service_charge_total}")

            # Create new ContractInfo object
            unit_contract = ContractInfo(**contract_dict)
            unit_contracts.append(unit_contract)

            logger.info(f"Created contract for unit: {unit.unit} with GFA: {unit.gfa}")

        logger.info(f"Successfully created {len(unit_contracts)} unit-specific contracts")

        # Print summary
        print("\n" + "="*80)
        print(f"📋 CREATED {len(unit_contracts)} UNIT-SPECIFIC CONTRACTS:")
        print("="*80)
        for i, (unit, contract) in enumerate(zip(unit_breakdown_result.units, unit_contracts), 1):
            print(f"{i}. Unit: {contract.unit_for_lease} | GFA: {contract.gfa} sqm")
        print("="*80 + "\n")

        return unit_contracts

    def process_contract_with_unit_breakdown(
        self,
        contract_file_path: Union[str, Path],
        unit_breakdown_file_path: Union[str, Path],
        validate_gfa: bool = True,
        gfa_tolerance: float = 0.01
    ) -> Dict[str, any]:
        """
        Process a contract PDF and create unit-specific contracts from breakdown Excel.

        Args:
            contract_file_path: Path to the contract PDF
            unit_breakdown_file_path: Path to the unit breakdown Excel file
            validate_gfa: Whether to validate total GFA matches contract GLA
            gfa_tolerance: Tolerance percentage for GFA validation

        Returns:
            Dictionary containing:
                - base_result: ContractExtractionResult from PDF
                - breakdown_result: UnitBreakdownResult from Excel
                - unit_contracts: List of unit-specific ContractInfo objects
                - success: Overall success status
        """
        logger.info("Processing contract with unit breakdown")
        logger.info(f"Contract file: {contract_file_path}")
        logger.info(f"Unit breakdown file: {unit_breakdown_file_path}")

        # Step 1: Process the contract PDF
        base_result = self.process_contract(contract_file_path)

        if not base_result.success:
            logger.error("Failed to process contract PDF")
            return {
                'success': False,
                'base_result': base_result,
                'breakdown_result': None,
                'unit_contracts': [],
                'error': 'Failed to process contract PDF'
            }

        # Step 2: Read unit breakdown Excel
        reader = UnitBreakdownReader()
        breakdown_result = reader.read_unit_breakdown(unit_breakdown_file_path)

        if not breakdown_result.success:
            logger.error("Failed to read unit breakdown Excel")
            return {
                'success': False,
                'base_result': base_result,
                'breakdown_result': breakdown_result,
                'unit_contracts': [],
                'error': 'Failed to read unit breakdown Excel'
            }

        # Step 3: Create unit-specific contracts
        unit_contracts = self.create_unit_specific_contracts(
            base_result.data,
            breakdown_result,
            validate_gfa=validate_gfa,
            gfa_tolerance=gfa_tolerance
        )

        success = len(unit_contracts) > 0

        logger.info(f"Contract processing with unit breakdown complete. Success: {success}")

        return {
            'success': success,
            'base_result': base_result,
            'breakdown_result': breakdown_result,
            'unit_contracts': unit_contracts
        }
