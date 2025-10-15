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

from ..models.contract_schemas import ContractInfo, ContractExtractionResult, Party
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ContractOCRService:
    """Service for reading and extracting information from contract documents using OpenAI's vision API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OCR service with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.pdf_max_pages = 10  # Maximum pages to process from PDF

        logger.info(f"Contract OCR Service initialized with model: {self.model}")

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _pdf_to_images(self, pdf_path: Union[str, Path]) -> List[str]:
        """
        Convert PDF pages to base64-encoded images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of base64-encoded image strings
        """
        pdf_document = fitz.open(pdf_path)
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Render page to image (higher DPI for better OCR)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            images.append(base64_image)

        pdf_document.close()
        return images

    def _create_extraction_prompt(self) -> str:
        """Create the prompt for information extraction."""
        return f"""You are an expert contract analyst specializing in lease and commercial contracts. Analyze this contract document image and extract the following information in JSON format:

GENERAL CONTRACT FIELDS (extract if available):
1. contract_title: The title or name of the contract
2. contract_type: Type of contract (e.g., Service Agreement, NDA, Lease, Purchase Agreement)
3. parties_involved: List of all parties with their names, roles, and addresses (format: {{"name": "...", "role": "...", "address": "..."}})
4. effective_date: When the contract becomes effective
5. expiration_date: When the contract expires
6. contract_value: Total monetary value
7. payment_terms: Payment schedule and terms
8. key_obligations: Main obligations of each party (as a list)
9. termination_clauses: Conditions for contract termination
10. governing_law: Jurisdiction or law governing the contract
11. signatures_present: Whether signatures are visible (true/false)
12. special_conditions: Any special conditions or clauses (as a list)

LEASE-SPECIFIC FIELDS (extract if this is a lease contract and the information is available):
13. internal_id: Internal reference ID
14. id: ID number
15. historical: Whether this is a historical record (true/false)
16. master_record_id: Master Record ID
17. plc_id: PLC ID
18. unit_for_lease: The unit/property being leased (unit number, description)
19. type: Type of lease
20. start_date: Lease start date
21. end_date: Lease end date
22. tenant: Tenant name
23. monthly_rate_per_sqm: Monthly rate per square meter
24. gla_for_lease: GLA (Gross Leasable Area) for lease
25. total_monthly_rate: Total monthly rate amount
26. months: Total number of months
27. total_rate: Total rate for entire lease period
28. status: Status of the lease
29. historical_journal_entry: Historical journal entry reference
30. amortization_journal_entry: Amortization journal entry reference
31. related_billing_schedule: Related billing schedule reference
32. subsidiary: Subsidiary information
33. ccs_product_type: CCS Product Type
34. bwid_project: BWID Project name/reference
35. phase: Phase information

IMPORTANT INSTRUCTIONS:
- Return the information as a valid JSON object
- If a field cannot be determined or does not exist in the document, use null for strings/objects, false for booleans, or empty arrays for lists
- Leave fields blank (null) rather than guessing
- Extract all monetary values with their currency symbols
- Extract all dates in a consistent format (YYYY-MM-DD preferred)
- For lease contracts, pay special attention to all lease-specific fields

Read the entire document carefully and extract all relevant information."""

    def process_contract(self, file_path: Union[str, Path]) -> ContractExtractionResult:
        """
        Read and extract information from a contract document.

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
            # Handle PDF conversion to images
            if file_path.suffix.lower() == '.pdf':
                logger.info(f"Converting PDF to images...")
                all_pdf_images = self._pdf_to_images(file_path)
                total_pages = len(all_pdf_images)
                logger.info(f"Found {total_pages} page(s)")

                # Limit pages to process
                pdf_images = all_pdf_images[:self.pdf_max_pages]
                if self.pdf_max_pages < total_pages:
                    logger.info(f"Processing first {self.pdf_max_pages} page(s) only")

                # Prepare multi-page content for the API
                prompt = self._create_extraction_prompt()

                # Build content with all pages as images
                message_content = [{"type": "text", "text": prompt}]
                for page_num, base64_image in enumerate(pdf_images, 1):
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })

                logger.info(f"Analyzing {len(pdf_images)} page(s) with OpenAI...")

                # Call OpenAI API once with all pages
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": message_content
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.1,
                )
                content = response.choices[0].message.content

            else:
                # Encode image
                logger.info(f"Processing image file...")
                base64_image = self._encode_image(file_path)

                # Create prompt
                prompt = self._create_extraction_prompt()

                # Call OpenAI API with vision
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.1,  # Low temperature for more consistent extraction
                )
                content = response.choices[0].message.content

            # Parse JSON from response
            # Sometimes the model wraps JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            extracted_data = json.loads(content)

            # Convert parties_involved to Party objects
            parties = []
            if "parties_involved" in extracted_data and extracted_data["parties_involved"]:
                for party_data in extracted_data["parties_involved"]:
                    parties.append(Party(**party_data))
            extracted_data["parties_involved"] = parties

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
