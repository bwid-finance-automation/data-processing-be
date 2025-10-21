"""Contract OCR API endpoints."""
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from ..models.contract_schemas import (
    ContractExtractionResult,
    BatchContractResult,
    SupportedFormatsResponse
)
from ..services.contract_ocr_service import ContractOCRService
from ..services.contract_excel_export import ContractExcelExporter
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/contract-ocr", tags=["Contract OCR"])

# Supported file formats
SUPPORTED_FORMATS = [".pdf", ".png", ".jpg", ".jpeg"]


def get_ocr_service() -> ContractOCRService:
    """Get or create OCR service instance."""
    try:
        return ContractOCRService()
    except ValueError as e:
        logger.error(f"Failed to initialize OCR service: {e}")
        raise HTTPException(
            status_code=500,
            detail="OCR service not available. Please check OpenAI API key configuration."
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for Contract OCR service."""
    return {
        "status": "healthy",
        "service": "Contract OCR",
        "supported_formats": SUPPORTED_FORMATS
    }


@router.get("/supported-formats", response_model=SupportedFormatsResponse)
async def get_supported_formats():
    """Get list of supported file formats."""
    return SupportedFormatsResponse(
        formats=SUPPORTED_FORMATS,
        description="Supported file formats for contract documents"
    )


@router.post("/process-contract", response_model=ContractExtractionResult)
async def process_single_contract(file: UploadFile = File(...)):
    """
    Process a single contract document and extract information.

    - **file**: Contract document (PDF, PNG, JPG, JPEG)

    Returns extracted contract information including:
    - General contract fields (title, type, parties, dates, etc.)
    - Lease-specific fields (if applicable)
    - Processing metadata
    """
    logger.info(f"Received contract processing request: {file.filename}")

    # Validate file format
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Save uploaded file to temp directory
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Saved temporary file: {temp_file_path}")

        # Process the contract
        ocr_service = get_ocr_service()
        result = ocr_service.process_contract(temp_file_path)

        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")

        if not result.success:
            logger.error(f"Contract processing failed: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)

        logger.info(f"Successfully processed contract: {file.filename}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing contract: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process contract: {str(e)}"
        )


@router.post("/process-contracts-batch", response_model=BatchContractResult)
async def process_multiple_contracts(files: List[UploadFile] = File(...)):
    """
    Process multiple contract documents in batch.

    - **files**: List of contract documents (PDF, PNG, JPG, JPEG)

    Returns:
    - Summary of batch processing (total, successful, failed)
    - Individual results for each contract
    """
    logger.info(f"Received batch processing request: {len(files)} files")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_files = []
    results = []

    try:
        # Save all uploaded files to temp directory
        for file in files:
            file_ext = Path(file.filename).suffix.lower()

            # Validate file format
            if file_ext not in SUPPORTED_FORMATS:
                results.append(
                    ContractExtractionResult(
                        success=False,
                        error=f"Unsupported file format: {file_ext}",
                        source_file=file.filename
                    )
                )
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append((temp_file.name, file.filename))

        # Process all valid contracts
        if temp_files:
            ocr_service = get_ocr_service()
            file_paths = [temp_path for temp_path, _ in temp_files]
            processing_results = ocr_service.process_contracts_batch(file_paths)

            # Update source file names
            for result, (_, original_name) in zip(processing_results, temp_files):
                result.source_file = original_name

            results.extend(processing_results)

        # Clean up temp files
        for temp_path, _ in temp_files:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")

        # Calculate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        logger.info(f"Batch processing complete: {successful}/{len(results)} successful")

        return BatchContractResult(
            success=True,
            total_files=len(results),
            successful=successful,
            failed=failed,
            results=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)

        # Clean up temp files on error
        for temp_path, _ in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.post("/export-to-excel")
async def export_contracts_to_excel(files: List[UploadFile] = File(...)):
    """
    Process multiple contracts and export results to Excel.

    - **files**: List of contract documents (PDF, PNG, JPG, JPEG)

    Returns:
    - Excel file with one row per rate period (normalized format)
    - Includes all 42+ extracted fields
    - Multilingual support (Vietnamese/English/Chinese)
    """
    logger.info(f"Received Excel export request: {len(files)} files")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_files = []
    results = []

    try:
        # Save all uploaded files to temp directory
        for file in files:
            file_ext = Path(file.filename).suffix.lower()

            # Validate file format
            if file_ext not in SUPPORTED_FORMATS:
                results.append(
                    ContractExtractionResult(
                        success=False,
                        error=f"Unsupported file format: {file_ext}",
                        source_file=file.filename
                    )
                )
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append((temp_file.name, file.filename))

        # Process all valid contracts
        if temp_files:
            ocr_service = get_ocr_service()
            file_paths = [temp_path for temp_path, _ in temp_files]
            processing_results = ocr_service.process_contracts_batch(file_paths)

            # Update source file names
            for result, (_, original_name) in zip(processing_results, temp_files):
                result.source_file = original_name

            results.extend(processing_results)

        # Clean up temp contract files
        for temp_path, _ in temp_files:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")

        # Export to Excel
        exporter = ContractExcelExporter()

        # Create temp Excel file
        excel_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        excel_path = Path(excel_temp.name)
        excel_temp.close()

        # Export results
        exporter.export_to_excel(results, excel_path, include_failed=True)

        # Return the Excel file
        logger.info(f"Returning Excel file: {excel_path}")

        return FileResponse(
            path=str(excel_path),
            filename=f"contract_extractions_{len(results)}_contracts.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=None  # File will be deleted after response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Excel export: {e}", exc_info=True)

        # Clean up temp files on error
        for temp_path, _ in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"Excel export failed: {str(e)}"
        )


@router.get("/")
async def contract_ocr_info():
    """Get Contract OCR API information."""
    return {
        "service": "Contract OCR API",
        "version": "2.0.0",
        "description": "Extract information from multilingual contract documents using AI-powered OCR",
        "features": [
            "Single contract processing",
            "Batch contract processing",
            "Excel export (normalized format)",
            "42+ fields extraction",
            "Multilingual support (Vietnamese/English/Chinese)",
            "Service charge calculation",
            "One row per rate period export",
            "Lease-specific fields",
            "PDF and image support"
        ],
        "new_fields": [
            "customer_name",
            "contract_number",
            "contract_date",
            "payment_terms_details",
            "deposit_amount",
            "handover_date",
            "gfa (Gross Floor Area)",
            "service_charge_rate",
            "service_charge_applies_to",
            "service_charge_total (calculated)"
        ],
        "supported_formats": SUPPORTED_FORMATS,
        "endpoints": {
            "health": "GET /api/v1/contract-ocr/health",
            "supported_formats": "GET /api/v1/contract-ocr/supported-formats",
            "process_single": "POST /api/v1/contract-ocr/process-contract",
            "process_batch": "POST /api/v1/contract-ocr/process-contracts-batch",
            "export_to_excel": "POST /api/v1/contract-ocr/export-to-excel"
        }
    }
