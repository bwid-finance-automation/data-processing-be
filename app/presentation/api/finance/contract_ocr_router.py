"""Contract OCR API endpoints."""
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from app.presentation.schemas.contract_schemas import (
    ContractExtractionResult,
    BatchContractResult,
    SupportedFormatsResponse,
    TokenUsage,
    ContractWithUnitsResult,
    UnitBreakdownSummary,
    UnitBreakdownInfo
)
from app.application.finance.contract_ocr.process_contracts import ContractOCRService
from app.application.finance.contract_ocr.contract_excel_export import ContractExcelExporter
from app.application.finance.contract_ocr.unit_breakdown_reader import UnitBreakdownReader
from app.shared.utils.logging_config import get_logger

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

        # Aggregate token usage and costs across all contracts
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        for result in results:
            if result.token_usage:
                total_prompt_tokens += result.token_usage.prompt_tokens
                total_completion_tokens += result.token_usage.completion_tokens
                total_tokens += result.token_usage.total_tokens

        # Create aggregated token usage object
        aggregated_token_usage = None
        if total_tokens > 0:
            aggregated_token_usage = TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens
            )
            logger.info(f"Total token usage for batch - Prompt: {total_prompt_tokens}, "
                       f"Completion: {total_completion_tokens}, Total: {total_tokens}")

        logger.info(f"Batch processing complete: {successful}/{len(results)} successful")

        return BatchContractResult(
            success=True,
            total_files=len(results),
            successful=successful,
            failed=failed,
            results=results,
            total_token_usage=aggregated_token_usage
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

        # Calculate totals for successful contracts
        successful_results = [r for r in results if r.success]
        total_prompt_tokens = sum(r.token_usage.prompt_tokens for r in successful_results if r.token_usage)
        total_completion_tokens = sum(r.token_usage.completion_tokens for r in successful_results if r.token_usage)
        total_tokens = sum(r.token_usage.total_tokens for r in successful_results if r.token_usage)

        # Print summary banner
        print("\n" + "="*80)
        print("✅ BATCH EXPORT COMPLETE - SUMMARY")
        print("="*80)
        print(f"📄 Contracts processed: {len(successful_results)} successful, {len(results) - len(successful_results)} failed")
        print(f"📊 Total contracts exported: {len(results)}")
        print("")
        print("📊 TOKEN USAGE:")
        print(f"   • Input tokens:      {total_prompt_tokens:,}")
        print(f"   • Output tokens:     {total_completion_tokens:,}")
        print(f"   • TOTAL TOKENS:      {total_tokens:,}")
        print("")
        print(f"📁 Output file: contract_extractions_{len(results)}_contracts.xlsx")
        print("="*80 + "\n")

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


@router.post("/process-contract-with-units", response_model=ContractWithUnitsResult)
async def process_contract_with_unit_breakdown(
    contract_file: UploadFile = File(..., description="Contract PDF/image file"),
    unit_breakdown_file: UploadFile = File(..., description="Unit breakdown Excel file (.xlsx)")
):
    """
    Process a contract PDF and create individual contracts for each unit from breakdown Excel.

    This endpoint is useful when a single contract covers multiple units, and you have a separate
    Excel file that lists each unit with its GFA (Gross Floor Area).

    **Workflow:**
    1. Process the contract PDF to extract general contract information
    2. Read the unit breakdown Excel to get list of units and their GFAs
    3. Validate that sum of unit GFAs matches the contract's total GLA
    4. Create individual contract records for each unit (duplicating contract data with unit-specific GFA)

    **Required Excel columns:**
    - `Unit`: Unit code/name (e.g., "WA1.1", "WB1.2")
    - `GFA`: Gross Floor Area for this unit in sqm

    **Optional Excel columns:**
    - `Customer Code`: Customer code
    - `Customer Name`: Customer name
    - `Tax rate`: Tax rate

    **Returns:**
    - Base contract extraction result
    - Unit breakdown summary
    - Individual contracts for each unit (with recalculated totals)
    - GFA validation result
    """
    logger.info(f"Received contract with units request: {contract_file.filename}, {unit_breakdown_file.filename}")

    # Validate contract file format
    contract_ext = Path(contract_file.filename).suffix.lower()
    if contract_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported contract file format: {contract_ext}. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Validate Excel file format
    excel_ext = Path(unit_breakdown_file.filename).suffix.lower()
    if excel_ext not in ['.xlsx', '.xls']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Excel format: {excel_ext}. Use .xlsx or .xls"
        )

    contract_temp_path = None
    excel_temp_path = None

    try:
        # Save contract file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=contract_ext) as temp_file:
            content = await contract_file.read()
            temp_file.write(content)
            contract_temp_path = temp_file.name

        logger.info(f"Saved contract temp file: {contract_temp_path}")

        # Save Excel file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=excel_ext) as temp_file:
            content = await unit_breakdown_file.read()
            temp_file.write(content)
            excel_temp_path = temp_file.name

        logger.info(f"Saved Excel temp file: {excel_temp_path}")

        # Process contract with unit breakdown
        ocr_service = get_ocr_service()
        result = ocr_service.process_contract_with_unit_breakdown(
            contract_file_path=contract_temp_path,
            unit_breakdown_file_path=excel_temp_path,
            validate_gfa=True,
            gfa_tolerance=0.01  # 1% tolerance
        )

        # Clean up temp files
        for temp_path in [contract_temp_path, excel_temp_path]:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

        if not result['success']:
            error_msg = result.get('error', 'Unknown error processing contract with units')
            logger.error(f"Processing failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        # Convert result to response model
        unit_breakdown_summary = None
        if result['breakdown_result'] and result['breakdown_result'].success:
            unit_breakdown_summary = UnitBreakdownSummary(
                success=result['breakdown_result'].success,
                total_units=len(result['breakdown_result'].units),
                total_gfa=result['breakdown_result'].total_gfa,
                units=[
                    UnitBreakdownInfo(
                        customer_code=unit.customer_code,
                        customer_name=unit.customer_name,
                        tax_rate=unit.tax_rate,
                        unit=unit.unit,
                        gfa=unit.gfa
                    )
                    for unit in result['breakdown_result'].units
                ],
                error=result['breakdown_result'].error
            )

        # Create GFA validation info
        gfa_validation = None
        if result['base_result'] and result['base_result'].data and result['breakdown_result']:
            try:
                contract_gla = float(result['base_result'].data.gla_for_lease) if result['base_result'].data.gla_for_lease else None
                if contract_gla and result['breakdown_result'].total_gfa:
                    reader = UnitBreakdownReader()
                    gfa_validation = reader.validate_gfa_match(
                        result['breakdown_result'],
                        contract_gla,
                        tolerance=0.01
                    )
            except Exception as e:
                logger.warning(f"Could not create GFA validation: {e}")

        response = ContractWithUnitsResult(
            success=result['success'],
            base_contract=result['base_result'],
            unit_breakdown=unit_breakdown_summary,
            unit_contracts=result['unit_contracts'],
            total_units=len(result['unit_contracts']),
            gfa_validation=gfa_validation,
            error=result.get('error')
        )

        logger.info(f"Successfully processed contract with {len(result['unit_contracts'])} units")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing contract with units: {e}", exc_info=True)

        # Clean up temp files on error
        for temp_path in [contract_temp_path, excel_temp_path]:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process contract with units: {str(e)}"
        )


@router.post("/export-contract-with-units-to-excel")
async def export_contract_with_units_to_excel(
    contract_file: UploadFile = File(..., description="Contract PDF/image file"),
    unit_breakdown_file: UploadFile = File(..., description="Unit breakdown Excel file (.xlsx)")
):
    """
    Process a contract with unit breakdown and export unit-specific contracts to Excel.

    This endpoint combines contract processing with unit breakdown and Excel export:
    1. Processes the contract PDF to extract general contract information
    2. Reads the unit breakdown Excel to get list of units and their GFAs
    3. Creates individual contract records for each unit
    4. Exports all unit-specific contracts to Excel (one row per unit per rate period)

    **Required Excel columns:**
    - `Unit`: Unit code/name (e.g., "WA1.1", "WB1.2")
    - `GFA`: Gross Floor Area for this unit in sqm

    **Optional Excel columns:**
    - `Customer Code`: Customer code
    - `Customer Name`: Customer name
    - `Tax rate`: Tax rate

    **Returns:**
    - Excel file with one row per unit per rate period
    - Unit column properly filled from breakdown Excel
    - All rate calculations based on unit-specific GFA
    """
    logger.info(f"Received Excel export with units request: {contract_file.filename}, {unit_breakdown_file.filename}")

    # Validate contract file format
    contract_ext = Path(contract_file.filename).suffix.lower()
    if contract_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported contract file format: {contract_ext}. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Validate Excel file format
    excel_ext = Path(unit_breakdown_file.filename).suffix.lower()
    if excel_ext not in ['.xlsx', '.xls']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Excel format: {excel_ext}. Use .xlsx or .xls"
        )

    contract_temp_path = None
    excel_temp_path = None
    output_excel_path = None

    try:
        # Save contract file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=contract_ext) as temp_file:
            content = await contract_file.read()
            temp_file.write(content)
            contract_temp_path = temp_file.name

        logger.info(f"Saved contract temp file: {contract_temp_path}")

        # Save Excel file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=excel_ext) as temp_file:
            content = await unit_breakdown_file.read()
            temp_file.write(content)
            excel_temp_path = temp_file.name

        logger.info(f"Saved Excel temp file: {excel_temp_path}")

        # Process contract with unit breakdown
        ocr_service = get_ocr_service()
        result = ocr_service.process_contract_with_unit_breakdown(
            contract_file_path=contract_temp_path,
            unit_breakdown_file_path=excel_temp_path,
            validate_gfa=True,
            gfa_tolerance=0.01  # 1% tolerance
        )

        if not result['success']:
            error_msg = result.get('error', 'Unknown error processing contract with units')
            logger.error(f"Processing failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        # Convert unit contracts to ContractExtractionResult objects for Excel export
        unit_contracts = result['unit_contracts']

        if not unit_contracts:
            raise HTTPException(
                status_code=500,
                detail="No unit-specific contracts were created"
            )

        # Create ContractExtractionResult objects from unit contracts
        extraction_results = []
        for unit_contract in unit_contracts:
            extraction_result = ContractExtractionResult(
                success=True,
                data=unit_contract,
                processing_time=result['base_result'].processing_time if result['base_result'] else None,
                source_file=f"{contract_file.filename} - Unit {unit_contract.unit_for_lease}",
                token_usage=result['base_result'].token_usage if result['base_result'] else None
            )
            extraction_results.append(extraction_result)

        logger.info(f"Created {len(extraction_results)} extraction results for Excel export")

        # Export to Excel
        exporter = ContractExcelExporter()

        # Create temp Excel file for output
        excel_output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        output_excel_path = Path(excel_output_temp.name)
        excel_output_temp.close()

        # Export results
        exporter.export_to_excel(extraction_results, output_excel_path, include_failed=False)

        logger.info(f"Successfully exported {len(extraction_results)} unit contracts to Excel")

        # Print summary with token usage
        if result['base_result'] and result['base_result'].token_usage:
            print("\n" + "="*80)
            print("✅ EXPORT COMPLETE - SUMMARY")
            print("="*80)
            print(f"📄 Units exported: {len(unit_contracts)}")
            print(f"📊 Total rows in Excel: {len(unit_contracts) * len(result['base_result'].data.rate_periods) if result['base_result'].data.rate_periods else len(unit_contracts)}")
            print("")
            print("📊 TOKEN USAGE:")
            print(f"   • Input tokens:      {result['base_result'].token_usage.prompt_tokens:,}")
            print(f"   • Output tokens:     {result['base_result'].token_usage.completion_tokens:,}")
            print(f"   • TOTAL TOKENS:      {result['base_result'].token_usage.total_tokens:,}")
            print("")
            print(f"📁 Output file: contract_with_units_{len(unit_contracts)}_units.xlsx")
            print("="*80 + "\n")

        # Clean up input temp files
        for temp_path in [contract_temp_path, excel_temp_path]:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

        # Return the Excel file
        return FileResponse(
            path=str(output_excel_path),
            filename=f"contract_with_units_{len(unit_contracts)}_units.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=None  # File will be deleted after response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Excel export with units: {e}", exc_info=True)

        # Clean up temp files on error
        for temp_path in [contract_temp_path, excel_temp_path, output_excel_path]:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        raise HTTPException(
            status_code=500,
            detail=f"Excel export with units failed: {str(e)}"
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
            "Contract with unit breakdown (NEW)",
            "Excel export (normalized format)",
            "Excel export with unit breakdown (NEW)",
            "42+ fields extraction",
            "Multilingual support (Vietnamese/English/Chinese)",
            "Service charge calculation",
            "One row per rate period export",
            "Lease-specific fields",
            "PDF and image support",
            "GFA validation"
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
            "process_with_units": "POST /api/v1/contract-ocr/process-contract-with-units",
            "export_to_excel": "POST /api/v1/contract-ocr/export-to-excel",
            "export_with_units_to_excel": "POST /api/v1/contract-ocr/export-contract-with-units-to-excel"
        }
    }
