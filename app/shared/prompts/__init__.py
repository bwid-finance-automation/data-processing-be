"""
Centralized AI Prompts Module
=============================

All AI prompts organized by department for easy management and maintenance.

Departments:
- FPA (Financial Planning & Analysis): GLA variance, leasing analysis
- Finance: Contract OCR, financial document processing
"""

from .fpa_prompts import (
    # GLA Variance Analysis
    GLA_VARIANCE_SYSTEM_PROMPT,
    GLA_NOTES_SYSTEM_PROMPT,
    GLA_FILE_STRUCTURE_SYSTEM_PROMPT,
    format_variance_data,
    get_variance_user_prompt,
    get_notes_user_prompt,
    get_file_structure_user_prompt,
    # NTM EBITDA Variance Analysis
    NTM_EBITDA_SYSTEM_PROMPT,
    NTM_COMMENTARY_SYSTEM_PROMPT,
    get_ntm_commentary_prompt,
)

from .finance_prompts import (
    # Contract OCR
    CONTRACT_EXTRACTION_SYSTEM_PROMPT,
    get_contract_extraction_prompt,
    # Variance Analysis (22 Rules)
    VARIANCE_ANALYSIS_22_RULES_SYSTEM_PROMPT,
    VARIANCE_ANALYSIS_SYSTEM_PROMPT,
    # Sheet Detection
    SHEET_DETECTION_SYSTEM_PROMPT,
    create_sheet_detection_prompt,
    # Account Extraction
    ACCOUNT_EXTRACTION_SYSTEM_PROMPT,
    create_account_extraction_prompt,
    # Consolidation
    CONSOLIDATION_SYSTEM_PROMPT,
    # Revenue Analysis
    REVENUE_ANALYSIS_SYSTEM_PROMPT,
    create_revenue_analysis_prompt,
)

__all__ = [
    # FPA Prompts - GLA Variance
    "GLA_VARIANCE_SYSTEM_PROMPT",
    "GLA_NOTES_SYSTEM_PROMPT",
    "GLA_FILE_STRUCTURE_SYSTEM_PROMPT",
    "format_variance_data",
    "get_variance_user_prompt",
    "get_notes_user_prompt",
    "get_file_structure_user_prompt",
    # FPA Prompts - NTM EBITDA
    "NTM_EBITDA_SYSTEM_PROMPT",
    "NTM_COMMENTARY_SYSTEM_PROMPT",
    "get_ntm_commentary_prompt",
    # Finance Prompts - Contract OCR
    "CONTRACT_EXTRACTION_SYSTEM_PROMPT",
    "get_contract_extraction_prompt",
    # Finance Prompts - Variance Analysis
    "VARIANCE_ANALYSIS_22_RULES_SYSTEM_PROMPT",
    "VARIANCE_ANALYSIS_SYSTEM_PROMPT",
    # Finance Prompts - Sheet Detection
    "SHEET_DETECTION_SYSTEM_PROMPT",
    "create_sheet_detection_prompt",
    # Finance Prompts - Account Extraction
    "ACCOUNT_EXTRACTION_SYSTEM_PROMPT",
    "create_account_extraction_prompt",
    # Finance Prompts - Consolidation
    "CONSOLIDATION_SYSTEM_PROMPT",
    # Finance Prompts - Revenue Analysis
    "REVENUE_ANALYSIS_SYSTEM_PROMPT",
    "create_revenue_analysis_prompt",
]
