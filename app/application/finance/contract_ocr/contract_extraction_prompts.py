"""
Prompts for contract OCR extraction service.

This module re-exports from the centralized prompts module for backward compatibility.
All prompts are now managed in app/shared/prompts/finance_prompts.py
"""

from app.shared.prompts import get_contract_extraction_prompt

__all__ = ["get_contract_extraction_prompt"]
