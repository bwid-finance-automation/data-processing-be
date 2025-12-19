"""
GLA AI Analyzer Service
Uses LLM to analyze GLA variances and provide business explanations.
"""
import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
import pathlib
project_root = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

# Try to import AI clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from app.shared.utils.logging_config import get_logger
from app.shared.prompts import (
    GLA_VARIANCE_SYSTEM_PROMPT,
    GLA_NOTES_SYSTEM_PROMPT,
    GLA_FILE_STRUCTURE_SYSTEM_PROMPT,
    format_variance_data,
    get_variance_user_prompt,
    get_notes_user_prompt,
    get_file_structure_user_prompt,
)
from ..models.gla_models import GLAVarianceResult, GLAAnalysisSummary, TenantChange

logger = get_logger(__name__)


class GLAAIAnalyzer:
    """
    AI-powered analyzer for GLA variance data.
    Provides business context and explanations for GLA changes.
    """

    # Supported models
    # To switch providers: set AI_PROVIDER=anthropic or AI_PROVIDER=openai in .env
    # To change model: set ANTHROPIC_MODEL or OPENAI_MODEL in .env
    CLAUDE_MODELS = [
        "claude-opus-4-5-20251101",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
    ]

    GPT_MODELS = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    def __init__(self, progress_callback=None):
        """Initialize the AI analyzer with configured provider."""
        self.progress_callback = progress_callback

        # Determine AI provider
        self.ai_provider = os.getenv("AI_PROVIDER", "openai").lower()

        if self.ai_provider == "anthropic":
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.api_key = os.getenv("OPENAI_API_KEY")

        self.is_claude = self.model.startswith("claude")

        # Initialize client
        self.client = None
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                if self.is_claude and ANTHROPIC_AVAILABLE:
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                    logger.info(f"Initialized Anthropic client with model: {self.model}")
                elif OPENAI_AVAILABLE:
                    self.client = OpenAI(api_key=self.api_key, timeout=120.0)
                    logger.info(f"Initialized OpenAI client with model: {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize AI client: {e}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for GLA variance analysis."""
        return GLA_VARIANCE_SYSTEM_PROMPT

    def _format_variance_data(self, summary: GLAAnalysisSummary) -> str:
        """Format variance data for the AI prompt."""
        return format_variance_data(summary)

    def _get_user_prompt(self, summary: GLAAnalysisSummary) -> str:
        """Generate the user prompt with variance data."""
        return get_variance_user_prompt(summary)

    def analyze_variance(
        self,
        summary: GLAAnalysisSummary,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Analyze GLA variance using AI and generate explanations.

        Args:
            summary: GLAAnalysisSummary with variance results
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with AI analysis results
        """
        if not self.client:
            logger.warning("AI client not initialized, returning basic analysis")
            return self._generate_basic_analysis(summary)

        callback = progress_callback or self.progress_callback

        try:
            if callback:
                callback(10, "Preparing data for AI analysis...")

            system_prompt = self._get_system_prompt()
            user_prompt = self._get_user_prompt(summary)

            if callback:
                callback(20, f"Sending to {self.model} for analysis...")

            # Call AI
            if self.is_claude:
                response = self._call_claude(system_prompt, user_prompt)
            else:
                response = self._call_openai(system_prompt, user_prompt)

            if callback:
                callback(80, "Processing AI response...")

            # Parse response
            analysis_text = response.get("content", "")

            if callback:
                callback(90, "Analysis complete!")

            return {
                "status": "success",
                "analysis": analysis_text,
                "model": self.model,
                "provider": "anthropic" if self.is_claude else "openai"
            }

        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "analysis": self._generate_basic_analysis(summary).get("analysis", ""),
                "model": self.model
            }

    def _call_openai(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        try:
            # Build parameters
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 4000,
            }

            # GPT-5 doesn't support temperature
            if not self.model.startswith("gpt-5"):
                params["temperature"] = 0.3

            response = self.client.chat.completions.create(**params)

            return {
                "content": response.choices[0].message.content,
                "tokens": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens
                }
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _call_claude(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            return {
                "content": response.content[0].text,
                "tokens": {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _generate_basic_analysis(self, summary: GLAAnalysisSummary) -> Dict[str, Any]:
        """Generate basic analysis without AI when API is not available."""
        lines = []
        lines.append("## GLA Variance Analysis (Basic Mode)")
        lines.append("")
        lines.append("*Note: AI analysis not available. Showing automated summary.*")
        lines.append("")

        # Executive summary
        lines.append("### Executive Summary")
        total_committed = summary.total_portfolio_committed_variance
        total_handover = summary.total_portfolio_handover_variance

        if total_committed > 0:
            lines.append(f"- Portfolio Committed GLA **increased** by {total_committed:,.2f} sqm")
        else:
            lines.append(f"- Portfolio Committed GLA **decreased** by {abs(total_committed):,.2f} sqm")

        if total_handover > 0:
            lines.append(f"- Portfolio Handover GLA **increased** by {total_handover:,.2f} sqm")
        else:
            lines.append(f"- Portfolio Handover GLA **decreased** by {abs(total_handover):,.2f} sqm")

        lines.append("")

        # Significant changes
        lines.append("### Significant Changes")
        sorted_results = sorted(
            summary.results,
            key=lambda r: abs(r.committed_variance) + abs(r.handover_variance),
            reverse=True
        )

        count = 0
        for r in sorted_results[:10]:
            if r.committed_variance != 0 or r.handover_variance != 0:
                count += 1
                change_type = "increased" if (r.committed_variance + r.handover_variance) > 0 else "decreased"
                lines.append(f"- **{r.project_name}** ({r.product_type}, {r.region}): {change_type}")
                if r.committed_variance != 0:
                    lines.append(f"  - Committed: {r.committed_variance:+,.2f} sqm")
                if r.handover_variance != 0:
                    lines.append(f"  - Handover: {r.handover_variance:+,.2f} sqm")

        if count == 0:
            lines.append("- No significant changes detected")

        lines.append("")

        # Product type breakdown
        lines.append("### By Product Type")
        lines.append(f"- **RBF**: Committed {summary.total_rbf_committed_variance:+,.2f} sqm, Handover {summary.total_rbf_handover_variance:+,.2f} sqm")
        lines.append(f"- **RBW**: Committed {summary.total_rbw_committed_variance:+,.2f} sqm, Handover {summary.total_rbw_handover_variance:+,.2f} sqm")

        return {
            "status": "success",
            "analysis": "\n".join(lines),
            "model": "basic"
        }

    def generate_project_notes(
        self,
        results: List[GLAVarianceResult],
        progress_callback=None
    ) -> List[GLAVarianceResult]:
        """
        Generate AI-powered notes for each project's variance based on tenant changes.

        Args:
            results: List of GLAVarianceResult to annotate
            progress_callback: Optional callback for progress updates

        Returns:
            Updated list with notes populated
        """
        callback = progress_callback or self.progress_callback

        # Filter to only projects with variance
        projects_with_variance = [
            r for r in results
            if r.committed_variance != 0 or r.handover_variance != 0
        ]

        if not projects_with_variance:
            return results

        # First, generate notes from tenant data directly (no AI needed for basic info)
        for r in projects_with_variance:
            # Generate committed note from tenant changes
            if r.committed_variance != 0 and r.committed_tenant_changes:
                r.committed_note = self._format_tenant_changes(r.committed_tenant_changes)

            # Generate handover note from tenant changes
            if r.handover_variance != 0 and r.handover_tenant_changes:
                r.handover_note = self._format_tenant_changes(r.handover_tenant_changes)

        # If AI client available, enhance notes with context
        if self.client:
            try:
                if callback:
                    callback(30, f"Enhancing notes with AI for {len(projects_with_variance)} projects...")

                # Create a batch prompt for efficiency
                prompt = self._create_notes_prompt(projects_with_variance)

                if self.is_claude:
                    response = self._call_claude(GLA_NOTES_SYSTEM_PROMPT, prompt)
                else:
                    response = self._call_openai(GLA_NOTES_SYSTEM_PROMPT, prompt)

                # Parse JSON response
                notes = self._parse_notes_response(response.get("content", ""))

                # Apply AI-enhanced notes to results
                notes_map = {(n.get("project_name", ""), n.get("product_type", "")): n for n in notes}
                for r in results:
                    key = (r.project_name, r.product_type)
                    if key in notes_map:
                        note = notes_map[key]
                        if note.get("committed_note"):
                            r.committed_note = note["committed_note"]
                        if note.get("handover_note"):
                            r.handover_note = note["handover_note"]

                if callback:
                    callback(70, "Notes generated successfully")

            except Exception as e:
                logger.warning(f"AI note enhancement failed, using basic notes: {e}")

        return results

    def _format_tenant_changes(self, changes: List[TenantChange]) -> str:
        """Format tenant changes into a readable note string."""
        if not changes:
            return ""

        parts = []
        for change in changes[:3]:  # Limit to top 3 tenants
            tenant = change.tenant_name
            if len(tenant) > 20:
                tenant = tenant[:18] + ".."

            if change.change_type == "new":
                parts.append(f"{tenant} (new +{change.variance:,.0f})")
            elif change.change_type == "terminated":
                parts.append(f"{tenant} (term {change.variance:,.0f})")
            elif change.change_type == "expanded":
                parts.append(f"{tenant} (+{change.variance:,.0f})")
            elif change.change_type == "reduced":
                parts.append(f"{tenant} ({change.variance:,.0f})")

        return "; ".join(parts)

    def _create_notes_prompt(self, results: List[GLAVarianceResult]) -> str:
        """Create prompt for generating project notes with tenant details."""
        return get_notes_user_prompt(results)

    def _parse_notes_response(self, content: str) -> List[Dict[str, str]]:
        """Parse AI response for project notes."""
        try:
            # Try to extract JSON from response
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            # Find JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                content = content[start:end]

            return json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to parse notes JSON: {e}")
            return []

    def detect_file_structure(self, file_path: str, sheet_name: str = None) -> Dict[str, Any]:
        """
        Use AI to dynamically detect the structure of a GLA Excel file.

        This is a FULLY AI-POWERED detection that sends all relevant Excel data
        to the AI and receives complete structure analysis including all monthly
        column mappings.

        Args:
            file_path: Path to the Excel file
            sheet_name: Optional specific sheet to analyze

        Returns:
            Dict with detected structure including monthly_columns mapping
        """
        import pandas as pd

        logger.info(f"AI detecting file structure for: {file_path}")

        if not self.client:
            logger.error("No AI client available - AI detection requires an API key")
            raise ValueError("AI client required for file structure detection. Please configure OPENAI_API_KEY or ANTHROPIC_API_KEY.")

        try:
            xl = pd.ExcelFile(file_path)
            sheets = xl.sheet_names if sheet_name is None else [sheet_name]

            # Find target sheet
            target_sheet = None
            for s in sheets:
                if 'handover' in s.lower() or 'committed' in s.lower():
                    target_sheet = s
                    break
            if not target_sheet:
                target_sheet = sheets[0]

            # Read raw data for AI analysis
            df_raw = pd.read_excel(file_path, sheet_name=target_sheet, header=None, nrows=10)

            # Build comprehensive data for AI - include ALL date columns with headers
            ai_data = self._extract_complete_file_data(df_raw, file_path, target_sheet)

            # Send to AI for complete analysis
            result = self._ai_analyze_file_structure(ai_data)

            logger.info(f"AI detection complete: format={result.get('format')}, monthly_columns={len(result.get('monthly_columns', {}))}")
            return result

        except Exception as e:
            logger.error(f"AI file structure detection failed: {e}")
            raise

    def _extract_complete_file_data(self, df_raw, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """Extract complete file data for AI analysis - includes ALL date columns."""
        import pandas as pd
        import os

        data = {
            'file_name': os.path.basename(file_path),
            'sheet_name': sheet_name,
            'total_columns': len(df_raw.columns),
            'first_rows': [],
            'date_columns': []
        }

        # Extract first 6 rows (first 15 columns each)
        for row_idx in range(min(6, len(df_raw))):
            row_data = []
            for col_idx in range(min(15, len(df_raw.columns))):
                val = df_raw.iloc[row_idx, col_idx]
                if pd.notna(val):
                    row_data.append({'col': col_idx, 'value': str(val)[:30]})
            data['first_rows'].append({'row': row_idx, 'cells': row_data})

        # Find ALL date columns with their headers
        for row_idx in range(min(5, len(df_raw))):
            row = df_raw.iloc[row_idx]
            header_row = df_raw.iloc[row_idx + 1] if row_idx + 1 < len(df_raw) else None

            for col_idx, val in enumerate(row):
                if pd.notna(val) and hasattr(val, 'month'):
                    header = ""
                    if header_row is not None and col_idx < len(header_row):
                        header = str(header_row.iloc[col_idx]).strip() if pd.notna(header_row.iloc[col_idx]) else ""

                    data['date_columns'].append({
                        'column_index': col_idx,
                        'date_row': row_idx,
                        'date_value': val.strftime('%Y-%m-%d') if hasattr(val, 'strftime') else str(val),
                        'year': val.year if hasattr(val, 'year') else None,
                        'month': val.month if hasattr(val, 'month') else None,
                        'header': header
                    })

        return data

    def _ai_analyze_file_structure(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send file data to AI for complete structure analysis."""
        user_prompt = get_file_structure_user_prompt(file_data)

        try:
            if self.is_claude:
                response = self._call_claude(GLA_FILE_STRUCTURE_SYSTEM_PROMPT, user_prompt)
            else:
                response = self._call_openai(GLA_FILE_STRUCTURE_SYSTEM_PROMPT, user_prompt)

            content = response.get("content", "")
            result = self._parse_structure_response(content)

            # Convert monthly_columns from "YYYY-MM" string keys to (year, month) tuples
            if result.get("monthly_columns"):
                converted = {}
                for key, col_idx in result["monthly_columns"].items():
                    if isinstance(key, str) and '-' in key:
                        parts = key.split('-')
                        year = int(parts[0])
                        month = int(parts[1])
                        converted[(year, month)] = col_idx
                result["monthly_columns"] = converted

            # If AI didn't return monthly_columns but detected pivot_table,
            # extract from the date_columns data we already have
            if result.get("format") == "pivot_table" and not result.get("monthly_columns"):
                gla_header = result.get("monthly_gla_header", "Handover GLA")
                monthly_cols = {}
                for dc in file_data['date_columns']:
                    if dc['header'] == gla_header and dc['year'] and dc['month']:
                        monthly_cols[(dc['year'], dc['month'])] = dc['column_index']
                result["monthly_columns"] = monthly_cols
                logger.info(f"Extracted {len(monthly_cols)} monthly columns from file data")

            return result

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            raise

    def _parse_structure_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response for structure detection."""
        try:
            # Try to extract JSON from response
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            # Find JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]

            return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse structure JSON: {e}")
            raise ValueError(f"AI returned invalid JSON response: {e}")

    def detect_comparison_months(self, file_path: str, sheet_names: List[str] = None) -> Dict[str, Any]:
        """
        Detect which months should be compared based on file name and sheet structure.

        Priority:
        1. Extract months from filename (Dec-Nov, T11-T12, etc.)
        2. Validate against available data columns
        3. Find matching year from available data
        4. Fallback to most recent months in data

        Args:
            file_path: Path to the Excel file
            sheet_names: Optional list of sheet names to analyze

        Returns:
            Dict with:
            {
                "previous_month": (year, month),
                "current_month": (year, month),
                "source": "filename" | "sheet_data" | "default"
            }
        """
        import re
        import os

        filename = os.path.basename(file_path)

        # First, get available months from the data
        structure = self.detect_file_structure(file_path)
        available_months = []
        if structure.get("format") == "pivot_table" and structure.get("monthly_columns"):
            available_months = sorted(structure["monthly_columns"].keys())
            logger.info(f"Available months in data: {available_months}")

        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        # Try "Month1-Month2" pattern (e.g., "Dec-Nov")
        # Convention: First month is CURRENT, second is PREVIOUS
        # Example: "Dec-Nov" means comparing Dec (current) vs Nov (previous)
        pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[^\w]*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        match = re.search(pattern, filename.lower())

        if match:
            current_month_num = month_map[match.group(1)]  # First month = current
            previous_month_num = month_map[match.group(2)]  # Second month = previous

            # Find the correct year from available data
            prev_found = None
            curr_found = None

            for year, month in available_months:
                if month == previous_month_num and prev_found is None:
                    prev_found = (year, month)
                if month == current_month_num and curr_found is None:
                    curr_found = (year, month)

            if prev_found and curr_found:
                logger.info(f"Matched filename months to data: {prev_found} (prev) -> {curr_found} (curr)")
                return {
                    "previous_month": prev_found,
                    "current_month": curr_found,
                    "source": "filename"
                }

        # Try "T##-T##" pattern (Vietnamese month notation)
        # Convention: First T## is CURRENT, second is PREVIOUS
        # Example: "T12-T11" means comparing T12 (current) vs T11 (previous)
        t_pattern = r'[Tt](\d{1,2})[^\d]+[Tt](\d{1,2})'
        match = re.search(t_pattern, filename)

        if match:
            current_month_num = int(match.group(1))  # First T## = current
            previous_month_num = int(match.group(2))  # Second T## = previous

            # Find the correct year from available data
            prev_found = None
            curr_found = None

            for year, month in available_months:
                if month == previous_month_num and prev_found is None:
                    prev_found = (year, month)
                if month == current_month_num and curr_found is None:
                    curr_found = (year, month)

            if prev_found and curr_found:
                logger.info(f"Matched T-notation months to data: {prev_found} (prev) -> {curr_found} (curr)")
                return {
                    "previous_month": prev_found,
                    "current_month": curr_found,
                    "source": "filename"
                }

        # Fallback: use the two most recent months from the data
        if len(available_months) >= 2:
            logger.info(f"Using most recent months from data: {available_months[-2]} -> {available_months[-1]}")
            return {
                "previous_month": available_months[-2],
                "current_month": available_months[-1],
                "source": "sheet_data"
            }

        # Default to Nov-Dec 2024
        logger.warning("Could not detect months, using default Nov-Dec 2024")
        return {
            "previous_month": (2024, 11),
            "current_month": (2024, 12),
            "source": "default"
        }
