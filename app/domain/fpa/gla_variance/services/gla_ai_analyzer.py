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
        return """You are a senior real estate analyst specializing in industrial properties in Vietnam.
You are analyzing Gross Leasable Area (GLA) variance data for BW Industrial Development.

Your role is to:
1. Analyze changes in Handover GLA (units already handed over to tenants)
2. Analyze changes in Committed GLA (units with signed contracts - Open + Handed Over)
3. Provide business explanations for significant variances
4. Identify patterns and trends across the portfolio

Key concepts:
- RBF = Ready Built Factory (industrial manufacturing space)
- RBW = Ready Built Warehouse (logistics and storage space)
- Positive variance = GLA increased (new leases, handovers)
- Negative variance = GLA decreased (terminations, lease expirations)
- Committed > Handover typically indicates pipeline deals not yet handed over

Common reasons for GLA changes:
- New tenant handovers
- Lease terminations (early or normal)
- Contract expansions
- New committed deals (signed but not yet handed over)
- Project completions adding new inventory

When analyzing, consider:
- Materiality (changes > 1,000 sqm are significant)
- Regional trends (North vs South Vietnam)
- Product type trends (RBF vs RBW demand)
- Individual project performance"""

    def _format_variance_data(self, summary: GLAAnalysisSummary) -> str:
        """Format variance data for the AI prompt."""
        lines = []
        lines.append("GLA VARIANCE DATA")
        lines.append("=" * 60)
        lines.append(f"Period: {summary.previous_period} to {summary.current_period}")
        lines.append("")
        lines.append("PROJECT DETAILS:")
        lines.append("-" * 60)
        lines.append(f"{'Project':<30} {'Type':<5} {'Region':<8} {'Committed Var':>15} {'Handover Var':>15}")
        lines.append("-" * 60)

        # Sort by absolute variance for importance
        sorted_results = sorted(
            summary.results,
            key=lambda r: abs(r.committed_variance) + abs(r.handover_variance),
            reverse=True
        )

        for r in sorted_results:
            if r.committed_variance != 0 or r.handover_variance != 0:
                lines.append(
                    f"{r.project_name:<30} {r.product_type:<5} {r.region:<8} "
                    f"{r.committed_variance:>15,.2f} {r.handover_variance:>15,.2f}"
                )

        lines.append("")
        lines.append("PORTFOLIO TOTALS:")
        lines.append("-" * 60)
        lines.append(f"Total RBF Committed Variance: {summary.total_rbf_committed_variance:,.2f} sqm")
        lines.append(f"Total RBF Handover Variance: {summary.total_rbf_handover_variance:,.2f} sqm")
        lines.append(f"Total RBW Committed Variance: {summary.total_rbw_committed_variance:,.2f} sqm")
        lines.append(f"Total RBW Handover Variance: {summary.total_rbw_handover_variance:,.2f} sqm")
        lines.append(f"Total Portfolio Committed Variance: {summary.total_portfolio_committed_variance:,.2f} sqm")
        lines.append(f"Total Portfolio Handover Variance: {summary.total_portfolio_handover_variance:,.2f} sqm")

        return "\n".join(lines)

    def _get_user_prompt(self, summary: GLAAnalysisSummary) -> str:
        """Generate the user prompt with variance data."""
        data = self._format_variance_data(summary)

        return f"""{data}

Please analyze this GLA variance data and provide:

1. **Executive Summary** (2-3 sentences)
   - Overall portfolio performance
   - Key highlights

2. **Significant Changes** (top 5-10 projects by impact)
   For each significant project, provide:
   - Project name and product type
   - Variance amounts (Committed and Handover)
   - Likely business explanation
   - Risk or opportunity assessment

3. **Regional Analysis**
   - North vs South performance comparison
   - Regional trends

4. **Product Type Analysis**
   - RBF vs RBW demand trends
   - Market insights

5. **Recommendations**
   - Areas requiring attention
   - Opportunities to pursue

Format your response in clear markdown with headers and bullet points.
Be specific and quantitative where possible."""

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

                system_prompt = """You are a real estate analyst for BW Industrial.
Based on the tenant change data provided, generate concise notes explaining each variance.
Focus on the most significant tenant changes (largest GLA impact).

Format each note as: "[Tenant1] (+X sqm), [Tenant2] (-Y sqm)" or similar brief explanation.
Keep each note under 100 characters.

Return a JSON array with objects containing:
- project_name: string
- product_type: string
- committed_note: string (brief explanation of committed GLA change)
- handover_note: string (brief explanation of handover GLA change)

Only include projects where you have meaningful tenant information."""

                if self.is_claude:
                    response = self._call_claude(system_prompt, prompt)
                else:
                    response = self._call_openai(system_prompt, prompt)

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
        lines = ["Generate brief explanations for these GLA variances. Tenant change data is provided:", ""]

        for r in results:
            lines.append(f"Project: {r.project_name} ({r.product_type}, {r.region})")
            lines.append(f"  Committed: {r.committed_previous:,.0f} -> {r.committed_current:,.0f} ({r.committed_variance:+,.0f} sqm)")

            # Include committed tenant changes
            if r.committed_tenant_changes:
                lines.append("  Committed tenant changes:")
                for tc in r.committed_tenant_changes[:5]:
                    lines.append(f"    - {tc.tenant_name}: {tc.previous_gla:,.0f} -> {tc.current_gla:,.0f} ({tc.change_type})")

            lines.append(f"  Handover: {r.handover_previous:,.0f} -> {r.handover_current:,.0f} ({r.handover_variance:+,.0f} sqm)")

            # Include handover tenant changes
            if r.handover_tenant_changes:
                lines.append("  Handover tenant changes:")
                for tc in r.handover_tenant_changes[:5]:
                    lines.append(f"    - {tc.tenant_name}: {tc.previous_gla:,.0f} -> {tc.current_gla:,.0f} ({tc.change_type})")

            lines.append("")

        return "\n".join(lines)

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

        system_prompt = """You are an expert Excel file analyzer for GLA (Gross Leasable Area) data.

Analyze the provided Excel file structure and return a COMPLETE analysis including:
1. File format: "standard" (single GLA column) or "pivot_table" (monthly GLA columns)
2. Row positions: date_row, header_row, data_start_row
3. For pivot_table format: the complete mapping of ALL monthly "Handover GLA" columns

IMPORTANT: For pivot_table format, you MUST return the "monthly_columns" object with ALL months that have "Handover GLA" header.

Return ONLY valid JSON in this exact format:
{
    "format": "standard" or "pivot_table",
    "date_row": <row index with dates, or null>,
    "header_row": <row index with headers>,
    "data_start_row": <first data row>,
    "monthly_gla_header": "Handover GLA",
    "monthly_columns": {
        "2024-01": <column_index>,
        "2024-02": <column_index>,
        ... (include ALL months with Handover GLA header)
    },
    "reasoning": "<brief explanation>"
}

For the monthly_columns, use format "YYYY-MM" as keys and column indices as values.
Only include columns where the header is exactly "Handover GLA"."""

        # Format file data for AI
        prompt_lines = [
            f"File: {file_data['file_name']}",
            f"Sheet: {file_data['sheet_name']}",
            f"Total columns: {file_data['total_columns']}",
            "",
            "First rows preview:"
        ]

        for row_info in file_data['first_rows']:
            cells = [f"[{c['col']}]'{c['value']}'" for c in row_info['cells'][:8]]
            prompt_lines.append(f"  Row {row_info['row']}: {', '.join(cells)}")

        if file_data['date_columns']:
            prompt_lines.append("")
            prompt_lines.append(f"Found {len(file_data['date_columns'])} date columns:")
            prompt_lines.append("")

            # Group by header for clarity
            by_header = {}
            for dc in file_data['date_columns']:
                header = dc['header'] or 'unknown'
                if header not in by_header:
                    by_header[header] = []
                by_header[header].append(dc)

            for header, cols in by_header.items():
                prompt_lines.append(f"Header '{header}' ({len(cols)} columns):")
                for dc in cols[:5]:  # Show first 5 of each type
                    prompt_lines.append(f"  Column {dc['column_index']}: {dc['date_value']} (Year: {dc['year']}, Month: {dc['month']})")
                if len(cols) > 5:
                    prompt_lines.append(f"  ... and {len(cols) - 5} more")
                prompt_lines.append("")

            # Also provide complete list for Handover GLA columns
            handover_cols = [dc for dc in file_data['date_columns'] if dc['header'] == 'Handover GLA']
            if handover_cols:
                prompt_lines.append("COMPLETE LIST of 'Handover GLA' columns:")
                for dc in handover_cols:
                    prompt_lines.append(f"  Column {dc['column_index']}: {dc['year']}-{dc['month']:02d}")

        user_prompt = "\n".join(prompt_lines) + "\n\nAnalyze this structure and return the complete JSON response."

        try:
            if self.is_claude:
                response = self._call_claude(system_prompt, user_prompt)
            else:
                response = self._call_openai(system_prompt, user_prompt)

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

        Args:
            file_path: Path to the Excel file
            sheet_names: Optional list of sheet names to analyze

        Returns:
            Dict with:
            {
                "previous_month": (year, month),
                "current_month": (year, month),
                "source": "filename" | "sheet_data" | "ai_detected"
            }
        """
        import re
        import os

        filename = os.path.basename(file_path)

        # Try to detect from filename (e.g., "Dec-Nov" or "T11-T12")
        # Pattern: Dec-Nov, Nov-Dec, T11-T12, etc.
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        # Try "Month1-Month2" pattern
        pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[^\w]*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        match = re.search(pattern, filename.lower())

        if match:
            month1 = month_map[match.group(1)]
            month2 = month_map[match.group(2)]
            # Assume current year for now
            from datetime import datetime
            year = datetime.now().year

            # Determine which is previous/current based on order
            # "Dec-Nov" means comparing Dec (current) vs Nov (previous)
            return {
                "previous_month": (year, month1),
                "current_month": (year, month2),
                "source": "filename"
            }

        # Try "T##-T##" pattern (Vietnamese month notation)
        t_pattern = r'[Tt](\d{1,2})[^\d]+[Tt](\d{1,2})'
        match = re.search(t_pattern, filename)

        if match:
            month1 = int(match.group(1))
            month2 = int(match.group(2))
            from datetime import datetime
            year = datetime.now().year

            return {
                "previous_month": (year, month1),
                "current_month": (year, month2),
                "source": "filename"
            }

        # Fallback: use the two most recent months from the data
        structure = self.detect_file_structure(file_path)
        if structure.get("format") == "pivot_table" and structure.get("monthly_columns"):
            monthly_cols = structure["monthly_columns"]
            if len(monthly_cols) >= 2:
                sorted_months = sorted(monthly_cols.keys())
                return {
                    "previous_month": sorted_months[-2],
                    "current_month": sorted_months[-1],
                    "source": "sheet_data"
                }

        # Default to Nov-Dec 2024
        return {
            "previous_month": (2024, 11),
            "current_month": (2024, 12),
            "source": "default"
        }
