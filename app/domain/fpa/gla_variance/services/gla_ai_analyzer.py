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
