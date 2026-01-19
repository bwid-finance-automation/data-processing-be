"""
NTM AI Analyzer Service
Uses LLM to analyze NTM EBITDA variances and generate professional commentary.
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
    NTM_EBITDA_SYSTEM_PROMPT,
    NTM_COMMENTARY_SYSTEM_PROMPT,
    get_ntm_commentary_prompt,
)
from ..models.ntm_ebitda_models import (
    NTMVarianceResult,
    NTMAnalysisSummary,
    AnalysisConfig,
)

logger = get_logger(__name__)


class NTMAIAnalyzer:
    """
    AI-powered analyzer for NTM EBITDA variance data.
    Generates professional commentary for lease changes.
    """

    # Supported models
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

    def __init__(self, config: Optional[AnalysisConfig] = None, progress_callback=None):
        """
        Initialize the AI analyzer.

        Args:
            config: Optional analysis configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or AnalysisConfig()
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

    def _call_openai(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        try:
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

    def generate_commentary(
        self,
        results: List[NTMVarianceResult],
        progress_callback=None
    ) -> List[NTMVarianceResult]:
        """
        Generate AI-powered commentary for each project's variance.

        Args:
            results: List of NTMVarianceResult to annotate
            progress_callback: Optional callback for progress updates

        Returns:
            Updated list with commentary populated
        """
        callback = progress_callback or self.progress_callback

        # Filter to significant variances
        significant = [r for r in results if r.is_significant(self.config.variance_threshold)]

        if not significant:
            logger.info("No significant variances to comment on")
            return results

        logger.info(f"Generating commentary for {len(significant)} significant variances")

        if not self.client:
            logger.warning("No AI client available - using rule-based commentary")
            return self._generate_rule_based_commentary(results)

        # Batch projects (max 15 per batch to avoid token limits)
        BATCH_SIZE = 15
        total_batches = (len(significant) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(significant))
            batch = significant[start_idx:end_idx]

            if callback:
                progress = 40 + int(40 * batch_idx / total_batches)
                callback(progress, f"Generating commentary batch {batch_idx + 1}/{total_batches}...")

            # Create prompt
            prompt = get_ntm_commentary_prompt(batch)

            try:
                if self.is_claude:
                    response = self._call_claude(NTM_COMMENTARY_SYSTEM_PROMPT, prompt)
                else:
                    response = self._call_openai(NTM_COMMENTARY_SYSTEM_PROMPT, prompt)

                # Parse response
                commentaries = self._parse_commentary_response(response.get("content", ""))

                # Apply to results
                for r in batch:
                    key = r.project_name
                    if key in commentaries:
                        r.commentary = commentaries[key]

                logger.info(f"Batch {batch_idx + 1}: Generated {len(commentaries)} commentaries")

            except Exception as e:
                logger.error(f"AI commentary generation failed: {e}")
                # Fall back to rule-based for this batch
                for r in batch:
                    r.commentary = self._generate_single_rule_commentary(r)

        if callback:
            callback(80, f"Commentary generated for {len(significant)} projects")

        return results

    def _parse_commentary_response(self, content: str) -> Dict[str, str]:
        """
        Parse AI response for commentaries.

        Args:
            content: Raw AI response content

        Returns:
            Dict mapping project name -> commentary
        """
        commentaries = {}

        try:
            # Try to extract JSON
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            # Try parsing as JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                content = content[start:end]
                items = json.loads(content)
                for item in items:
                    if isinstance(item, dict) and "project_name" in item and "commentary" in item:
                        commentaries[item["project_name"]] = item["commentary"]
                return commentaries

            # Try parsing as JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
                data = json.loads(content)
                if isinstance(data, dict):
                    return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        # Fallback: try to extract project:commentary pairs from text
        import re
        pattern = r'"?([^":\n]+)"?\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, content)
        for project, commentary in matches:
            commentaries[project.strip()] = commentary.strip()

        return commentaries

    def _generate_rule_based_commentary(
        self,
        results: List[NTMVarianceResult]
    ) -> List[NTMVarianceResult]:
        """
        Generate rule-based commentary when AI is not available.

        Args:
            results: List of NTMVarianceResult to annotate

        Returns:
            Updated list with commentary populated
        """
        for result in results:
            if result.is_significant(self.config.variance_threshold):
                result.commentary = self._generate_single_rule_commentary(result)
        return results

    def _generate_single_rule_commentary(self, result: NTMVarianceResult) -> str:
        """
        Generate commentary for a single result using rules.

        Args:
            result: NTMVarianceResult to generate commentary for

        Returns:
            Commentary string
        """
        parts = []

        # New leases
        new_leases = [lc for lc in result.lease_changes if lc.change_type.value == "new"]
        if new_leases:
            for lc in new_leases[:3]:  # Top 3
                if lc.gla_sqm > 0:
                    parts.append(f"New sign {lc.gla_sqm/1000:.0f}k sqm {lc.tenant_name}")
                    if lc.term_months:
                        parts[-1] += f", {lc.term_months//12}Y lease"
                    if lc.lease_start:
                        date_str = lc.lease_start.strftime("%b'%y")
                        parts[-1] += f" starting from {date_str}"

        # Terminations
        terminated = [lc for lc in result.lease_changes if lc.change_type.value == "terminated"]
        if terminated:
            total_sqm = sum(lc.gla_sqm for lc in terminated)
            if total_sqm > 1000:
                tenants = ", ".join([f"{lc.gla_sqm/1000:.0f}k {lc.tenant_name}" for lc in terminated[:3]])
                parts.append(f"Will expire: {tenants}")

        # Renewals
        renewed = [lc for lc in result.lease_changes if lc.change_type.value == "renewed"]
        if renewed:
            for lc in renewed[:2]:
                parts.append(f"{lc.gla_sqm/1000:.0f}k sqm {lc.tenant_name} renewed")

        if not parts:
            if result.ebitda_variance > 0:
                parts.append("EBITDA increased due to timing adjustments")
            else:
                parts.append("EBITDA decreased due to timing adjustments")

        return "; ".join(parts)

    def analyze_portfolio(
        self,
        summary: NTMAnalysisSummary,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Generate overall portfolio analysis.

        Args:
            summary: NTMAnalysisSummary with all results
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with analysis results
        """
        callback = progress_callback or self.progress_callback

        if not self.client:
            logger.warning("No AI client available - returning basic analysis")
            return self._generate_basic_analysis(summary)

        try:
            if callback:
                callback(85, "Generating portfolio analysis...")

            # Build prompt with summary data
            prompt = self._build_portfolio_prompt(summary)

            if self.is_claude:
                response = self._call_claude(NTM_EBITDA_SYSTEM_PROMPT, prompt)
            else:
                response = self._call_openai(NTM_EBITDA_SYSTEM_PROMPT, prompt)

            if callback:
                callback(95, "Analysis complete!")

            return {
                "status": "success",
                "analysis": response.get("content", ""),
                "model": self.model,
                "provider": "anthropic" if self.is_claude else "openai"
            }

        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return self._generate_basic_analysis(summary)

    def _build_portfolio_prompt(self, summary: NTMAnalysisSummary) -> str:
        """Build prompt for portfolio analysis."""
        fx = self.config.fx_rate

        lines = [
            "NTM EBITDA VARIANCE ANALYSIS",
            "=" * 60,
            f"Period: {summary.previous_period} to {summary.current_period}",
            f"FX Rate: {fx:,.0f} VND/USD",
            "",
            "PORTFOLIO TOTALS (USD millions):",
            f"  Revenue: {summary.total_revenue_previous/fx/1e6:.2f} -> {summary.total_revenue_current/fx/1e6:.2f} ({summary.total_revenue_variance/fx/1e6:+.2f})",
            f"  EBITDA:  {summary.total_ebitda_previous/fx/1e6:.2f} -> {summary.total_ebitda_current/fx/1e6:.2f} ({summary.total_ebitda_variance/fx/1e6:+.2f}, {summary.total_ebitda_variance_pct:+.1%})",
            "",
            f"SIGNIFICANT VARIANCES ({len(summary.significant_results)} projects):",
        ]

        for r in summary.significant_results[:15]:
            lines.append(f"  {r.project_name}: EBITDA {r.ebitda_variance/fx/1e6:+.2f} USD mn ({r.ebitda_variance_pct:+.1%})")
            if r.commentary:
                lines.append(f"    -> {r.commentary}")

        lines.append("")
        lines.append("Please provide:")
        lines.append("1. Executive Summary (2-3 sentences)")
        lines.append("2. Key Drivers of variance")
        lines.append("3. Risk areas to monitor")
        lines.append("4. Recommendations")

        return "\n".join(lines)

    def _generate_basic_analysis(self, summary: NTMAnalysisSummary) -> Dict[str, Any]:
        """Generate basic analysis without AI."""
        fx = self.config.fx_rate

        lines = [
            "## NTM EBITDA Variance Analysis (Basic Mode)",
            "",
            "*Note: AI analysis not available. Showing automated summary.*",
            "",
            "### Executive Summary",
        ]

        variance_pct = summary.total_ebitda_variance_pct
        if variance_pct > 0:
            lines.append(f"- Portfolio EBITDA NTM **increased** by {summary.total_ebitda_variance/fx/1e6:.2f} USD mn ({variance_pct:.1%})")
        else:
            lines.append(f"- Portfolio EBITDA NTM **decreased** by {abs(summary.total_ebitda_variance)/fx/1e6:.2f} USD mn ({variance_pct:.1%})")

        lines.append(f"- {len(summary.significant_results)} projects with significant variance (>{summary.variance_threshold:.0%})")
        lines.append(f"- {len(summary.projects_with_increase)} projects increased, {len(summary.projects_with_decrease)} decreased")
        lines.append("")
        lines.append("### Top Variances")

        for r in summary.significant_results[:10]:
            direction = "+" if r.ebitda_variance > 0 else ""
            lines.append(f"- **{r.project_name}**: {direction}{r.ebitda_variance/fx/1e6:.2f} USD mn ({r.ebitda_variance_pct:+.1%})")
            if r.commentary:
                lines.append(f"  - {r.commentary}")

        return {
            "status": "success",
            "analysis": "\n".join(lines),
            "model": "basic"
        }
