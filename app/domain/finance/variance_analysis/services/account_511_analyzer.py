"""
Account 511 Analyzer - Comprehensive revenue drill-down analysis service.

This module provides Account 511 (Revenue) drill-down analysis including:
- Sub-account breakdown
- Project-level analysis
- Tenant matching with unit data
- AI-generated narratives
"""

import io
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field

from openai import OpenAI
from anthropic import Anthropic

from .account_511_parser import (
    parse_revenue_breakdown,
    parse_unit_for_lease_list,
    RevenueLineItem,
    Account511SubAccount,
    UnitForLease,
    Account511Analysis,
    ACCOUNT_511_HIERARCHY,
)
from .tenant_matcher import TenantMatcher, TenantMatch

logger = logging.getLogger(__name__)


# AI Prompt for Account 511 variance narrative
ACCOUNT_511_SYSTEM_PROMPT = """You are a senior financial analyst for BW Industrial, a real estate company in Vietnam.
Analyze Account 511 (Revenue) variance data and generate a concise executive narrative.

Your analysis should:
1. Identify the TOP 3 sub-accounts contributing most to the variance
2. Explain what drove the variance (new tenants, terminations, rate changes, etc.)
3. Highlight any concerning patterns or positive trends
4. Connect revenue changes to occupancy when data is available

IMPORTANT RULES:
- Focus on MATERIAL variances (>5% or >100M VND)
- Use specific numbers in your explanation
- Keep the narrative under 300 words
- Format amounts in billions (B) or millions (M) VND
- Be direct and actionable

Output JSON format:
{
    "executive_summary": "2-3 sentence summary of the overall variance",
    "key_drivers": [
        {"sub_account": "511xxxxx", "description": "explanation of what drove this variance", "impact": "+X.XB VND"}
    ],
    "occupancy_correlation": "How occupancy changes relate to revenue (if data available)",
    "recommendations": ["actionable recommendation 1", "recommendation 2"]
}
"""


@dataclass
class Account511DrillDownResult:
    """Complete drill-down result for Account 511."""
    # Summary
    total_current: float = 0.0
    total_previous: float = 0.0
    total_variance: float = 0.0
    total_variance_pct: float = 0.0

    # Period info
    current_period: Tuple[int, int] = (2026, 1)
    previous_period: Tuple[int, int] = (2025, 12)

    # Sub-account breakdown
    sub_accounts: List[Dict[str, Any]] = field(default_factory=list)

    # By project breakdown
    by_project: List[Dict[str, Any]] = field(default_factory=list)

    # Tenant matches
    tenant_matches: List[Dict[str, Any]] = field(default_factory=list)

    # Unit summary
    unit_summary: Dict[str, Any] = field(default_factory=dict)

    # AI narrative
    ai_narrative: Dict[str, Any] = field(default_factory=dict)


class Account511Analyzer:
    """
    Comprehensive Account 511 analyzer with AI-powered insights.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize analyzer.

        Args:
            api_key: OpenAI or Anthropic API key
            model: Model to use (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("AI_MODEL", "gpt-4o")

        # Determine API type
        self.is_claude = "claude" in self.model.lower() if self.model else False

        # Initialize client
        self.client = None
        if self.api_key:
            if self.is_claude:
                self.client = Anthropic(api_key=self.api_key)
            else:
                self.client = OpenAI(api_key=self.api_key)

        self.tenant_matcher = TenantMatcher(min_confidence=0.6)

    def analyze(
        self,
        revenue_breakdown_bytes: bytes,
        unit_for_lease_bytes: bytes,
        callback: Optional[Callable[[int, str], None]] = None
    ) -> Account511DrillDownResult:
        """
        Perform comprehensive Account 511 analysis.

        Args:
            revenue_breakdown_bytes: Raw bytes of RevenueBreakdown file
            unit_for_lease_bytes: Raw bytes of UnitForLeaseList file
            callback: Optional progress callback(percentage, message)

        Returns:
            Account511DrillDownResult with complete analysis
        """
        result = Account511DrillDownResult()

        if callback:
            callback(5, "Parsing RevenueBreakdown file...")

        # Parse revenue breakdown
        line_items, sub_accounts, current_period, previous_period = parse_revenue_breakdown(
            revenue_breakdown_bytes
        )
        result.current_period = current_period
        result.previous_period = previous_period

        if callback:
            callback(20, "Parsing UnitForLeaseList file...")

        # Parse unit list
        units = parse_unit_for_lease_list(unit_for_lease_bytes)

        if callback:
            callback(35, "Analyzing sub-account variances...")

        # Calculate totals and format sub-accounts
        result.sub_accounts = self._format_sub_accounts(sub_accounts)
        result.total_current = sum(s["current_month_amount"] for s in result.sub_accounts if s["account_code"] == "511000000")
        result.total_previous = sum(s["previous_month_amount"] for s in result.sub_accounts if s["account_code"] == "511000000")
        result.total_variance = result.total_current - result.total_previous
        if result.total_previous != 0:
            result.total_variance_pct = (result.total_variance / abs(result.total_previous)) * 100

        if callback:
            callback(50, "Aggregating by project...")

        # Aggregate by project
        result.by_project = self._aggregate_by_project(line_items, current_period)

        if callback:
            callback(60, "Matching tenants to units...")

        # Match tenants
        result.tenant_matches = self._match_tenants(line_items, units)

        if callback:
            callback(70, "Summarizing unit data...")

        # Summarize units
        result.unit_summary = self._summarize_units(units)

        if callback:
            callback(80, "Generating AI narrative...")

        # Generate AI narrative
        if self.client:
            result.ai_narrative = self._generate_ai_narrative(result)
        else:
            logger.warning("No AI client available, skipping narrative generation")
            result.ai_narrative = {
                "executive_summary": "AI analysis not available - API key not configured",
                "key_drivers": [],
                "occupancy_correlation": "N/A",
                "recommendations": []
            }

        if callback:
            callback(100, "Analysis complete")

        logger.info(f"Account 511 analysis complete: {len(result.sub_accounts)} sub-accounts, "
                   f"{len(result.by_project)} projects, {len(result.tenant_matches)} tenant matches")

        return result

    def _format_sub_accounts(self, sub_accounts: Dict[str, Account511SubAccount]) -> List[Dict[str, Any]]:
        """Format sub-accounts for output."""
        formatted = []

        # Sort by hierarchy level (parent accounts first)
        def get_level(code: str) -> int:
            if code == "511000000":
                return 0
            elif code.endswith("0000"):
                return 1
            else:
                return 2

        sorted_codes = sorted(sub_accounts.keys(), key=lambda c: (get_level(c), c))

        for code in sorted_codes:
            sa = sub_accounts[code]
            # Only include if there's activity
            if sa.current_month_amount != 0 or sa.previous_month_amount != 0:
                formatted.append({
                    "account_code": sa.account_code,
                    "account_name": sa.account_name,
                    "parent_code": sa.parent_code,
                    "level": get_level(sa.account_code),
                    "current_month_amount": sa.current_month_amount,
                    "previous_month_amount": sa.previous_month_amount,
                    "variance": sa.variance,
                    "variance_pct": sa.variance_pct,
                    "by_project": sa.by_project
                })

        return formatted

    def _aggregate_by_project(
        self,
        line_items: List[RevenueLineItem],
        current_period: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Aggregate revenue by project."""
        project_totals: Dict[str, float] = {}

        for item in line_items:
            if item.project:
                # Extract project code from format "PVC3: VC3"
                project_key = item.project.split(":")[0].strip() if ":" in item.project else item.project

                current_amount = item.monthly_amounts.get(current_period, 0.0)
                project_totals[project_key] = project_totals.get(project_key, 0.0) + current_amount

        # Sort by amount descending
        sorted_projects = sorted(project_totals.items(), key=lambda x: abs(x[1]), reverse=True)

        return [
            {"project": proj, "current_amount": amt, "formatted": self._format_amount(amt)}
            for proj, amt in sorted_projects if amt != 0
        ]

    def _match_tenants(
        self,
        line_items: List[RevenueLineItem],
        units: List[UnitForLease]
    ) -> List[Dict[str, Any]]:
        """Match revenue entities to unit tenants."""
        # Build list of unique entities from revenue
        revenue_entities: List[Tuple[str, str]] = []
        seen = set()
        for item in line_items:
            if item.entity_name and item.entity_name not in seen:
                revenue_entities.append((item.entity_name, item.entity_code))
                seen.add(item.entity_name)

        # Build list of unique tenants from units
        unit_tenants: List[Tuple[str, str]] = []
        seen_tenants = set()
        for unit in units:
            if unit.tenant and unit.tenant not in seen_tenants:
                unit_tenants.append((unit.tenant, unit.tenant_code))
                seen_tenants.add(unit.tenant)

        # Match
        matches = self.tenant_matcher.match_all_tenants(revenue_entities, unit_tenants)

        return [
            {
                "revenue_entity": m.revenue_entity,
                "unit_tenant": m.unit_tenant,
                "confidence": round(m.confidence * 100, 1),
                "match_type": m.match_type
            }
            for m in matches.values()
        ]

    def _summarize_units(self, units: List[UnitForLease]) -> Dict[str, Any]:
        """Summarize unit data by status."""
        summary = {
            "total_units": len(units),
            "by_status": {},
            "total_gla_by_status": {},
            "by_region": {},
            "by_project": {}
        }

        for unit in units:
            # By status
            status = unit.status or "Unknown"
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            summary["total_gla_by_status"][status] = (
                summary["total_gla_by_status"].get(status, 0.0) + unit.gla
            )

            # By region
            region = unit.region or "Unknown"
            summary["by_region"][region] = summary["by_region"].get(region, 0) + 1

            # By project
            project = unit.project_code or unit.project_name or "Unknown"
            if project not in summary["by_project"]:
                summary["by_project"][project] = {"count": 0, "gla": 0.0}
            summary["by_project"][project]["count"] += 1
            summary["by_project"][project]["gla"] += unit.gla

        return summary

    def _generate_ai_narrative(self, result: Account511DrillDownResult) -> Dict[str, Any]:
        """Generate AI narrative for the analysis."""
        # Build context for AI
        context = self._build_ai_context(result)

        try:
            if self.is_claude:
                response = self._call_claude(context)
            else:
                response = self._call_openai(context)

            # Parse JSON response
            return self._parse_ai_response(response)
        except Exception as e:
            logger.error(f"AI narrative generation failed: {e}")
            return {
                "executive_summary": f"AI analysis error: {str(e)}",
                "key_drivers": [],
                "occupancy_correlation": "N/A",
                "recommendations": []
            }

    def _build_ai_context(self, result: Account511DrillDownResult) -> str:
        """Build context string for AI analysis."""
        lines = [
            "## Account 511 Revenue Variance Analysis",
            f"Period: {result.previous_period[1]:02d}/{result.previous_period[0]} → {result.current_period[1]:02d}/{result.current_period[0]}",
            "",
            "### Summary",
            f"Total Revenue Current: {self._format_amount(result.total_current)}",
            f"Total Revenue Previous: {self._format_amount(result.total_previous)}",
            f"Total Variance: {self._format_amount(result.total_variance)} ({result.total_variance_pct:+.1f}%)",
            "",
            "### Sub-Account Breakdown (sorted by variance magnitude)",
        ]

        # Top variances
        sorted_sub = sorted(result.sub_accounts, key=lambda x: abs(x["variance"]), reverse=True)
        for sa in sorted_sub[:10]:
            if sa["variance"] != 0:
                lines.append(
                    f"- {sa['account_code']} ({sa['account_name']}): "
                    f"{self._format_amount(sa['variance'])} ({sa['variance_pct']:+.1f}%)"
                )

        lines.extend(["", "### Top Projects by Revenue"])
        for proj in result.by_project[:10]:
            lines.append(f"- {proj['project']}: {proj['formatted']}")

        lines.extend(["", "### Unit Occupancy Summary"])
        for status, count in result.unit_summary.get("by_status", {}).items():
            gla = result.unit_summary.get("total_gla_by_status", {}).get(status, 0)
            lines.append(f"- {status}: {count} units, {gla:,.0f} sqm GLA")

        lines.extend(["", "### Tenant Matches"])
        lines.append(f"Matched {len(result.tenant_matches)} revenue entities to unit tenants")

        return "\n".join(lines)

    def _call_openai(self, context: str) -> str:
        """Call OpenAI API."""
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ACCOUNT_511_SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            "max_tokens": 2000,
        }

        # GPT-5 doesn't support temperature
        if not self.model.startswith("gpt-5"):
            params["temperature"] = 0.3

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def _call_claude(self, context: str) -> str:
        """Call Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=ACCOUNT_511_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": context}
            ]
        )
        return response.content[0].text

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI JSON response."""
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0].strip()

            return json.loads(json_match)
        except (json.JSONDecodeError, IndexError):
            # Fallback: return raw response as summary
            return {
                "executive_summary": response[:500] if len(response) > 500 else response,
                "key_drivers": [],
                "occupancy_correlation": "See summary",
                "recommendations": []
            }

    def _format_amount(self, amount: float) -> str:
        """Format amount in VND (billions/millions)."""
        if abs(amount) >= 1e9:
            return f"{amount / 1e9:,.2f}B VND"
        elif abs(amount) >= 1e6:
            return f"{amount / 1e6:,.2f}M VND"
        else:
            return f"{amount:,.0f} VND"


def analyze_account_511(
    revenue_breakdown_bytes: bytes,
    unit_for_lease_bytes: bytes,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    callback: Optional[Callable[[int, str], None]] = None
) -> Account511DrillDownResult:
    """
    Convenience function to analyze Account 511.

    Args:
        revenue_breakdown_bytes: Raw bytes of RevenueBreakdown file
        unit_for_lease_bytes: Raw bytes of UnitForLeaseList file
        api_key: Optional API key
        model: Optional model name
        callback: Optional progress callback

    Returns:
        Account511DrillDownResult
    """
    analyzer = Account511Analyzer(api_key=api_key, model=model)
    return analyzer.analyze(revenue_breakdown_bytes, unit_for_lease_bytes, callback)
