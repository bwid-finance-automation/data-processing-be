"""
FPA (Financial Planning & Analysis) Department AI Prompts
==========================================================

Contains prompts for:
- GLA Variance Analysis
- Leasing Summary Comparison
- Other FPA-related AI tasks
"""

from typing import List, Dict, Any

# =============================================================================
# GLA VARIANCE ANALYSIS PROMPTS
# =============================================================================

GLA_VARIANCE_SYSTEM_PROMPT = """You are a senior real estate analyst specializing in industrial properties in Vietnam.
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


GLA_NOTES_SYSTEM_PROMPT = """You are a real estate analyst for BW Industrial.
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


GLA_FILE_STRUCTURE_SYSTEM_PROMPT = """You are an expert Excel file analyzer for GLA (Gross Leasable Area) data.

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


# =============================================================================
# GLA VARIANCE USER PROMPT BUILDERS
# =============================================================================

def format_variance_data(summary) -> str:
    """
    Format variance data for the AI prompt.

    Args:
        summary: GLAAnalysisSummary object with variance results

    Returns:
        Formatted string for AI prompt
    """
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


def get_variance_user_prompt(summary) -> str:
    """
    Generate the user prompt for GLA variance analysis.

    Args:
        summary: GLAAnalysisSummary object with variance results

    Returns:
        User prompt string
    """
    data = format_variance_data(summary)

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


def get_notes_user_prompt(results: List) -> str:
    """
    Create prompt for generating project notes with tenant details.

    Args:
        results: List of GLAVarianceResult objects

    Returns:
        User prompt string for notes generation
    """
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


def get_file_structure_user_prompt(file_data: Dict[str, Any]) -> str:
    """
    Generate user prompt for file structure detection.

    Args:
        file_data: Dict containing file metadata and sample data

    Returns:
        User prompt string for structure detection
    """
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

    if file_data.get('date_columns'):
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

    return "\n".join(prompt_lines) + "\n\nAnalyze this structure and return the complete JSON response."
