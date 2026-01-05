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


GLA_NOTES_SYSTEM_PROMPT = """You are a senior real estate analyst for BW Industrial in Vietnam.
Analyze raw tenant data to detect changes and generate concise notes.

YOUR TASK:
1. Compare PREVIOUS vs CURRENT tenant lists for each project
2. Detect: new tenants, terminated tenants, expanded/reduced space, entity changes
3. Generate concise notes for: committed_note, handover_note, wale_note, gross_rent_note

DETECTION RULES:
- NEW: Tenant in current but not in previous (or previous=0)
- TERMINATED: Tenant in previous but not in current (or current=0)
- EXPANDED: Same tenant name, current > previous
- REDUCED: Same tenant name, current < previous
- ENTITY CHANGE: Similar company name with legal suffix change (e.g., "GA HEALTH" → "GA HEALTH VN") - use "replaced (entity change)"
- ESCALATION: Same tenant, rate increased ~5% (common annual escalation)

IMPORTANT - "REPLACED" RULES:
- ONLY use "replaced" for ENTITY CHANGES (same company, different legal name)
- DO NOT use "replaced" just because sqm values are similar

FORMATTING FOR committed_note AND handover_note:
- Include sqm for tenants (e.g., "+6,102 sqm" or "-1,921 sqm")
- Examples: "KUKAHOME expanded (+21,280 sqm); DASEN terminated (-20,226 sqm)"

FORMATTING FOR gross_rent_note (IMPORTANT - use this specific format):
- For escalations: "+5% esc TENANT1, TENANT2" (list tenants with ~5% rate increase)
- For new handovers: "Handover: TENANT (XXXk VND)" (show rate in thousands VND)
- For terminations: "Terminate: TENANT (XXXk VND)" (show rate in thousands VND)
- Combine multiple: "+5% esc TENANT1; Handover: TENANT2 (110k VND); Terminate: TENANT3 (105k VND)"
- If no rent variance: leave empty or "No change"

GROSS RENT EXAMPLES:
- "+5% esc KANEPACKAGE, VALSPAR"
- "Handover: Aerotech (110.7k VND), Drinda (104.4k VND)"
- "Terminate: Ceva (164.8k VND)"
- "+5% esc Qing Yi; Handover: Aerotech (110.7k VND), Drinda (104.4k VND)"

Return JSON array:
[{"project_name": "X", "product_type": "RBF", "committed_note": "...", "handover_note": "...", "wale_note": "...", "gross_rent_note": "..."}]

IMPORTANT: Keep response compact. Only include projects with actual changes."""


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
    Create prompt for generating project notes with RAW tenant data.

    AI will:
    1. Compare previous vs current tenant lists
    2. Detect changes (new, terminated, expanded, reduced, replaced)
    3. Generate concise notes

    Args:
        results: List of GLAVarianceResult objects with raw tenant data

    Returns:
        User prompt string for notes generation
    """
    lines = ["Analyze these projects. Compare PREVIOUS vs CURRENT tenants to detect changes.", ""]

    for r in results:
        # Skip projects with no variance
        has_changes = (
            abs(r.committed_variance) > 100 or
            abs(r.handover_variance) > 100 or
            r.committed_tenants_previous or
            r.committed_tenants_current or
            r.handover_tenants_previous or
            r.handover_tenants_current
        )
        if not has_changes:
            continue

        lines.append(f"=== {r.project_name} ({r.product_type}) ===")

        # Committed GLA section
        if abs(r.committed_variance) > 0 or r.committed_tenants_previous or r.committed_tenants_current:
            lines.append(f"COMMITTED GLA: {r.committed_previous:,.0f} → {r.committed_current:,.0f} (Var: {r.committed_variance:+,.0f})")

            # Build tenant dict for comparison
            prev_dict = {t.tenant_name: t.committed_gla for t in (r.committed_tenants_previous or []) if t.committed_gla > 0}
            curr_dict = {t.tenant_name: t.committed_gla for t in (r.committed_tenants_current or []) if t.committed_gla > 0}

            # Find tenants with changes (new, terminated, or variance > 1000)
            changed_tenants = set()
            for name in set(prev_dict.keys()) | set(curr_dict.keys()):
                prev_gla = prev_dict.get(name, 0)
                curr_gla = curr_dict.get(name, 0)
                if prev_gla == 0 or curr_gla == 0 or abs(curr_gla - prev_gla) > 1000:
                    changed_tenants.add(name)

            # Previous tenants: top 10 by GLA + any with changes
            if r.committed_tenants_previous:
                prev_sorted = sorted(r.committed_tenants_previous, key=lambda t: t.committed_gla, reverse=True)
                top_10 = [t for t in prev_sorted[:10] if t.committed_gla > 0]
                changed = [t for t in prev_sorted[10:] if t.tenant_name in changed_tenants and t.committed_gla > 0]
                prev_strs = [f"{t.tenant_name}:{t.committed_gla:,.0f}" for t in (top_10 + changed)]
                if prev_strs:
                    lines.append(f"  PREV: {'; '.join(prev_strs)}")

            # Current tenants: top 10 by GLA + any with changes
            if r.committed_tenants_current:
                curr_sorted = sorted(r.committed_tenants_current, key=lambda t: t.committed_gla, reverse=True)
                top_10 = [t for t in curr_sorted[:10] if t.committed_gla > 0]
                changed = [t for t in curr_sorted[10:] if t.tenant_name in changed_tenants and t.committed_gla > 0]
                curr_strs = [f"{t.tenant_name}:{t.committed_gla:,.0f}" for t in (top_10 + changed)]
                if curr_strs:
                    lines.append(f"  CURR: {'; '.join(curr_strs)}")

        # Handover GLA section (includes rent rate data for gross_rent_note)
        if abs(r.handover_variance) > 0 or r.handover_tenants_previous or r.handover_tenants_current:
            lines.append(f"HANDOVER GLA: {r.handover_previous:,.0f} → {r.handover_current:,.0f} (Var: {r.handover_variance:+,.0f})")

            # Build tenant dicts for comparison (GLA and rate)
            prev_dict = {t.tenant_name: (t.handover_gla, t.monthly_rate) for t in (r.handover_tenants_previous or []) if t.handover_gla > 0}
            curr_dict = {t.tenant_name: (t.handover_gla, t.monthly_rate) for t in (r.handover_tenants_current or []) if t.handover_gla > 0}

            # Find tenants with changes (new, terminated, or variance > 1000)
            changed_tenants = set()
            for name in set(prev_dict.keys()) | set(curr_dict.keys()):
                prev_gla = prev_dict.get(name, (0, 0))[0]
                curr_gla = curr_dict.get(name, (0, 0))[0]
                if prev_gla == 0 or curr_gla == 0 or abs(curr_gla - prev_gla) > 1000:
                    changed_tenants.add(name)

            # Previous tenants: top 10 by GLA + any with changes (include rate)
            if r.handover_tenants_previous:
                prev_sorted = sorted(r.handover_tenants_previous, key=lambda t: t.handover_gla, reverse=True)
                top_10 = [t for t in prev_sorted[:10] if t.handover_gla > 0]
                changed = [t for t in prev_sorted[10:] if t.tenant_name in changed_tenants and t.handover_gla > 0]
                prev_strs = [f"{t.tenant_name}:{t.handover_gla:,.0f}@{t.monthly_rate/1000:.1f}k" for t in (top_10 + changed) if t.monthly_rate > 0]
                if not prev_strs:
                    prev_strs = [f"{t.tenant_name}:{t.handover_gla:,.0f}" for t in (top_10 + changed)]
                if prev_strs:
                    lines.append(f"  PREV: {'; '.join(prev_strs)}")

            # Current tenants: top 10 by GLA + any with changes (include rate)
            if r.handover_tenants_current:
                curr_sorted = sorted(r.handover_tenants_current, key=lambda t: t.handover_gla, reverse=True)
                top_10 = [t for t in curr_sorted[:10] if t.handover_gla > 0]
                changed = [t for t in curr_sorted[10:] if t.tenant_name in changed_tenants and t.handover_gla > 0]
                curr_strs = [f"{t.tenant_name}:{t.handover_gla:,.0f}@{t.monthly_rate/1000:.1f}k" for t in (top_10 + changed) if t.monthly_rate > 0]
                if not curr_strs:
                    curr_strs = [f"{t.tenant_name}:{t.handover_gla:,.0f}" for t in (top_10 + changed)]
                if curr_strs:
                    lines.append(f"  CURR: {'; '.join(curr_strs)}")

            # Add rent rate changes summary for gross_rent_note
            escalations = []
            for name in set(prev_dict.keys()) & set(curr_dict.keys()):
                prev_rate = prev_dict[name][1]
                curr_rate = curr_dict[name][1]
                if prev_rate > 0 and curr_rate > 0:
                    pct_change = (curr_rate - prev_rate) / prev_rate * 100
                    if 3 <= pct_change <= 10:  # Escalation range (3-10%)
                        escalations.append(f"{name}(+{pct_change:.0f}%)")
            if escalations:
                lines.append(f"  ESCALATIONS: {', '.join(escalations[:5])}")

        lines.append("")

    lines.append("Return JSON array with committed_note, handover_note, wale_note, and gross_rent_note for each project.")

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
