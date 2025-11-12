# app/revenue_analysis.py
"""Revenue impact analysis functions."""

from __future__ import annotations

import io
import pandas as pd
from typing import Dict, List, Any

from app.infrastructure.persistence.excel.data_utils import DEFAULT_CONFIG, REVENUE_ANALYSIS
from app.infrastructure.persistence.excel.excel_processing import extract_subsidiary_name_from_bytes
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Import AI analyzer for AI mode
try:
    from .llm_analyzer import LLMFinancialAnalyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("⚠️  AI analyzer not available - AI mode will be disabled")

def clean_numeric_value(val):
    """Convert value to numeric, handling various formats"""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace(',', '').replace(' ', ''))
    except:
        return 0.0

def fill_down_accounts(df: pd.DataFrame, account_col: str = 'Account_Description') -> pd.DataFrame:
    """
    Implement fill-down function for Column A (revenue accounts).
    Forward-fills account descriptions when cells are empty.
    """
    df = df.copy()

    # Forward fill the account column to handle merged cells
    if account_col in df.columns:
        # Only fill down if the cell is empty/NaN but not if it contains data
        mask = df[account_col].isna() | (df[account_col].astype(str).str.strip() == '')
        df.loc[mask, account_col] = df[account_col].ffill()

    return df

def identify_contribution_accounts(df: pd.DataFrame, contribution_col: str = 'Account_Code') -> pd.DataFrame:
    """
    Identify revenue accounts that have '01' in the contribution column (Column C).
    These represent the actual revenue contribution accounts.
    """
    df = df.copy()

    if contribution_col in df.columns:
        # Mark rows where contribution code is '01'
        df['Is_Revenue_Contribution'] = df[contribution_col].astype(str).str.strip() == '01'
    else:
        df['Is_Revenue_Contribution'] = False

    return df

def calculate_total_revenue_excluding_base(df: pd.DataFrame, month_cols: List[str],
                                         account_col: str = 'Account_Description',
                                         contribution_col: str = 'Account_Code',
                                         entity_col: str = 'Entity') -> dict:
    """
    Calculate total revenue excluding 511000000 (Revenue from sale and service provider).
    Only include accounts with '01' in contribution column (Column C).
    """
    # Filter for revenue accounts (511*) with contribution code '01'
    revenue_mask = (
        df[account_col].astype(str).str.contains('511', na=False) &
        (df[contribution_col].astype(str).str.strip() == '01') &
        ~df[account_col].astype(str).str.contains('511000000', na=False)  # Exclude base revenue
    )

    revenue_df = df[revenue_mask].copy()

    # Calculate monthly totals
    monthly_totals = {}
    for month in month_cols:
        if month in revenue_df.columns:
            monthly_totals[month] = revenue_df[month].sum()
        else:
            monthly_totals[month] = 0.0

    return {
        'monthly_totals': monthly_totals,
        'contributing_accounts': revenue_df[[account_col, entity_col] + month_cols].to_dict('records')
    }

def analyze_month_over_month_variance(monthly_totals: dict) -> List[dict]:
    """
    Perform month-by-month variance analysis for total revenue changes.

    Args:
        monthly_totals: Dictionary with months as keys and revenue totals as values

    Returns:
        List of variance analysis dictionaries for each month-to-month change
    """
    months = list(monthly_totals.keys())
    variance_analysis = []

    for i in range(1, len(months)):
        prev_month = months[i-1]
        curr_month = months[i]
        prev_value = monthly_totals[prev_month]
        curr_value = monthly_totals[curr_month]

        # Calculate variance
        absolute_change = curr_value - prev_value
        percentage_change = (absolute_change / prev_value * 100) if prev_value != 0 else 0

        # Skip if both absolute and percentage change are 0
        if absolute_change == 0 and percentage_change == 0:
            continue

        variance_analysis.append({
            'period_from': prev_month,
            'period_to': curr_month,
            'previous_revenue': prev_value,
            'current_revenue': curr_value,
            'absolute_change': absolute_change,
            'percentage_change': percentage_change,
            'change_direction': 'Increase' if absolute_change > 0 else 'Decrease' if absolute_change < 0 else 'No Change'
        })

    return variance_analysis

def analyze_revenue_stream_contributions(df: pd.DataFrame, month_cols: List[str],
                                       account_col: str = 'Account_Description',
                                       contribution_col: str = 'Account_Code',
                                       entity_col: str = 'Entity') -> dict:
    """
    Identify which revenue streams cause changes between months.
    Focus on accounts with '01' contribution code.

    Args:
        df: DataFrame with revenue data
        month_cols: List of month column names
        account_col: Column name for account descriptions
        contribution_col: Column name for contribution codes
        entity_col: Column name for vendor/customer entities

    Returns:
        Dictionary with revenue stream analysis by account
    """
    # Filter for contributing revenue accounts (511* with '01' code, excluding 511000000)
    revenue_mask = (
        df[account_col].astype(str).str.contains('511', na=False) &
        (df[contribution_col].astype(str).str.strip() == '01') &
        ~df[account_col].astype(str).str.contains('511000000', na=False)
    )

    revenue_df = df[revenue_mask].copy()

    # Group by account to analyze each revenue stream
    revenue_streams = {}

    for account in revenue_df[account_col].unique():
        if pd.isna(account):
            continue

        account_data = revenue_df[revenue_df[account_col] == account].copy()

        # Calculate monthly totals for this account
        monthly_totals = {}
        for month in month_cols:
            if month in account_data.columns:
                monthly_totals[month] = account_data[month].sum()
            else:
                monthly_totals[month] = 0.0

        # Calculate month-over-month changes for this stream
        month_changes = []
        months = list(monthly_totals.keys())

        for i in range(1, len(months)):
            prev_month = months[i-1]
            curr_month = months[i]
            prev_value = monthly_totals[prev_month]
            curr_value = monthly_totals[curr_month]

            absolute_change = curr_value - prev_value
            percentage_change = (absolute_change / prev_value * 100) if prev_value != 0 else 0

            # Skip if both absolute and percentage change are 0
            if absolute_change == 0 and percentage_change == 0:
                continue

            month_changes.append({
                'period_from': prev_month,
                'period_to': curr_month,
                'previous_value': prev_value,
                'current_value': curr_value,
                'absolute_change': absolute_change,
                'percentage_change': percentage_change
            })

        # Store entities/vendors for this account
        entities_data = []
        for _, row in account_data.iterrows():
            entity = row[entity_col] if entity_col in row and pd.notna(row[entity_col]) else 'Unknown'
            entity_monthly_values = {}
            for month in month_cols:
                if month in row:
                    entity_monthly_values[month] = row[month]

            entities_data.append({
                'entity': entity,
                'monthly_values': entity_monthly_values
            })

        revenue_streams[account] = {
            'monthly_totals': monthly_totals,
            'month_changes': month_changes,
            'entities': entities_data,
            'total_entities': len(entities_data)
        }

    return revenue_streams

def analyze_vendor_customer_impact(revenue_streams: dict, significance_threshold: float = 100000) -> dict:
    """
    Drill down to specific vendors/customers causing changes within each revenue stream.
    Shows net effect analysis: how positive and negative contributions combine to reach total delta.

    Args:
        revenue_streams: Dictionary from analyze_revenue_stream_contributions
        significance_threshold: Minimum change amount to be considered significant (VND)

    Returns:
        Dictionary with vendor/customer impact analysis including net effect breakdown
    """
    vendor_impact_analysis = {}

    for account, stream_data in revenue_streams.items():
        entities = stream_data['entities']
        month_changes = stream_data['month_changes']

        # For each month-to-month change period, identify contributing entities
        period_impacts = []

        for change_period in month_changes:
            prev_month = change_period['period_from']
            curr_month = change_period['period_to']
            total_change = change_period['absolute_change']

            # Calculate entity-level contributions for this period
            entity_contributions = []
            positive_contributors = []
            negative_contributors = []

            for entity_data in entities:
                entity_name = entity_data['entity']
                monthly_values = entity_data['monthly_values']

                prev_value = monthly_values.get(prev_month, 0)
                curr_value = monthly_values.get(curr_month, 0)
                entity_change = curr_value - prev_value

                if abs(entity_change) >= significance_threshold:
                    entity_percentage_change = (entity_change / prev_value * 100) if prev_value != 0 else 0
                    contribution_to_total_pct = (entity_change / total_change * 100) if total_change != 0 else 0

                    contribution_data = {
                        'entity': entity_name,
                        'previous_value': prev_value,
                        'current_value': curr_value,
                        'absolute_change': entity_change,
                        'percentage_change': entity_percentage_change,
                        'contribution_to_period_change': contribution_to_total_pct
                    }

                    entity_contributions.append(contribution_data)

                    # Separate positive and negative contributors
                    if entity_change > 0:
                        positive_contributors.append(contribution_data)
                    elif entity_change < 0:
                        negative_contributors.append(contribution_data)

            # Sort contributors by absolute change (most significant first)
            entity_contributions.sort(key=lambda x: abs(x['absolute_change']), reverse=True)
            positive_contributors.sort(key=lambda x: x['absolute_change'], reverse=True)  # Largest positive first
            negative_contributors.sort(key=lambda x: x['absolute_change'])  # Most negative first

            # Calculate net effect summary
            total_positive_change = sum(c['absolute_change'] for c in positive_contributors)
            total_negative_change = sum(c['absolute_change'] for c in negative_contributors)
            net_effect = total_positive_change + total_negative_change

            # Create net effect explanation
            net_effect_explanation = create_net_effect_explanation(
                total_change, positive_contributors, negative_contributors
            )

            # Skip if total_change is 0
            if total_change != 0:
                period_impacts.append({
                    'period_from': prev_month,
                    'period_to': curr_month,
                    'total_change': total_change,
                    'net_effect_calculated': net_effect,
                    'net_effect_explanation': net_effect_explanation,
                    'positive_contributors': positive_contributors[:3],  # Top 3 positive
                    'negative_contributors': negative_contributors[:3],  # Top 3 negative
                    'total_positive_change': total_positive_change,
                    'total_negative_change': total_negative_change,
                    'all_contributing_entities': entity_contributions[:10],  # Top 10 overall
                    'entities_with_significant_change': len(entity_contributions)
                })

        vendor_impact_analysis[account] = {
            'account_name': account,
            'period_impacts': period_impacts,
            'total_periods_analyzed': len(period_impacts)
        }

    return vendor_impact_analysis

def create_net_effect_explanation(total_change: float, positive_contributors: list, negative_contributors: list) -> str:
    """
    Create a human-readable explanation of how positive and negative contributions
    combine to reach the total delta.

    Example: "Net change +20,000 VND: Increase +35,000 VND (Customer A), Decrease -15,000 VND (Customer B)"
    """
    explanation_parts = []

    # Start with total change
    direction = "increase" if total_change >= 0 else "decrease"
    explanation_parts.append(f"Net {direction} {abs(total_change):,.0f} VND")

    # Add top positive contributors
    if positive_contributors:
        top_positive = positive_contributors[:2]  # Top 2 positive
        for contrib in top_positive:
            explanation_parts.append(f"increase +{contrib['absolute_change']:,.0f} VND ({contrib['entity']})")

    # Add top negative contributors
    if negative_contributors:
        top_negative = negative_contributors[:2]  # Top 2 negative
        for contrib in top_negative:
            explanation_parts.append(f"decrease {contrib['absolute_change']:,.0f} VND ({contrib['entity']})")

    # Handle case with more contributors
    total_contributors = len(positive_contributors) + len(negative_contributors)
    if total_contributors > 4:
        explanation_parts.append(f"...and {total_contributors - 4} other entities")

    # Join with appropriate separators
    if len(explanation_parts) == 1:
        return explanation_parts[0]
    elif len(explanation_parts) == 2:
        return explanation_parts[0] + ": " + explanation_parts[1]
    else:
        return explanation_parts[0] + ": " + ", ".join(explanation_parts[1:])

def generate_contribution_summary(positive_change: float, negative_change: float,
                                positive_contributors: list, negative_contributors: list) -> dict:
    """Generate a summary of positive vs negative contributions."""
    summary = {
        'total_positive': positive_change,
        'total_negative': negative_change,
        'net_effect': positive_change + negative_change,
        'positive_entity_count': len(positive_contributors),
        'negative_entity_count': len(negative_contributors),
        'dominant_direction': 'Positive' if positive_change > abs(negative_change) else 'Negative' if abs(negative_change) > positive_change else 'Balanced'
    }

    return summary

def analyze_revenue_variance_comprehensive(xl_bytes: bytes, filename: str, CONFIG: dict = DEFAULT_CONFIG) -> dict:
    """
    Comprehensive month-by-month revenue variance analysis as requested by the user.

    This function provides:
    1. Total revenue changes current month vs previous month
    2. Revenue stream analysis - which stream causes change (accounts with '01' code)
    3. Vendor/customer analysis - which specific vendors/customers cause change

    Args:
        xl_bytes: Excel file bytes
        filename: Name of the file
        CONFIG: Configuration dictionary

    Returns:
        Dictionary with comprehensive variance analysis results
    """
    try:
        from app.shared.utils.sheet_detection import detect_sheets_from_bytes

        # Detect PL sheet using fuzzy matching
        _, pl_sheet = detect_sheets_from_bytes(xl_bytes)

        if not pl_sheet:
            return {"error": "PL sheet not found in file"}

        xls = pd.ExcelFile(io.BytesIO(xl_bytes))

        if pl_sheet not in xls.sheet_names:
            return {"error": f"Detected PL sheet '{pl_sheet}' not found"}

        pl_df = pd.read_excel(xls, sheet_name=pl_sheet)

        # Find data start row
        data_start_row = None
        for i, row in pl_df.iterrows():
            if str(row.iloc[1]).strip().lower() == 'entity':
                data_start_row = i
                break

        if data_start_row is None:
            return {"error": "Could not find data start row with 'Entity' header"}

        # Extract month columns
        month_headers = pl_df.iloc[data_start_row + 1].fillna('').astype(str).tolist()
        MONTH_TOKENS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_cols = []
        for h2 in month_headers:
            h2s = str(h2)
            if any(tok in h2s for tok in MONTH_TOKENS):
                month_cols.append(h2s.strip())

        # Extract data
        data_df = pl_df.iloc[data_start_row + 2:].copy()
        actual_col_count = len(data_df.columns)
        new_columns = ['Account_Description', 'Entity', 'Account_Code']
        new_columns.extend(month_cols)
        while len(new_columns) < actual_col_count:
            new_columns.append(f'Extra_{len(new_columns)}')
        data_df.columns = new_columns[:actual_col_count]
        data_df = data_df.dropna(how='all')

        # Apply fill-down for Column A (Account_Description)
        logger.info("🔄 Applying fill-down function for revenue accounts...")
        data_df = fill_down_accounts(data_df, 'Account_Description')

        # Clean numeric columns
        for month in month_cols:
            if month in data_df.columns:
                data_df[month] = data_df[month].apply(clean_numeric_value)

        # Identify contribution accounts (Column C with '01')
        logger.info("🔍 Identifying contribution accounts with code '01'...")
        data_df = identify_contribution_accounts(data_df, 'Account_Code')

        # Extract subsidiary name
        subsidiary = extract_subsidiary_name_from_bytes(xl_bytes, filename)

        logger.info("📊 Starting comprehensive revenue variance analysis...")

        # 1. Calculate total revenue (excluding 511000000, only '01' contributions)
        logger.info("💰 Calculating total revenue excluding base account...")
        total_revenue_data = calculate_total_revenue_excluding_base(
            data_df, month_cols[:CONFIG["months_to_analyze"]],
            'Account_Description', 'Account_Code', 'Entity'
        )

        # 2. Perform month-over-month variance analysis
        logger.info("📈 Performing month-by-month variance analysis...")
        variance_analysis = analyze_month_over_month_variance(total_revenue_data['monthly_totals'])

        # 3. Analyze revenue stream contributions
        logger.info("🔍 Analyzing revenue stream contributions...")
        revenue_streams = analyze_revenue_stream_contributions(
            data_df, month_cols[:CONFIG["months_to_analyze"]],
            'Account_Description', 'Account_Code', 'Entity'
        )

        # 4. Analyze vendor/customer impacts
        logger.info("👥 Analyzing vendor/customer impacts...")
        vendor_impact = analyze_vendor_customer_impact(
            revenue_streams,
            CONFIG.get("revenue_entity_threshold_vnd", 100000)
        )

        # 5. Prepare comprehensive results
        results = {
            'subsidiary': subsidiary,
            'filename': filename,
            'months_analyzed': month_cols,  # Include ALL month columns (even if some are zero)
            'analysis_summary': {
                'total_revenue_streams': len(revenue_streams),
                'total_variance_periods': len(variance_analysis),
                'accounts_with_vendor_impact': len(vendor_impact)
            },

            # Core Analysis Results
            'total_revenue_analysis': {
                'monthly_totals': total_revenue_data['monthly_totals'],
                'month_over_month_changes': variance_analysis,
                'contributing_accounts_count': len(total_revenue_data['contributing_accounts'])
            },

            'revenue_stream_analysis': {
                'streams': revenue_streams,
                'stream_count': len(revenue_streams),
                'summary': {
                    stream_name: {
                        'total_periods': len(stream_data['month_changes']),
                        'total_entities': stream_data['total_entities'],
                        'latest_monthly_total': list(stream_data['monthly_totals'].values())[-1] if stream_data['monthly_totals'] else 0
                    }
                    for stream_name, stream_data in revenue_streams.items()
                }
            },

            'vendor_customer_impact': {
                'detailed_analysis': vendor_impact,
                'summary': {
                    account: {
                        'total_periods_analyzed': analysis_data['total_periods_analyzed'],
                        'periods_with_significant_entities': sum(1 for p in analysis_data['period_impacts'] if p['entities_with_significant_change'] > 0)
                    }
                    for account, analysis_data in vendor_impact.items()
                }
            },

            # High-level insights
            'key_insights': generate_key_insights(variance_analysis, revenue_streams, vendor_impact),

            'configuration_used': {
                'months_analyzed': CONFIG["months_to_analyze"],
                'revenue_entity_threshold_vnd': CONFIG.get("revenue_entity_threshold_vnd", 100000),
                'excluded_base_account': '511000000'
            }
        }

        logger.info("✅ Comprehensive revenue variance analysis completed successfully!")
        return results

    except Exception as e:
        logger.error(f"❌ Revenue variance analysis failed: {str(e)}", exc_info=True)
        return {"error": f"Revenue variance analysis failed: {str(e)}"}

def generate_key_insights(variance_analysis: List[dict], revenue_streams: dict, vendor_impact: dict) -> List[str]:
    """Generate high-level insights from the analysis results including net effect analysis."""
    insights = []

    # Total revenue insights
    if variance_analysis:
        largest_change = max(variance_analysis, key=lambda x: abs(x['absolute_change']))
        insights.append(
            f"Largest month-over-month change: {largest_change['change_direction']} of "
            f"{abs(largest_change['absolute_change']):,.0f} VND "
            f"({largest_change['percentage_change']:+.1f}%) from {largest_change['period_from']} to {largest_change['period_to']}"
        )

    # Revenue stream insights
    if revenue_streams:
        most_volatile_stream = None
        max_volatility = 0

        for stream_name, stream_data in revenue_streams.items():
            if stream_data['month_changes']:
                avg_abs_change = sum(abs(change['absolute_change']) for change in stream_data['month_changes']) / len(stream_data['month_changes'])
                if avg_abs_change > max_volatility:
                    max_volatility = avg_abs_change
                    most_volatile_stream = stream_name

        if most_volatile_stream:
            insights.append(f"Most volatile revenue stream: {most_volatile_stream} with average change of {max_volatility:,.0f} VND per month")

    # Net effect insights - find the most interesting net effect examples
    best_net_effect_example = None
    max_net_complexity = 0

    for account_data in vendor_impact.values():
        for period_impact in account_data['period_impacts']:
            positive_count = len(period_impact.get('positive_contributors', []))
            negative_count = len(period_impact.get('negative_contributors', []))
            total_complexity = positive_count + negative_count

            # Look for cases with both positive and negative contributors
            if positive_count > 0 and negative_count > 0 and total_complexity > max_net_complexity:
                max_net_complexity = total_complexity
                best_net_effect_example = period_impact

    if best_net_effect_example:
        net_explanation = best_net_effect_example.get('net_effect_explanation', '')
        insights.append(f"Complex net effect example: {net_explanation}")

    # Entity contribution insights
    total_significant_entities = sum(
        sum(period['entities_with_significant_change'] for period in account_data['period_impacts'])
        for account_data in vendor_impact.values()
    )

    if total_significant_entities > 0:
        insights.append(f"Total entities with significant changes across all periods: {total_significant_entities}")

    # Net effect balance insights
    total_periods_with_mixed_effects = 0
    for account_data in vendor_impact.values():
        for period_impact in account_data['period_impacts']:
            has_positive = len(period_impact.get('positive_contributors', [])) > 0
            has_negative = len(period_impact.get('negative_contributors', [])) > 0
            if has_positive and has_negative:
                total_periods_with_mixed_effects += 1

    if total_periods_with_mixed_effects > 0:
        insights.append(f"Mixed effects detected in {total_periods_with_mixed_effects} period(s) - changes driven by offsetting customer impacts")

    return insights

def _build_total_trend(group_accounts: dict, months: list[str]) -> dict:
    """
    Collapse {account: {'monthly_totals': {m: v}, ...}, ...}
    into a single {'monthly_totals': {...}, 'changes': [...] }.
    """
    if not group_accounts:
        return {'monthly_totals': {}, 'changes': []}

    monthly_totals = {m: 0.0 for m in months}
    for acc in group_accounts.values():
        mt = acc.get('monthly_totals', {})
        for m in months:
            monthly_totals[m] += float(mt.get(m, 0.0))

    changes = []
    for i in range(1, len(months)):
        prev_m, curr_m = months[i-1], months[i]
        prev_val, curr_val = monthly_totals[prev_m], monthly_totals[curr_m]
        delta = curr_val - prev_val
        pct = (delta / prev_val * 100.0) if prev_val != 0 else 0.0
        changes.append({
            'from': prev_m, 'to': curr_m,
            'prev_value': prev_val,
            'curr_value': curr_val,
            'change': delta,
            'pct_change': pct
        })
    return {'monthly_totals': monthly_totals, 'changes': changes}


def _build_total_trend_with_account_explanations(group_accounts: dict, months: list[str]) -> dict:
    """
    Enhanced version of _build_total_trend that includes detailed account-level explanations
    for what drives each VND change in the total trend.
    """
    if not group_accounts:
        return {'monthly_totals': {}, 'changes': [], 'account_explanations': {}}

    monthly_totals = {m: 0.0 for m in months}
    account_monthly_data = {}

    # Track each account's contribution by month
    for account_name, account_data in group_accounts.items():
        account_monthly_data[account_name] = {}
        mt = account_data.get('monthly_totals', {})
        for m in months:
            val = float(mt.get(m, 0.0))
            account_monthly_data[account_name][m] = val
            monthly_totals[m] += val

    changes = []
    account_explanations = {}

    for i in range(1, len(months)):
        prev_m, curr_m = months[i-1], months[i]
        prev_val, curr_val = monthly_totals[prev_m], monthly_totals[curr_m]
        delta = curr_val - prev_val
        pct = (delta / prev_val * 100.0) if prev_val != 0 else 0.0

        # Calculate each account's contribution to the total change
        account_contributions = []
        for account_name, account_months in account_monthly_data.items():
            account_prev = account_months.get(prev_m, 0.0)
            account_curr = account_months.get(curr_m, 0.0)
            account_change = account_curr - account_prev

            if abs(account_change) > 0:  # Only include accounts with changes
                account_pct_change = (account_change / account_prev * 100.0) if account_prev != 0 else 0.0
                contribution_to_total_pct = (account_change / abs(delta) * 100.0) if delta != 0 else 0.0

                account_contributions.append({
                    'account': account_name,
                    'change': account_change,
                    'pct_change': account_pct_change,
                    'prev_value': account_prev,
                    'curr_value': account_curr,
                    'contribution_to_total_pct': contribution_to_total_pct
                })

        # Sort by absolute change amount (largest changes first)
        account_contributions.sort(key=lambda x: abs(x['change']), reverse=True)

        # Create explanation text
        period_key = f"{prev_m}_to_{curr_m}"
        if account_contributions:
            top_3_accounts = account_contributions[:3]  # Show top 3 contributors
            explanation_parts = []

            for contrib in top_3_accounts:
                direction = "increased" if contrib['change'] > 0 else "decreased"
                explanation_parts.append(
                    f"• {contrib['account']}: {direction} by {abs(contrib['change']):,.0f} VND "
                    f"({contrib['pct_change']:+.1f}%, {abs(contrib['contribution_to_total_pct']):.1f}% of total change)"
                )

            if len(account_contributions) > 3:
                remaining_change = sum(abs(c['change']) for c in account_contributions[3:])
                explanation_parts.append(f"• Other {len(account_contributions) - 3} accounts: {remaining_change:,.0f} VND")

            account_explanations[period_key] = {
                'summary': f"Total change of {delta:+,.0f} VND ({pct:+.1f}%) explained by:",
                'detailed_breakdown': explanation_parts,
                'all_account_contributions': account_contributions
            }
        else:
            account_explanations[period_key] = {
                'summary': f"No significant account changes detected for {delta:+,.0f} VND ({pct:+.1f}%) total change",
                'detailed_breakdown': [],
                'all_account_contributions': []
            }

        # Create account breakdown text for the extra column
        account_breakdown_text = ""
        if account_contributions:
            top_contributors = account_contributions[:3]
            breakdown_parts = []
            for contrib in top_contributors:
                direction = "↑" if contrib['change'] > 0 else "↓"
                breakdown_parts.append(f"{contrib['account']}: {direction}{abs(contrib['change']):,.0f} VND ({contrib['pct_change']:+.1f}%)")

            if len(account_contributions) > 3:
                breakdown_parts.append(f"...+{len(account_contributions) - 3} more accounts")

            account_breakdown_text = " | ".join(breakdown_parts)

        changes.append({
            'from': prev_m, 'to': curr_m,
            'prev_value': prev_val,
            'curr_value': curr_val,
            'change': delta,
            'pct_change': pct,
            'account_breakdown': account_breakdown_text,  # New extra column data
            'account_contributions': account_contributions[:5]  # Top 5 for API response
        })

    return {
        'monthly_totals': monthly_totals,
        'changes': changes,
        'account_explanations': account_explanations
    }


def analyze_comprehensive_revenue_impact_from_bytes(xl_bytes: bytes, filename: str, CONFIG: dict = DEFAULT_CONFIG) -> dict:
    """
    Comprehensive Revenue Impact Analysis
    Answers specific questions:
    1. If revenue increases (511*), which specific revenue accounts drive the increase?
    2. Which customers/entities drive the revenue changes for each account?
    3. Gross margin analysis: (Revenue - Cost)/Revenue and risk identification
    4. Utility revenue vs cost pairing analysis
    5. SG&A expense analysis (641* and 642*) for expense management insights
    """
    try:
        from app.shared.utils.sheet_detection import detect_sheets_from_bytes

        # Detect PL sheet using fuzzy matching
        _, pl_sheet = detect_sheets_from_bytes(xl_bytes)

        if not pl_sheet:
            return {"error": "PL sheet not found in file"}

        xls = pd.ExcelFile(io.BytesIO(xl_bytes))

        if pl_sheet not in xls.sheet_names:
            return {"error": f"Detected PL sheet '{pl_sheet}' not found"}

        pl_df = pd.read_excel(xls, sheet_name=pl_sheet)

        # Find data start row
        data_start_row = None
        for i, row in pl_df.iterrows():
            if str(row.iloc[1]).strip().lower() == 'entity':
                data_start_row = i
                break

        if data_start_row is None:
            return {"error": "Could not find data start row with 'Entity' header"}

        # Extract month columns
        # Replace your month detection block with:
        month_headers = pl_df.iloc[data_start_row + 1].fillna('').astype(str).tolist()
        MONTH_TOKENS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_cols = []
        for h2 in month_headers:
            h2s = str(h2)
            if any(tok in h2s for tok in MONTH_TOKENS):
                month_cols.append(h2s.strip())
        # --- End replacement ---

        # Extract data
        data_df = pl_df.iloc[data_start_row + 2:].copy()
        actual_col_count = len(data_df.columns)
        new_columns = ['Account_Description', 'Entity', 'Account_Code']
        new_columns.extend(month_cols)
        while len(new_columns) < actual_col_count:
            new_columns.append(f'Extra_{len(new_columns)}')
        data_df.columns = new_columns[:actual_col_count]
        data_df = data_df.dropna(how='all')

        # Extract subsidiary name
        subsidiary = extract_subsidiary_name_from_bytes(xl_bytes, filename)

        analysis_result = {
            'subsidiary': subsidiary,
            'filename': filename,
            'months_analyzed': month_cols,  # Include ALL month columns (even if some are zero)
            'total_revenue_analysis': {},
            'revenue_by_account': {},
            'gross_margin_analysis': {},
            'utility_analysis': {},
            'sga_641_analysis': {},
            'sga_642_analysis': {},
            'combined_sga_analysis': {},
            'risk_assessment': []
        }

        # =====================================
        # 1. TOTAL REVENUE ANALYSIS (511*)
        # =====================================

        total_revenue_by_month = {}
        for month in month_cols[:CONFIG["months_to_analyze"]]:
            month_total = 0
            for i, row in data_df.iterrows():
                entity = str(row['Entity']) if 'Entity' in row and pd.notna(row['Entity']) else ''
                if entity and entity != 'nan' and not entity.startswith('Total'):
                    # Check if under a 511* revenue account
                    for prev_i in range(max(0, i-CONFIG["lookback_periods"]), i):
                        if prev_i < len(data_df):
                            prev_desc = str(data_df.iloc[prev_i]['Account_Description']) if pd.notna(data_df.iloc[prev_i]['Account_Description']) else ''
                            if '511' in prev_desc and 'revenue' in prev_desc.lower():
                                val = clean_numeric_value(row[month])
                                month_total += val
                                break
            total_revenue_by_month[month] = month_total

        # Calculate month-over-month changes (basic version first)
        months = list(total_revenue_by_month.keys())
        total_revenue_changes = []
        for i in range(1, len(months)):
            prev_month = months[i-1]
            curr_month = months[i]
            prev_revenue = total_revenue_by_month[prev_month]
            curr_revenue = total_revenue_by_month[curr_month]
            change = curr_revenue - prev_revenue
            pct_change = (change / prev_revenue * 100) if prev_revenue != 0 else 0

            total_revenue_changes.append({
                'from': prev_month,
                'to': curr_month,
                'prev_value': prev_revenue,
                'curr_value': curr_revenue,
                'change': change,
                'pct_change': pct_change,
                'account_breakdown': ''  # Will be populated later after revenue_accounts is built
            })

        # Store initial analysis result - will be enhanced after revenue_accounts is populated
        analysis_result['total_revenue_analysis'] = {
            'monthly_totals': total_revenue_by_month,
            'changes': total_revenue_changes
        }

        # =====================================
        # 2. REVENUE BY ACCOUNT TYPE (511.xxx)
        # =====================================

        revenue_accounts = {}
        current_account = None

        for i, row in data_df.iterrows():
            account_desc = str(row['Account_Description']) if pd.notna(row['Account_Description']) else ''
            entity = str(row['Entity']) if pd.notna(row['Entity']) else ''

            # Revenue account headers
            if '511' in account_desc and 'revenue' in account_desc.lower():
                current_account = account_desc
                if current_account not in revenue_accounts:
                    revenue_accounts[current_account] = {
                        'entities': {},
                        'monthly_totals': {month: 0 for month in month_cols[:CONFIG["months_to_analyze"]]}
                    }

            # Entity data under current account
            elif current_account and entity and entity != 'nan' and not entity.startswith('Total'):
                if entity not in revenue_accounts[current_account]['entities']:
                    revenue_accounts[current_account]['entities'][entity] = {}

                for month in month_cols[:CONFIG["months_to_analyze"]]:
                    val = clean_numeric_value(row[month])
                    revenue_accounts[current_account]['entities'][entity][month] = val
                    revenue_accounts[current_account]['monthly_totals'][month] += val

        # Analyze each revenue account
        for account, data in revenue_accounts.items():
            months = list(data['monthly_totals'].keys())
            account_changes = []

            for i in range(1, len(months)):
                prev_month = months[i-1]
                curr_month = months[i]
                prev_val = data['monthly_totals'][prev_month]
                curr_val = data['monthly_totals'][curr_month]
                change = curr_val - prev_val
                pct_change = (change / prev_val * 100) if prev_val != 0 else 0

                # Skip if both change and pct_change are 0
                if change == 0 and pct_change == 0:
                    continue

                account_changes.append({
                    'from': prev_month,
                    'to': curr_month,
                    'change': change,
                    'pct_change': pct_change,
                    'prev_val': prev_val,
                    'curr_val': curr_val
                })

            # Find biggest change for customer analysis
            biggest_change = max(account_changes, key=lambda x: abs(x['change'])) if account_changes else None
            customer_impacts = []

            if biggest_change and abs(biggest_change['change']) > CONFIG["revenue_change_threshold_vnd"]:
                for entity, entity_data in data['entities'].items():
                    prev_val = entity_data.get(biggest_change['from'], 0)
                    curr_val = entity_data.get(biggest_change['to'], 0)
                    entity_change = curr_val - prev_val

                    if abs(entity_change) > CONFIG["revenue_entity_threshold_vnd"]:
                        customer_impacts.append({
                            'entity': entity,
                            'change': entity_change,
                            'prev_val': prev_val,
                            'curr_val': curr_val,
                            'pct_change': (entity_change / prev_val * 100) if prev_val != 0 else 0
                        })

                customer_impacts.sort(key=lambda x: abs(x['change']), reverse=True)

            revenue_accounts[account]['changes'] = account_changes
            revenue_accounts[account]['biggest_change'] = biggest_change
            revenue_accounts[account]['customer_impacts'] = customer_impacts[:CONFIG["top_entity_impacts"]]

        analysis_result['revenue_by_account'] = revenue_accounts

        # Now that revenue_accounts is built, create enhanced trend analysis with account breakdown
        revenue_accounts_for_trend = {}
        for account_name, account_data in revenue_accounts.items():
            if '511' in account_name:
                revenue_accounts_for_trend[account_name] = account_data

        # Use enhanced trend analysis for total revenue (511*)
        enhanced_revenue_trend = _build_total_trend_with_account_explanations(revenue_accounts_for_trend, months)

        # Update the total_revenue_changes with account breakdown info
        if enhanced_revenue_trend and enhanced_revenue_trend.get('changes'):
            for i, total_change in enumerate(analysis_result['total_revenue_analysis']['changes']):
                for period_change in enhanced_revenue_trend['changes']:
                    if (period_change['from'] == total_change['from'] and
                        period_change['to'] == total_change['to']):
                        analysis_result['total_revenue_analysis']['changes'][i]['account_breakdown'] = period_change.get('account_breakdown', '')
                        break

        # Add enhanced trend data to total_revenue_analysis
        analysis_result['total_revenue_analysis']['enhanced_trend'] = enhanced_revenue_trend
        analysis_result['total_revenue_analysis']['account_explanations'] = enhanced_revenue_trend.get('account_explanations', {})
        analysis_result['total_revenue_analysis']['detailed_account_contributions'] = enhanced_revenue_trend.get('changes', [])

        # =====================================
        # 3. GROSS MARGIN ANALYSIS
        # =====================================

        cost_accounts = {}
        current_cost_account = None

        # Look for 632* cost accounts
        for i, row in data_df.iterrows():
            account_desc = str(row['Account_Description']) if pd.notna(row['Account_Description']) else ''
            entity = str(row['Entity']) if pd.notna(row['Entity']) else ''

            if '632' in account_desc and 'cost' in account_desc.lower():
                current_cost_account = account_desc
                if current_cost_account not in cost_accounts:
                    cost_accounts[current_cost_account] = {
                        'entities': {},
                        'monthly_totals': {month: 0 for month in month_cols[:CONFIG["months_to_analyze"]]}
                    }

            elif current_cost_account and entity and entity != 'nan' and not entity.startswith('Total'):
                if entity not in cost_accounts[current_cost_account]['entities']:
                    cost_accounts[current_cost_account]['entities'][entity] = {}

                for month in month_cols[:CONFIG["months_to_analyze"]]:
                    val = clean_numeric_value(row[month])
                    cost_accounts[current_cost_account]['entities'][entity][month] = val
                    cost_accounts[current_cost_account]['monthly_totals'][month] += val

        # Calculate gross margins
        gross_margin_trend = []
        for i in range(len(months)):
            month = months[i]
            total_revenue = total_revenue_by_month[month]
            total_cost = sum([cost_data['monthly_totals'][month] for cost_data in cost_accounts.values()])

            if total_revenue > 0:
                gross_margin_pct = ((total_revenue - total_cost) / total_revenue) * 100
                gross_margin_trend.append({
                    'month': month,
                    'revenue': total_revenue,
                    'cost': total_cost,
                    'gross_margin_pct': gross_margin_pct
                })

                if i > 0:
                    prev_gm = gross_margin_trend[i-1]['gross_margin_pct']
                    gm_change = gross_margin_pct - prev_gm

                    if abs(gm_change) > CONFIG["gross_margin_change_threshold_pct"]:
                        risk_level = "HIGH" if gm_change < CONFIG["high_gross_margin_risk_threshold_pct"] else "MEDIUM"
                        analysis_result['risk_assessment'].append({
                            'type': 'Gross Margin Change',
                            'period': f"{gross_margin_trend[i-1]['month']} → {month}",
                            'change': gm_change,
                            'risk_level': risk_level,
                            'description': f"Gross margin changed by {gm_change:+.1f}%"
                        })

        analysis_result['gross_margin_analysis'] = {
            'trend': gross_margin_trend,
            'cost_accounts': cost_accounts
        }

        # --- NEW: COGS 632* account analysis (place after cost_accounts is populated) ---
        cogs_632_accounts = {}
        for account, data in cost_accounts.items():
            # only consider true 632* buckets (avoid "Total ..." rows)
            if account and '632' in account and not account.strip().lower().startswith('total'):
                months_list = list(data['monthly_totals'].keys())
                account_changes = []
                for i in range(1, len(months_list)):
                    prev_m = months_list[i-1]
                    curr_m = months_list[i]
                    prev_val = data['monthly_totals'][prev_m]
                    curr_val = data['monthly_totals'][curr_m]
                    delta = curr_val - prev_val
                    pct = (delta / prev_val * 100.0) if prev_val != 0 else 0.0

                    # Skip if both delta and pct are 0
                    if delta == 0 and pct == 0:
                        continue

                    account_changes.append({
                        'from': prev_m, 'to': curr_m,
                        'change': delta, 'pct_change': pct,
                        'prev_val': prev_val, 'curr_val': curr_val
                    })

                # biggest movement period for entity drilldown (lower thresholds than revenue)
                biggest = max(account_changes, key=lambda x: abs(x['change'])) if account_changes else None
                entity_impacts = []
                if biggest and abs(biggest['change']) > CONFIG["cogs_change_threshold_vnd"]:
                    for entity, entity_months in data.get('entities', {}).items():
                        prev_v = entity_months.get(biggest['from'], 0.0)
                        curr_v = entity_months.get(biggest['to'], 0.0)
                        e_delta = curr_v - prev_v
                        if abs(e_delta) > CONFIG["cogs_entity_threshold_vnd"]:
                            entity_impacts.append({
                                'entity': entity,
                                'change': e_delta,
                                'prev_val': prev_v,
                                'curr_val': curr_v,
                                'pct_change': (e_delta / prev_v * 100.0) if prev_v != 0 else 0.0
                            })
                entity_impacts.sort(key=lambda x: abs(x['change']), reverse=True)

                cogs_632_accounts[account] = {
                    'monthly_totals': data['monthly_totals'],
                    'changes': account_changes,
                    'biggest_change': biggest,
                    'entity_impacts': entity_impacts[:CONFIG["top_entity_impacts"]],
                }

        # attach to analysis result so the Excel writer can render it
        analysis_result['cogs_632_analysis'] = cogs_632_accounts

        analysis_result['total_632_trend'] = _build_total_trend_with_account_explanations(cogs_632_accounts, months)

        # =====================================
        # 4. UTILITY ANALYSIS
        # =====================================

        utility_revenue = None
        utility_cost = None

        for account, data in revenue_accounts.items():
            if 'utilit' in account.lower():
                utility_revenue = data
                break

        for account, data in cost_accounts.items():
            if 'utilit' in account.lower():
                utility_cost = data
                break

        if utility_revenue and utility_cost:
            utility_margins = []
            for month in months:
                rev = utility_revenue['monthly_totals'][month]
                cost = utility_cost['monthly_totals'][month]
                if rev > 0:
                    gm_pct = ((rev - cost) / rev) * 100
                    utility_margins.append({
                        'month': month,
                        'revenue': rev,
                        'cost': cost,
                        'margin_pct': gm_pct
                    })

            analysis_result['utility_analysis'] = {
                'available': True,
                'margins': utility_margins
            }
        else:
            analysis_result['utility_analysis'] = {
                'available': False,
                'reason': 'Could not find matching utility revenue/cost accounts'
            }

        # =====================================
        # 5. SG&A ANALYSIS (641* Accounts)
        # =====================================

        sga_641_accounts = {}
        current_sga_641_account = None

        # Look for 641* SG&A accounts
        for i, row in data_df.iterrows():
            account_desc = str(row['Account_Description']) if pd.notna(row['Account_Description']) else ''
            entity = str(row['Entity']) if pd.notna(row['Entity']) else ''

            if '641' in account_desc:
                current_sga_641_account = account_desc
                if current_sga_641_account not in sga_641_accounts:
                    sga_641_accounts[current_sga_641_account] = {
                        'entities': {},
                        'monthly_totals': {month: 0 for month in month_cols[:CONFIG["months_to_analyze"]]}
                    }

            elif current_sga_641_account and entity and entity != 'nan' and not entity.startswith('Total'):
                if entity not in sga_641_accounts[current_sga_641_account]['entities']:
                    sga_641_accounts[current_sga_641_account]['entities'][entity] = {}

                for month in month_cols[:CONFIG["months_to_analyze"]]:
                    val = clean_numeric_value(row[month])
                    sga_641_accounts[current_sga_641_account]['entities'][entity][month] = val
                    sga_641_accounts[current_sga_641_account]['monthly_totals'][month] += val

        # Analyze each 641 account for changes
        for account, data in sga_641_accounts.items():
            months_list = list(data['monthly_totals'].keys())
            account_changes = []

            for i in range(1, len(months_list)):
                prev_month = months_list[i-1]
                curr_month = months_list[i]
                prev_val = data['monthly_totals'][prev_month]
                curr_val = data['monthly_totals'][curr_month]
                change = curr_val - prev_val
                pct_change = (change / prev_val * 100) if prev_val != 0 else 0

                # Skip if both change and pct_change are 0
                if change == 0 and pct_change == 0:
                    continue

                account_changes.append({
                    'from': prev_month,
                    'to': curr_month,
                    'change': change,
                    'pct_change': pct_change,
                    'prev_val': prev_val,
                    'curr_val': curr_val
                })

            # Find biggest change for entity analysis
            biggest_change = max(account_changes, key=lambda x: abs(x['change'])) if account_changes else None
            entity_impacts = []

            if biggest_change and abs(biggest_change['change']) > CONFIG["sga_change_threshold_vnd"]:
                for entity, entity_data in data['entities'].items():
                    prev_val = entity_data.get(biggest_change['from'], 0)
                    curr_val = entity_data.get(biggest_change['to'], 0)
                    entity_change = curr_val - prev_val

                    if abs(entity_change) > CONFIG["sga_entity_threshold_vnd"]:
                        entity_impacts.append({
                            'entity': entity,
                            'change': entity_change,
                            'prev_val': prev_val,
                            'curr_val': curr_val,
                            'pct_change': (entity_change / prev_val * 100) if prev_val != 0 else 0
                        })

                entity_impacts.sort(key=lambda x: abs(x['change']), reverse=True)

            sga_641_accounts[account]['changes'] = account_changes
            sga_641_accounts[account]['biggest_change'] = biggest_change
            sga_641_accounts[account]['entity_impacts'] = entity_impacts[:CONFIG["top_entity_impacts"]]

        analysis_result['sga_641_analysis'] = sga_641_accounts

        analysis_result['total_641_trend'] = _build_total_trend_with_account_explanations(sga_641_accounts, months)

        # =====================================
        # 6. SG&A ANALYSIS (642* Accounts)
        # =====================================

        sga_642_accounts = {}
        current_sga_642_account = None

        # Look for 642* SG&A accounts
        for i, row in data_df.iterrows():
            account_desc = str(row['Account_Description']) if pd.notna(row['Account_Description']) else ''
            entity = str(row['Entity']) if pd.notna(row['Entity']) else ''

            if '642' in account_desc:
                current_sga_642_account = account_desc
                if current_sga_642_account not in sga_642_accounts:
                    sga_642_accounts[current_sga_642_account] = {
                        'entities': {},
                        'monthly_totals': {month: 0 for month in month_cols[:CONFIG["months_to_analyze"]]}
                    }

            elif current_sga_642_account and entity and entity != 'nan' and not entity.startswith('Total'):
                if entity not in sga_642_accounts[current_sga_642_account]['entities']:
                    sga_642_accounts[current_sga_642_account]['entities'][entity] = {}

                for month in month_cols[:CONFIG["months_to_analyze"]]:
                    val = clean_numeric_value(row[month])
                    sga_642_accounts[current_sga_642_account]['entities'][entity][month] = val
                    sga_642_accounts[current_sga_642_account]['monthly_totals'][month] += val

        # Analyze each 642 account for changes
        for account, data in sga_642_accounts.items():
            months_list = list(data['monthly_totals'].keys())
            account_changes = []

            for i in range(1, len(months_list)):
                prev_month = months_list[i-1]
                curr_month = months_list[i]
                prev_val = data['monthly_totals'][prev_month]
                curr_val = data['monthly_totals'][curr_month]
                change = curr_val - prev_val
                pct_change = (change / prev_val * 100) if prev_val != 0 else 0

                # Skip if both change and pct_change are 0
                if change == 0 and pct_change == 0:
                    continue

                account_changes.append({
                    'from': prev_month,
                    'to': curr_month,
                    'change': change,
                    'pct_change': pct_change,
                    'prev_val': prev_val,
                    'curr_val': curr_val
                })

            # Find biggest change for entity analysis
            biggest_change = max(account_changes, key=lambda x: abs(x['change'])) if account_changes else None
            entity_impacts = []

            if biggest_change and abs(biggest_change['change']) > CONFIG["sga_change_threshold_vnd"]:
                for entity, entity_data in data['entities'].items():
                    prev_val = entity_data.get(biggest_change['from'], 0)
                    curr_val = entity_data.get(biggest_change['to'], 0)
                    entity_change = curr_val - prev_val

                    if abs(entity_change) > CONFIG["sga_entity_threshold_vnd"]:
                        entity_impacts.append({
                            'entity': entity,
                            'change': entity_change,
                            'prev_val': prev_val,
                            'curr_val': curr_val,
                            'pct_change': (entity_change / prev_val * 100) if prev_val != 0 else 0
                        })

                entity_impacts.sort(key=lambda x: abs(x['change']), reverse=True)

            sga_642_accounts[account]['changes'] = account_changes
            sga_642_accounts[account]['biggest_change'] = biggest_change
            sga_642_accounts[account]['entity_impacts'] = entity_impacts[:CONFIG["top_entity_impacts"]]

        analysis_result['sga_642_analysis'] = sga_642_accounts

        analysis_result['total_642_trend'] = _build_total_trend_with_account_explanations(sga_642_accounts, months)

        # =====================================
        # 7. COMBINED SG&A ANALYSIS (641* + 642*)
        # =====================================

        # Calculate total SG&A expenses by month
        total_sga_by_month = {}
        for month in month_cols[:CONFIG["months_to_analyze"]]:
            total_641 = sum([account_data['monthly_totals'][month] for account_data in sga_641_accounts.values()])
            total_642 = sum([account_data['monthly_totals'][month] for account_data in sga_642_accounts.values()])
            total_sga_by_month[month] = total_641 + total_642

        # Calculate SG&A as percentage of revenue
        sga_ratio_trend = []
        for month in months:
            total_revenue = total_revenue_by_month[month]
            total_sga = total_sga_by_month[month]

            if total_revenue > 0:
                sga_ratio_pct = (total_sga / total_revenue) * 100
                sga_ratio_trend.append({
                    'month': month,
                    'revenue': total_revenue,
                    'sga_641_total': sum([account_data['monthly_totals'][month] for account_data in sga_641_accounts.values()]),
                    'sga_642_total': sum([account_data['monthly_totals'][month] for account_data in sga_642_accounts.values()]),
                    'total_sga': total_sga,
                    'sga_ratio_pct': sga_ratio_pct
                })

                # Risk assessment for SG&A ratio changes
                if len(sga_ratio_trend) > 1:
                    prev_ratio = sga_ratio_trend[-2]['sga_ratio_pct']
                    ratio_change = sga_ratio_pct - prev_ratio

                    if abs(ratio_change) > CONFIG["sga_ratio_change_threshold_pct"]:
                        risk_level = "HIGH" if ratio_change > CONFIG["high_sga_ratio_threshold_pct"] else "MEDIUM"
                        analysis_result['risk_assessment'].append({
                            'type': 'SG&A Ratio Change',
                            'period': f"{sga_ratio_trend[-2]['month']} → {month}",
                            'change': ratio_change,
                            'risk_level': risk_level,
                            'description': f"SG&A ratio changed by {ratio_change:+.1f}% (now {sga_ratio_pct:.1f}% of revenue)"
                        })

        analysis_result['combined_sga_analysis'] = {
            'monthly_totals': total_sga_by_month,
            'ratio_trend': sga_ratio_trend,
            'total_641_accounts': len(sga_641_accounts),
            'total_642_accounts': len(sga_642_accounts)
        }

        # =====================================
        # 8. SUMMARY METRICS
        # =====================================

        analysis_result['summary'] = {
            'total_accounts': len([a for a in revenue_accounts if not a.startswith('Total')]),
            'highest_variance_account': max(revenue_accounts.items(),
                                          key=lambda x: max([abs(c['change']) for c in x[1].get('changes', [])], default=0))[0] if revenue_accounts else None,
            'total_revenue_latest': total_revenue_by_month[months[-1]] if months else 0,
            'gross_margin_latest': gross_margin_trend[-1]['gross_margin_pct'] if gross_margin_trend else 0,
            'total_sga_641_accounts': len(sga_641_accounts),
            'total_sga_642_accounts': len(sga_642_accounts),
            'total_sga_latest': total_sga_by_month[months[-1]] if months and total_sga_by_month else 0,
            'sga_ratio_latest': sga_ratio_trend[-1]['sga_ratio_pct'] if sga_ratio_trend else 0,
            'risk_periods': [r for r in analysis_result['risk_assessment'] if r['risk_level'] == 'HIGH']
        }

        return analysis_result

    except Exception as e:
        return {"error": f"Comprehensive analysis failed: {str(e)}"}

def analyze_revenue_impact_from_bytes(xl_bytes: bytes, filename: str, CONFIG: dict = DEFAULT_CONFIG) -> dict:
    """
    Comprehensive revenue impact analysis from Excel bytes
    Returns structured data for frontend display
    """
    try:
        from app.shared.utils.sheet_detection import detect_sheets_from_bytes

        # Detect PL sheet using fuzzy matching
        _, pl_sheet = detect_sheets_from_bytes(xl_bytes)

        if not pl_sheet:
            return {"error": "PL sheet not found in file"}

        xls = pd.ExcelFile(io.BytesIO(xl_bytes))

        if pl_sheet not in xls.sheet_names:
            return {"error": f"Detected PL sheet '{pl_sheet}' not found"}

        pl_df = pd.read_excel(xls, sheet_name=pl_sheet)

        # Find data start row
        data_start_row = None
        for i, row in pl_df.iterrows():
            if str(row.iloc[1]).strip().lower() == 'entity':
                data_start_row = i
                break

        if data_start_row is None:
            return {"error": "Could not find data start row with 'Entity' header"}

        # Extract month columns
        month_headers = pl_df.iloc[data_start_row + 1].fillna('').astype(str).tolist()
        month_cols = []
        for h2 in month_headers:
            if any(month in str(h2) for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                month_cols.append(str(h2).strip())

        # Extract data
        data_df = pl_df.iloc[data_start_row + 2:].copy()
        actual_col_count = len(data_df.columns)
        new_columns = ['Account_Description', 'Entity', 'Account_Code']
        new_columns.extend(month_cols)
        while len(new_columns) < actual_col_count:
            new_columns.append(f'Extra_{len(new_columns)}')
        data_df.columns = new_columns[:actual_col_count]
        data_df = data_df.dropna(how='all')

        # Extract subsidiary name
        subsidiary = extract_subsidiary_name_from_bytes(xl_bytes, filename)

        # 1. Calculate total revenue by month
        total_revenue_by_month = {}
        for month in month_cols[:CONFIG["months_to_analyze"]]:
            month_total = 0
            for i, row in data_df.iterrows():
                entity = str(row['Entity']) if 'Entity' in row and pd.notna(row['Entity']) else ''
                if entity and entity != 'nan' and not entity.startswith('Total'):
                    # Check if under a 511* revenue account
                    for prev_i in range(max(0, i-CONFIG["lookback_periods"]), i):
                        if prev_i < len(data_df):
                            prev_desc = str(data_df.iloc[prev_i]['Account_Description']) if pd.notna(data_df.iloc[prev_i]['Account_Description']) else ''
                            if '511' in prev_desc and 'revenue' in prev_desc.lower():
                                val = clean_numeric_value(row[month])
                                month_total += val
                                break
            total_revenue_by_month[month] = month_total

        # 2. Analyze revenue by account type
        revenue_accounts = {}
        current_account = None

        for i, row in data_df.iterrows():
            account_desc = str(row['Account_Description']) if pd.notna(row['Account_Description']) else ''
            entity = str(row['Entity']) if pd.notna(row['Entity']) else ''

            # Revenue account headers
            if '511' in account_desc and 'revenue' in account_desc.lower():
                current_account = account_desc
                if current_account not in revenue_accounts:
                    revenue_accounts[current_account] = {
                        'entities': {},
                        'monthly_totals': {month: 0 for month in month_cols[:CONFIG["months_to_analyze"]]}
                    }

            # Entity data under current account
            elif current_account and entity and entity != 'nan' and not entity.startswith('Total'):
                if entity not in revenue_accounts[current_account]['entities']:
                    revenue_accounts[current_account]['entities'][entity] = {}

                for month in month_cols[:CONFIG["months_to_analyze"]]:
                    val = clean_numeric_value(row[month])
                    revenue_accounts[current_account]['entities'][entity][month] = val
                    revenue_accounts[current_account]['monthly_totals'][month] += val

        # 3. Calculate changes and impacts
        months = list(total_revenue_by_month.keys())
        total_revenue_changes = []

        for i in range(1, len(months)):
            prev_month = months[i-1]
            curr_month = months[i]
            prev_val = total_revenue_by_month[prev_month]
            curr_val = total_revenue_by_month[curr_month]
            change = curr_val - prev_val
            pct_change = (change / prev_val * 100) if prev_val != 0 else 0

            total_revenue_changes.append({
                'from': prev_month,
                'to': curr_month,
                'prev_value': prev_val,
                'curr_value': curr_val,
                'change': change,
                'pct_change': pct_change,
                'account_breakdown': ''  # Empty for basic analysis - could be enhanced later
            })

        # 4. Account-level analysis with customer breakdowns
        account_analysis = []

        for account, data in revenue_accounts.items():
            if not account.startswith('Total'):  # Skip total rows
                months = list(data['monthly_totals'].keys())
                account_changes = []

                for i in range(1, len(months)):
                    prev_month = months[i-1]
                    curr_month = months[i]
                    prev_val = data['monthly_totals'][prev_month]
                    curr_val = data['monthly_totals'][curr_month]
                    change = curr_val - prev_val
                    pct_change = (change / prev_val * 100) if prev_val != 0 else 0

                    # Skip if both change and pct_change are 0
                    if change == 0 and pct_change == 0:
                        continue

                    account_changes.append({
                        'from': prev_month,
                        'to': curr_month,
                        'change': change,
                        'pct_change': pct_change,
                        'prev_val': prev_val,
                        'curr_val': curr_val
                    })

                # Find biggest change for customer analysis
                biggest_change = max(account_changes, key=lambda x: abs(x['change'])) if account_changes else None
                customer_impacts = []

                if biggest_change and abs(biggest_change['change']) > CONFIG["revenue_change_threshold_vnd"]:
                    for entity, entity_data in data['entities'].items():
                        prev_val = entity_data.get(biggest_change['from'], 0)
                        curr_val = entity_data.get(biggest_change['to'], 0)
                        entity_change = curr_val - prev_val

                        if abs(entity_change) > CONFIG["revenue_entity_threshold_vnd"]:
                            customer_impacts.append({
                                'entity': entity,
                                'change': entity_change,
                                'prev_val': prev_val,
                                'curr_val': curr_val,
                                'pct_change': (entity_change / prev_val * 100) if prev_val != 0 else 0
                            })

                    customer_impacts.sort(key=lambda x: abs(x['change']), reverse=True)

                account_analysis.append({
                    'account': account,
                    'changes': account_changes,
                    'biggest_change': biggest_change,
                    'customer_impacts': customer_impacts[:CONFIG["top_entity_impacts"]]
                })

        # 5. Risk analysis and gross margin calculation
        risk_periods = []

        # Simple gross margin estimation (would need cost data for full analysis)
        for change in total_revenue_changes:
            if abs(change['pct_change']) > CONFIG["revenue_pct_change_risk_threshold"]:
                risk_level = "HIGH" if abs(change['pct_change']) > CONFIG["high_revenue_pct_change_threshold"] else "MEDIUM"
                risk_periods.append({
                    'period': f"{change['from']} → {change['to']}",
                    'change': change['change'],
                    'pct_change': change['pct_change'],
                    'risk_level': risk_level,
                    'description': f"Revenue changed by {change['pct_change']:+.1f}%"
                })

        return {
            "subsidiary": subsidiary,
            "months_analyzed": months,
            "total_revenue_trend": total_revenue_by_month,
            "total_revenue_changes": total_revenue_changes,
            "account_analysis": account_analysis,
            "risk_periods": risk_periods,
            "summary": {
                "total_accounts": len([a for a in account_analysis if not a['account'].startswith('Total')]),
                "highest_variance_account": max(account_analysis, key=lambda x: max([abs(c['change']) for c in x['changes']], default=0))['account'] if account_analysis else None,
                "total_revenue_latest": total_revenue_by_month[months[-1]] if months else 0
            }
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def analyze_comprehensive_revenue_impact_ai(
    excel_bytes: bytes,
    filename: str,
    sub: str,
    CONFIG: dict = DEFAULT_CONFIG
) -> dict:
    """
    AI-powered comprehensive revenue impact analysis matching core.py functionality.
    Uses enhanced AI prompts to provide detailed 511/641/642 analysis with entity-level insights.
    """
    logger.info("🎯 ===== AI COMPREHENSIVE REVENUE IMPACT ANALYSIS =====")
    logger.info(f"📁 File: {filename}")
    logger.info(f"🏢 Subsidiary: {sub}")

    if not AI_AVAILABLE:
        return {
            "error": "AI analyzer not available - install required dependencies",
            "subsidiary": sub,
            "filename": filename
        }

    try:
        # Initialize AI analyzer
        llm_analyzer = LLMFinancialAnalyzer(CONFIG.get("llm_model", "gpt-4o"))
        logger.info(f"✅ AI analyzer initialized with model: {CONFIG.get('llm_model', 'gpt-4o')}")

        # Run comprehensive revenue impact analysis
        logger.info("🔍 Running AI comprehensive revenue impact analysis...")
        ai_analysis_results = llm_analyzer.analyze_comprehensive_revenue_impact(
            excel_bytes, filename, sub, CONFIG
        )
        logger.info(f"✅ AI comprehensive analysis completed with {len(ai_analysis_results)} insights")

        # Convert AI results to structured format matching core.py output
        logger.info("📊 Converting AI results to comprehensive revenue analysis structure...")

        analysis_result = {
            'subsidiary': sub,
            'filename': filename,
            'months_analyzed': [],  # Will be populated from AI analysis
            'total_revenue_analysis': {},
            'revenue_by_account': {},
            'gross_margin_analysis': {},
            'utility_analysis': {},
            'sga_641_analysis': {},
            'sga_642_analysis': {},
            'combined_sga_analysis': {},
            'risk_assessment': [],
            'summary': {},
            'ai_insights': ai_analysis_results  # Include raw AI insights
        }

        # Process AI results and populate analysis structure
        revenue_accounts = {}
        sga_641_accounts = {}
        sga_642_accounts = {}
        total_revenue_changes = []
        risk_assessment = []

        for insight in ai_analysis_results:
            analysis_type = insight.get('analysis_type', 'general')

            if analysis_type == 'total_revenue_trend':
                # Extract total revenue trend data
                details = insight.get('details', {})
                monthly_totals = details.get('monthly_totals', {})
                biggest_changes = details.get('biggest_changes', [])

                analysis_result['total_revenue_analysis'] = {
                    'monthly_totals': monthly_totals,
                    'changes': biggest_changes
                }

                if monthly_totals:
                    analysis_result['months_analyzed'] = list(monthly_totals.keys())

            elif analysis_type == 'revenue_by_account':
                # Extract individual revenue account data
                account_name = insight.get('account', 'Unknown')
                details = insight.get('details', {})

                revenue_accounts[account_name] = {
                    'monthly_totals': details.get('monthly_totals', {}),
                    'entity_impacts': details.get('entity_impacts', []),
                    'changes': [],  # Could be enhanced
                    'biggest_change': {
                        'change': insight.get('change_amount', 0),
                        'pct_change': insight.get('change_percent', 0),
                        'from': 'Previous',
                        'to': 'Current'
                    }
                }

            elif analysis_type == 'sga_641_analysis':
                # Extract SG&A 641 account data
                account_name = insight.get('account', 'Unknown')
                details = insight.get('details', {})

                sga_641_accounts[account_name] = {
                    'monthly_totals': details.get('monthly_totals', {}),
                    'entity_impacts': details.get('entity_impacts', []),
                    'changes': [],
                    'biggest_change': {
                        'change': insight.get('change_amount', 0),
                        'pct_change': insight.get('change_percent', 0),
                        'from': 'Previous',
                        'to': 'Current'
                    }
                }

            elif analysis_type == 'sga_642_analysis':
                # Extract SG&A 642 account data
                account_name = insight.get('account', 'Unknown')
                details = insight.get('details', {})

                sga_642_accounts[account_name] = {
                    'monthly_totals': details.get('monthly_totals', {}),
                    'entity_impacts': details.get('entity_impacts', []),
                    'changes': [],
                    'biggest_change': {
                        'change': insight.get('change_amount', 0),
                        'pct_change': insight.get('change_percent', 0),
                        'from': 'Previous',
                        'to': 'Current'
                    }
                }

            elif analysis_type == 'combined_sga_ratio':
                # Extract combined SG&A ratio analysis
                details = insight.get('details', {})
                sga_ratio_trend = details.get('sga_ratio_trend', [])
                ratio_changes = details.get('ratio_changes', [])

                analysis_result['combined_sga_analysis'] = {
                    'ratio_trend': sga_ratio_trend,
                    'monthly_totals': {},  # Could be calculated from ratio_trend
                    'total_641_accounts': len(sga_641_accounts),
                    'total_642_accounts': len(sga_642_accounts)
                }

            # Add to risk assessment if severity is high
            if insight.get('severity', '').lower() in ['high', 'medium']:
                risk_assessment.append({
                    'type': analysis_type.replace('_', ' ').title(),
                    'period': 'AI Analysis',
                    'risk_level': insight.get('severity', 'Medium').upper(),
                    'description': insight.get('explanation', insight.get('description', ''))
                })

        # Populate the analysis result structure
        analysis_result['revenue_by_account'] = revenue_accounts
        analysis_result['sga_641_analysis'] = sga_641_accounts
        analysis_result['sga_642_analysis'] = sga_642_accounts
        analysis_result['risk_assessment'] = risk_assessment

        # Create summary
        latest_month = analysis_result['months_analyzed'][-1] if analysis_result['months_analyzed'] else 'N/A'
        total_revenue_latest = 0
        total_sga_latest = 0

        if analysis_result['total_revenue_analysis'].get('monthly_totals'):
            monthly_totals = analysis_result['total_revenue_analysis']['monthly_totals']
            if latest_month in monthly_totals:
                total_revenue_latest = monthly_totals[latest_month]

        analysis_result['summary'] = {
            'total_accounts': len(revenue_accounts),
            'total_sga_641_accounts': len(sga_641_accounts),
            'total_sga_642_accounts': len(sga_642_accounts),
            'total_revenue_latest': total_revenue_latest,
            'total_sga_latest': total_sga_latest,
            'sga_ratio_latest': (total_sga_latest / total_revenue_latest * 100) if total_revenue_latest > 0 else 0,
            'risk_periods': [r for r in risk_assessment if r['risk_level'] == 'HIGH'],
            'ai_insights_count': len(ai_analysis_results)
        }

        logger.info("✅ AI comprehensive revenue analysis conversion completed")
        logger.info(f"📊 Structure: {len(revenue_accounts)} revenue accounts, {len(sga_641_accounts)} SG&A 641 accounts, {len(sga_642_accounts)} SG&A 642 accounts")

        return analysis_result

    except Exception as e:
        logger.error(f"❌ AI comprehensive revenue analysis failed: {str(e)}", exc_info=True)
        return {
            "error": f"AI comprehensive analysis failed: {str(e)}",
            "subsidiary": sub,
            "filename": filename
        }