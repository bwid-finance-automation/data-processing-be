"""
GLA PDF Report Generator
Generates PDF reports with GLA variance analysis and AI insights.
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime
from typing import Dict, Any

from app.shared.utils.logging_config import get_logger
from ..models.gla_models import GLAAnalysisSummary

logger = get_logger(__name__)


class GLAPDFGenerator:
    """
    Generates professional PDF reports for GLA variance analysis.
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1F4E79')
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2E75B6'),
            borderPadding=5
        ))

        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#1F4E79')
        ))

        # Custom body text style (use different name to avoid conflict)
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14
        ))

        # AI Analysis style
        self.styles.add(ParagraphStyle(
            name='AIAnalysis',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leading=14,
            leftIndent=10,
            rightIndent=10
        ))

        # Table header style
        self.styles.add(ParagraphStyle(
            name='ReportTableHeader',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.white,
            alignment=TA_CENTER
        ))

    def generate_pdf(
        self,
        summary: GLAAnalysisSummary,
        statistics: Dict[str, Any],
        ai_analysis: str,
        output_path: str,
        previous_period: str = "Previous",
        current_period: str = "Current"
    ) -> str:
        """
        Generate a PDF report with GLA variance analysis and AI insights.

        Args:
            summary: GLAAnalysisSummary with all variance results
            statistics: Summary statistics dictionary
            ai_analysis: AI-generated analysis text
            output_path: Path to save the PDF file
            previous_period: Label for previous period
            current_period: Label for current period

        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating PDF report: {output_path}")

        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )

        # Build content
        story = []

        # Title
        story.append(Paragraph(
            "GLA Variance Analysis Report",
            self.styles['ReportTitle']
        ))

        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y at %H:%M")
        story.append(Paragraph(
            f"<b>Generated:</b> {report_date}<br/>"
            f"<b>Period Comparison:</b> {previous_period} → {current_period}",
            self.styles['ReportBody']
        ))
        story.append(Spacer(1, 20))

        # Executive Summary Section
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(self._create_executive_summary(statistics, previous_period, current_period))
        story.append(Spacer(1, 15))

        # Portfolio Statistics Table
        story.append(Paragraph("Portfolio Statistics", self.styles['SubsectionHeader']))
        story.append(self._create_statistics_table(statistics, previous_period, current_period))
        story.append(Spacer(1, 20))

        # AI Analysis Section
        if ai_analysis:
            story.append(Paragraph("AI-Powered Analysis", self.styles['SectionHeader']))
            story.append(self._format_ai_analysis(ai_analysis))
            story.append(Spacer(1, 20))

        # Variance Details Section
        story.append(PageBreak())
        story.append(Paragraph("Detailed Variance Analysis", self.styles['SectionHeader']))
        story.append(self._create_variance_table(summary, previous_period, current_period))

        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated successfully: {output_path}")

        return output_path

    def _create_executive_summary(
        self,
        statistics: Dict[str, Any],
        previous_period: str,
        current_period: str
    ) -> Paragraph:
        """Create executive summary paragraph."""
        total_projects = statistics.get('total_projects', 0)
        committed = statistics.get('committed', {})
        handover = statistics.get('handover', {})

        committed_var = committed.get('total_variance', 0)
        handover_var = handover.get('total_variance', 0)

        committed_direction = "increased" if committed_var > 0 else "decreased" if committed_var < 0 else "remained unchanged"
        handover_direction = "increased" if handover_var > 0 else "decreased" if handover_var < 0 else "remained unchanged"

        summary_text = f"""
        This report analyzes GLA (Gross Leasable Area) variance across <b>{total_projects} projects</b>
        comparing {previous_period} to {current_period}.<br/><br/>

        <b>Key Findings:</b><br/>
        • Total Committed GLA {committed_direction} by <b>{abs(committed_var):,.2f} sqm</b>
          ({committed.get('increased', 0)} projects increased, {committed.get('decreased', 0)} decreased)<br/>
        • Total Handover GLA {handover_direction} by <b>{abs(handover_var):,.2f} sqm</b>
          ({handover.get('increased', 0)} projects increased, {handover.get('decreased', 0)} decreased)
        """

        return Paragraph(summary_text, self.styles['ReportBody'])

    def _create_statistics_table(
        self,
        statistics: Dict[str, Any],
        previous_period: str,
        current_period: str
    ) -> Table:
        """Create statistics summary table."""
        committed = statistics.get('committed', {})
        handover = statistics.get('handover', {})
        by_type = statistics.get('by_type', {})

        data = [
            ['Metric', previous_period, current_period, 'Variance'],
            ['Committed GLA (Total)',
             f"{committed.get('total_previous', 0):,.2f}",
             f"{committed.get('total_current', 0):,.2f}",
             f"{committed.get('total_variance', 0):,.2f}"],
            ['Handover GLA (Total)',
             f"{handover.get('total_previous', 0):,.2f}",
             f"{handover.get('total_current', 0):,.2f}",
             f"{handover.get('total_variance', 0):,.2f}"],
            ['', '', '', ''],
            ['RBF Committed Variance', '', '', f"{by_type.get('rbf', {}).get('committed_variance', 0):,.2f}"],
            ['RBF Handover Variance', '', '', f"{by_type.get('rbf', {}).get('handover_variance', 0):,.2f}"],
            ['RBW Committed Variance', '', '', f"{by_type.get('rbw', {}).get('committed_variance', 0):,.2f}"],
            ['RBW Handover Variance', '', '', f"{by_type.get('rbw', {}).get('handover_variance', 0):,.2f}"],
        ]

        table = Table(data, colWidths=[150, 100, 100, 100])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),

            # Alternating row colors
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#E8F0FE')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.white),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#E8F0FE')),
            ('BACKGROUND', (0, 5), (-1, 5), colors.white),
            ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor('#E8F0FE')),
            ('BACKGROUND', (0, 7), (-1, 7), colors.white),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1F4E79')),

            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        return table

    def _format_ai_analysis(self, ai_analysis: str) -> Paragraph:
        """Format AI analysis text into readable paragraphs."""
        import re

        # Sanitize input - remove any existing HTML-like tags that could break ReportLab
        formatted = ai_analysis

        # Convert markdown bold **text** to <b>text</b> using regex
        formatted = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', formatted)

        # Convert markdown headers (# Header) to bold
        formatted = re.sub(r'^#{1,3}\s*(.+)$', r'<b>\1</b>', formatted, flags=re.MULTILINE)

        # Handle bullet points before joining lines
        lines = formatted.split('\n')
        processed_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('• '):
                line = '&nbsp;&nbsp;• ' + line[2:]
            elif line.startswith('* '):
                line = '&nbsp;&nbsp;• ' + line[2:]
            processed_lines.append(line)

        # Join lines with <br/> and handle paragraph breaks
        formatted = '<br/>'.join(processed_lines)
        formatted = formatted.replace('<br/><br/><br/>', '<br/><br/>')  # Clean up extra breaks

        return Paragraph(formatted, self.styles['AIAnalysis'])

    def _create_variance_table(
        self,
        summary: GLAAnalysisSummary,
        previous_period: str,
        current_period: str
    ) -> Table:
        """Create detailed variance table."""
        # Header row
        header = [
            'Project',
            'Type',
            'Region',
            f'Committed\n{previous_period}',
            f'Committed\n{current_period}',
            'Committed\nVariance',
            f'Handover\n{previous_period}',
            f'Handover\n{current_period}',
            'Handover\nVariance'
        ]

        data = [header]

        # Data rows
        for result in summary.results:
            row = [
                result.project_name[:20],  # Truncate long names
                result.product_type,
                result.region[:6] if result.region else '',
                f"{result.committed_previous:,.0f}",
                f"{result.committed_current:,.0f}",
                f"{result.committed_variance:,.0f}",
                f"{result.handover_previous:,.0f}",
                f"{result.handover_current:,.0f}",
                f"{result.handover_variance:,.0f}"
            ]
            data.append(row)

        # Totals row
        data.append([
            'TOTAL',
            '',
            '',
            f"{summary.total_portfolio_committed_previous:,.0f}",
            f"{summary.total_portfolio_committed_current:,.0f}",
            f"{summary.total_portfolio_committed_variance:,.0f}",
            f"{summary.total_portfolio_handover_previous:,.0f}",
            f"{summary.total_portfolio_handover_current:,.0f}",
            f"{summary.total_portfolio_handover_variance:,.0f}"
        ])

        # Create table with appropriate column widths
        col_widths = [80, 35, 40, 55, 55, 55, 55, 55, 55]
        table = Table(data, colWidths=col_widths, repeatRows=1)

        # Base style
        style = [
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),

            # Data
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('ALIGN', (3, 1), (-1, -1), 'RIGHT'),

            # Totals row
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#D9E2F3')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1F4E79')),

            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ]

        # Add alternating row colors
        for i in range(1, len(data) - 1):
            if i % 2 == 0:
                style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F5F5F5')))

        # Color variance cells based on positive/negative
        for i, result in enumerate(summary.results, start=1):
            # Committed variance column (index 5)
            if result.committed_variance > 0:
                style.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#006400')))
            elif result.committed_variance < 0:
                style.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#8B0000')))

            # Handover variance column (index 8)
            if result.handover_variance > 0:
                style.append(('TEXTCOLOR', (8, i), (8, i), colors.HexColor('#006400')))
            elif result.handover_variance < 0:
                style.append(('TEXTCOLOR', (8, i), (8, i), colors.HexColor('#8B0000')))

        table.setStyle(TableStyle(style))
        return table
