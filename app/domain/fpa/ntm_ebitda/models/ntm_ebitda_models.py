"""
NTM EBITDA Variance Analysis Domain Models
Defines the core business entities for NTM (Next Twelve Months) EBITDA variance analysis.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import date
from enum import Enum


class ChangeType(str, Enum):
    """Types of lease changes detected between periods"""
    NEW = "new"
    TERMINATED = "terminated"
    RENEWED = "renewed"
    EXPANDED = "expanded"
    REDUCED = "reduced"
    TIMING_SHIFT = "timing_shift"
    UNCHANGED = "unchanged"


@dataclass
class AnalysisConfig:
    """Configuration for NTM EBITDA analysis"""
    fx_rate: float = 25500  # VND/USD exchange rate
    variance_threshold: float = 0.05  # 5% threshold for flagging
    ntm_months: int = 12  # NTM period length
    min_lease_ntm_for_comment: float = 0.3  # Min Bn VND for commentary
    min_gla_for_comment: float = 500  # Min sqm for commentary


@dataclass
class LeaseRecord:
    """
    Represents a single lease/tenant record from the leasing model.
    """
    project_code: str  # e.g., "BWD-Bau Bang"
    project_name: str  # Mapped reporting name
    phase: str  # e.g., "Phase 1", "All Phase"
    metric_type: str  # e.g., "Accounting revenue", "OPEX"
    tenant_name: Optional[str] = None
    gla_sqm: float = 0.0
    lease_start_date: Optional[date] = None
    lease_end_date: Optional[date] = None
    term_months: int = 0
    monthly_ntm: List[float] = field(default_factory=list)  # 12 months of NTM data
    total_ntm: float = 0.0  # Sum of monthly NTM

    # Additional lease details
    unit_price: float = 0.0  # VND/sqm/month
    stake: float = 1.0  # Ownership stake (0-1)

    def calculate_total_ntm(self):
        """Calculate total NTM from monthly values"""
        self.total_ntm = sum(self.monthly_ntm) if self.monthly_ntm else 0.0
        return self.total_ntm


@dataclass
class NTMRecord:
    """
    Represents NTM data for a specific metric type (revenue, OPEX, SG&A, EBITDA).
    """
    project_name: str
    metric_type: str  # 'revenue', 'opex', 'sga', 'ebitda'
    total_ntm: float = 0.0
    monthly_ntm: List[float] = field(default_factory=list)
    stake: float = 1.0

    # For EBITDA: revenue - opex - sga
    revenue_ntm: float = 0.0
    opex_ntm: float = 0.0
    sga_ntm: float = 0.0


@dataclass
class LeaseChange:
    """Represents a lease change between periods"""
    tenant_name: str
    change_type: ChangeType
    gla_sqm: float = 0.0
    previous_ntm: float = 0.0
    current_ntm: float = 0.0
    variance: float = 0.0
    lease_start: Optional[date] = None
    lease_end: Optional[date] = None
    term_months: int = 0
    unit_price: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_name": self.tenant_name,
            "change_type": self.change_type.value,
            "gla_sqm": self.gla_sqm,
            "previous_ntm": self.previous_ntm,
            "current_ntm": self.current_ntm,
            "variance": self.variance,
            "lease_start": self.lease_start.isoformat() if self.lease_start else None,
            "lease_end": self.lease_end.isoformat() if self.lease_end else None,
            "term_months": self.term_months,
            "unit_price": self.unit_price,
            "description": self.description,
        }


@dataclass
class ProjectNTMSummary:
    """
    Aggregated NTM summary for a project.
    """
    project_name: str
    project_code: str = ""
    stake: float = 1.0

    # Revenue NTM (Contracted Revenue)
    revenue_ntm: float = 0.0

    # Cost components
    opex_ntm: float = 0.0
    sga_ntm: float = 0.0

    # EBITDA = Revenue - OPEX - SG&A
    ebitda_ntm: float = 0.0

    # Monthly breakdown (for trend analysis)
    monthly_revenue: List[float] = field(default_factory=list)
    monthly_opex: List[float] = field(default_factory=list)
    monthly_sga: List[float] = field(default_factory=list)
    monthly_ebitda: List[float] = field(default_factory=list)

    # Lease-level details for variance explanation
    leases: List[LeaseRecord] = field(default_factory=list)

    def calculate_ebitda(self):
        """Calculate EBITDA from components"""
        self.ebitda_ntm = self.revenue_ntm - self.opex_ntm - self.sga_ntm
        return self.ebitda_ntm

    def __hash__(self):
        return hash(self.project_name)

    def __eq__(self, other):
        if not isinstance(other, ProjectNTMSummary):
            return False
        return self.project_name == other.project_name


@dataclass
class NTMVarianceResult:
    """
    Variance result comparing previous and current NTM values for a project.
    """
    project_name: str
    project_code: str = ""
    stake: float = 1.0

    # Revenue NTM variance
    revenue_previous: float = 0.0
    revenue_current: float = 0.0
    revenue_variance: float = 0.0
    revenue_variance_pct: float = 0.0

    # OPEX NTM variance
    opex_previous: float = 0.0
    opex_current: float = 0.0
    opex_variance: float = 0.0

    # SG&A NTM variance
    sga_previous: float = 0.0
    sga_current: float = 0.0
    sga_variance: float = 0.0

    # EBITDA NTM variance
    ebitda_previous: float = 0.0
    ebitda_current: float = 0.0
    ebitda_variance: float = 0.0
    ebitda_variance_pct: float = 0.0

    # AI-generated commentary
    commentary: str = ""

    # Lease-level changes for explanation
    lease_changes: List[LeaseChange] = field(default_factory=list)

    # Raw lease data for AI analysis
    leases_previous: List[LeaseRecord] = field(default_factory=list)
    leases_current: List[LeaseRecord] = field(default_factory=list)

    def calculate_variances(self):
        """Calculate variance values and percentages"""
        self.revenue_variance = self.revenue_current - self.revenue_previous
        self.opex_variance = self.opex_current - self.opex_previous
        self.sga_variance = self.sga_current - self.sga_previous
        self.ebitda_variance = self.ebitda_current - self.ebitda_previous

        # Calculate percentages (avoid division by zero)
        if self.revenue_previous != 0:
            self.revenue_variance_pct = self.revenue_variance / abs(self.revenue_previous)
        else:
            self.revenue_variance_pct = 1.0 if self.revenue_current != 0 else 0.0

        if self.ebitda_previous != 0:
            self.ebitda_variance_pct = self.ebitda_variance / abs(self.ebitda_previous)
        else:
            self.ebitda_variance_pct = 1.0 if self.ebitda_current != 0 else 0.0

    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if variance exceeds threshold"""
        return abs(self.ebitda_variance_pct) >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output"""
        return {
            "project_name": self.project_name,
            "project_code": self.project_code,
            "stake": self.stake,
            "revenue_previous": self.revenue_previous,
            "revenue_current": self.revenue_current,
            "revenue_variance": self.revenue_variance,
            "revenue_variance_pct": self.revenue_variance_pct,
            "opex_previous": self.opex_previous,
            "opex_current": self.opex_current,
            "opex_variance": self.opex_variance,
            "sga_previous": self.sga_previous,
            "sga_current": self.sga_current,
            "sga_variance": self.sga_variance,
            "ebitda_previous": self.ebitda_previous,
            "ebitda_current": self.ebitda_current,
            "ebitda_variance": self.ebitda_variance,
            "ebitda_variance_pct": self.ebitda_variance_pct,
            "commentary": self.commentary,
            "lease_changes": [lc.to_dict() for lc in self.lease_changes],
        }


@dataclass
class NTMAnalysisSummary:
    """
    Overall summary of NTM EBITDA variance analysis.
    """
    results: List[NTMVarianceResult] = field(default_factory=list)

    # Portfolio totals
    total_revenue_previous: float = 0.0
    total_revenue_current: float = 0.0
    total_revenue_variance: float = 0.0

    total_opex_previous: float = 0.0
    total_opex_current: float = 0.0
    total_opex_variance: float = 0.0

    total_sga_previous: float = 0.0
    total_sga_current: float = 0.0
    total_sga_variance: float = 0.0

    total_ebitda_previous: float = 0.0
    total_ebitda_current: float = 0.0
    total_ebitda_variance: float = 0.0
    total_ebitda_variance_pct: float = 0.0

    # Period labels
    previous_period: str = ""
    current_period: str = ""

    # Analysis metadata
    fx_rate: float = 25500
    variance_threshold: float = 0.05

    def calculate_totals(self):
        """Calculate portfolio totals from results"""
        self.total_revenue_previous = sum(r.revenue_previous for r in self.results)
        self.total_revenue_current = sum(r.revenue_current for r in self.results)
        self.total_revenue_variance = self.total_revenue_current - self.total_revenue_previous

        self.total_opex_previous = sum(r.opex_previous for r in self.results)
        self.total_opex_current = sum(r.opex_current for r in self.results)
        self.total_opex_variance = self.total_opex_current - self.total_opex_previous

        self.total_sga_previous = sum(r.sga_previous for r in self.results)
        self.total_sga_current = sum(r.sga_current for r in self.results)
        self.total_sga_variance = self.total_sga_current - self.total_sga_previous

        self.total_ebitda_previous = sum(r.ebitda_previous for r in self.results)
        self.total_ebitda_current = sum(r.ebitda_current for r in self.results)
        self.total_ebitda_variance = self.total_ebitda_current - self.total_ebitda_previous

        if self.total_ebitda_previous != 0:
            self.total_ebitda_variance_pct = self.total_ebitda_variance / abs(self.total_ebitda_previous)
        else:
            self.total_ebitda_variance_pct = 1.0 if self.total_ebitda_current != 0 else 0.0

    @property
    def significant_results(self) -> List[NTMVarianceResult]:
        """Get results with significant variance"""
        return [r for r in self.results if r.is_significant(self.variance_threshold)]

    @property
    def projects_with_increase(self) -> List[NTMVarianceResult]:
        """Get projects with EBITDA increase"""
        return [r for r in self.results if r.ebitda_variance > 0]

    @property
    def projects_with_decrease(self) -> List[NTMVarianceResult]:
        """Get projects with EBITDA decrease"""
        return [r for r in self.results if r.ebitda_variance < 0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "previous_period": self.previous_period,
            "current_period": self.current_period,
            "fx_rate": self.fx_rate,
            "variance_threshold": self.variance_threshold,
            "totals": {
                "revenue": {
                    "previous": self.total_revenue_previous,
                    "current": self.total_revenue_current,
                    "variance": self.total_revenue_variance,
                },
                "opex": {
                    "previous": self.total_opex_previous,
                    "current": self.total_opex_current,
                    "variance": self.total_opex_variance,
                },
                "sga": {
                    "previous": self.total_sga_previous,
                    "current": self.total_sga_current,
                    "variance": self.total_sga_variance,
                },
                "ebitda": {
                    "previous": self.total_ebitda_previous,
                    "current": self.total_ebitda_current,
                    "variance": self.total_ebitda_variance,
                    "variance_pct": self.total_ebitda_variance_pct,
                },
            },
            "results": [r.to_dict() for r in self.results],
            "significant_count": len(self.significant_results),
            "increase_count": len(self.projects_with_increase),
            "decrease_count": len(self.projects_with_decrease),
        }
