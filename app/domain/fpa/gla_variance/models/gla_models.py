"""
GLA Variance Analysis Domain Models
Defines the core business entities for GLA (Gross Leasable Area) variance analysis.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ProductType(str, Enum):
    """Product types for industrial properties"""
    RBF = "RBF"  # Ready Built Factory
    RBW = "RBW"  # Ready Built Warehouse
    BTSW = "BTSW"  # Build-to-Suit Warehouse
    OFFICE = "Office"
    SERVICE = "Service"


class Region(str, Enum):
    """Geographic regions"""
    NORTH = "North"
    SOUTH = "South"
    CENTRAL = "Central"


class LeaseStatus(str, Enum):
    """Unit for Lease status values"""
    HANDED_OVER = "Handed Over"
    OPEN = "Open"
    TERMINATED = "Terminated"
    VOIDED = "Voided"
    NONE = "- None -"


@dataclass
class GLARecord:
    """
    Represents a single Unit for Lease (UFL) record from the input data.
    """
    ufl_id: int
    project_code: str  # Short code like 'PBBA'
    project_name: str  # Full name from 'BWID Project.' like 'PBBA: Bau Bang'
    product_type: str  # RBF, RBW, etc.
    region: str  # North, South, Central
    gla_for_lease: float  # Area in sqm
    status: str  # Unit for Lease Status
    tenant: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @property
    def readable_project_name(self) -> str:
        """Extract readable project name from 'BWID Project.' format"""
        if ':' in self.project_name:
            return self.project_name.split(':', 1)[1].strip()
        return self.project_name


@dataclass
class TenantGLA:
    """GLA information for a single tenant."""
    tenant_name: str
    gla_sqm: float
    status: str  # 'Open' or 'Handed Over'
    # Handover sheet attributes
    handover_gla: float = 0.0
    monthly_gross_rent: float = 0.0
    monthly_rate: float = 0.0
    # Committed sheet attributes
    committed_gla: float = 0.0
    months_to_expire: float = 0.0
    months_to_expire_x_committed_gla: float = 0.0


@dataclass
class ProjectGLASummary:
    """
    Aggregated GLA summary for a project + product type combination.
    """
    project_name: str  # Readable name (e.g., 'Bau Bang')
    product_type: str  # RBF or RBW
    region: str
    gla_sqm: float = 0.0
    tenants: List['TenantGLA'] = field(default_factory=list)  # List of tenants with GLA

    # Handover sheet aggregated attributes
    handover_gla: float = 0.0  # Sum of Handover GLA
    monthly_gross_rent: float = 0.0  # Sum of Monthly Gross rent
    monthly_rate: float = 0.0  # Average Monthly rate (weighted by GLA)

    # Committed sheet aggregated attributes
    committed_gla: float = 0.0  # Sum of Committed GLA
    months_to_expire: float = 0.0  # Average months to expire (weighted)
    months_to_expire_x_committed_gla: float = 0.0  # Sum of Month to expire x committed GLA

    def __hash__(self):
        return hash((self.project_name, self.product_type))

    def __eq__(self, other):
        if not isinstance(other, ProjectGLASummary):
            return False
        return (self.project_name == other.project_name and
                self.product_type == other.product_type)


@dataclass
class TenantChange:
    """Represents a tenant's GLA change between periods."""
    tenant_name: str
    previous_gla: float
    current_gla: float
    variance: float
    change_type: str  # 'new', 'terminated', 'expanded', 'reduced', 'unchanged'


@dataclass
class GLAVarianceResult:
    """
    Variance result comparing previous and current GLA values for a project.
    """
    project_name: str
    product_type: str
    region: str

    # Committed GLA (Open + Handed Over)
    committed_previous: float = 0.0
    committed_current: float = 0.0
    committed_variance: float = 0.0
    committed_note: str = ""

    # Handover GLA (Handed Over only)
    handover_previous: float = 0.0
    handover_current: float = 0.0
    handover_variance: float = 0.0
    handover_note: str = ""

    # Handover sheet attributes - Monthly Gross rent
    monthly_gross_rent_previous: float = 0.0
    monthly_gross_rent_current: float = 0.0
    monthly_gross_rent_variance: float = 0.0

    # Handover sheet attributes - Monthly rate
    monthly_rate_previous: float = 0.0
    monthly_rate_current: float = 0.0
    monthly_rate_variance: float = 0.0

    # Committed sheet attributes - Months to expire (WALE)
    months_to_expire_previous: float = 0.0
    months_to_expire_current: float = 0.0
    months_to_expire_variance: float = 0.0
    wale_note: str = ""  # AI-generated note for WALE variance

    # Committed sheet attributes - Month to expire x committed GLA
    months_to_expire_x_gla_previous: float = 0.0
    months_to_expire_x_gla_current: float = 0.0
    months_to_expire_x_gla_variance: float = 0.0

    # Gross rent note
    gross_rent_note: str = ""  # AI-generated note for gross rent variance

    # Accounting Net Rent (straight-line basis)
    accounting_net_rent_previous: float = 0.0
    accounting_net_rent_current: float = 0.0
    accounting_net_rent_variance: float = 0.0
    accounting_net_rent_note: str = ""  # AI-generated note for accounting net rent variance

    # Tenant-level changes for explanation (legacy - will be removed)
    committed_tenant_changes: List[TenantChange] = field(default_factory=list)
    handover_tenant_changes: List[TenantChange] = field(default_factory=list)

    # Raw tenant data for AI analysis
    committed_tenants_previous: List['TenantGLA'] = field(default_factory=list)
    committed_tenants_current: List['TenantGLA'] = field(default_factory=list)
    handover_tenants_previous: List['TenantGLA'] = field(default_factory=list)
    handover_tenants_current: List['TenantGLA'] = field(default_factory=list)

    def calculate_variances(self):
        """Calculate variance values"""
        self.committed_variance = self.committed_current - self.committed_previous
        self.handover_variance = self.handover_current - self.handover_previous
        self.monthly_gross_rent_variance = self.monthly_gross_rent_current - self.monthly_gross_rent_previous
        self.monthly_rate_variance = self.monthly_rate_current - self.monthly_rate_previous
        self.months_to_expire_variance = self.months_to_expire_current - self.months_to_expire_previous
        self.months_to_expire_x_gla_variance = self.months_to_expire_x_gla_current - self.months_to_expire_x_gla_previous
        self.accounting_net_rent_variance = self.accounting_net_rent_current - self.accounting_net_rent_previous

    def to_dict(self) -> dict:
        """Convert to dictionary for output"""
        return {
            'project_name': self.project_name,
            'product_type': self.product_type,
            'region': self.region,
            'committed_previous': self.committed_previous,
            'committed_current': self.committed_current,
            'committed_variance': self.committed_variance,
            'committed_note': self.committed_note,
            'handover_previous': self.handover_previous,
            'handover_current': self.handover_current,
            'handover_variance': self.handover_variance,
            'handover_note': self.handover_note,
            # WALE attributes
            'months_to_expire_previous': self.months_to_expire_previous,
            'months_to_expire_current': self.months_to_expire_current,
            'months_to_expire_variance': self.months_to_expire_variance,
            'wale_note': self.wale_note,
            # Gross rent attributes
            'monthly_gross_rent_previous': self.monthly_gross_rent_previous,
            'monthly_gross_rent_current': self.monthly_gross_rent_current,
            'monthly_gross_rent_variance': self.monthly_gross_rent_variance,
            'gross_rent_note': self.gross_rent_note,
            # Other attributes
            'monthly_rate_previous': self.monthly_rate_previous,
            'monthly_rate_current': self.monthly_rate_current,
            'monthly_rate_variance': self.monthly_rate_variance,
            'months_to_expire_x_gla_previous': self.months_to_expire_x_gla_previous,
            'months_to_expire_x_gla_current': self.months_to_expire_x_gla_current,
            'months_to_expire_x_gla_variance': self.months_to_expire_x_gla_variance,
            # Accounting Net Rent
            'accounting_net_rent_previous': self.accounting_net_rent_previous,
            'accounting_net_rent_current': self.accounting_net_rent_current,
            'accounting_net_rent_variance': self.accounting_net_rent_variance,
            'accounting_net_rent_note': self.accounting_net_rent_note,
        }


@dataclass
class GLAAnalysisSummary:
    """
    Overall summary of GLA variance analysis.
    """
    results: List[GLAVarianceResult] = field(default_factory=list)

    # Totals
    total_rbf_committed_previous: float = 0.0
    total_rbf_committed_current: float = 0.0
    total_rbf_committed_variance: float = 0.0

    total_rbw_committed_previous: float = 0.0
    total_rbw_committed_current: float = 0.0
    total_rbw_committed_variance: float = 0.0

    total_rbf_handover_previous: float = 0.0
    total_rbf_handover_current: float = 0.0
    total_rbf_handover_variance: float = 0.0

    total_rbw_handover_previous: float = 0.0
    total_rbw_handover_current: float = 0.0
    total_rbw_handover_variance: float = 0.0

    # WALE totals (weighted average by committed GLA)
    total_rbf_wale_previous: float = 0.0
    total_rbf_wale_current: float = 0.0
    total_rbf_wale_variance: float = 0.0

    total_rbw_wale_previous: float = 0.0
    total_rbw_wale_current: float = 0.0
    total_rbw_wale_variance: float = 0.0

    # Gross Rent totals (weighted average by handover GLA)
    total_rbf_gross_rent_previous: float = 0.0
    total_rbf_gross_rent_current: float = 0.0
    total_rbf_gross_rent_variance: float = 0.0

    total_rbw_gross_rent_previous: float = 0.0
    total_rbw_gross_rent_current: float = 0.0
    total_rbw_gross_rent_variance: float = 0.0

    previous_period: str = ""
    current_period: str = ""

    def calculate_totals(self):
        """Calculate totals from results"""
        # Accumulators for weighted averages
        rbf_wale_x_gla_prev = 0.0
        rbf_wale_x_gla_curr = 0.0
        rbw_wale_x_gla_prev = 0.0
        rbw_wale_x_gla_curr = 0.0

        rbf_rent_x_gla_prev = 0.0
        rbf_rent_x_gla_curr = 0.0
        rbw_rent_x_gla_prev = 0.0
        rbw_rent_x_gla_curr = 0.0

        for result in self.results:
            if result.product_type == ProductType.RBF.value:
                self.total_rbf_committed_previous += result.committed_previous
                self.total_rbf_committed_current += result.committed_current
                self.total_rbf_handover_previous += result.handover_previous
                self.total_rbf_handover_current += result.handover_current
                # WALE weighted by committed GLA
                rbf_wale_x_gla_prev += result.months_to_expire_previous * result.committed_previous
                rbf_wale_x_gla_curr += result.months_to_expire_current * result.committed_current
                # Gross Rent weighted by handover GLA
                rbf_rent_x_gla_prev += result.monthly_rate_previous * result.handover_previous
                rbf_rent_x_gla_curr += result.monthly_rate_current * result.handover_current
            elif result.product_type == ProductType.RBW.value:
                self.total_rbw_committed_previous += result.committed_previous
                self.total_rbw_committed_current += result.committed_current
                self.total_rbw_handover_previous += result.handover_previous
                self.total_rbw_handover_current += result.handover_current
                # WALE weighted by committed GLA
                rbw_wale_x_gla_prev += result.months_to_expire_previous * result.committed_previous
                rbw_wale_x_gla_curr += result.months_to_expire_current * result.committed_current
                # Gross Rent weighted by handover GLA
                rbw_rent_x_gla_prev += result.monthly_rate_previous * result.handover_previous
                rbw_rent_x_gla_curr += result.monthly_rate_current * result.handover_current

        # GLA variances
        self.total_rbf_committed_variance = self.total_rbf_committed_current - self.total_rbf_committed_previous
        self.total_rbw_committed_variance = self.total_rbw_committed_current - self.total_rbw_committed_previous
        self.total_rbf_handover_variance = self.total_rbf_handover_current - self.total_rbf_handover_previous
        self.total_rbw_handover_variance = self.total_rbw_handover_current - self.total_rbw_handover_previous

        # WALE weighted averages (in months, will convert to years in Excel output)
        if self.total_rbf_committed_previous > 0:
            self.total_rbf_wale_previous = rbf_wale_x_gla_prev / self.total_rbf_committed_previous
        if self.total_rbf_committed_current > 0:
            self.total_rbf_wale_current = rbf_wale_x_gla_curr / self.total_rbf_committed_current
        self.total_rbf_wale_variance = self.total_rbf_wale_current - self.total_rbf_wale_previous

        if self.total_rbw_committed_previous > 0:
            self.total_rbw_wale_previous = rbw_wale_x_gla_prev / self.total_rbw_committed_previous
        if self.total_rbw_committed_current > 0:
            self.total_rbw_wale_current = rbw_wale_x_gla_curr / self.total_rbw_committed_current
        self.total_rbw_wale_variance = self.total_rbw_wale_current - self.total_rbw_wale_previous

        # Gross Rent weighted averages
        if self.total_rbf_handover_previous > 0:
            self.total_rbf_gross_rent_previous = rbf_rent_x_gla_prev / self.total_rbf_handover_previous
        if self.total_rbf_handover_current > 0:
            self.total_rbf_gross_rent_current = rbf_rent_x_gla_curr / self.total_rbf_handover_current
        self.total_rbf_gross_rent_variance = self.total_rbf_gross_rent_current - self.total_rbf_gross_rent_previous

        if self.total_rbw_handover_previous > 0:
            self.total_rbw_gross_rent_previous = rbw_rent_x_gla_prev / self.total_rbw_handover_previous
        if self.total_rbw_handover_current > 0:
            self.total_rbw_gross_rent_current = rbw_rent_x_gla_curr / self.total_rbw_handover_current
        self.total_rbw_gross_rent_variance = self.total_rbw_gross_rent_current - self.total_rbw_gross_rent_previous

    @property
    def total_portfolio_committed_previous(self) -> float:
        return self.total_rbf_committed_previous + self.total_rbw_committed_previous

    @property
    def total_portfolio_committed_current(self) -> float:
        return self.total_rbf_committed_current + self.total_rbw_committed_current

    @property
    def total_portfolio_committed_variance(self) -> float:
        return self.total_rbf_committed_variance + self.total_rbw_committed_variance

    @property
    def total_portfolio_handover_previous(self) -> float:
        return self.total_rbf_handover_previous + self.total_rbw_handover_previous

    @property
    def total_portfolio_handover_current(self) -> float:
        return self.total_rbf_handover_current + self.total_rbw_handover_current

    @property
    def total_portfolio_handover_variance(self) -> float:
        return self.total_rbf_handover_variance + self.total_rbw_handover_variance

    @property
    def total_portfolio_wale_previous(self) -> float:
        """Portfolio WALE weighted by committed GLA"""
        total_gla = self.total_portfolio_committed_previous
        if total_gla <= 0:
            return 0.0
        weighted = (self.total_rbf_wale_previous * self.total_rbf_committed_previous +
                    self.total_rbw_wale_previous * self.total_rbw_committed_previous)
        return weighted / total_gla

    @property
    def total_portfolio_wale_current(self) -> float:
        """Portfolio WALE weighted by committed GLA"""
        total_gla = self.total_portfolio_committed_current
        if total_gla <= 0:
            return 0.0
        weighted = (self.total_rbf_wale_current * self.total_rbf_committed_current +
                    self.total_rbw_wale_current * self.total_rbw_committed_current)
        return weighted / total_gla

    @property
    def total_portfolio_wale_variance(self) -> float:
        return self.total_portfolio_wale_current - self.total_portfolio_wale_previous

    @property
    def total_portfolio_gross_rent_previous(self) -> float:
        """Portfolio Gross Rent weighted by handover GLA"""
        total_gla = self.total_portfolio_handover_previous
        if total_gla <= 0:
            return 0.0
        weighted = (self.total_rbf_gross_rent_previous * self.total_rbf_handover_previous +
                    self.total_rbw_gross_rent_previous * self.total_rbw_handover_previous)
        return weighted / total_gla

    @property
    def total_portfolio_gross_rent_current(self) -> float:
        """Portfolio Gross Rent weighted by handover GLA"""
        total_gla = self.total_portfolio_handover_current
        if total_gla <= 0:
            return 0.0
        weighted = (self.total_rbf_gross_rent_current * self.total_rbf_handover_current +
                    self.total_rbw_gross_rent_current * self.total_rbw_handover_current)
        return weighted / total_gla

    @property
    def total_portfolio_gross_rent_variance(self) -> float:
        return self.total_portfolio_gross_rent_current - self.total_portfolio_gross_rent_previous
