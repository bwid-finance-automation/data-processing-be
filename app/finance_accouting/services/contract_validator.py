"""Validation service for contract extraction results."""
from typing import List, Dict, Optional
from datetime import datetime

from ..models.contract_schemas import ContractInfo, RatePeriod
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ContractValidator:
    """Validates extracted contract data for quality and consistency."""

    def __init__(self):
        """Initialize the validator."""
        self.critical_fields = [
            'customer_name',
            'contract_number',
            'contract_date',
            'gfa',
            'rate_periods'
        ]

    def validate_contract(self, contract: ContractInfo) -> Dict[str, List[str]]:
        """
        Validate a single contract extraction result.

        Args:
            contract: ContractInfo object to validate

        Returns:
            Dictionary with 'errors', 'warnings', and 'info' lists
        """
        errors = []
        warnings = []
        info = []

        # Critical field validation
        errors.extend(self._validate_critical_fields(contract))

        # Date validation
        date_issues = self._validate_dates(contract)
        errors.extend(date_issues['errors'])
        warnings.extend(date_issues['warnings'])

        # Numeric field validation
        numeric_issues = self._validate_numeric_fields(contract)
        errors.extend(numeric_issues['errors'])
        warnings.extend(numeric_issues['warnings'])

        # Rate period validation
        rate_issues = self._validate_rate_periods(contract)
        errors.extend(rate_issues['errors'])
        warnings.extend(rate_issues['warnings'])
        info.extend(rate_issues['info'])

        # Cross-field validation
        cross_issues = self._validate_cross_field_logic(contract)
        warnings.extend(cross_issues['warnings'])
        info.extend(cross_issues['info'])

        # Service charge validation
        sc_issues = self._validate_service_charge(contract)
        warnings.extend(sc_issues['warnings'])
        info.extend(sc_issues['info'])

        return {
            'errors': errors,
            'warnings': warnings,
            'info': info
        }

    def _validate_critical_fields(self, contract: ContractInfo) -> List[str]:
        """Validate that critical fields are present."""
        errors = []

        if not contract.customer_name and not contract.tenant:
            errors.append("CRITICAL: No customer/tenant name found")

        if not contract.contract_number:
            errors.append("CRITICAL: Contract number missing")

        if not contract.contract_date:
            errors.append("CRITICAL: Contract date missing")

        if not contract.gfa and not contract.gla_for_lease:
            errors.append("CRITICAL: No area information (GFA or GLA) found")

        if not contract.rate_periods or len(contract.rate_periods) == 0:
            errors.append("CRITICAL: No rate periods extracted")

        return errors

    def _validate_dates(self, contract: ContractInfo) -> Dict[str, List[str]]:
        """Validate date fields for format and logical consistency."""
        errors = []
        warnings = []

        date_fields = {
            'contract_date': contract.contract_date,
            'handover_date': contract.handover_date,
            'effective_date': contract.effective_date,
            'expiration_date': contract.expiration_date
        }

        parsed_dates = {}

        # Validate date formats
        for field_name, date_value in date_fields.items():
            if date_value:
                try:
                    parsed = datetime.strptime(date_value, '%m-%d-%Y')
                    parsed_dates[field_name] = parsed
                except ValueError:
                    errors.append(f"Invalid date format for {field_name}: '{date_value}' (expected MM-DD-YYYY)")

        # Logical date checks
        if 'contract_date' in parsed_dates and 'handover_date' in parsed_dates:
            if parsed_dates['handover_date'] < parsed_dates['contract_date']:
                warnings.append("Handover date is before contract date (unusual)")

        if 'effective_date' in parsed_dates and 'expiration_date' in parsed_dates:
            if parsed_dates['expiration_date'] <= parsed_dates['effective_date']:
                errors.append("Expiration date must be after effective date")

        # Check if dates are in reasonable range (not too old or too far in future)
        current_year = datetime.now().year
        for field_name, date_obj in parsed_dates.items():
            if date_obj.year < 2000:
                warnings.append(f"{field_name} is before year 2000: {date_obj.year}")
            elif date_obj.year > current_year + 20:
                warnings.append(f"{field_name} is more than 20 years in future: {date_obj.year}")

        return {'errors': errors, 'warnings': warnings}

    def _validate_numeric_fields(self, contract: ContractInfo) -> Dict[str, List[str]]:
        """Validate numeric fields are actually numeric and reasonable."""
        errors = []
        warnings = []

        numeric_fields = {
            'gfa': contract.gfa,
            'gla_for_lease': contract.gla_for_lease,
            'deposit_amount': contract.deposit_amount,
            'service_charge_rate': contract.service_charge_rate,
            'service_charge_total': contract.service_charge_total
        }

        for field_name, value in numeric_fields.items():
            if value is not None:
                try:
                    num_value = float(value)

                    # Check for negative values
                    if num_value < 0:
                        errors.append(f"{field_name} cannot be negative: {value}")

                    # Reasonableness checks
                    if field_name in ['gfa', 'gla_for_lease']:
                        if num_value == 0:
                            warnings.append(f"{field_name} is zero (unusual)")
                        elif num_value > 10000:
                            warnings.append(f"{field_name} is very large: {value} sqm (verify accuracy)")

                    if field_name == 'deposit_amount':
                        if num_value == 0:
                            warnings.append("Deposit amount is zero (verify if correct)")

                except ValueError:
                    errors.append(f"{field_name} is not a valid number: '{value}'")

        return {'errors': errors, 'warnings': warnings}

    def _validate_rate_periods(self, contract: ContractInfo) -> Dict[str, List[str]]:
        """Validate rate periods for completeness and consistency."""
        errors = []
        warnings = []
        info = []

        if not contract.rate_periods:
            return {'errors': errors, 'warnings': warnings, 'info': info}

        info.append(f"Total rate periods: {len(contract.rate_periods)}")

        rent_free_count = 0
        total_months = 0

        for i, period in enumerate(contract.rate_periods, start=1):
            # Check required fields
            if not period.start_date:
                errors.append(f"Period {i}: Missing start_date")
            if not period.end_date:
                errors.append(f"Period {i}: Missing end_date")
            if not period.monthly_rate_per_sqm:
                warnings.append(f"Period {i}: Missing monthly_rate_per_sqm")

            # Validate dates if present
            if period.start_date and period.end_date:
                try:
                    start = datetime.strptime(period.start_date, '%m-%d-%Y')
                    end = datetime.strptime(period.end_date, '%m-%d-%Y')

                    if end <= start:
                        errors.append(f"Period {i}: End date must be after start date")

                    # Calculate months
                    months = ((end.year - start.year) * 12 + end.month - start.month) + 1
                    total_months += months

                    if months > 120:
                        warnings.append(f"Period {i}: Very long period ({months} months)")

                except ValueError as e:
                    errors.append(f"Period {i}: Invalid date format - {e}")

            # Check for rent-free periods using foc_from/foc_to fields
            if period.foc_from and period.foc_to:
                rent_free_count += 1
                # Validate FOC dates if present
                try:
                    foc_start = datetime.strptime(period.foc_from, '%m-%d-%Y')
                    foc_end = datetime.strptime(period.foc_to, '%m-%d-%Y')

                    if foc_end <= foc_start:
                        errors.append(f"Period {i}: FOC end date must be after FOC start date")

                    # Check that FOC period is within the rate period
                    if period.start_date and period.end_date:
                        period_start = datetime.strptime(period.start_date, '%m-%d-%Y')
                        period_end = datetime.strptime(period.end_date, '%m-%d-%Y')

                        if foc_start < period_start or foc_end > period_end:
                            warnings.append(f"Period {i}: FOC dates ({period.foc_from} to {period.foc_to}) are outside rate period dates")

                except ValueError as e:
                    errors.append(f"Period {i}: Invalid FOC date format - {e}")

            # Validate monthly_rate_per_sqm is not "0" (FOC should use foc_from/foc_to fields)
            if period.monthly_rate_per_sqm:
                try:
                    rate = float(period.monthly_rate_per_sqm)
                    if rate == 0:
                        warnings.append(f"Period {i}: monthly_rate_per_sqm is 0 (use foc_from/foc_to fields for FOC periods instead)")
                except ValueError:
                    errors.append(f"Period {i}: Invalid monthly_rate_per_sqm: '{period.monthly_rate_per_sqm}'")

            # Validate total_monthly_rate if present
            if period.total_monthly_rate and period.monthly_rate_per_sqm:
                area = float(contract.gfa or contract.gla_for_lease or 0)
                if area > 0:
                    try:
                        rate = float(period.monthly_rate_per_sqm)
                        total = float(period.total_monthly_rate)
                        expected_total = rate * area

                        # Allow 1% tolerance for rounding
                        if abs(total - expected_total) > expected_total * 0.01:
                            warnings.append(
                                f"Period {i}: total_monthly_rate ({total}) doesn't match "
                                f"rate_per_sqm × area ({expected_total:.2f})"
                            )
                    except ValueError:
                        pass  # Already caught in numeric validation

        # Check for period gaps/overlaps
        if len(contract.rate_periods) > 1:
            sorted_periods = sorted(contract.rate_periods, key=lambda p: p.start_date or '')
            for i in range(len(sorted_periods) - 1):
                try:
                    end1 = datetime.strptime(sorted_periods[i].end_date, '%m-%d-%Y')
                    start2 = datetime.strptime(sorted_periods[i+1].start_date, '%m-%d-%Y')

                    gap_days = (start2 - end1).days
                    if gap_days > 1:
                        warnings.append(f"Gap detected between period {i+1} and {i+2}: {gap_days} days")
                    elif gap_days < 0:
                        warnings.append(f"Overlap detected between period {i+1} and {i+2}")
                except (ValueError, AttributeError):
                    pass  # Skip if dates invalid

        if rent_free_count > 0:
            info.append(f"Rent-free periods: {rent_free_count}")

        if total_months > 0:
            info.append(f"Total contract duration: {total_months} months")

        return {'errors': errors, 'warnings': warnings, 'info': info}

    def _validate_cross_field_logic(self, contract: ContractInfo) -> Dict[str, List[str]]:
        """Validate logical relationships between fields."""
        warnings = []
        info = []

        # Check if customer_name and tenant are consistent
        if contract.customer_name and contract.tenant:
            if contract.customer_name.lower() != contract.tenant.lower():
                info.append(f"customer_name ('{contract.customer_name}') differs from tenant ('{contract.tenant}')")

        # Check if GFA and GLA are both present and similar
        if contract.gfa and contract.gla_for_lease:
            try:
                gfa = float(contract.gfa)
                gla = float(contract.gla_for_lease)

                if abs(gfa - gla) > max(gfa, gla) * 0.1:  # More than 10% difference
                    warnings.append(f"GFA ({gfa}) and GLA ({gla}) differ by more than 10%")
            except ValueError:
                pass

        # Validate deposit amount reasonableness (typically 1-6 months rent)
        if contract.deposit_amount and contract.rate_periods:
            try:
                deposit = float(contract.deposit_amount)
                # Find first non-zero monthly rate
                for period in contract.rate_periods:
                    if period.total_monthly_rate:
                        try:
                            monthly_rate = float(period.total_monthly_rate)
                            if monthly_rate > 0:
                                months_equivalent = deposit / monthly_rate

                                if months_equivalent > 12:
                                    warnings.append(
                                        f"Deposit is equivalent to {months_equivalent:.1f} months rent (unusually high)"
                                    )
                                elif months_equivalent < 0.5:
                                    warnings.append(
                                        f"Deposit is equivalent to {months_equivalent:.1f} months rent (unusually low)"
                                    )
                                else:
                                    info.append(f"Deposit: {months_equivalent:.1f} months rent equivalent")
                                break
                        except ValueError:
                            pass
            except ValueError:
                pass

        return {'warnings': warnings, 'info': info}

    def _validate_service_charge(self, contract: ContractInfo) -> Dict[str, List[str]]:
        """Validate service charge calculation and logic."""
        warnings = []
        info = []

        applies_to = contract.service_charge_applies_to

        if applies_to == 'not_applicable':
            if contract.service_charge_rate or contract.service_charge_total:
                warnings.append("Service charge marked as 'not_applicable' but rate/total present")
            return {'warnings': warnings, 'info': info}

        if contract.service_charge_rate and not contract.service_charge_total:
            warnings.append("Service charge rate present but total not calculated")

        if contract.service_charge_total:
            info.append(f"Service charge total: {contract.service_charge_total}")

        if applies_to == 'rent_free_only':
            # Check if there are actually rent-free periods (using foc_from/foc_to fields)
            has_rent_free = False
            if contract.rate_periods:
                for period in contract.rate_periods:
                    if period.foc_from and period.foc_to:
                        has_rent_free = True
                        break

            if not has_rent_free:
                warnings.append("Service charge applies to 'rent_free_only' but no FOC periods found (check foc_from/foc_to fields)")

        return {'warnings': warnings, 'info': info}

    def get_validation_summary(self, validation_result: Dict[str, List[str]]) -> str:
        """
        Generate a human-readable validation summary.

        Args:
            validation_result: Result from validate_contract()

        Returns:
            Formatted summary string
        """
        errors = validation_result.get('errors', [])
        warnings = validation_result.get('warnings', [])
        info = validation_result.get('info', [])

        summary_lines = []

        if errors:
            summary_lines.append(f"❌ ERRORS ({len(errors)}):")
            for error in errors:
                summary_lines.append(f"  - {error}")

        if warnings:
            summary_lines.append(f"⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                summary_lines.append(f"  - {warning}")

        if info:
            summary_lines.append(f"ℹ️  INFO ({len(info)}):")
            for item in info:
                summary_lines.append(f"  - {item}")

        if not errors and not warnings:
            summary_lines.append("✅ All validations passed")

        return "\n".join(summary_lines)
