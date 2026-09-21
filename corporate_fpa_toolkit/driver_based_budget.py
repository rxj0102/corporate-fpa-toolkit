"""Driver-based annual budget for a synthetic multi-BU retail/distribution company.

**FP&A concept.** A driver-based budget is built bottom-up from the
operational levers that actually generate revenue and cost (units sold,
price per unit, headcount, cost per head, cost per unit handled) rather
than a top-down "grow last year by X%". Each business unit (BU) owns an
explicit, auditable set of drivers; the P&L is a *derived* output of
those drivers, and monthly phasing is applied through a seasonality
curve so the annual budget ties exactly to twelve months that sum back
to it.

This module defines the four synthetic business units, builds each
one's monthly budget, and rolls them up into a consolidated P&L from
Revenue through EBITDA.

.. warning:: All figures are synthetic -- see ``corporate_fpa_toolkit.SYNTHETIC_DATA_DISCLAIMER``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

from .common import MONTH_NAMES, MONTHS_PER_YEAR, blended_seasonality, even_seasonality, peaked_seasonality

REVENUE_SEGMENTS = ("eCommerce", "B2B Contract Sales", "Retail")
ALL_SEGMENTS = ("eCommerce", "B2B Contract Sales", "Retail", "Supply Chain & Distribution")


@dataclass(frozen=True)
class BusinessUnitDrivers:
    """Operational drivers for one business unit's annual budget.

    ``Supply Chain & Distribution`` is modeled as an internal cost
    center (``is_cost_center=True``, ``price_per_unit=None``): it has no
    external revenue, but its per-unit handling cost, headcount, and
    fixed opex flow into the consolidated COGS/SG&A the same way a
    revenue-generating BU's costs do.
    """

    name: str
    is_cost_center: bool
    annual_units: float
    price_per_unit: Optional[float]
    variable_cost_per_unit: float
    headcount: float
    cost_per_head: float
    other_fixed_opex_annual: float
    volume_seasonality: np.ndarray = field(default_factory=even_seasonality)
    opex_seasonality: np.ndarray = field(default_factory=even_seasonality)

    def __post_init__(self) -> None:
        if self.is_cost_center and self.price_per_unit is not None:
            raise ValueError(f"{self.name}: cost centers must have price_per_unit=None")
        if not self.is_cost_center and self.price_per_unit is None:
            raise ValueError(f"{self.name}: revenue units must set price_per_unit")
        if self.annual_units < 0 or self.variable_cost_per_unit < 0:
            raise ValueError(f"{self.name}: units and variable cost must be non-negative")
        if len(self.volume_seasonality) != MONTHS_PER_YEAR or len(self.opex_seasonality) != MONTHS_PER_YEAR:
            raise ValueError(f"{self.name}: seasonality curves must have 12 entries")
        if not np.isclose(self.volume_seasonality.sum(), 1.0):
            raise ValueError(f"{self.name}: volume_seasonality must sum to 1.0")
        if not np.isclose(self.opex_seasonality.sum(), 1.0):
            raise ValueError(f"{self.name}: opex_seasonality must sum to 1.0")

    @property
    def annual_revenue(self) -> float:
        if self.is_cost_center:
            return 0.0
        return self.annual_units * self.price_per_unit

    @property
    def annual_variable_cost(self) -> float:
        return self.annual_units * self.variable_cost_per_unit

    @property
    def annual_headcount_cost(self) -> float:
        return self.headcount * self.cost_per_head

    @property
    def annual_sga(self) -> float:
        return self.annual_headcount_cost + self.other_fixed_opex_annual

    @property
    def annual_ebitda(self) -> float:
        if self.is_cost_center:
            return -(self.annual_variable_cost + self.annual_sga)
        return self.annual_revenue - self.annual_variable_cost - self.annual_sga

    def with_overrides(self, **kwargs) -> "BusinessUnitDrivers":
        """Return a copy of this BU with the given driver fields replaced.

        Convenient for building "actual" or scenario variants of a BU
        without mutating the budget baseline.
        """
        return replace(self, **kwargs)


def default_business_units() -> list[BusinessUnitDrivers]:
    """Construct the four synthetic business units used throughout the toolkit.

    Loosely modeled (for flavor only -- see module docstring) on a
    multi-segment retail/distribution company: an eCommerce channel, a
    B2B/contract sales channel, a store-based Retail channel, and an
    internal Supply Chain & Distribution function that fulfills all
    three.
    """
    ecommerce_seasonality = peaked_seasonality(peak_months=[11, 12], peak_multiplier=2.2, base=1.0)
    b2b_seasonality = peaked_seasonality(peak_months=[3, 12], peak_multiplier=1.35, base=1.0)
    retail_seasonality = peaked_seasonality(peak_months=[8, 9, 11, 12], peak_multiplier=1.6, base=1.0)

    ecommerce = BusinessUnitDrivers(
        name="eCommerce",
        is_cost_center=False,
        annual_units=18_000_000,
        price_per_unit=85.0,
        variable_cost_per_unit=54.0,
        headcount=380,
        cost_per_head=92_000.0,
        other_fixed_opex_annual=140_000_000.0,
        volume_seasonality=ecommerce_seasonality,
        opex_seasonality=even_seasonality(),
    )
    b2b = BusinessUnitDrivers(
        name="B2B Contract Sales",
        is_cost_center=False,
        annual_units=120_000,
        price_per_unit=9_500.0,
        variable_cost_per_unit=6_800.0,
        headcount=520,
        cost_per_head=100_000.0,
        other_fixed_opex_annual=30_000_000.0,
        volume_seasonality=b2b_seasonality,
        opex_seasonality=even_seasonality(),
    )
    retail = BusinessUnitDrivers(
        name="Retail",
        is_cost_center=False,
        annual_units=95_000_000,
        price_per_unit=14.5,
        variable_cost_per_unit=9.6,
        headcount=12_000,
        cost_per_head=30_000.0,
        other_fixed_opex_annual=220_000_000.0,
        volume_seasonality=retail_seasonality,
        opex_seasonality=peaked_seasonality(peak_months=[11, 12], peak_multiplier=1.4, base=1.0),
    )
    distribution_seasonality = blended_seasonality(
        curves=[ecommerce_seasonality, b2b_seasonality, retail_seasonality],
        weights=[ecommerce.annual_units, b2b.annual_units, retail.annual_units],
    )
    distribution = BusinessUnitDrivers(
        name="Supply Chain & Distribution",
        is_cost_center=True,
        annual_units=190_000_000,
        price_per_unit=None,
        variable_cost_per_unit=0.95,
        headcount=1_800,
        cost_per_head=46_000.0,
        other_fixed_opex_annual=110_000_000.0,
        volume_seasonality=distribution_seasonality,
        opex_seasonality=even_seasonality(),
    )
    return [ecommerce, b2b, retail, distribution]


def build_unit_monthly_budget(unit: BusinessUnitDrivers) -> pd.DataFrame:
    """Phase one business unit's annual drivers into a 12-month budget.

    Volume (units, revenue, variable cost) is phased with
    ``unit.volume_seasonality``; headcount cost and fixed opex are
    phased with ``unit.opex_seasonality``. Each monthly column ties
    exactly (to floating-point precision) back to the BU's annual total.
    """
    months = np.arange(1, MONTHS_PER_YEAR + 1)
    units = unit.annual_units * unit.volume_seasonality
    revenue = units * unit.price_per_unit if not unit.is_cost_center else np.zeros(MONTHS_PER_YEAR)
    variable_cost = units * unit.variable_cost_per_unit
    headcount_cost = unit.annual_headcount_cost * unit.opex_seasonality
    fixed_opex = unit.other_fixed_opex_annual * unit.opex_seasonality
    cogs = variable_cost
    sga = headcount_cost + fixed_opex
    gross_profit = revenue - cogs
    ebitda = gross_profit - sga

    df = pd.DataFrame(
        {
            "business_unit": unit.name,
            "month": months,
            "month_name": [MONTH_NAMES[m - 1] for m in months],
            "units": units,
            "revenue": revenue,
            "cogs": cogs,
            "gross_profit": gross_profit,
            "headcount_cost": headcount_cost,
            "fixed_opex": fixed_opex,
            "sga": sga,
            "ebitda": ebitda,
        }
    )
    return df


def build_all_unit_budgets(units: list[BusinessUnitDrivers]) -> pd.DataFrame:
    """Long-format (tidy) monthly budget for every business unit, stacked."""
    return pd.concat([build_unit_monthly_budget(u) for u in units], ignore_index=True)


def consolidate_monthly(unit_budgets: pd.DataFrame) -> pd.DataFrame:
    """Roll a long-format per-BU monthly budget up into a consolidated
    monthly P&L (Revenue through EBITDA).
    """
    agg = (
        unit_budgets.groupby(["month", "month_name"], sort=True)[
            ["units", "revenue", "cogs", "gross_profit", "headcount_cost", "fixed_opex", "sga", "ebitda"]
        ]
        .sum()
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )
    return agg


def annual_summary(unit_budgets: pd.DataFrame) -> pd.DataFrame:
    """Annual (full-year) total per business unit plus a Consolidated row."""
    by_unit = (
        unit_budgets.groupby("business_unit", sort=False)[
            ["units", "revenue", "cogs", "gross_profit", "headcount_cost", "fixed_opex", "sga", "ebitda"]
        ]
        .sum()
        .reset_index()
    )
    consolidated = by_unit.drop(columns="business_unit").sum().to_frame().T
    consolidated.insert(0, "business_unit", "Consolidated")
    return pd.concat([by_unit, consolidated], ignore_index=True)


def build_budget(units: Optional[list[BusinessUnitDrivers]] = None) -> dict[str, pd.DataFrame]:
    """Convenience entry point: build the full driver-based budget.

    Returns a dict with keys ``"by_unit_monthly"`` (tidy long format),
    ``"consolidated_monthly"``, and ``"annual_summary"``.
    """
    units = units if units is not None else default_business_units()
    by_unit_monthly = build_all_unit_budgets(units)
    return {
        "by_unit_monthly": by_unit_monthly,
        "consolidated_monthly": consolidate_monthly(by_unit_monthly),
        "annual_summary": annual_summary(by_unit_monthly),
    }


if __name__ == "__main__":
    from . import SYNTHETIC_DATA_DISCLAIMER

    print(SYNTHETIC_DATA_DISCLAIMER)
    result = build_budget()
    print(result["annual_summary"].to_string(index=False))
