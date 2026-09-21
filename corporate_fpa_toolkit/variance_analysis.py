"""Actual-vs-budget variance analysis: volume / price / mix / spend.

**FP&A concept -- the variance bridge.** Once a year is closed, FP&A
explains *why* actual results differed from budget by decomposing the
gap into named drivers rather than reporting a single "we missed by
$X" number:

* **Volume variance** -- what the revenue change would have been if
  only unit volumes had moved (at budget prices/mix).
* **Price variance** -- the revenue impact of selling at a different
  price than budgeted, at actual volumes.
* **Mix variance** -- a *portfolio* effect: even with total volume
  flat, revenue moves if the business sells relatively more of a
  higher-priced/higher-margin unit and less of a lower-priced one.
  Mix only exists when there is more than one product/segment to mix
  across, which is why it is computed at the consolidated
  (multi-business-unit) level rather than within a single BU.
* **Spend (cost) variance** -- for opex, the analogous split between
  "we had more/fewer people or ran more/less fixed spend than
  budgeted" (a volume/headcount effect) and "the rate we paid was
  different than budgeted" (a spend/rate effect).

Combined, the four effects reconcile exactly back to the total
actual-vs-budget change -- that reconciliation ("variance walk") is the
core deliverable FP&A presents to leadership each period.

.. warning:: All figures are synthetic -- see ``corporate_fpa_toolkit.SYNTHETIC_DATA_DISCLAIMER``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .driver_based_budget import (
    BusinessUnitDrivers,
    annual_summary,
    build_all_unit_budgets,
    consolidate_monthly,
    default_business_units,
)

REVENUE_BU_NAMES = ("eCommerce", "B2B Contract Sales", "Retail")


def generate_actual_business_units(
    budget_units: list[BusinessUnitDrivers],
    seed: int = 42,
    volume_shock_sd: float = 0.06,
    price_shock_sd: float = 0.025,
    variable_cost_shock_sd: float = 0.035,
    headcount_shock_sd: float = 0.03,
    cost_per_head_shock_sd: float = 0.02,
    fixed_opex_shock_sd: float = 0.045,
) -> list[BusinessUnitDrivers]:
    """Generate a synthetic "actual" year by perturbing each BU's drivers.

    Each driver is shocked independently and multiplicatively with a
    normally-distributed percentage shock (clipped to +/-3 standard
    deviations to avoid pathological draws). This produces a full
    actual-year driver set with the *same* structure (and seasonality)
    as the budget, so the same phasing/roll-up functions can build an
    "actual" monthly P&L exactly as they build the budget.
    """
    rng = np.random.default_rng(seed)

    def shock(sd: float) -> float:
        return float(np.clip(rng.normal(loc=0.0, scale=sd), -3 * sd, 3 * sd))

    actuals: list[BusinessUnitDrivers] = []
    for unit in budget_units:
        new_units = unit.annual_units * (1 + shock(volume_shock_sd))
        new_price = None if unit.is_cost_center else unit.price_per_unit * (1 + shock(price_shock_sd))
        new_var_cost = unit.variable_cost_per_unit * (1 + shock(variable_cost_shock_sd))
        new_headcount = max(0.0, unit.headcount * (1 + shock(headcount_shock_sd)))
        new_cost_per_head = unit.cost_per_head * (1 + shock(cost_per_head_shock_sd))
        new_fixed_opex = unit.other_fixed_opex_annual * (1 + shock(fixed_opex_shock_sd))
        actuals.append(
            unit.with_overrides(
                annual_units=new_units,
                price_per_unit=new_price,
                variable_cost_per_unit=new_var_cost,
                headcount=new_headcount,
                cost_per_head=new_cost_per_head,
                other_fixed_opex_annual=new_fixed_opex,
            )
        )
    return actuals


def build_actual_results(actual_units: list[BusinessUnitDrivers]) -> dict[str, pd.DataFrame]:
    """Build the actual-year monthly/annual P&L using the same roll-up
    logic used for the budget (so budget and actual are perfectly
    comparable line-for-line).
    """
    by_unit_monthly = build_all_unit_budgets(actual_units)
    return {
        "by_unit_monthly": by_unit_monthly,
        "consolidated_monthly": consolidate_monthly(by_unit_monthly),
        "annual_summary": annual_summary(by_unit_monthly),
    }


def _unit_lookup(units: list[BusinessUnitDrivers]) -> dict[str, BusinessUnitDrivers]:
    return {u.name: u for u in units}


def revenue_variance_by_unit(
    budget_units: list[BusinessUnitDrivers],
    actual_units: list[BusinessUnitDrivers],
) -> pd.DataFrame:
    """Business-unit-level 2-way revenue variance: volume and price.

    For a single BU (single "product"), revenue variance decomposes
    exactly into a volume effect (units delta priced at the *budget*
    price) and a price effect (price delta applied to *actual* units):

    ``total_variance == volume_variance + price_variance``  (per row)
    """
    b_lookup, a_lookup = _unit_lookup(budget_units), _unit_lookup(actual_units)
    rows = []
    for name in REVENUE_BU_NAMES:
        b, a = b_lookup[name], a_lookup[name]
        volume_variance = (a.annual_units - b.annual_units) * b.price_per_unit
        price_variance = (a.price_per_unit - b.price_per_unit) * a.annual_units
        rows.append(
            {
                "business_unit": name,
                "budget_units": b.annual_units,
                "actual_units": a.annual_units,
                "budget_price": b.price_per_unit,
                "actual_price": a.price_per_unit,
                "budget_revenue": b.annual_revenue,
                "actual_revenue": a.annual_revenue,
                "volume_variance": volume_variance,
                "price_variance": price_variance,
                "total_variance": volume_variance + price_variance,
            }
        )
    return pd.DataFrame(rows)


def consolidated_revenue_variance(
    budget_units: list[BusinessUnitDrivers],
    actual_units: list[BusinessUnitDrivers],
) -> dict[str, object]:
    """Consolidated (portfolio) 3-way revenue variance: volume, mix, price.

    Uses the standard FP&A volume/mix/price framework:

    * ``avg_budget_price`` -- unit-weighted average budget price across
      revenue business units.
    * ``volume_variance``  -- ``avg_budget_price * (total_actual_units - total_budget_units)``
    * ``mix_variance``     -- for each BU, its budget price times the gap
      between its actual units and what it "should" have sold if the
      *budget mix* had simply scaled up/down to the actual total volume.
    * ``price_variance``   -- sum of each BU's own price variance.

    These three reconcile exactly to the total consolidated revenue
    variance: ``volume + mix + price == actual_revenue - budget_revenue``.
    """
    b_lookup, a_lookup = _unit_lookup(budget_units), _unit_lookup(actual_units)

    total_budget_units = sum(b_lookup[n].annual_units for n in REVENUE_BU_NAMES)
    total_actual_units = sum(a_lookup[n].annual_units for n in REVENUE_BU_NAMES)
    total_budget_revenue = sum(b_lookup[n].annual_revenue for n in REVENUE_BU_NAMES)
    total_actual_revenue = sum(a_lookup[n].annual_revenue for n in REVENUE_BU_NAMES)
    avg_budget_price = total_budget_revenue / total_budget_units

    volume_variance = avg_budget_price * (total_actual_units - total_budget_units)

    mix_rows = []
    price_variance = 0.0
    for name in REVENUE_BU_NAMES:
        b, a = b_lookup[name], a_lookup[name]
        budget_mix_share = b.annual_units / total_budget_units
        expected_units_at_budget_mix = budget_mix_share * total_actual_units
        mix_variance_bu = b.price_per_unit * (a.annual_units - expected_units_at_budget_mix)
        mix_rows.append({"business_unit": name, "mix_variance": mix_variance_bu})
        price_variance += (a.price_per_unit - b.price_per_unit) * a.annual_units

    mix_variance = sum(r["mix_variance"] for r in mix_rows)
    total_variance = total_actual_revenue - total_budget_revenue

    return {
        "avg_budget_price": avg_budget_price,
        "total_budget_units": total_budget_units,
        "total_actual_units": total_actual_units,
        "total_budget_revenue": total_budget_revenue,
        "total_actual_revenue": total_actual_revenue,
        "volume_variance": volume_variance,
        "mix_variance": mix_variance,
        "price_variance": price_variance,
        "total_variance": total_variance,
        "mix_by_unit": pd.DataFrame(mix_rows),
    }


def cogs_spend_bridge_by_unit(
    budget_units: list[BusinessUnitDrivers],
    actual_units: list[BusinessUnitDrivers],
) -> pd.DataFrame:
    """Per-BU COGS/variable-cost bridge: volume effect + rate (spend) effect.

    ``total_variance == volume_variance + rate_variance`` for every BU
    (revenue-generating BUs and the Supply Chain & Distribution cost
    center alike, since all four carry a units x variable-cost-per-unit
    structure).
    """
    b_lookup, a_lookup = _unit_lookup(budget_units), _unit_lookup(actual_units)
    rows = []
    for unit in budget_units:
        b, a = b_lookup[unit.name], a_lookup[unit.name]
        volume_variance = (a.annual_units - b.annual_units) * b.variable_cost_per_unit
        rate_variance = (a.variable_cost_per_unit - b.variable_cost_per_unit) * a.annual_units
        rows.append(
            {
                "business_unit": unit.name,
                "budget_cogs": b.annual_variable_cost,
                "actual_cogs": a.annual_variable_cost,
                "volume_variance": volume_variance,
                "rate_variance": rate_variance,
                "total_variance": volume_variance + rate_variance,
            }
        )
    return pd.DataFrame(rows)


def opex_spend_bridge_by_unit(
    budget_units: list[BusinessUnitDrivers],
    actual_units: list[BusinessUnitDrivers],
) -> pd.DataFrame:
    """Per-BU SG&A spend variance bridge: headcount volume, pay-rate, and
    fixed-opex spend effects, reconciling to the total SG&A variance.

    ``total_variance == headcount_volume_variance + rate_variance + fixed_opex_spend_variance``
    """
    b_lookup, a_lookup = _unit_lookup(budget_units), _unit_lookup(actual_units)
    rows = []
    for unit in budget_units:
        b, a = b_lookup[unit.name], a_lookup[unit.name]
        headcount_volume_variance = (a.headcount - b.headcount) * b.cost_per_head
        rate_variance = (a.cost_per_head - b.cost_per_head) * a.headcount
        fixed_opex_spend_variance = a.other_fixed_opex_annual - b.other_fixed_opex_annual
        total = headcount_volume_variance + rate_variance + fixed_opex_spend_variance
        rows.append(
            {
                "business_unit": unit.name,
                "budget_sga": b.annual_sga,
                "actual_sga": a.annual_sga,
                "headcount_volume_variance": headcount_volume_variance,
                "rate_variance": rate_variance,
                "fixed_opex_spend_variance": fixed_opex_spend_variance,
                "total_variance": total,
            }
        )
    return pd.DataFrame(rows)


def ebitda_variance_walk(
    budget_units: list[BusinessUnitDrivers],
    actual_units: list[BusinessUnitDrivers],
) -> pd.DataFrame:
    """Consolidated EBITDA variance walk: Budget -> Volume -> Mix -> Price -> Cost -> Actual.

    Volume and Mix are computed on *contribution margin* (price less
    variable cost) rather than on revenue alone, because a unit-volume
    change also drags variable cost along with it; Price is the pure
    price-rate effect at actual volumes; Cost effect captures every
    remaining rate/spend variance (COGS rate for the revenue BUs, the
    Supply Chain & Distribution cost center's full cost variance, and
    SG&A spend variance across all four BUs). The five deltas
    reconcile exactly:

    ``budget_ebitda + volume + mix + price + cost == actual_ebitda``
    """
    b_lookup, a_lookup = _unit_lookup(budget_units), _unit_lookup(actual_units)

    budget_ebitda = sum(u.annual_ebitda for u in budget_units)
    actual_ebitda = sum(u.annual_ebitda for u in actual_units)

    total_budget_units = sum(b_lookup[n].annual_units for n in REVENUE_BU_NAMES)
    total_actual_units = sum(a_lookup[n].annual_units for n in REVENUE_BU_NAMES)

    budget_cm_total = sum(b_lookup[n].annual_revenue - b_lookup[n].annual_variable_cost for n in REVENUE_BU_NAMES)
    avg_budget_cm = budget_cm_total / total_budget_units

    volume_effect = avg_budget_cm * (total_actual_units - total_budget_units)

    mix_effect = 0.0
    price_effect = 0.0
    for name in REVENUE_BU_NAMES:
        b, a = b_lookup[name], a_lookup[name]
        budget_cm_unit = b.price_per_unit - b.variable_cost_per_unit
        budget_mix_share = b.annual_units / total_budget_units
        expected_units_at_budget_mix = budget_mix_share * total_actual_units
        mix_effect += budget_cm_unit * (a.annual_units - expected_units_at_budget_mix)
        price_effect += (a.price_per_unit - b.price_per_unit) * a.annual_units

    cogs_bridge = cogs_spend_bridge_by_unit(budget_units, actual_units).set_index("business_unit")
    opex_bridge = opex_spend_bridge_by_unit(budget_units, actual_units).set_index("business_unit")

    cogs_rate_variance_revenue_bus = sum(cogs_bridge.loc[n, "rate_variance"] for n in REVENUE_BU_NAMES)
    sga_variance_all_bus = opex_bridge["total_variance"].sum()
    distribution_cost_variance = cogs_bridge.loc["Supply Chain & Distribution", "total_variance"]

    cost_effect = -(cogs_rate_variance_revenue_bus + sga_variance_all_bus + distribution_cost_variance)

    walk = pd.DataFrame(
        [
            {"step": "Budget EBITDA", "delta": np.nan, "value": budget_ebitda},
            {"step": "Volume Effect", "delta": volume_effect, "value": np.nan},
            {"step": "Mix Effect", "delta": mix_effect, "value": np.nan},
            {"step": "Price Effect", "delta": price_effect, "value": np.nan},
            {"step": "Cost Effect", "delta": cost_effect, "value": np.nan},
            {"step": "Actual EBITDA", "delta": np.nan, "value": actual_ebitda},
        ]
    )
    running = budget_ebitda
    values = [budget_ebitda]
    for delta in walk.loc[1:4, "delta"]:
        running += delta
        values.append(running)
    values.append(actual_ebitda)
    walk["value"] = values
    return walk


def full_variance_report(
    budget_units: Optional[list[BusinessUnitDrivers]] = None,
    seed: int = 42,
) -> dict[str, object]:
    """Convenience entry point bundling every variance analysis output."""
    budget_units = budget_units if budget_units is not None else default_business_units()
    actual_units = generate_actual_business_units(budget_units, seed=seed)
    return {
        "budget_units": budget_units,
        "actual_units": actual_units,
        "actual_results": build_actual_results(actual_units),
        "revenue_variance_by_unit": revenue_variance_by_unit(budget_units, actual_units),
        "consolidated_revenue_variance": consolidated_revenue_variance(budget_units, actual_units),
        "cogs_spend_bridge_by_unit": cogs_spend_bridge_by_unit(budget_units, actual_units),
        "opex_spend_bridge_by_unit": opex_spend_bridge_by_unit(budget_units, actual_units),
        "ebitda_variance_walk": ebitda_variance_walk(budget_units, actual_units),
    }


if __name__ == "__main__":
    from . import SYNTHETIC_DATA_DISCLAIMER

    print(SYNTHETIC_DATA_DISCLAIMER)
    report = full_variance_report()
    print("\n--- EBITDA Variance Walk ---")
    print(report["ebitda_variance_walk"].to_string(index=False))
