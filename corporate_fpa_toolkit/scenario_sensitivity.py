"""Scenario analysis and tornado/sensitivity analysis.

**FP&A concept -- scenarios vs. sensitivities.** These are two related
but distinct exercises:

* A **scenario** (Base/Upside/Downside) moves a *coherent bundle* of
  macro and operational assumptions together -- e.g. "a mild recession"
  means volume down, price growth soft, and input costs up, all at
  once -- to answer "what does the P&L and cash flow look like under
  this story."
* A **sensitivity / tornado analysis** instead flexes *one driver at a
  time*, holding everything else at the base case, to answer "which
  individual assumption, if we got it wrong, would hurt us the most."
  Ranking drivers by the size of their swing (high case minus low
  case) and plotting them widest-bar-on-top produces the classic
  "tornado chart" shape FP&A teams use to focus diligence and
  monitoring on the few drivers that actually move the outcome.

Both build on the driver-based budget (:mod:`corporate_fpa_toolkit.driver_based_budget`)
and the cash flow model (:mod:`corporate_fpa_toolkit.cash_flow_model`).

.. warning:: All figures are synthetic -- see ``corporate_fpa_toolkit.SYNTHETIC_DATA_DISCLAIMER``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .cash_flow_model import (
    CapexAssumptions,
    DebtAssumptions,
    WorkingCapitalAssumptions,
    build_cash_flow_model,
)
from .driver_based_budget import (
    BusinessUnitDrivers,
    annual_summary,
    build_all_unit_budgets,
    consolidate_monthly,
    default_business_units,
)


@dataclass(frozen=True)
class ScenarioAssumptions:
    """A coherent bundle of macro/operational assumptions defining one scenario."""

    name: str
    volume_growth: float
    price_growth: float
    cost_inflation: float


BASE_SCENARIO = ScenarioAssumptions(name="Base", volume_growth=0.0, price_growth=0.0, cost_inflation=0.0)
UPSIDE_SCENARIO = ScenarioAssumptions(name="Upside", volume_growth=0.04, price_growth=0.015, cost_inflation=0.01)
DOWNSIDE_SCENARIO = ScenarioAssumptions(name="Downside", volume_growth=-0.05, price_growth=-0.01, cost_inflation=0.035)
DEFAULT_SCENARIOS: tuple[ScenarioAssumptions, ...] = (BASE_SCENARIO, UPSIDE_SCENARIO, DOWNSIDE_SCENARIO)


def apply_scenario(base_units: list[BusinessUnitDrivers], scenario: ScenarioAssumptions) -> list[BusinessUnitDrivers]:
    """Apply a scenario's macro assumptions to every business unit's drivers.

    Volume growth is applied to unit volume for *all* BUs, including the
    Supply Chain & Distribution cost center (its throughput follows
    overall demand). Price growth applies only to revenue-generating
    BUs. Cost inflation applies to variable cost per unit, cost per
    head, and fixed opex across all BUs.
    """
    flexed = []
    for unit in base_units:
        new_units = unit.annual_units * (1 + scenario.volume_growth)
        new_price = None if unit.is_cost_center else unit.price_per_unit * (1 + scenario.price_growth)
        new_var_cost = unit.variable_cost_per_unit * (1 + scenario.cost_inflation)
        new_cost_per_head = unit.cost_per_head * (1 + scenario.cost_inflation)
        new_fixed_opex = unit.other_fixed_opex_annual * (1 + scenario.cost_inflation)
        flexed.append(
            unit.with_overrides(
                annual_units=new_units,
                price_per_unit=new_price,
                variable_cost_per_unit=new_var_cost,
                cost_per_head=new_cost_per_head,
                other_fixed_opex_annual=new_fixed_opex,
            )
        )
    return flexed


def _annual_free_cash_flow(consolidated_monthly: pd.DataFrame) -> float:
    cf = build_cash_flow_model(consolidated_monthly)["monthly_cash_flow"]
    return float(cf["free_cash_flow"].sum())


def run_scenarios(
    base_units: Optional[list[BusinessUnitDrivers]] = None,
    scenarios: tuple[ScenarioAssumptions, ...] = DEFAULT_SCENARIOS,
) -> pd.DataFrame:
    """Run each scenario's P&L and cash flow, returning one comparison row per scenario."""
    base_units = base_units if base_units is not None else default_business_units()
    rows = []
    for scenario in scenarios:
        flexed_units = apply_scenario(base_units, scenario)
        consolidated_monthly = consolidate_monthly(build_all_unit_budgets(flexed_units))
        annual = annual_summary(build_all_unit_budgets(flexed_units))
        consolidated_annual = annual[annual["business_unit"] == "Consolidated"].iloc[0]
        annual_fcf = _annual_free_cash_flow(consolidated_monthly)
        rows.append(
            {
                "scenario": scenario.name,
                "volume_growth": scenario.volume_growth,
                "price_growth": scenario.price_growth,
                "cost_inflation": scenario.cost_inflation,
                "revenue": consolidated_annual["revenue"],
                "ebitda": consolidated_annual["ebitda"],
                "ebitda_margin_pct": consolidated_annual["ebitda"] / consolidated_annual["revenue"] * 100,
                "free_cash_flow": annual_fcf,
            }
        )
    return pd.DataFrame(rows)


# --- Tornado / one-at-a-time sensitivity analysis --------------------------

DEFAULT_SENSITIVITY_SHOCKS: dict[str, float] = {
    "Unit Volume": 0.05,
    "Price": 0.03,
    "Variable Cost per Unit": 0.04,
    "Cost per Head": 0.03,
    "Fixed Opex": 0.05,
    "DSO (days)": 10.0,
    "DPO (days)": 10.0,
    "DIO (days)": 10.0,
    "Capex % of Revenue": 0.005,
}


def _flex_units_for_driver(
    base_units: list[BusinessUnitDrivers], driver: str, shock: float
) -> list[BusinessUnitDrivers]:
    flexed = []
    for unit in base_units:
        if driver == "Unit Volume":
            flexed.append(unit.with_overrides(annual_units=unit.annual_units * (1 + shock)))
        elif driver == "Price":
            if unit.is_cost_center:
                flexed.append(unit)
            else:
                flexed.append(unit.with_overrides(price_per_unit=unit.price_per_unit * (1 + shock)))
        elif driver == "Variable Cost per Unit":
            flexed.append(unit.with_overrides(variable_cost_per_unit=unit.variable_cost_per_unit * (1 + shock)))
        elif driver == "Cost per Head":
            flexed.append(unit.with_overrides(cost_per_head=unit.cost_per_head * (1 + shock)))
        elif driver == "Fixed Opex":
            flexed.append(unit.with_overrides(other_fixed_opex_annual=unit.other_fixed_opex_annual * (1 + shock)))
        else:
            flexed.append(unit)
    return flexed


def _flex_cash_assumptions_for_driver(
    driver: str, shock: float, wc: WorkingCapitalAssumptions, capex_assump: CapexAssumptions
) -> tuple[WorkingCapitalAssumptions, CapexAssumptions]:
    if driver == "DSO (days)":
        wc = WorkingCapitalAssumptions(dso_days=wc.dso_days + shock, dpo_days=wc.dpo_days, dio_days=wc.dio_days)
    elif driver == "DPO (days)":
        wc = WorkingCapitalAssumptions(dso_days=wc.dso_days, dpo_days=wc.dpo_days + shock, dio_days=wc.dio_days)
    elif driver == "DIO (days)":
        wc = WorkingCapitalAssumptions(dso_days=wc.dso_days, dpo_days=wc.dpo_days, dio_days=wc.dio_days + shock)
    elif driver == "Capex % of Revenue":
        capex_assump = CapexAssumptions(
            capex_pct_of_revenue=capex_assump.capex_pct_of_revenue + shock,
            useful_life_years=capex_assump.useful_life_years,
            beginning_net_ppe=capex_assump.beginning_net_ppe,
            beginning_ppe_remaining_life_years=capex_assump.beginning_ppe_remaining_life_years,
        )
    return wc, capex_assump


def _evaluate(
    units: list[BusinessUnitDrivers],
    wc: WorkingCapitalAssumptions,
    capex_assump: CapexAssumptions,
    debt: DebtAssumptions,
) -> tuple[float, float]:
    unit_budgets = build_all_unit_budgets(units)
    consolidated_monthly = consolidate_monthly(unit_budgets)
    annual = annual_summary(unit_budgets)
    ebitda = float(annual.loc[annual["business_unit"] == "Consolidated", "ebitda"].iloc[0])
    cf = build_cash_flow_model(consolidated_monthly, wc=wc, capex_assump=capex_assump, debt=debt)["monthly_cash_flow"]
    fcf = float(cf["free_cash_flow"].sum())
    return ebitda, fcf


def tornado_analysis(
    base_units: Optional[list[BusinessUnitDrivers]] = None,
    wc: WorkingCapitalAssumptions = WorkingCapitalAssumptions(),
    capex_assump: CapexAssumptions = CapexAssumptions(),
    debt: DebtAssumptions = DebtAssumptions(),
    shocks: dict[str, float] = DEFAULT_SENSITIVITY_SHOCKS,
) -> pd.DataFrame:
    """Flex each driver in ``shocks`` up and down (one at a time, all else
    held at base) and measure the resulting swing in consolidated
    annual EBITDA and annual free cash flow.

    Returns a DataFrame sorted by ``ebitda_range`` descending (the
    conventional tornado-chart order: the widest bar -- the driver that
    moves EBITDA the most -- listed first).
    """
    base_units = base_units if base_units is not None else default_business_units()
    base_ebitda, base_fcf = _evaluate(base_units, wc, capex_assump, debt)

    rows = []
    for driver, shock in shocks.items():
        low_units = _flex_units_for_driver(base_units, driver, -shock)
        high_units = _flex_units_for_driver(base_units, driver, shock)
        low_wc, low_capex = _flex_cash_assumptions_for_driver(driver, -shock, wc, capex_assump)
        high_wc, high_capex = _flex_cash_assumptions_for_driver(driver, shock, wc, capex_assump)

        low_ebitda, low_fcf = _evaluate(low_units, low_wc, low_capex, debt)
        high_ebitda, high_fcf = _evaluate(high_units, high_wc, high_capex, debt)

        rows.append(
            {
                "driver": driver,
                "shock": shock,
                "low_ebitda": low_ebitda,
                "base_ebitda": base_ebitda,
                "high_ebitda": high_ebitda,
                "ebitda_range": abs(high_ebitda - low_ebitda),
                "low_fcf": low_fcf,
                "base_fcf": base_fcf,
                "high_fcf": high_fcf,
                "fcf_range": abs(high_fcf - low_fcf),
            }
        )
    return pd.DataFrame(rows).sort_values("ebitda_range", ascending=False).reset_index(drop=True)


def full_scenario_sensitivity_report(base_units: Optional[list[BusinessUnitDrivers]] = None) -> dict[str, pd.DataFrame]:
    """Convenience entry point bundling scenario and tornado analysis."""
    base_units = base_units if base_units is not None else default_business_units()
    return {
        "scenarios": run_scenarios(base_units),
        "tornado": tornado_analysis(base_units),
    }


if __name__ == "__main__":
    from . import SYNTHETIC_DATA_DISCLAIMER

    print(SYNTHETIC_DATA_DISCLAIMER)
    report = full_scenario_sensitivity_report()
    print("\n--- Scenarios ---")
    print(report["scenarios"].to_string(index=False))
    print("\n--- Tornado (EBITDA sensitivity, widest first) ---")
    print(report["tornado"][["driver", "low_ebitda", "base_ebitda", "high_ebitda", "ebitda_range"]].to_string(index=False))
