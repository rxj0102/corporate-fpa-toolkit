"""Board-style monthly reporting pack.

**FP&A concept -- the monthly reporting pack.** Once actuals close each
month, FP&A assembles a standard package for leadership/the board: the
consolidated P&L against both the current budget and the prior year,
a variance bridge explaining the budget gap in the same
volume/mix/price/cost language used throughout this toolkit, a cash
flow summary, and a short written narrative that calls out the handful
of variances that actually matter rather than making the reader scan
every line. This module assembles that package from the other modules'
outputs and auto-generates the narrative text directly from the
computed numbers -- nothing here is pre-written commentary.

.. warning:: All figures are synthetic -- see ``corporate_fpa_toolkit.SYNTHETIC_DATA_DISCLAIMER``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cash_flow_model import build_cash_flow_model, build_thirteen_week_liquidity_view
from .driver_based_budget import BusinessUnitDrivers, build_budget, default_business_units
from .scenario_sensitivity import ScenarioAssumptions, apply_scenario
from .variance_analysis import (
    build_actual_results,
    consolidated_revenue_variance,
    ebitda_variance_walk,
    generate_actual_business_units,
)

PRIOR_YEAR_SCENARIO = ScenarioAssumptions(name="Prior Year", volume_growth=-0.06, price_growth=-0.025, cost_inflation=-0.03)


def build_prior_year_units(budget_units: list[BusinessUnitDrivers]) -> list[BusinessUnitDrivers]:
    """Synthesize a prior-year driver set by un-growing the budget.

    Reuses :func:`corporate_fpa_toolkit.scenario_sensitivity.apply_scenario`
    with a negative-growth "Prior Year" bundle, i.e. the current budget
    is treated as prior year grown forward by the scenario's
    (positive) inverse.
    """
    return apply_scenario(budget_units, PRIOR_YEAR_SCENARIO)


def monthly_pnl_comparison(
    budget_monthly: pd.DataFrame, actual_monthly: pd.DataFrame, py_monthly: pd.DataFrame
) -> pd.DataFrame:
    """Consolidated monthly Revenue and EBITDA: Budget vs. Actual vs. Prior Year."""
    b = budget_monthly.set_index("month")
    a = actual_monthly.set_index("month")
    py = py_monthly.set_index("month")

    rows = []
    for month in b.index:
        rev_b, rev_a, rev_py = b.loc[month, "revenue"], a.loc[month, "revenue"], py.loc[month, "revenue"]
        eb_b, eb_a, eb_py = b.loc[month, "ebitda"], a.loc[month, "ebitda"], py.loc[month, "ebitda"]
        rows.append(
            {
                "month": month,
                "month_name": b.loc[month, "month_name"],
                "revenue_budget": rev_b,
                "revenue_actual": rev_a,
                "revenue_py": rev_py,
                "revenue_var_vs_budget": rev_a - rev_b,
                "revenue_var_vs_budget_pct": (rev_a - rev_b) / rev_b * 100 if rev_b else np.nan,
                "revenue_var_vs_py_pct": (rev_a - rev_py) / rev_py * 100 if rev_py else np.nan,
                "ebitda_budget": eb_b,
                "ebitda_actual": eb_a,
                "ebitda_py": eb_py,
                "ebitda_var_vs_budget": eb_a - eb_b,
                "ebitda_var_vs_py": eb_a - eb_py,
            }
        )
    return pd.DataFrame(rows)


def variance_bridge_chart_data(walk_df: pd.DataFrame) -> list[dict]:
    """Reshape an EBITDA variance walk into waterfall-chart-ready records:
    each step carries its bar's ``start`` and ``end`` (for a floating
    bar) plus whether it is a ``total`` or an ``increase``/``decrease``
    step.
    """
    records = []
    running = None
    for _, row in walk_df.iterrows():
        if pd.isna(row["delta"]):
            records.append({"label": row["step"], "start": 0.0, "end": row["value"], "kind": "total"})
            running = row["value"]
        else:
            start = running
            end = running + row["delta"]
            records.append(
                {
                    "label": row["step"],
                    "start": float(min(start, end)),
                    "end": float(max(start, end)),
                    "delta": float(row["delta"]),
                    "kind": "increase" if row["delta"] >= 0 else "decrease",
                }
            )
            running = end
    return records


def cash_flow_summary(budget_monthly: pd.DataFrame, actual_monthly: pd.DataFrame) -> dict[str, object]:
    """Budget vs. actual cash flow model outputs, plus a 13-week liquidity
    view built off the actual-year cash flow.
    """
    budget_cf = build_cash_flow_model(budget_monthly)
    actual_cf = build_cash_flow_model(actual_monthly)
    comparison = pd.DataFrame(
        {
            "month": budget_cf["monthly_cash_flow"]["month"],
            "month_name": budget_cf["monthly_cash_flow"]["month_name"],
            "fcf_budget": budget_cf["monthly_cash_flow"]["free_cash_flow"],
            "fcf_actual": actual_cf["monthly_cash_flow"]["free_cash_flow"],
            "ending_cash_budget": budget_cf["monthly_cash_flow"]["ending_cash"],
            "ending_cash_actual": actual_cf["monthly_cash_flow"]["ending_cash"],
        }
    )
    comparison["fcf_variance"] = comparison["fcf_actual"] - comparison["fcf_budget"]
    return {
        "budget_cash_flow": budget_cf["monthly_cash_flow"],
        "actual_cash_flow": actual_cf["monthly_cash_flow"],
        "comparison": comparison,
        "thirteen_week_liquidity_actual": build_thirteen_week_liquidity_view(actual_cf["monthly_cash_flow"]),
    }


def flag_top_variances_by_period(
    by_unit_monthly_budget: pd.DataFrame, by_unit_monthly_actual: pd.DataFrame, top_n: int = 3
) -> pd.DataFrame:
    """For each month, rank every (business unit, line) pair's dollar
    variance and keep the ``top_n`` largest by absolute size, tagged
    favorable/unfavorable by sign. Covers both the Revenue and EBITDA
    lines, since a board pack cares about both the top line and the
    bottom line.
    """
    b = by_unit_monthly_budget.set_index(["business_unit", "month"])
    a = by_unit_monthly_actual.set_index(["business_unit", "month"])

    rows = []
    for (business_unit, month), b_row in b.iterrows():
        a_row = a.loc[(business_unit, month)]
        for line in ("revenue", "ebitda"):
            variance = a_row[line] - b_row[line]
            rows.append(
                {
                    "month": month,
                    "month_name": b_row["month_name"],
                    "business_unit": business_unit,
                    "line": line,
                    "variance_dollar": variance,
                    "direction": "favorable" if variance >= 0 else "unfavorable",
                }
            )
    all_variances = pd.DataFrame(rows)

    flagged = (
        all_variances.assign(abs_variance=all_variances["variance_dollar"].abs())
        .sort_values(["month", "abs_variance"], ascending=[True, False])
        .groupby("month", sort=True)
        .head(top_n)
        .drop(columns="abs_variance")
        .reset_index(drop=True)
    )
    return flagged


def _fmt_dollars(value: float) -> str:
    return f"${value / 1e6:,.1f}M"


def generate_annual_narrative(
    walk_df: pd.DataFrame, revenue_variance: dict[str, object], budget_annual_ebitda: float, actual_annual_ebitda: float
) -> str:
    """Auto-generate a full-year narrative paragraph directly from the
    computed EBITDA variance walk and consolidated revenue variance --
    no numbers here are hand-written.
    """
    delta = actual_annual_ebitda - budget_annual_ebitda
    pct = delta / budget_annual_ebitda * 100 if budget_annual_ebitda else float("nan")
    direction = "ahead of" if delta >= 0 else "behind"

    volume = float(walk_df.loc[walk_df["step"] == "Volume Effect", "delta"].iloc[0])
    mix = float(walk_df.loc[walk_df["step"] == "Mix Effect", "delta"].iloc[0])
    price = float(walk_df.loc[walk_df["step"] == "Price Effect", "delta"].iloc[0])
    cost = float(walk_df.loc[walk_df["step"] == "Cost Effect", "delta"].iloc[0])

    def _effect_phrase(name: str, value: float) -> str:
        tag = "favorable" if value >= 0 else "unfavorable"
        return f"a {_fmt_dollars(abs(value))} {tag} {name} effect"

    return (
        f"Full-year consolidated EBITDA of {_fmt_dollars(actual_annual_ebitda)} came in "
        f"{_fmt_dollars(abs(delta))} ({abs(pct):.1f}%) {direction} the {_fmt_dollars(budget_annual_ebitda)} budget. "
        f"The variance reflects {_effect_phrase('volume', volume)}, {_effect_phrase('mix', mix)}, "
        f"{_effect_phrase('price', price)}, and {_effect_phrase('cost', cost)}. "
        f"Consolidated revenue variance versus budget was {_fmt_dollars(revenue_variance['total_variance'])}."
    )


def generate_monthly_highlights(flagged: pd.DataFrame) -> str:
    """Turn the flagged top-variance table into readable bullet lines,
    grouped by month.
    """
    lines = []
    for month, group in flagged.groupby("month", sort=True):
        month_name = group["month_name"].iloc[0]
        lines.append(f"{month_name}:")
        for _, row in group.iterrows():
            lines.append(
                f"  - {row['business_unit']} {row['line']} was {_fmt_dollars(abs(row['variance_dollar']))} "
                f"{row['direction']} to budget."
            )
    return "\n".join(lines)


def full_executive_summary(
    walk_df: pd.DataFrame,
    revenue_variance: dict[str, object],
    budget_annual_ebitda: float,
    actual_annual_ebitda: float,
    flagged_variances: pd.DataFrame,
) -> str:
    """Combine the annual narrative and monthly highlight bullets into one
    written executive summary.
    """
    annual = generate_annual_narrative(walk_df, revenue_variance, budget_annual_ebitda, actual_annual_ebitda)
    monthly = generate_monthly_highlights(flagged_variances)
    return f"{annual}\n\nMonth-by-month highlights (largest variances vs. budget):\n{monthly}"


def build_reporting_pack(
    budget_units: Optional[list[BusinessUnitDrivers]] = None,
    variance_seed: int = 42,
    top_n_variances: int = 3,
) -> dict[str, object]:
    """Convenience entry point: assemble the full monthly board reporting pack."""
    budget_units = budget_units if budget_units is not None else default_business_units()
    budget = build_budget(budget_units)

    actual_units = generate_actual_business_units(budget_units, seed=variance_seed)
    actual = build_actual_results(actual_units)

    py_units = build_prior_year_units(budget_units)
    py = build_budget(py_units)

    pnl_comparison = monthly_pnl_comparison(
        budget["consolidated_monthly"], actual["consolidated_monthly"], py["consolidated_monthly"]
    )
    walk = ebitda_variance_walk(budget_units, actual_units)
    revenue_variance = consolidated_revenue_variance(budget_units, actual_units)
    budget_annual_ebitda = float(walk.loc[walk["step"] == "Budget EBITDA", "value"].iloc[0])
    actual_annual_ebitda = float(walk.loc[walk["step"] == "Actual EBITDA", "value"].iloc[0])

    flagged = flag_top_variances_by_period(budget["by_unit_monthly"], actual["by_unit_monthly"], top_n=top_n_variances)
    cash_summary = cash_flow_summary(budget["consolidated_monthly"], actual["consolidated_monthly"])

    return {
        "budget": budget,
        "actual": actual,
        "prior_year": py,
        "pnl_comparison": pnl_comparison,
        "ebitda_variance_walk": walk,
        "variance_bridge_chart_data": variance_bridge_chart_data(walk),
        "flagged_variances": flagged,
        "cash_flow_summary": cash_summary,
        "executive_summary": full_executive_summary(
            walk, revenue_variance, budget_annual_ebitda, actual_annual_ebitda, flagged
        ),
    }


if __name__ == "__main__":
    from . import SYNTHETIC_DATA_DISCLAIMER

    print(SYNTHETIC_DATA_DISCLAIMER)
    pack = build_reporting_pack()
    print("\n--- Executive Summary ---")
    print(pack["executive_summary"])
