"""Quarterly rolling forecast vs. static annual budget.

**FP&A concept -- the rolling forecast.** A once-a-year budget locks in
assumptions in month 0 and never updates them; by month 9, if volumes or
costs have drifted, the budget is simply wrong and stops being useful
for decision-making. A *rolling forecast* re-estimates the remainder of
the year at each checkpoint (commonly quarterly) by blending two
things: (1) the trend embedded in actuals-to-date, de-seasonalized back
to an annual run-rate, and (2) the original budget assumption -- with
more weight shifting to the observed trend as more of the year's
actuals become known. This module builds that quarterly reforecast
process and then scores it against the static budget using the two
standard forecast-accuracy metrics:

* **MAPE** (mean absolute percentage error) -- average size of the miss,
  regardless of direction; lower is more accurate.
* **Bias** (mean percentage error, signed) -- whether a forecast
  systematically runs high (positive) or low (negative); a good
  forecast has bias near zero even if MAPE is not.

.. warning:: All figures are synthetic -- see ``corporate_fpa_toolkit.SYNTHETIC_DATA_DISCLAIMER``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .common import MONTH_NAMES, MONTHS_PER_YEAR
from .driver_based_budget import (
    BusinessUnitDrivers,
    build_all_unit_budgets,
    consolidate_monthly,
    default_business_units,
)
from .variance_analysis import generate_actual_business_units

DEFAULT_CUTOFFS: tuple[int, ...] = (3, 6, 9)


def simulate_noisy_actual_monthly(
    actual_units: list[BusinessUnitDrivers],
    seed: int = 7,
    unit_noise_sd: float = 0.05,
    price_noise_sd: float = 0.015,
    cost_noise_sd: float = 0.02,
) -> pd.DataFrame:
    """Simulate month-by-month actuals with idiosyncratic noise layered
    on top of a BU's already-perturbed annual "actual" drivers.

    Units, price, and variable cost each get independent monthly noise
    (headcount cost and fixed opex are left smooth, as those tend to be
    the more predictable line items in a real company). This is what
    makes mid-year actuals-to-date an *estimate* of the full-year
    run-rate rather than a perfect reveal of it -- which is what makes
    a rolling reforecast a genuinely different (and, if it is doing its
    job, more accurate) exercise than just re-reading the original
    budget.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in actual_units:
        unit_noise = rng.normal(0, unit_noise_sd, MONTHS_PER_YEAR)
        price_noise = rng.normal(0, price_noise_sd, MONTHS_PER_YEAR)
        cost_noise = rng.normal(0, cost_noise_sd, MONTHS_PER_YEAR)
        units_m = unit.annual_units * unit.volume_seasonality * (1 + unit_noise)
        price_m = (
            np.full(MONTHS_PER_YEAR, unit.price_per_unit) * (1 + price_noise)
            if not unit.is_cost_center
            else np.zeros(MONTHS_PER_YEAR)
        )
        var_cost_m = unit.variable_cost_per_unit * (1 + cost_noise)
        revenue_m = units_m * price_m if not unit.is_cost_center else np.zeros(MONTHS_PER_YEAR)
        cogs_m = units_m * var_cost_m
        headcount_cost_m = unit.annual_headcount_cost * unit.opex_seasonality
        fixed_opex_m = unit.other_fixed_opex_annual * unit.opex_seasonality
        sga_m = headcount_cost_m + fixed_opex_m
        gross_profit_m = revenue_m - cogs_m
        ebitda_m = gross_profit_m - sga_m
        for m in range(MONTHS_PER_YEAR):
            rows.append(
                {
                    "business_unit": unit.name,
                    "month": m + 1,
                    "month_name": MONTH_NAMES[m],
                    "units": units_m[m],
                    "price_per_unit": price_m[m] if not unit.is_cost_center else np.nan,
                    "variable_cost_per_unit": var_cost_m[m],
                    "revenue": revenue_m[m],
                    "cogs": cogs_m[m],
                    "gross_profit": gross_profit_m[m],
                    "headcount_cost": headcount_cost_m[m],
                    "fixed_opex": fixed_opex_m[m],
                    "sga": sga_m[m],
                    "ebitda": ebitda_m[m],
                }
            )
    return pd.DataFrame(rows)


def reforecast_business_units(
    budget_units: list[BusinessUnitDrivers],
    actual_monthly: pd.DataFrame,
    cutoff_month: int,
) -> list[BusinessUnitDrivers]:
    """Re-forecast full-year drivers at a quarterly cutoff.

    Units and price/variable-cost run-rates are de-seasonalized from
    actuals-to-date and blended with the original budget assumption,
    with the blend weight on the observed trend growing linearly with
    how much of the year has actually been observed
    (``alpha = cutoff_month / 12``). Headcount, comp-per-head, and
    fixed opex are read directly off actuals-to-date (these are
    assumed to be smooth, so no noise blending is needed for them).
    """
    if not 1 <= cutoff_month <= 12:
        raise ValueError("cutoff_month must be between 1 and 12")
    alpha = cutoff_month / MONTHS_PER_YEAR
    ytd = actual_monthly[actual_monthly["month"] <= cutoff_month]

    reforecast_units = []
    for unit in budget_units:
        bu_ytd = ytd[ytd["business_unit"] == unit.name]
        seasonality_ytd = unit.volume_seasonality[:cutoff_month].sum()
        run_rate_units = bu_ytd["units"].sum() / seasonality_ytd

        if unit.is_cost_center:
            run_rate_price = None
        else:
            run_rate_price = np.average(bu_ytd["price_per_unit"], weights=bu_ytd["units"])
        run_rate_var_cost = np.average(bu_ytd["variable_cost_per_unit"], weights=bu_ytd["units"])

        blended_units = alpha * run_rate_units + (1 - alpha) * unit.annual_units
        blended_price = (
            None if unit.is_cost_center else alpha * run_rate_price + (1 - alpha) * unit.price_per_unit
        )
        blended_var_cost = alpha * run_rate_var_cost + (1 - alpha) * unit.variable_cost_per_unit

        actual_headcount_cost = bu_ytd["headcount_cost"].sum() / unit.opex_seasonality[:cutoff_month].sum()
        actual_fixed_opex = bu_ytd["fixed_opex"].sum() / unit.opex_seasonality[:cutoff_month].sum()
        actual_headcount = actual_headcount_cost / unit.cost_per_head

        reforecast_units.append(
            unit.with_overrides(
                annual_units=blended_units,
                price_per_unit=blended_price,
                variable_cost_per_unit=blended_var_cost,
                headcount=actual_headcount,
                cost_per_head=unit.cost_per_head,
                other_fixed_opex_annual=actual_fixed_opex,
            )
        )
    return reforecast_units


def build_latest_estimate_monthly(
    budget_units: list[BusinessUnitDrivers],
    actual_monthly: pd.DataFrame,
    cutoff_month: int,
) -> pd.DataFrame:
    """Consolidated monthly "latest estimate": actuals through
    ``cutoff_month``, reforecast values for the remaining months.
    """
    reforecast_units = reforecast_business_units(budget_units, actual_monthly, cutoff_month)
    reforecast_full_year = consolidate_monthly(build_all_unit_budgets(reforecast_units))
    actual_consolidated = (
        actual_monthly.groupby(["month", "month_name"], sort=True)[
            ["units", "revenue", "cogs", "gross_profit", "headcount_cost", "fixed_opex", "sga", "ebitda"]
        ]
        .sum()
        .reset_index()
    )
    le = pd.concat(
        [
            actual_consolidated[actual_consolidated["month"] <= cutoff_month].assign(source="actual"),
            reforecast_full_year[reforecast_full_year["month"] > cutoff_month].assign(source="forecast"),
        ],
        ignore_index=True,
    ).sort_values("month").reset_index(drop=True)
    return le


def mape(forecast: Sequence[float], actual: Sequence[float]) -> float:
    """Mean absolute percentage error, in percent (averaged period-by-period).

    Best suited to lines that stay comfortably away from zero (e.g.
    revenue). For a line like EBITDA that can sit near zero in a given
    month, a single small-denominator month can dominate the average;
    see :func:`aggregate_mape` for a more stable alternative on such
    lines.
    """
    f, a = np.asarray(forecast, dtype=float), np.asarray(actual, dtype=float)
    return float(np.mean(np.abs((f - a) / a)) * 100)


def bias(forecast: Sequence[float], actual: Sequence[float]) -> float:
    """Mean percentage error (signed), in percent. Positive = forecast runs high."""
    f, a = np.asarray(forecast, dtype=float), np.asarray(actual, dtype=float)
    return float(np.mean((f - a) / a) * 100)


def aggregate_mape(forecast: Sequence[float], actual: Sequence[float]) -> float:
    """Percentage error of the *summed* forecast vs. summed actual over the
    period, in percent.

    Unlike :func:`mape`, this sums forecast and actual across all
    periods first and takes one percentage error of the totals, so it
    is not distorted by any single period's value sitting near zero.
    This is the more appropriate accuracy measure for a line like
    EBITDA that can be small (or cross zero) in an individual month --
    what a forecast consumer actually cares about is usually "how far
    off was the projected full-period total", not the average of
    noisy monthly ratios.
    """
    f_total, a_total = float(np.sum(forecast)), float(np.sum(actual))
    return float(abs(f_total - a_total) / abs(a_total) * 100)


def aggregate_bias(forecast: Sequence[float], actual: Sequence[float]) -> float:
    """Signed percentage error of the summed forecast vs. summed actual.
    See :func:`aggregate_mape` for why summed totals are used instead
    of a period-by-period average.
    """
    f_total, a_total = float(np.sum(forecast)), float(np.sum(actual))
    return float((f_total - a_total) / abs(a_total) * 100)


def forecast_accuracy_by_vintage(
    budget_units: list[BusinessUnitDrivers],
    actual_monthly: pd.DataFrame,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> pd.DataFrame:
    """Score each forecast vintage's accuracy over its own remaining-year
    horizon against the eventual actuals, for both the static original
    budget and each quarterly reforecast -- an apples-to-apples
    comparison of "what would you have predicted for months N..12 at
    this point in the year" between the two approaches.
    """
    budget_monthly = consolidate_monthly(build_all_unit_budgets(budget_units))
    actual_consolidated = (
        actual_monthly.groupby(["month", "month_name"], sort=True)[["revenue", "ebitda"]].sum().reset_index()
    )

    rows = []
    for cutoff in cutoffs:
        horizon = actual_consolidated["month"] > cutoff
        actual_horizon = actual_consolidated[horizon]

        reforecast_units = reforecast_business_units(budget_units, actual_monthly, cutoff)
        reforecast_monthly = consolidate_monthly(build_all_unit_budgets(reforecast_units))
        reforecast_horizon = reforecast_monthly[reforecast_monthly["month"] > cutoff]

        budget_horizon = budget_monthly[budget_monthly["month"] > cutoff]

        # Revenue stays comfortably positive every month, so a
        # period-by-period MAPE is stable; EBITDA can sit near zero in a
        # given month, so its accuracy is measured on the horizon total
        # (see aggregate_mape's docstring).
        metric_fns = {
            "revenue": (mape, bias),
            "ebitda": (aggregate_mape, aggregate_bias),
        }
        for metric, (mape_fn, bias_fn) in metric_fns.items():
            rows.append(
                {
                    "cutoff_month": cutoff,
                    "vintage": "Static Budget",
                    "metric": metric,
                    "mape_pct": mape_fn(budget_horizon[metric], actual_horizon[metric]),
                    "bias_pct": bias_fn(budget_horizon[metric], actual_horizon[metric]),
                }
            )
            rows.append(
                {
                    "cutoff_month": cutoff,
                    "vintage": f"Q{(cutoff // 3)} Reforecast",
                    "metric": metric,
                    "mape_pct": mape_fn(reforecast_horizon[metric], actual_horizon[metric]),
                    "bias_pct": bias_fn(reforecast_horizon[metric], actual_horizon[metric]),
                }
            )
    return pd.DataFrame(rows)


def full_rolling_forecast_report(
    budget_units: Optional[list[BusinessUnitDrivers]] = None,
    variance_seed: int = 42,
    monthly_noise_seed: int = 7,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, object]:
    """Convenience entry point bundling the full rolling-forecast workflow."""
    budget_units = budget_units if budget_units is not None else default_business_units()
    actual_units = generate_actual_business_units(budget_units, seed=variance_seed)
    actual_monthly = simulate_noisy_actual_monthly(actual_units, seed=monthly_noise_seed)

    latest_estimates = {
        f"Q{cutoff // 3} Reforecast": build_latest_estimate_monthly(budget_units, actual_monthly, cutoff)
        for cutoff in cutoffs
    }
    return {
        "budget_units": budget_units,
        "actual_units": actual_units,
        "actual_monthly": actual_monthly,
        "latest_estimates": latest_estimates,
        "accuracy": forecast_accuracy_by_vintage(budget_units, actual_monthly, cutoffs),
    }


if __name__ == "__main__":
    from . import SYNTHETIC_DATA_DISCLAIMER

    print(SYNTHETIC_DATA_DISCLAIMER)
    report = full_rolling_forecast_report()
    print("\n--- Forecast Accuracy: Static Budget vs. Rolling Reforecast ---")
    print(report["accuracy"].to_string(index=False))
