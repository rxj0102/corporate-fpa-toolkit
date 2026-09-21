import numpy as np
import pytest

from corporate_fpa_toolkit.rolling_forecast import (
    aggregate_bias,
    aggregate_mape,
    bias,
    build_latest_estimate_monthly,
    forecast_accuracy_by_vintage,
    mape,
    reforecast_business_units,
    simulate_noisy_actual_monthly,
)
from corporate_fpa_toolkit.variance_analysis import generate_actual_business_units


@pytest.fixture
def actual_units(budget_units):
    return generate_actual_business_units(budget_units, seed=42)


@pytest.fixture
def actual_monthly(actual_units):
    return simulate_noisy_actual_monthly(actual_units, seed=7)


def test_mape_zero_for_perfect_forecast():
    actual = [100.0, 200.0, 300.0]
    assert mape(actual, actual) == pytest.approx(0.0)
    assert bias(actual, actual) == pytest.approx(0.0)


def test_mape_known_value():
    forecast = [110.0]
    actual = [100.0]
    assert mape(forecast, actual) == pytest.approx(10.0)
    assert bias(forecast, actual) == pytest.approx(10.0)


def test_bias_is_signed_mape_is_not():
    forecast = [90.0, 110.0]
    actual = [100.0, 100.0]
    assert bias(forecast, actual) == pytest.approx(0.0)
    assert mape(forecast, actual) == pytest.approx(10.0)


def test_aggregate_mape_matches_manual_calc():
    forecast = [50.0, 60.0]
    actual = [55.0, 55.0]
    expected = abs(110 - 110) / 110 * 100
    assert aggregate_mape(forecast, actual) == pytest.approx(expected)
    assert aggregate_bias(forecast, actual) == pytest.approx((110 - 110) / 110 * 100)


def test_simulated_actual_monthly_has_144_rows(actual_monthly):
    assert len(actual_monthly) == 4 * 12  # 4 business units x 12 months


def test_reforecast_at_month_12_uses_full_year_actuals(budget_units, actual_units, actual_monthly):
    reforecast = reforecast_business_units(budget_units, actual_monthly, cutoff_month=12)
    a_lookup = {u.name: u for u in actual_units}
    for unit in reforecast:
        # At cutoff=12, alpha=1.0, so the reforecast is entirely the
        # de-seasonalized full-year run-rate, which should sit close to
        # the (noisy) full-year actual it was estimated from.
        assert unit.annual_units == pytest.approx(a_lookup[unit.name].annual_units, rel=0.05)
        assert unit.headcount == pytest.approx(a_lookup[unit.name].headcount, rel=0.05)


def test_latest_estimate_first_months_match_actuals(budget_units, actual_monthly):
    cutoff = 3
    le = build_latest_estimate_monthly(budget_units, actual_monthly, cutoff)
    actual_consolidated = (
        actual_monthly.groupby(["month", "month_name"], sort=True)[["revenue"]].sum().reset_index()
    )
    for month in range(1, cutoff + 1):
        le_val = le.loc[le["month"] == month, "revenue"].iloc[0]
        actual_val = actual_consolidated.loc[actual_consolidated["month"] == month, "revenue"].iloc[0]
        assert le_val == pytest.approx(actual_val)


def test_latest_estimate_has_actual_and_forecast_source_tags(budget_units, actual_monthly):
    le = build_latest_estimate_monthly(budget_units, actual_monthly, cutoff_month=6)
    assert set(le["source"]) == {"actual", "forecast"}
    assert (le.loc[le["month"] <= 6, "source"] == "actual").all()
    assert (le.loc[le["month"] > 6, "source"] == "forecast").all()


def test_forecast_accuracy_report_covers_every_cutoff(budget_units, actual_monthly):
    accuracy = forecast_accuracy_by_vintage(budget_units, actual_monthly, cutoffs=(3, 6, 9))
    assert set(accuracy["cutoff_month"]) == {3, 6, 9}
    assert set(accuracy["vintage"]) == {"Static Budget", "Q1 Reforecast", "Q2 Reforecast", "Q3 Reforecast"}
    assert (accuracy["mape_pct"] >= 0).all()


def test_forecast_accuracy_metrics_are_finite(budget_units, actual_monthly):
    accuracy = forecast_accuracy_by_vintage(budget_units, actual_monthly, cutoffs=(3, 6, 9))
    assert np.isfinite(accuracy["mape_pct"]).all()
    assert np.isfinite(accuracy["bias_pct"]).all()
