import numpy as np
import pytest

from corporate_fpa_toolkit.reporting import (
    build_prior_year_units,
    build_reporting_pack,
    flag_top_variances_by_period,
    monthly_pnl_comparison,
    variance_bridge_chart_data,
)
from corporate_fpa_toolkit.driver_based_budget import build_budget
from corporate_fpa_toolkit.variance_analysis import build_actual_results, ebitda_variance_walk, generate_actual_business_units


def test_prior_year_is_smaller_than_budget(budget_units):
    py_units = build_prior_year_units(budget_units)
    py_revenue = sum(u.annual_revenue for u in py_units)
    budget_revenue = sum(u.annual_revenue for u in budget_units)
    assert py_revenue < budget_revenue


def test_monthly_pnl_comparison_has_twelve_rows(budget_units):
    budget = build_budget(budget_units)
    actual_units = generate_actual_business_units(budget_units, seed=1)
    actual = build_actual_results(actual_units)
    py_units = build_prior_year_units(budget_units)
    py = build_budget(py_units)

    comparison = monthly_pnl_comparison(budget["consolidated_monthly"], actual["consolidated_monthly"], py["consolidated_monthly"])
    assert len(comparison) == 12
    assert np.allclose(
        comparison["revenue_var_vs_budget"], comparison["revenue_actual"] - comparison["revenue_budget"]
    )


def test_variance_bridge_chart_data_has_six_steps(budget_units):
    actual_units = generate_actual_business_units(budget_units, seed=1)
    walk = ebitda_variance_walk(budget_units, actual_units)
    chart_data = variance_bridge_chart_data(walk)
    assert len(chart_data) == 6
    assert chart_data[0]["kind"] == "total"
    assert chart_data[-1]["kind"] == "total"


def test_variance_bridge_chart_data_ends_at_actual_ebitda(budget_units):
    actual_units = generate_actual_business_units(budget_units, seed=1)
    walk = ebitda_variance_walk(budget_units, actual_units)
    chart_data = variance_bridge_chart_data(walk)
    actual_ebitda = walk.loc[walk["step"] == "Actual EBITDA", "value"].iloc[0]
    assert chart_data[-1]["end"] == pytest.approx(actual_ebitda)


def test_flag_top_variances_returns_top_n_per_month(budget_units):
    budget = build_budget(budget_units)
    actual_units = generate_actual_business_units(budget_units, seed=1)
    actual = build_actual_results(actual_units)
    flagged = flag_top_variances_by_period(budget["by_unit_monthly"], actual["by_unit_monthly"], top_n=3)
    counts = flagged.groupby("month").size()
    assert (counts == 3).all()


def test_flag_top_variances_direction_matches_sign(budget_units):
    budget = build_budget(budget_units)
    actual_units = generate_actual_business_units(budget_units, seed=1)
    actual = build_actual_results(actual_units)
    flagged = flag_top_variances_by_period(budget["by_unit_monthly"], actual["by_unit_monthly"], top_n=3)
    favorable = flagged[flagged["direction"] == "favorable"]
    unfavorable = flagged[flagged["direction"] == "unfavorable"]
    assert (favorable["variance_dollar"] >= 0).all()
    assert (unfavorable["variance_dollar"] < 0).all()


def test_build_reporting_pack_contains_all_sections(budget_units):
    pack = build_reporting_pack(budget_units, variance_seed=1, top_n_variances=2)
    expected_keys = {
        "budget", "actual", "prior_year", "pnl_comparison", "ebitda_variance_walk",
        "variance_bridge_chart_data", "flagged_variances", "cash_flow_summary", "executive_summary",
    }
    assert expected_keys.issubset(pack.keys())
    assert isinstance(pack["executive_summary"], str)
    assert len(pack["executive_summary"]) > 0


def test_executive_summary_mentions_ebitda(budget_units):
    pack = build_reporting_pack(budget_units, variance_seed=1)
    assert "EBITDA" in pack["executive_summary"]
