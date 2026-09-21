import numpy as np
import pytest

from corporate_fpa_toolkit.scenario_sensitivity import (
    BASE_SCENARIO,
    DOWNSIDE_SCENARIO,
    UPSIDE_SCENARIO,
    ScenarioAssumptions,
    apply_scenario,
    run_scenarios,
    tornado_analysis,
)
from corporate_fpa_toolkit.driver_based_budget import annual_summary, build_all_unit_budgets


def test_base_scenario_matches_original_budget(budget_units):
    flexed = apply_scenario(budget_units, BASE_SCENARIO)
    base_ebitda = sum(u.annual_ebitda for u in budget_units)
    flexed_ebitda = sum(u.annual_ebitda for u in flexed)
    assert flexed_ebitda == pytest.approx(base_ebitda)


def test_upside_scenario_beats_downside_on_ebitda(budget_units):
    df = run_scenarios(budget_units, scenarios=(BASE_SCENARIO, UPSIDE_SCENARIO, DOWNSIDE_SCENARIO))
    upside_ebitda = df.loc[df["scenario"] == "Upside", "ebitda"].iloc[0]
    base_ebitda = df.loc[df["scenario"] == "Base", "ebitda"].iloc[0]
    downside_ebitda = df.loc[df["scenario"] == "Downside", "ebitda"].iloc[0]
    assert upside_ebitda > base_ebitda > downside_ebitda


def test_scenario_distribution_units_scale_with_volume(budget_units):
    scenario = ScenarioAssumptions(name="Volume+10", volume_growth=0.10, price_growth=0.0, cost_inflation=0.0)
    flexed = apply_scenario(budget_units, scenario)
    b_lookup = {u.name: u for u in budget_units}
    distribution = next(u for u in flexed if u.is_cost_center)
    assert distribution.annual_units == pytest.approx(b_lookup[distribution.name].annual_units * 1.10)


def test_scenario_cost_center_price_stays_none(budget_units):
    flexed = apply_scenario(budget_units, UPSIDE_SCENARIO)
    distribution = next(u for u in flexed if u.is_cost_center)
    assert distribution.price_per_unit is None


def test_run_scenarios_revenue_matches_direct_calc(budget_units):
    df = run_scenarios(budget_units, scenarios=(BASE_SCENARIO,))
    flexed = apply_scenario(budget_units, BASE_SCENARIO)
    annual = annual_summary(build_all_unit_budgets(flexed))
    expected_revenue = annual.loc[annual["business_unit"] == "Consolidated", "revenue"].iloc[0]
    assert df.loc[0, "revenue"] == pytest.approx(expected_revenue)


def test_tornado_sorted_descending_by_ebitda_range(budget_units):
    tornado = tornado_analysis(budget_units, shocks={"Unit Volume": 0.05, "Price": 0.03, "DSO (days)": 10.0})
    ranges = tornado["ebitda_range"].tolist()
    assert ranges == sorted(ranges, reverse=True)


def test_tornado_working_capital_drivers_do_not_move_ebitda(budget_units):
    tornado = tornado_analysis(budget_units, shocks={"DSO (days)": 10.0, "DPO (days)": 10.0, "DIO (days)": 10.0})
    assert np.allclose(tornado["ebitda_range"], 0.0, atol=1e-6)
    assert (tornado["fcf_range"] > 0).all()


def test_tornado_price_driver_moves_ebitda_in_expected_direction(budget_units):
    tornado = tornado_analysis(budget_units, shocks={"Price": 0.03})
    row = tornado.iloc[0]
    assert row["high_ebitda"] > row["base_ebitda"] > row["low_ebitda"]
