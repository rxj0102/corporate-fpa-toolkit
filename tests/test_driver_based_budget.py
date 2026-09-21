import numpy as np
import pytest

from corporate_fpa_toolkit.driver_based_budget import (
    ALL_SEGMENTS,
    BusinessUnitDrivers,
    annual_summary,
    build_all_unit_budgets,
    build_budget,
    build_unit_monthly_budget,
    consolidate_monthly,
)


def test_default_business_units_cover_all_segments(budget_units):
    names = {u.name for u in budget_units}
    assert names == set(ALL_SEGMENTS)


def test_cost_center_requires_no_price():
    with pytest.raises(ValueError):
        BusinessUnitDrivers(
            name="Bad",
            is_cost_center=True,
            annual_units=100,
            price_per_unit=5.0,
            variable_cost_per_unit=1.0,
            headcount=1,
            cost_per_head=1.0,
            other_fixed_opex_annual=0.0,
        )


def test_revenue_unit_requires_price():
    with pytest.raises(ValueError):
        BusinessUnitDrivers(
            name="Bad",
            is_cost_center=False,
            annual_units=100,
            price_per_unit=None,
            variable_cost_per_unit=1.0,
            headcount=1,
            cost_per_head=1.0,
            other_fixed_opex_annual=0.0,
        )


def test_seasonality_must_sum_to_one():
    bad_curve = np.full(12, 1.0)  # sums to 12, not 1
    with pytest.raises(ValueError):
        BusinessUnitDrivers(
            name="Bad",
            is_cost_center=False,
            annual_units=100,
            price_per_unit=5.0,
            variable_cost_per_unit=1.0,
            headcount=1,
            cost_per_head=1.0,
            other_fixed_opex_annual=0.0,
            volume_seasonality=bad_curve,
        )


def test_monthly_units_tie_to_annual(budget_units):
    for unit in budget_units:
        monthly = build_unit_monthly_budget(unit)
        assert monthly["units"].sum() == pytest.approx(unit.annual_units)
        assert monthly["revenue"].sum() == pytest.approx(unit.annual_revenue)
        assert monthly["ebitda"].sum() == pytest.approx(unit.annual_ebitda)


def test_monthly_budget_has_twelve_rows_per_unit(budget_units):
    for unit in budget_units:
        monthly = build_unit_monthly_budget(unit)
        assert len(monthly) == 12
        assert list(monthly["month"]) == list(range(1, 13))


def test_cost_center_has_zero_revenue(budget_units):
    distribution = next(u for u in budget_units if u.is_cost_center)
    monthly = build_unit_monthly_budget(distribution)
    assert (monthly["revenue"] == 0).all()
    assert distribution.annual_revenue == 0.0


def test_gross_profit_and_sga_arithmetic(budget_units):
    for unit in budget_units:
        monthly = build_unit_monthly_budget(unit)
        assert np.allclose(monthly["gross_profit"], monthly["revenue"] - monthly["cogs"])
        assert np.allclose(monthly["sga"], monthly["headcount_cost"] + monthly["fixed_opex"])
        assert np.allclose(monthly["ebitda"], monthly["gross_profit"] - monthly["sga"])


def test_consolidated_monthly_equals_sum_of_units(budget_units):
    by_unit = build_all_unit_budgets(budget_units)
    consolidated = consolidate_monthly(by_unit)
    assert len(consolidated) == 12
    for month in range(1, 13):
        expected_revenue = by_unit.loc[by_unit["month"] == month, "revenue"].sum()
        actual_revenue = consolidated.loc[consolidated["month"] == month, "revenue"].iloc[0]
        assert actual_revenue == pytest.approx(expected_revenue)


def test_annual_summary_consolidated_row_matches_bu_sum(budget_units):
    by_unit = build_all_unit_budgets(budget_units)
    summary = annual_summary(by_unit)
    consolidated_row = summary[summary["business_unit"] == "Consolidated"].iloc[0]
    bu_rows = summary[summary["business_unit"] != "Consolidated"]
    assert consolidated_row["revenue"] == pytest.approx(bu_rows["revenue"].sum())
    assert consolidated_row["ebitda"] == pytest.approx(bu_rows["ebitda"].sum())


def test_build_budget_convenience_entry_point(budget_units):
    result = build_budget(budget_units)
    assert set(result.keys()) == {"by_unit_monthly", "consolidated_monthly", "annual_summary"}
    assert len(result["consolidated_monthly"]) == 12


def test_with_overrides_does_not_mutate_original(budget_units):
    unit = budget_units[0]
    original_units = unit.annual_units
    overridden = unit.with_overrides(annual_units=unit.annual_units * 2)
    assert unit.annual_units == original_units
    assert overridden.annual_units == pytest.approx(original_units * 2)
