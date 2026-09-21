import pytest

from corporate_fpa_toolkit.variance_analysis import (
    REVENUE_BU_NAMES,
    cogs_spend_bridge_by_unit,
    consolidated_revenue_variance,
    ebitda_variance_walk,
    generate_actual_business_units,
    opex_spend_bridge_by_unit,
    revenue_variance_by_unit,
)


@pytest.fixture
def actual_units(budget_units):
    return generate_actual_business_units(budget_units, seed=42)


def test_actual_units_differ_from_budget(budget_units, actual_units):
    b_lookup = {u.name: u for u in budget_units}
    for unit in actual_units:
        assert unit.annual_units != pytest.approx(b_lookup[unit.name].annual_units)


def test_actual_units_preserve_cost_center_flag(budget_units, actual_units):
    b_lookup = {u.name: u for u in budget_units}
    for unit in actual_units:
        assert unit.is_cost_center == b_lookup[unit.name].is_cost_center
        if unit.is_cost_center:
            assert unit.price_per_unit is None


def test_bu_level_volume_plus_price_equals_total(budget_units, actual_units):
    df = revenue_variance_by_unit(budget_units, actual_units)
    for _, row in df.iterrows():
        assert row["volume_variance"] + row["price_variance"] == pytest.approx(row["total_variance"])
        assert row["total_variance"] == pytest.approx(row["actual_revenue"] - row["budget_revenue"])


def test_consolidated_volume_mix_price_reconciles(budget_units, actual_units):
    result = consolidated_revenue_variance(budget_units, actual_units)
    total = result["volume_variance"] + result["mix_variance"] + result["price_variance"]
    assert total == pytest.approx(result["total_variance"], rel=1e-6)


def test_consolidated_price_variance_equals_sum_of_bu_price_variances(budget_units, actual_units):
    bu_df = revenue_variance_by_unit(budget_units, actual_units)
    consolidated = consolidated_revenue_variance(budget_units, actual_units)
    assert consolidated["price_variance"] == pytest.approx(bu_df["price_variance"].sum())


def test_mix_variance_is_zero_when_all_bus_grow_uniformly(budget_units):
    # If every revenue BU's units scale by the exact same factor, the
    # portfolio mix hasn't shifted -- mix variance should be ~0.
    uniform_actual = [u.with_overrides(annual_units=u.annual_units * 1.1) for u in budget_units]
    result = consolidated_revenue_variance(budget_units, uniform_actual)
    assert result["mix_variance"] == pytest.approx(0.0, abs=1.0)


def test_cogs_spend_bridge_reconciles(budget_units, actual_units):
    df = cogs_spend_bridge_by_unit(budget_units, actual_units)
    for _, row in df.iterrows():
        assert row["volume_variance"] + row["rate_variance"] == pytest.approx(row["total_variance"])
        assert row["total_variance"] == pytest.approx(row["actual_cogs"] - row["budget_cogs"])


def test_opex_spend_bridge_reconciles(budget_units, actual_units):
    df = opex_spend_bridge_by_unit(budget_units, actual_units)
    for _, row in df.iterrows():
        components = row["headcount_volume_variance"] + row["rate_variance"] + row["fixed_opex_spend_variance"]
        assert components == pytest.approx(row["total_variance"])
        assert row["total_variance"] == pytest.approx(row["actual_sga"] - row["budget_sga"])


def test_ebitda_variance_walk_reconciles(budget_units, actual_units):
    walk = ebitda_variance_walk(budget_units, actual_units)
    budget_ebitda = walk.loc[walk["step"] == "Budget EBITDA", "value"].iloc[0]
    actual_ebitda = walk.loc[walk["step"] == "Actual EBITDA", "value"].iloc[0]
    deltas = walk.loc[walk["step"].isin(["Volume Effect", "Mix Effect", "Price Effect", "Cost Effect"]), "delta"]
    assert budget_ebitda + deltas.sum() == pytest.approx(actual_ebitda, rel=1e-6)


def test_ebitda_walk_matches_direct_ebitda_totals(budget_units, actual_units):
    walk = ebitda_variance_walk(budget_units, actual_units)
    budget_ebitda = walk.loc[walk["step"] == "Budget EBITDA", "value"].iloc[0]
    actual_ebitda = walk.loc[walk["step"] == "Actual EBITDA", "value"].iloc[0]
    assert budget_ebitda == pytest.approx(sum(u.annual_ebitda for u in budget_units))
    assert actual_ebitda == pytest.approx(sum(u.annual_ebitda for u in actual_units))


def test_revenue_variance_only_covers_revenue_bus(budget_units, actual_units):
    df = revenue_variance_by_unit(budget_units, actual_units)
    assert set(df["business_unit"]) == set(REVENUE_BU_NAMES)
