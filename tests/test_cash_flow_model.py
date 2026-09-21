import numpy as np
import pandas as pd
import pytest

from corporate_fpa_toolkit.cash_flow_model import (
    CapexAssumptions,
    DebtAssumptions,
    WorkingCapitalAssumptions,
    build_cash_flow_model,
    build_thirteen_week_liquidity_view,
    full_cash_flow_report,
)
from corporate_fpa_toolkit.driver_based_budget import build_budget


@pytest.fixture
def consolidated_monthly(budget_units):
    return build_budget(budget_units)["consolidated_monthly"]


def test_balance_sheet_ties_out_every_month(consolidated_monthly):
    model = build_cash_flow_model(consolidated_monthly)
    bs = model["monthly_balance_sheet"]
    diff = bs["total_assets"] - bs["total_liabilities_and_equity"]
    assert (diff.abs() < 1.0).all()  # ties within $1 (floating point noise)


def test_opening_balance_sheet_ties_out(consolidated_monthly):
    model = build_cash_flow_model(consolidated_monthly)
    opening = model["opening_balance_sheet"]
    assert opening["total_assets"] == pytest.approx(opening["total_liabilities"] + opening["equity"])


def test_free_cash_flow_equals_cfo_plus_cfi(consolidated_monthly):
    model = build_cash_flow_model(consolidated_monthly)
    cf = model["monthly_cash_flow"]
    assert np.allclose(cf["free_cash_flow"], cf["cfo"] + cf["cfi"])


def test_net_change_in_cash_rolls_forward(consolidated_monthly):
    model = build_cash_flow_model(consolidated_monthly)
    cf = model["monthly_cash_flow"]
    for i in range(1, len(cf)):
        assert cf.loc[i, "beginning_cash"] == pytest.approx(cf.loc[i - 1, "ending_cash"])
    assert cf.loc[0, "ending_cash"] == pytest.approx(cf.loc[0, "beginning_cash"] + cf.loc[0, "net_change_in_cash"])


def test_ending_cash_never_below_zero_within_revolver_capacity(consolidated_monthly):
    # With a large enough revolver, cash should never go negative (the
    # model draws the revolver, up to its limit, to defend minimum cash).
    debt = DebtAssumptions(revolver_limit=2_000_000_000.0)
    model = build_cash_flow_model(consolidated_monthly, debt=debt)
    assert (model["monthly_cash_flow"]["ending_cash"] >= 0).all()


def test_higher_dso_reduces_operating_cash_flow(consolidated_monthly):
    low_dso = build_cash_flow_model(consolidated_monthly, wc=WorkingCapitalAssumptions(dso_days=20, dpo_days=48, dio_days=55))
    high_dso = build_cash_flow_model(consolidated_monthly, wc=WorkingCapitalAssumptions(dso_days=80, dpo_days=48, dio_days=55))
    assert high_dso["monthly_cash_flow"]["cfo"].sum() < low_dso["monthly_cash_flow"]["cfo"].sum()


def test_depreciation_increases_with_capex_rate(consolidated_monthly):
    low_capex = build_cash_flow_model(
        consolidated_monthly, capex_assump=CapexAssumptions(capex_pct_of_revenue=0.01)
    )
    high_capex = build_cash_flow_model(
        consolidated_monthly, capex_assump=CapexAssumptions(capex_pct_of_revenue=0.06)
    )
    assert high_capex["monthly_cash_flow"]["da_total"].iloc[-1] > low_capex["monthly_cash_flow"]["da_total"].iloc[-1]


def test_thirteen_week_view_sums_to_q1_monthly_totals(consolidated_monthly):
    model = build_cash_flow_model(consolidated_monthly)
    weekly = build_thirteen_week_liquidity_view(model["monthly_cash_flow"])
    assert len(weekly) == 13
    q1_monthly_cfo = model["monthly_cash_flow"].loc[model["monthly_cash_flow"]["month"] <= 3, "cfo"].sum()
    assert weekly["operating_cash_flow"].sum() == pytest.approx(q1_monthly_cfo)


def test_thirteen_week_view_cash_rolls_forward(consolidated_monthly):
    model = build_cash_flow_model(consolidated_monthly)
    weekly = build_thirteen_week_liquidity_view(model["monthly_cash_flow"])
    for i in range(1, len(weekly)):
        assert weekly.loc[i, "beginning_cash"] == pytest.approx(weekly.loc[i - 1, "ending_cash"])


def test_full_cash_flow_report_bundles_liquidity_view(budget_units):
    report = full_cash_flow_report(build_budget(budget_units)["consolidated_monthly"])
    assert "thirteen_week_liquidity" in report
    assert isinstance(report["thirteen_week_liquidity"], pd.DataFrame)
