import pytest

from corporate_fpa_toolkit.capital_allocation import (
    CapexProposal,
    default_capex_pipeline,
    discounted_payback_period,
    evaluate_proposal,
    npv,
    payback_period,
    profitability_index,
    rank_and_select_greedy,
    score_pipeline,
    select_optimal_knapsack,
)


def test_pipeline_has_thirteen_to_fifteen_proposals():
    assert 13 <= len(default_capex_pipeline()) <= 15


def test_npv_known_case():
    # $100 today grows to $110 in a year at 10%: NPV of [-100, 110] at 10% == 0.
    assert npv(0.10, [-100.0, 110.0]) == pytest.approx(0.0, abs=1e-6)


def test_payback_period_simple_case():
    # -100 invested, +40/year: recovered mid-way through year 3.
    cashflows = [-100.0, 40.0, 40.0, 40.0]
    assert payback_period(cashflows) == pytest.approx(2.5)


def test_payback_period_zero_when_cashflow_zero_length_one_year():
    cashflows = [-100.0, 100.0]
    assert payback_period(cashflows) == pytest.approx(1.0)


def test_payback_period_infinite_when_never_recovered():
    cashflows = [-100.0, 10.0, 10.0]
    assert payback_period(cashflows) == float("inf")


def test_discounted_payback_is_longer_than_simple_payback():
    proposal = CapexProposal("Test", "IT Systems", 100_000, 30_000, 6, 3)
    flows = proposal.cashflows()
    simple = payback_period(flows)
    discounted = discounted_payback_period(flows, 0.10)
    assert discounted >= simple


def test_profitability_index_above_one_for_positive_npv():
    cashflows = [-100.0, 60.0, 60.0]
    pi = profitability_index(cashflows, 0.05)
    assert pi > 1.0
    assert pi == pytest.approx(1 + npv(0.05, cashflows) / 100.0)


def test_evaluate_proposal_returns_expected_keys():
    proposal = default_capex_pipeline()[0]
    result = evaluate_proposal(proposal)
    expected_keys = {
        "name", "category", "initial_investment", "life_years", "strategic_fit_score",
        "npv", "irr", "payback_years", "discounted_payback_years", "profitability_index",
    }
    assert expected_keys.issubset(result.keys())


def test_score_pipeline_composite_score_between_zero_and_one():
    scored = score_pipeline(default_capex_pipeline())
    assert (scored["composite_score"] >= 0).all()
    assert (scored["composite_score"] <= 1.0001).all()


def test_score_pipeline_sorted_descending():
    scored = score_pipeline(default_capex_pipeline())
    scores = scored["composite_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_greedy_selection_respects_budget():
    scored = score_pipeline(default_capex_pipeline())
    budget = 100_000_000.0
    selection = rank_and_select_greedy(scored, annual_capex_budget=budget)
    funded_total = selection.loc[selection["funded"], "initial_investment"].sum()
    assert funded_total <= budget


def test_greedy_selection_funds_in_rank_order():
    scored = score_pipeline(default_capex_pipeline())
    selection = rank_and_select_greedy(scored, annual_capex_budget=1e12)  # unlimited budget
    assert selection["funded"].all()
    assert list(selection["rank"]) == list(range(1, len(selection) + 1))


def test_knapsack_never_exceeds_budget():
    budget = 120_000_000.0
    result = select_optimal_knapsack(default_capex_pipeline(), annual_capex_budget=budget)
    selected_total = result.loc[result["selected_optimal"], "initial_investment"].sum()
    assert selected_total <= budget + 1  # small rounding tolerance from budget discretization


def test_knapsack_total_npv_at_least_as_good_as_greedy():
    proposals = default_capex_pipeline()
    budget = 120_000_000.0
    scored = score_pipeline(proposals)
    greedy = rank_and_select_greedy(scored, annual_capex_budget=budget)
    optimal = select_optimal_knapsack(proposals, annual_capex_budget=budget)

    greedy_npv = greedy.loc[greedy["funded"], "npv"].sum()
    optimal_npv = optimal.loc[optimal["selected_optimal"], "npv"].sum()
    assert optimal_npv >= greedy_npv - 1e-6
