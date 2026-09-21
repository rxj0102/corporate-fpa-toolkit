"""Capital allocation: scoring and ranking a capex/investment pipeline.

**FP&A concept -- capital prioritization under a constrained budget.**
Every company has more good capital projects than it has capital.
FP&A's role in the annual capital planning cycle is to evaluate a
pipeline of proposals on a common basis -- **NPV** (value created in
today's dollars), **payback period** (how fast the cash comes back,
a liquidity/risk lens NPV doesn't capture), and **strategic fit** (a
qualitative score for projects, like cybersecurity or compliance
infrastructure, whose value isn't fully captured by direct cash
inflows) -- and then rank them against a fixed annual capex budget.

Two ranking approaches are implemented here, deliberately, because
real capital committees usually use the first even though it is not
mathematically optimal:

* **Greedy priority ranking** -- rank every project by a single
  composite score and fund top-down until the budget runs out. This is
  transparent and easy to defend to a room of stakeholders, but can
  leave value on the table (e.g. skipping one large, slightly-lower-
  scored project in favor of several small high-scored ones might beat
  it on total NPV within the same budget).
* **0/1 knapsack optimization** -- choose the subset of projects that
  maximizes total NPV subject to the budget constraint, solved exactly
  by dynamic programming. This is the "what's the most value we could
  possibly extract from this budget" benchmark against which the
  greedy ranking's opportunity cost can be measured.

.. warning:: All figures are synthetic -- see ``corporate_fpa_toolkit.SYNTHETIC_DATA_DISCLAIMER``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .common import irr, npv

DEFAULT_DISCOUNT_RATE = 0.09
DEFAULT_ANNUAL_CAPEX_BUDGET = 150_000_000.0
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {"npv": 0.4, "payback": 0.3, "strategic_fit": 0.3}


@dataclass(frozen=True)
class CapexProposal:
    """One capital investment proposal in the annual planning pipeline."""

    name: str
    category: str
    initial_investment: float
    annual_cash_inflow: float
    life_years: int
    strategic_fit_score: int  # 1 (low) .. 5 (high)
    annual_growth: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.strategic_fit_score <= 5:
            raise ValueError(f"{self.name}: strategic_fit_score must be 1..5")
        if self.initial_investment <= 0 or self.life_years <= 0:
            raise ValueError(f"{self.name}: initial_investment and life_years must be positive")

    def cashflows(self) -> list[float]:
        """Cash flow stream: ``[-initial_investment, inflow_1, ..., inflow_n]``."""
        flows = [-self.initial_investment]
        inflow = self.annual_cash_inflow
        for _ in range(self.life_years):
            flows.append(inflow)
            inflow *= 1 + self.annual_growth
        return flows


def default_capex_pipeline() -> list[CapexProposal]:
    """13 synthetic capex proposals spanning IT systems, supply chain
    automation, and store/facility upgrades -- the kinds of projects a
    multi-segment retail/distribution company's annual capital
    planning cycle would actually see competing for the same budget.
    """
    return [
        CapexProposal("eCommerce Replatform", "IT Systems", 45_000_000, 12_000_000, 5, 5),
        CapexProposal("B2B Self-Service Ordering Portal", "IT Systems", 18_000_000, 5_500_000, 5, 4),
        CapexProposal("ERP Finance Modernization", "IT Systems", 32_000_000, 7_000_000, 6, 3),
        CapexProposal("Warehouse Robotics Sortation", "Supply Chain Automation", 60_000_000, 16_000_000, 6, 5),
        CapexProposal("Distribution Network Optimization", "Supply Chain Automation", 40_000_000, 9_000_000, 6, 4),
        CapexProposal("WMS / Inventory Automation Upgrade", "Supply Chain Automation", 15_000_000, 4_500_000, 5, 4),
        CapexProposal("Retail Store Refresh Program", "Store & Facility Upgrades", 50_000_000, 11_000_000, 5, 3),
        CapexProposal("Store Energy & HVAC Retrofit", "Store & Facility Upgrades", 20_000_000, 4_200_000, 7, 2),
        CapexProposal("Store POS Hardware Refresh", "Store & Facility Upgrades", 12_000_000, 3_500_000, 4, 3),
        CapexProposal("Last-Mile Fleet Electrification", "Supply Chain Automation", 28_000_000, 5_800_000, 6, 3),
        CapexProposal("Cybersecurity & Data Infrastructure", "IT Systems", 22_000_000, 4_000_000, 5, 5),
        CapexProposal("Store Footprint Rationalization", "Store & Facility Upgrades", 9_000_000, 3_800_000, 4, 3),
        CapexProposal("B2B Sales Analytics & CRM Platform", "IT Systems", 14_000_000, 4_600_000, 5, 4),
    ]


def payback_period(cashflows: Sequence[float]) -> float:
    """Simple (undiscounted) payback period, in years, linearly interpolated
    within the year cumulative cash flow first turns non-negative.
    Returns ``float('inf')`` if the investment never pays back within
    the cash flow stream's horizon.
    """
    cum = cashflows[0]
    if cum >= 0:
        return 0.0
    for year, flow in enumerate(cashflows[1:], start=1):
        prev_cum = cum
        cum += flow
        if cum >= 0:
            fraction = 0.0 if flow == 0 else -prev_cum / flow
            return (year - 1) + fraction
    return float("inf")


def discounted_payback_period(cashflows: Sequence[float], rate: float) -> float:
    """Payback period computed on discounted cash flows (present-valued
    at ``rate`` before cumulating), i.e. the year the *discounted*
    cumulative cash flow turns non-negative.
    """
    discounted = [cf / (1 + rate) ** t for t, cf in enumerate(cashflows)]
    return payback_period(discounted)


def profitability_index(cashflows: Sequence[float], rate: float) -> float:
    """Profitability index: ``1 + NPV / |initial investment|``.

    A value above 1.0 means the project creates value (positive NPV);
    below 1.0 means it destroys it.
    """
    investment = abs(cashflows[0])
    return 1 + npv(rate, cashflows) / investment


def evaluate_proposal(proposal: CapexProposal, discount_rate: float = DEFAULT_DISCOUNT_RATE) -> dict[str, float]:
    """Compute NPV, IRR, simple/discounted payback, and profitability index
    for one proposal.
    """
    flows = proposal.cashflows()
    return {
        "name": proposal.name,
        "category": proposal.category,
        "initial_investment": proposal.initial_investment,
        "life_years": proposal.life_years,
        "strategic_fit_score": proposal.strategic_fit_score,
        "npv": npv(discount_rate, flows),
        "irr": irr(flows),
        "payback_years": payback_period(flows),
        "discounted_payback_years": discounted_payback_period(flows, discount_rate),
        "profitability_index": profitability_index(flows, discount_rate),
    }


def _min_max_normalize(values: pd.Series, higher_is_better: bool) -> pd.Series:
    finite = values.replace([np.inf, -np.inf], np.nan)
    lo, hi = finite.min(), finite.max()
    if hi == lo:
        return pd.Series(1.0, index=values.index)
    capped = values.clip(upper=finite.max() * 1.5 if np.isfinite(finite.max()) else hi)
    norm = (capped - lo) / (hi - lo)
    return norm if higher_is_better else 1 - norm


def score_pipeline(
    proposals: Sequence[CapexProposal],
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    weights: dict[str, float] = DEFAULT_SCORE_WEIGHTS,
) -> pd.DataFrame:
    """Evaluate every proposal and compute a weighted composite score.

    Each of NPV (higher better), payback period (lower better), and
    strategic fit (higher better) is min-max normalized to a 0-1 scale
    *across this pipeline* before being weighted -- so the composite
    score reflects each project's relative standing among its peers,
    not an arbitrary absolute scale.
    """
    df = pd.DataFrame([evaluate_proposal(p, discount_rate) for p in proposals])
    df["norm_npv"] = _min_max_normalize(df["npv"], higher_is_better=True)
    df["norm_payback"] = _min_max_normalize(df["payback_years"], higher_is_better=False)
    df["norm_strategic_fit"] = _min_max_normalize(df["strategic_fit_score"], higher_is_better=True)
    df["composite_score"] = (
        weights["npv"] * df["norm_npv"]
        + weights["payback"] * df["norm_payback"]
        + weights["strategic_fit"] * df["norm_strategic_fit"]
    )
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def rank_and_select_greedy(scored: pd.DataFrame, annual_capex_budget: float = DEFAULT_ANNUAL_CAPEX_BUDGET) -> pd.DataFrame:
    """Rank by composite score (already the sort order of ``scored``) and
    fund top-down until the budget is exhausted.

    Adds ``rank``, ``cumulative_investment``, and ``funded`` columns.
    A lower-ranked project that happens to fit in leftover budget after
    a larger one is skipped is *not* back-filled -- this mirrors how a
    capital committee typically works down a priority list rather than
    bin-packing it.
    """
    out = scored.copy().reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    cumulative = out["initial_investment"].cumsum()
    out["cumulative_investment"] = cumulative
    out["funded"] = cumulative <= annual_capex_budget
    return out


def select_optimal_knapsack(
    proposals: Sequence[CapexProposal],
    annual_capex_budget: float = DEFAULT_ANNUAL_CAPEX_BUDGET,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    budget_increment: float = 100_000.0,
) -> pd.DataFrame:
    """Solve the 0/1 knapsack problem that maximizes total NPV subject to
    the capex budget, via dynamic programming.

    The budget axis is discretized to ``budget_increment`` (default
    $100k) so the DP table stays a manageable size; investments are
    rounded to the same grid. This is the exact-optimal benchmark that
    :func:`rank_and_select_greedy`'s transparent-but-suboptimal ranking
    is compared against.
    """
    evaluated = [evaluate_proposal(p, discount_rate) for p in proposals]
    weights = [max(1, round(e["initial_investment"] / budget_increment)) for e in evaluated]
    values = [max(0.0, e["npv"]) for e in evaluated]
    capacity = int(round(annual_capex_budget / budget_increment))

    n = len(proposals)
    dp = np.zeros((n + 1, capacity + 1))
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            dp[i, c] = dp[i - 1, c]
            if w <= c:
                dp[i, c] = max(dp[i, c], dp[i - 1, c - w] + v)

    selected = [False] * n
    c = capacity
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]:
            selected[i - 1] = True
            c -= weights[i - 1]

    df = pd.DataFrame(evaluated)
    df["selected_optimal"] = selected
    return df.sort_values("npv", ascending=False).reset_index(drop=True)


def full_capital_allocation_report(
    proposals: Optional[Sequence[CapexProposal]] = None,
    annual_capex_budget: float = DEFAULT_ANNUAL_CAPEX_BUDGET,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> dict[str, pd.DataFrame]:
    """Convenience entry point bundling scoring, greedy selection, and the
    optimal-knapsack benchmark.
    """
    proposals = proposals if proposals is not None else default_capex_pipeline()
    scored = score_pipeline(proposals, discount_rate=discount_rate)
    greedy = rank_and_select_greedy(scored, annual_capex_budget=annual_capex_budget)
    optimal = select_optimal_knapsack(proposals, annual_capex_budget=annual_capex_budget, discount_rate=discount_rate)
    return {"scored_pipeline": scored, "greedy_selection": greedy, "optimal_knapsack": optimal}


if __name__ == "__main__":
    from . import SYNTHETIC_DATA_DISCLAIMER

    print(SYNTHETIC_DATA_DISCLAIMER)
    report = full_capital_allocation_report()
    cols = ["rank", "name", "category", "initial_investment", "npv", "payback_years", "composite_score", "funded"]
    print("\n--- Greedy Priority Ranking (budget-constrained) ---")
    print(report["greedy_selection"][cols].to_string(index=False))
    funded_investment = report["greedy_selection"].loc[report["greedy_selection"]["funded"], "initial_investment"].sum()
    optimal_investment = report["optimal_knapsack"].loc[
        report["optimal_knapsack"]["selected_optimal"], "initial_investment"
    ].sum()
    print(f"\nGreedy funded investment: {funded_investment:,.0f}")
    print(f"Knapsack-optimal funded investment: {optimal_investment:,.0f}")
