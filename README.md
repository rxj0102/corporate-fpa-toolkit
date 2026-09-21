# corporate-fpa-toolkit

A from-scratch, fully synthetic Python toolkit that models a **driver-based
budgeting, forecasting, and variance analysis workflow** for a multi-business-unit
retail/distribution company. It was built as a portfolio project to demonstrate
core FP&A (Financial Planning & Analysis) concepts and skills for a corporate
finance rotational program application: driver-based budgeting, variance
bridges, rolling forecasts, indirect-method cash flow modeling, scenario &
sensitivity analysis, capital allocation scoring, and board-style reporting.

> **⚠️ Synthetic data disclaimer.** Every number in this repository — unit
> volumes, prices, headcount, costs, working-capital days, capex proposals,
> discount rates, all of it — is invented for this project. The four business
> units are loosely modeled (for narrative flavor only) on a multi-segment
> retail/distribution company along the lines of Staples: an eCommerce
> channel, a B2B/contract sales channel, a store-based Retail channel, and an
> internal Supply Chain & Distribution function. **None of it reflects the
> real financial results, strategy, or operations of any actual company.**
> Do not use any figure here as a real-world reference point.

## Why this exists

Rotational-program interviews for corporate FP&A roles want to see that a
candidate understands *how the numbers are actually built* — not just that
they can run a regression. This project builds an entire FP&A cycle end to
end, in code, so every output can be inspected, re-run, and tied back to its
inputs:

```
driver_based_budget  →  variance_analysis  →  rolling_forecast
        ↓                                            ↓
cash_flow_model  →  scenario_sensitivity  →  capital_allocation
        ↓
    reporting  (assembles everything into one board pack)
```

## Project structure

| Module | What it does |
|---|---|
| `driver_based_budget.py` | Builds a bottom-up annual budget for 4 business units from explicit operational drivers (units, price, headcount, cost/unit), phased into 12 months by seasonality, and rolled up into a consolidated P&L (Revenue → EBITDA). |
| `variance_analysis.py` | Generates a synthetic "actual" year, then decomposes actual-vs-budget revenue variance into **volume / price / mix**, builds a **spend variance bridge** for opex, and produces a full **EBITDA variance walk** that reconciles exactly. |
| `rolling_forecast.py` | Implements a **quarterly rolling reforecast** (actuals-to-date blended with the original budget) and scores its accuracy against the static budget using **MAPE** and **bias**. |
| `cash_flow_model.py` | An **indirect-method** cash flow model driven by working-capital assumptions (DSO/DPO/DIO), a capex schedule, and a term-loan + revolver debt schedule. Produces monthly free cash flow, a full monthly balance sheet that ties out exactly, and a **13-week liquidity view**. |
| `scenario_sensitivity.py` | Runs **Base/Upside/Downside** macro scenarios and a **tornado/sensitivity analysis** ranking which drivers move consolidated EBITDA and free cash flow the most. |
| `capital_allocation.py` | Scores a pipeline of 13 synthetic capex proposals on **NPV, payback period, and strategic fit**, ranks them, and selects a funded subset under a constrained annual capex budget — both by transparent priority ranking and by an exact NPV-maximizing knapsack optimization. |
| `reporting.py` | Assembles a board-style monthly reporting pack: P&L vs. budget vs. prior year, variance-bridge waterfall-chart data, a cash flow summary, and a **written executive summary generated directly from the computed numbers**. |
| `common.py` | Shared utilities: seasonality-curve construction, NPV, IRR. |

## FP&A concepts, in plain language

**Driver-based budgeting.** Instead of "grow last year by 5%," each business
unit's budget is built from the operational levers that actually create
revenue and cost — units sold, price per unit, headcount, cost per head, cost
per unit handled. The P&L is a *derived output* of those drivers, which makes
the budget auditable (anyone can ask "why does revenue assume that?" and get
a driver-level answer) and easy to re-flex under different assumptions.

**The variance bridge (volume / price / mix / cost).** When actuals differ
from budget, FP&A doesn't just report "$26M short" — it explains *why*, in
four named, reconciling pieces:
- **Volume** — the effect of selling more or fewer units, at budget prices.
- **Price** — the effect of selling at a different price, at actual volumes.
- **Mix** — a portfolio effect: even at flat total volume, revenue moves if
  the business sells relatively more of a higher-margin product line.
- **Cost** — everything on the spend side: pay rates, unit costs, fixed opex.

These four effects sum *exactly* back to the total budget-to-actual gap —
that reconciliation is what this project's variance walk (and its test
suite) proves.

**Rolling forecast.** A once-a-year budget goes stale. A rolling forecast
re-estimates the rest of the year at each checkpoint (here, quarterly) by
blending the trend in actuals-to-date with the original budget assumption,
trusting the trend more as more of the year becomes known. Its value is then
measured against the static budget using **MAPE** (how big the average miss
is) and **bias** (whether the forecast systematically runs high or low).

**Indirect-method cash flow.** Net income is an accrual number; cash is not
the same thing. The indirect method reconciles the two by adding back
non-cash D&A and adjusting for cash tied up or released by working capital
(AR, inventory, AP) — driven off the P&L using **DSO/DPO/DIO** day-count
ratios. Done correctly, this reconciliation keeps a full balance sheet in
balance every period, which this project's cash flow model does and tests.

**Scenario vs. sensitivity analysis.** A *scenario* moves a coherent bundle
of assumptions together (e.g. "a mild downturn": volume down, costs up). A
*sensitivity (tornado) analysis* instead flexes **one driver at a time** to
see which single assumption matters most — the basis for the classic tornado
chart used to focus diligence on the few drivers that actually move outcomes.

**Capital allocation under a constrained budget.** Every company has more
good projects than capital. This module scores a capex pipeline on NPV,
payback, and strategic fit, and compares two ranking philosophies: a
transparent top-down **priority ranking** (what real capital committees
mostly use) against an exact NPV-**maximizing knapsack optimization** (the
benchmark for what the priority ranking leaves on the table).

## Getting started

```bash
pip install -r requirements.txt

# Run any module directly to see its output
python3 -m corporate_fpa_toolkit.driver_based_budget
python3 -m corporate_fpa_toolkit.variance_analysis
python3 -m corporate_fpa_toolkit.rolling_forecast
python3 -m corporate_fpa_toolkit.cash_flow_model
python3 -m corporate_fpa_toolkit.scenario_sensitivity
python3 -m corporate_fpa_toolkit.capital_allocation
python3 -m corporate_fpa_toolkit.reporting

# Run the test suite
pytest
```

### Example

```python
from corporate_fpa_toolkit.reporting import build_reporting_pack

pack = build_reporting_pack()
print(pack["executive_summary"])
print(pack["pnl_comparison"])
print(pack["ebitda_variance_walk"])
```

## Test suite

`pytest` covers budget roll-up arithmetic (monthly phasing ties to annual
totals), variance decomposition correctness (volume/price/mix components sum
exactly to total variance, at both the business-unit and consolidated
level), rolling forecast logic (MAPE/bias definitions, reforecast blending),
cash flow model integrity (the monthly balance sheet ties out to the
penny), NPV/payback/IRR calculations, and capital allocation ranking logic
(greedy selection respects the budget constraint; the knapsack optimum is
never worse than the greedy ranking).

## License

MIT — see `LICENSE`.
