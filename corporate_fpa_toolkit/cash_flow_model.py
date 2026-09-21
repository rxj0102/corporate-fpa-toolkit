"""Indirect-method cash flow model and 13-week short-term liquidity view.

**FP&A concept -- the indirect method.** The indirect-method cash flow
statement starts from net income (an *accrual* number) and reconciles
it to cash by (1) adding back non-cash charges like depreciation &
amortization and (2) adjusting for the cash tied up or freed by changes
in working capital -- receivables, inventory, and payables. Investing
activities (capex) and financing activities (debt paydown, revolver
draws) are layered on top to arrive at the change in the cash balance.
Because this reconciliation is exactly how a balance sheet's cash line
is derived from the income statement and the other balance-sheet
accounts, a correctly built indirect cash flow statement should leave
the balance sheet in balance (Assets == Liabilities + Equity) every
period -- this module's test suite checks exactly that tie-out.

**Working capital drivers.** Rather than forecasting AR/Inventory/AP
balances directly, FP&A drives them off the P&L using day-count ratios:

* **DSO** (Days Sales Outstanding) -- ``AR = Revenue x DSO/30``
* **DIO** (Days Inventory Outstanding) -- ``Inventory = COGS x DIO/30``
* **DPO** (Days Payable Outstanding) -- ``AP = COGS x DPO/30``

(A 30-day month is used as the simplifying day-count convention
throughout, consistent with the model's monthly cadence.)

**13-week liquidity view.** Treasury/FP&A teams commonly track a
rolling 13-week *weekly* cash view for short-term liquidity monitoring,
independent of the monthly forecast cadence. This module derives one by
splitting the first quarter's monthly cash flow using the retail
**4-4-5 fiscal calendar convention** (4 weeks + 4 weeks + 5 weeks = 13
weeks per quarter, 52 weeks per year) -- a natural fit alongside this
project's retail/distribution business units.

.. warning:: All figures are synthetic -- see ``corporate_fpa_toolkit.SYNTHETIC_DATA_DISCLAIMER``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .driver_based_budget import build_budget

DAYS_PER_MONTH = 30.0
WEEKS_PER_MONTH_445: tuple[int, ...] = (4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5)


@dataclass(frozen=True)
class WorkingCapitalAssumptions:
    dso_days: float = 42.0
    dpo_days: float = 48.0
    dio_days: float = 55.0


@dataclass(frozen=True)
class CapexAssumptions:
    capex_pct_of_revenue: float = 0.028
    useful_life_years: float = 7.0
    beginning_net_ppe: float = 900_000_000.0
    beginning_ppe_remaining_life_years: float = 10.0


@dataclass(frozen=True)
class DebtAssumptions:
    beginning_term_loan: float = 500_000_000.0
    term_loan_annual_rate: float = 0.065
    term_loan_monthly_principal_payment: float = 4_500_000.0
    revolver_limit: float = 300_000_000.0
    revolver_annual_rate: float = 0.075
    beginning_cash: float = 150_000_000.0
    minimum_cash: float = 50_000_000.0
    tax_rate: float = 0.25


@dataclass(frozen=True)
class OpeningBalanceSheetPlugs:
    """Balance sheet items outside this model's scope, held flat so the
    monthly balance sheet is fully specified and can be tied out.
    """

    other_assets: float = 200_000_000.0
    other_liabilities: float = 150_000_000.0


def _working_capital_balance(flow: float, days: float) -> float:
    return flow * (days / DAYS_PER_MONTH)


def build_cash_flow_model(
    consolidated_monthly: pd.DataFrame,
    wc: WorkingCapitalAssumptions = WorkingCapitalAssumptions(),
    capex_assump: CapexAssumptions = CapexAssumptions(),
    debt: DebtAssumptions = DebtAssumptions(),
    opening_plugs: OpeningBalanceSheetPlugs = OpeningBalanceSheetPlugs(),
) -> dict[str, pd.DataFrame]:
    """Build the monthly indirect-method cash flow statement and balance sheet.

    Parameters
    ----------
    consolidated_monthly:
        A 12-row consolidated monthly P&L (as produced by
        :func:`corporate_fpa_toolkit.driver_based_budget.consolidate_monthly`)
        with at least ``month``, ``revenue``, ``cogs``, and ``ebitda``
        columns.

    Returns
    -------
    dict with keys ``"monthly_cash_flow"`` and ``"monthly_balance_sheet"``.
    """
    months = consolidated_monthly.sort_values("month").reset_index(drop=True)
    n = len(months)

    term_loan_monthly_rate = debt.term_loan_annual_rate / 12
    revolver_monthly_rate = debt.revolver_annual_rate / 12
    da_existing_monthly = capex_assump.beginning_net_ppe / (capex_assump.beginning_ppe_remaining_life_years * 12)
    useful_life_months = capex_assump.useful_life_years * 12

    ar0 = _working_capital_balance(months.loc[0, "revenue"], wc.dso_days)
    inv0 = _working_capital_balance(months.loc[0, "cogs"], wc.dio_days)
    ap0 = _working_capital_balance(months.loc[0, "cogs"], wc.dpo_days)

    opening_assets = debt.beginning_cash + ar0 + inv0 + capex_assump.beginning_net_ppe + opening_plugs.other_assets
    opening_liabilities = ap0 + debt.beginning_term_loan + 0.0 + opening_plugs.other_liabilities
    opening_equity = opening_assets - opening_liabilities

    cash_bal = debt.beginning_cash
    ar_bal, inv_bal, ap_bal = ar0, inv0, ap0
    ppe_bal = capex_assump.beginning_net_ppe
    term_loan_bal = debt.beginning_term_loan
    revolver_bal = 0.0
    equity_bal = opening_equity
    cumulative_capex = 0.0

    cf_rows = []
    bs_rows = []
    for i in range(n):
        row = months.loc[i]
        revenue, cogs, ebitda = row["revenue"], row["cogs"], row["ebitda"]

        capex = capex_assump.capex_pct_of_revenue * revenue
        da_new_capex = cumulative_capex / useful_life_months
        da_total = da_existing_monthly + da_new_capex
        cumulative_capex += capex

        term_loan_interest = term_loan_bal * term_loan_monthly_rate
        revolver_interest = revolver_bal * revolver_monthly_rate
        interest_expense = term_loan_interest + revolver_interest

        ebit = ebitda - da_total
        pretax_income = ebit - interest_expense
        taxes = max(0.0, pretax_income) * debt.tax_rate
        net_income = pretax_income - taxes

        new_ar = _working_capital_balance(revenue, wc.dso_days)
        new_inv = _working_capital_balance(cogs, wc.dio_days)
        new_ap = _working_capital_balance(cogs, wc.dpo_days)
        delta_ar, delta_inv, delta_ap = new_ar - ar_bal, new_inv - inv_bal, new_ap - ap_bal
        nwc_cash_effect = -delta_ar - delta_inv + delta_ap

        cfo = net_income + da_total + nwc_cash_effect
        cfi = -capex

        term_loan_payment = min(debt.term_loan_monthly_principal_payment, term_loan_bal)
        pre_revolver_cash = cash_bal + cfo + cfi - term_loan_payment

        revolver_draw, revolver_paydown = 0.0, 0.0
        if pre_revolver_cash < debt.minimum_cash:
            shortfall = debt.minimum_cash - pre_revolver_cash
            revolver_draw = min(shortfall, debt.revolver_limit - revolver_bal)
        elif revolver_bal > 0:
            surplus = pre_revolver_cash - debt.minimum_cash
            revolver_paydown = min(surplus, revolver_bal)

        cff = -term_loan_payment + revolver_draw - revolver_paydown
        ending_cash = cash_bal + cfo + cfi + cff

        term_loan_bal -= term_loan_payment
        revolver_bal += revolver_draw - revolver_paydown
        ppe_bal += capex - da_total
        equity_bal += net_income

        cf_rows.append(
            {
                "month": row["month"],
                "month_name": row["month_name"],
                "revenue": revenue,
                "ebitda": ebitda,
                "da_total": da_total,
                "ebit": ebit,
                "interest_expense": interest_expense,
                "pretax_income": pretax_income,
                "taxes": taxes,
                "net_income": net_income,
                "delta_ar": delta_ar,
                "delta_inventory": delta_inv,
                "delta_ap": delta_ap,
                "nwc_cash_effect": nwc_cash_effect,
                "cfo": cfo,
                "capex": capex,
                "cfi": cfi,
                "term_loan_payment": term_loan_payment,
                "revolver_draw": revolver_draw,
                "revolver_paydown": revolver_paydown,
                "cff": cff,
                "beginning_cash": cash_bal,
                "net_change_in_cash": cfo + cfi + cff,
                "ending_cash": ending_cash,
                "free_cash_flow": cfo + cfi,
            }
        )
        bs_rows.append(
            {
                "month": row["month"],
                "month_name": row["month_name"],
                "cash": ending_cash,
                "accounts_receivable": new_ar,
                "inventory": new_inv,
                "net_ppe": ppe_bal,
                "other_assets": opening_plugs.other_assets,
                "total_assets": ending_cash + new_ar + new_inv + ppe_bal + opening_plugs.other_assets,
                "accounts_payable": new_ap,
                "term_loan": term_loan_bal,
                "revolver": revolver_bal,
                "other_liabilities": opening_plugs.other_liabilities,
                "total_liabilities": new_ap + term_loan_bal + revolver_bal + opening_plugs.other_liabilities,
                "equity": equity_bal,
                "total_liabilities_and_equity": new_ap + term_loan_bal + revolver_bal
                + opening_plugs.other_liabilities + equity_bal,
            }
        )

        cash_bal, ar_bal, inv_bal, ap_bal = ending_cash, new_ar, new_inv, new_ap

    return {
        "monthly_cash_flow": pd.DataFrame(cf_rows),
        "monthly_balance_sheet": pd.DataFrame(bs_rows),
        "opening_balance_sheet": {
            "cash": debt.beginning_cash,
            "accounts_receivable": ar0,
            "inventory": inv0,
            "net_ppe": capex_assump.beginning_net_ppe,
            "other_assets": opening_plugs.other_assets,
            "total_assets": opening_assets,
            "accounts_payable": ap0,
            "term_loan": debt.beginning_term_loan,
            "revolver": 0.0,
            "other_liabilities": opening_plugs.other_liabilities,
            "total_liabilities": opening_liabilities,
            "equity": opening_equity,
        },
    }


def build_thirteen_week_liquidity_view(monthly_cash_flow: pd.DataFrame) -> pd.DataFrame:
    """Split the first quarter (months 1-3) of the monthly cash flow into a
    13-week liquidity view using the 4-4-5 retail fiscal calendar (4
    weeks in month 1, 4 in month 2, 5 in month 3).

    Each month's cash flow lines are spread evenly across its weeks, so
    the 13 weekly rows sum back exactly to the Q1 monthly totals.
    """
    q1 = monthly_cash_flow[monthly_cash_flow["month"].isin([1, 2, 3])].sort_values("month")
    weeks_by_month = WEEKS_PER_MONTH_445[:3]

    rows = []
    week_number = 1
    running_cash: Optional[float] = None
    for (_, month_row), n_weeks in zip(q1.iterrows(), weeks_by_month):
        weekly_cfo = month_row["cfo"] / n_weeks
        weekly_cfi = month_row["cfi"] / n_weeks
        weekly_cff = month_row["cff"] / n_weeks
        weekly_net_change = month_row["net_change_in_cash"] / n_weeks
        if running_cash is None:
            running_cash = month_row["beginning_cash"]
        for _ in range(n_weeks):
            beginning = running_cash
            ending = beginning + weekly_net_change
            rows.append(
                {
                    "week": week_number,
                    "month": month_row["month"],
                    "month_name": month_row["month_name"],
                    "beginning_cash": beginning,
                    "operating_cash_flow": weekly_cfo,
                    "investing_cash_flow": weekly_cfi,
                    "financing_cash_flow": weekly_cff,
                    "net_change_in_cash": weekly_net_change,
                    "ending_cash": ending,
                }
            )
            running_cash = ending
            week_number += 1
    return pd.DataFrame(rows)


def full_cash_flow_report(
    consolidated_monthly: Optional[pd.DataFrame] = None,
    wc: WorkingCapitalAssumptions = WorkingCapitalAssumptions(),
    capex_assump: CapexAssumptions = CapexAssumptions(),
    debt: DebtAssumptions = DebtAssumptions(),
) -> dict[str, object]:
    """Convenience entry point: budget's consolidated P&L -> full cash flow model."""
    if consolidated_monthly is None:
        consolidated_monthly = build_budget()["consolidated_monthly"]
    model = build_cash_flow_model(consolidated_monthly, wc=wc, capex_assump=capex_assump, debt=debt)
    model["thirteen_week_liquidity"] = build_thirteen_week_liquidity_view(model["monthly_cash_flow"])
    return model


if __name__ == "__main__":
    from . import SYNTHETIC_DATA_DISCLAIMER

    print(SYNTHETIC_DATA_DISCLAIMER)
    report = full_cash_flow_report()
    print("\n--- Monthly Free Cash Flow ---")
    print(report["monthly_cash_flow"][["month_name", "cfo", "cfi", "free_cash_flow", "ending_cash"]].to_string(index=False))
    print("\n--- 13-Week Liquidity View (first 5 weeks) ---")
    print(report["thirteen_week_liquidity"].head().to_string(index=False))
