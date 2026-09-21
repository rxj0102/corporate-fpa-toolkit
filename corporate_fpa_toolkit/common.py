"""Shared constants and small utilities used across the toolkit.

Nothing in this module contains business results -- it only provides
plumbing (month labels, seasonality-curve construction, discounting
helpers) that the model modules build on.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

MONTH_NAMES: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

MONTHS_PER_YEAR = 12
QUARTER_OF_MONTH: tuple[int, ...] = tuple((m - 1) // 3 + 1 for m in range(1, 13))


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Scale ``weights`` so they sum to exactly 1.0.

    Used to turn an arbitrary set of relative monthly weights into a
    proper seasonality index (a probability-simplex vector) so that
    downstream code can multiply an annual driver by the vector and get
    monthly phasing that ties exactly back to the annual total.
    """
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or arr.size != MONTHS_PER_YEAR:
        raise ValueError("weights must be a length-12 sequence")
    if np.any(arr < 0):
        raise ValueError("weights must be non-negative")
    total = arr.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return arr / total


def even_seasonality() -> np.ndarray:
    """A flat (non-seasonal) 12-month phasing curve."""
    return np.full(MONTHS_PER_YEAR, 1.0 / MONTHS_PER_YEAR)


def peaked_seasonality(
    peak_months: Iterable[int],
    peak_multiplier: float = 1.8,
    base: float = 1.0,
) -> np.ndarray:
    """Build a seasonality curve that is flat except for elevated peak months.

    Parameters
    ----------
    peak_months:
        1-indexed months (1=Jan .. 12=Dec) that should carry extra weight.
    peak_multiplier:
        Relative weight of a peak month versus a base month.
    base:
        Relative weight of a non-peak month.
    """
    weights = np.full(MONTHS_PER_YEAR, base, dtype=float)
    for m in peak_months:
        if not 1 <= m <= 12:
            raise ValueError("peak_months must be 1..12")
        weights[m - 1] = base * peak_multiplier
    return normalize_weights(weights)


def blended_seasonality(curves: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weight-average several seasonality curves (e.g. to derive a
    distribution/fulfillment throughput curve from the volume curves of
    the demand-generating business units it serves).
    """
    curves_arr = np.vstack([normalize_weights(c) for c in curves])
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    blended = (curves_arr * w[:, None]).sum(axis=0)
    return normalize_weights(blended)


def npv(rate: float, cashflows: Sequence[float]) -> float:
    """Net present value of ``cashflows`` (index 0 = time 0, undiscounted).

    ``cashflows[0]`` is conventionally the initial (negative) investment
    occurring at t=0 and is therefore not discounted; ``cashflows[t]``
    for t >= 1 is discounted by ``(1 + rate) ** t``.
    """
    flows = np.asarray(cashflows, dtype=float)
    periods = np.arange(flows.size)
    return float(np.sum(flows / (1.0 + rate) ** periods))


def irr(cashflows: Sequence[float], guess: float = 0.1) -> float:
    """Internal rate of return via Newton's method with a bisection fallback.

    Returns ``float('nan')`` if no sign change exists in ``cashflows``
    (no real IRR can bracket the cash flow stream).
    """
    flows = np.asarray(cashflows, dtype=float)
    if not np.any(flows > 0) or not np.any(flows < 0):
        return float("nan")

    def _npv_at(rate: float) -> float:
        return npv(rate, flows)

    rate = guess
    for _ in range(200):
        f = _npv_at(rate)
        eps = 1e-6
        f_prime = (_npv_at(rate + eps) - f) / eps
        if f_prime == 0:
            break
        new_rate = rate - f / f_prime
        if not np.isfinite(new_rate):
            break
        if abs(new_rate - rate) < 1e-9:
            return float(new_rate)
        rate = new_rate

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = _npv_at(lo), _npv_at(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = _npv_at(mid)
        if abs(f_mid) < 1e-8:
            return float(mid)
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return float((lo + hi) / 2)
