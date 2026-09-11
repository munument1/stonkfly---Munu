from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fitness:
    score: float
    return_pct: float
    max_drawdown_pct: float
    trade_count: int


def score_equity_curve(equity_curve, trade_count: int) -> Fitness:
    """Risk-aware score in percentage points.

    Fees are already reflected in equity. Drawdown and a small trade-count
    penalty discourage one-shot gambling and excessive churn from dominating
    selection purely by luck.
    """

    curve = np.asarray(equity_curve, dtype=float)
    if curve.ndim != 1 or len(curve) < 2 or not np.isfinite(curve).all():
        raise ValueError("equity curve must contain at least two finite values")
    if np.any(curve <= 0):
        raise ValueError("equity must stay positive")
    if type(trade_count) is not int or trade_count < 0:
        raise ValueError("trade_count must be a nonnegative integer")

    start = float(curve[0])
    return_pct = (float(curve[-1]) / start - 1.0) * 100.0
    peaks = np.maximum.accumulate(curve)
    drawdown = (peaks - curve) / peaks * 100.0
    max_drawdown_pct = float(drawdown.max())
    score = return_pct - 0.5 * max_drawdown_pct - 0.02 * trade_count
    return Fitness(
        score=float(score),
        return_pct=float(return_pct),
        max_drawdown_pct=max_drawdown_pct,
        trade_count=trade_count,
    )
