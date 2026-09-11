import numpy as np

from stonkfly.evolution.fitness import score_equity_curve
from stonkfly.evolution.genome import Genome


def test_mutation_is_seeded_and_bounded():
    a = Genome().mutate(np.random.default_rng(123), 0.2)
    b = Genome().mutate(np.random.default_rng(123), 0.2)
    assert a == b
    assert a != Genome()
    for key, (lo, hi) in Genome.BOUNDS.items():
        assert lo <= getattr(a, key) <= hi


def test_fitness_rewards_return_and_penalizes_drawdown():
    smooth = score_equity_curve([100, 101, 102], 2)
    rough = score_equity_curve([100, 90, 102], 2)
    assert smooth.return_pct == rough.return_pct
    assert smooth.score > rough.score
    assert rough.max_drawdown_pct > 0


def test_trade_count_penalty_breaks_equal_equity_ties():
    quiet = score_equity_curve([100, 101], 1)
    churn = score_equity_curve([100, 101], 10)
    assert quiet.score > churn.score
