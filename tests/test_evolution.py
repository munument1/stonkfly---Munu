import numpy as np

from stonkfly.evolution import __main__ as evolution_cli
from stonkfly.evolution.experiment import LearnedMemory
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


def test_lamarckian_memory_seed_applies_only_memory_arrays():
    class Brain:
        def __init__(self):
            self.memory_u = np.zeros(3, dtype=np.float64)
            self.memory_w = np.zeros(3, dtype=np.float64)
            self.baseline_plastic = np.array([2.0, 4.0, 8.0])
            self.weight = np.array([2.0, 99.0, 4.0, 8.0, 99.0])
            self.circuit = {"edges": np.array([0, 2, 3])}

    parent = Brain()
    parent.memory_u[:] = [0.1, -0.2, 0.3]
    parent.memory_w[:] = [0.25, -0.5, 0.5]
    seed = LearnedMemory.from_brain(parent)

    child = Brain()
    unrelated_before = child.weight[[1, 4]].copy()
    seed.apply(child)

    np.testing.assert_allclose(child.memory_u, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(child.memory_w, [0.25, -0.5, 0.5])
    np.testing.assert_allclose(child.weight[[0, 2, 3]], [2.5, 2.0, 12.0])
    np.testing.assert_allclose(child.weight[[1, 4]], unrelated_before)


def test_memory_fingerprint_is_stable_and_sensitive():
    a = LearnedMemory(np.array([1.0]), np.array([2.0]))
    b = LearnedMemory(np.array([1.0]), np.array([2.0]))
    c = LearnedMemory(np.array([1.0]), np.array([3.0]))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()

def test_two_individual_cli_uses_one_elite(monkeypatch, tmp_path):
    calls = []

    def fake_run_evolution(**kwargs):
        calls.append(kwargs)
        return {
            "fingerprint": "test",
            "fitness": {"score": 0.0},
            "genome": Genome().to_dict(),
        }

    monkeypatch.setattr(evolution_cli, "run_evolution", fake_run_evolution)
    evolution_cli.main(
        [
            "--inheritance",
            "both",
            "--population",
            "2",
            "--generations",
            "2",
            "--steps",
            "6",
            "--out",
            str(tmp_path),
        ]
    )

    assert [call["inheritance"] for call in calls] == ["darwinian", "lamarckian"]
    assert [call["elite_count"] for call in calls] == [1, 1]

