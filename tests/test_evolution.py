import json

import numpy as np

from stonkfly.config import D
from stonkfly.evolution import __main__ as evolution_cli
from stonkfly.evolution.experiment import Individual, LearnedMemory, PaperAccount
from stonkfly.evolution.fitness import score_equity_curve
from stonkfly.evolution.genome import Genome
from stonkfly.market import Quote


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

def _quote(bid="100", ask="100.1"):
    return Quote(
        "BTC-USDC", D(bid), D(ask), 1.0, D(".00000001"), D(".01"),
        D(".01"), D("1"), D(".00000001"),
    )


def test_paper_account_uses_exact_decimal_fees():
    account = PaperAccount(capital=100, order_usdc=10, fee_rate="0.006")
    quote = _quote()
    assert account.trade("BUY", quote)
    assert account.cash == D("89.940")
    assert account.position == D("10") / quote.ask
    assert account.equity(quote) < D("100")
    assert account.trade("SELL", quote)
    assert account.position == D("0")
    expected_sale = (D("10") / quote.ask) * quote.bid
    assert account.cash == D("89.940") + expected_sale * D("0.994")
    assert account.trade_count == 2


def test_memory_copy_is_independent():
    parent = LearnedMemory(np.array([1.0]), np.array([2.0]))
    first = parent.copy()
    second = parent.copy()
    first.memory_u[0] = 9.0
    first.memory_w[0] = 8.0
    assert parent.memory_u[0] == 1.0
    assert parent.memory_w[0] == 2.0
    assert second.memory_u[0] == 1.0
    assert second.memory_w[0] == 2.0
    assert not np.shares_memory(first.memory_u, second.memory_u)
    assert not np.shares_memory(first.memory_w, second.memory_w)


def test_compiler_command_accepts_multiword_cxx(monkeypatch):
    from stonkfly.neural.brain import compiler_command
    monkeypatch.setenv("CXX", "zig c++")
    assert compiler_command() == ["zig", "c++"]


def test_same_seed_produces_same_initial_genomes(monkeypatch, tmp_path):
    import stonkfly.evolution.experiment as experiment
    populations = []

    def fake_evaluate(individual: Individual, **kwargs):
        populations[-1].append(individual.genome)
        learned = LearnedMemory(np.zeros(1), np.zeros(1))
        score = float(individual.genome.reward_current)
        return ({
            "genome": individual.genome.to_dict(),
            "fingerprint": individual.genome.fingerprint(),
            "fitness": {"score": score, "return_pct": 0.0, "max_drawdown_pct": 0.0, "trade_count": 0},
        }, learned)

    monkeypatch.setattr(experiment, "_evaluate_individual", fake_evaluate)
    for name in ("first", "second"):
        populations.append([])
        experiment.run_evolution(
            out=tmp_path / name, population_size=4, generations=1, steps=2,
            elite_count=2, mutation_sigma=0.15, seed=77, product="BTC-USDC",
            neural_ms=1, order_usdc=10, paper_fee=0.006, reward_deadband="0.01",
        )
    assert populations[0] == populations[1]


def test_fitness_sorting_and_elite_preservation(monkeypatch, tmp_path):
    import stonkfly.evolution.experiment as experiment
    scores = iter([1.0, 3.0, 2.0, 0.0, 0.0, 0.0])

    def fake_evaluate(individual: Individual, **kwargs):
        learned = LearnedMemory(np.zeros(1), np.zeros(1))
        score = next(scores)
        return ({
            "genome": individual.genome.to_dict(),
            "fingerprint": individual.genome.fingerprint(),
            "fitness": {"score": score, "return_pct": 0.0, "max_drawdown_pct": 0.0, "trade_count": 0},
        }, learned)

    monkeypatch.setattr(experiment, "_evaluate_individual", fake_evaluate)
    out = tmp_path / "selection"
    experiment.run_evolution(
        out=out, population_size=3, generations=2, steps=2, elite_count=1,
        mutation_sigma=0.15, seed=11, product="BTC-USDC", neural_ms=1,
        order_usdc=10, paper_fee=0.006, reward_deadband="0.01",
    )
    first = json.loads((out / "generation-0000.json").read_text())["ranked"]
    second = json.loads((out / "generation-0001.json").read_text())["ranked"]
    assert [row["fitness"]["score"] for row in first] == [3.0, 2.0, 1.0]
    assert second[0]["genome"] == first[0]["genome"]

