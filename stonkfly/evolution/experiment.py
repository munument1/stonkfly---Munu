from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import D
from ..display import market_frame
from ..market import FixtureMarket
from ..neural.common import annotations
from ..neural.controller import Decoder
from ..neural.visual import VisualMemoryBrain
from ..reinforcement import reinforcement
from .fitness import score_equity_curve
from .genome import Genome
from .replay import ReplayMarket, recording_info


@dataclass(frozen=True)
class LearnedMemory:
    """Acquired KC/MBON memory state eligible for Lamarckian inheritance."""

    memory_u: np.ndarray
    memory_w: np.ndarray

    @classmethod
    def from_brain(cls, brain) -> "LearnedMemory":
        return cls(brain.memory_u.copy(), brain.memory_w.copy())

    def copy(self) -> "LearnedMemory":
        return LearnedMemory(self.memory_u.copy(), self.memory_w.copy())

    def apply(self, brain) -> None:
        if (
            self.memory_u.shape != brain.memory_u.shape
            or self.memory_w.shape != brain.memory_w.shape
            or not np.isfinite(self.memory_u).all()
            or not np.isfinite(self.memory_w).all()
        ):
            raise ValueError("Inherited memory is incompatible with this connectome")
        brain.memory_u[:] = self.memory_u
        brain.memory_w[:] = self.memory_w
        brain.weight[brain.circuit["edges"]] = brain.baseline_plastic * (
            1 + brain.memory_w
        )

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.memory_u.tobytes())
        h.update(self.memory_w.tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class Individual:
    genome: Genome
    inherited_memory: LearnedMemory | None = None


class EvolutionController:
    """Full connectome controller with heritable physiology parameters."""

    def __init__(
        self,
        genome: Genome,
        neural_ms: float,
        inherited_memory: LearnedMemory | None = None,
    ):
        if not math.isfinite(neural_ms) or neural_ms <= 0:
            raise ValueError("neural_ms must be positive and finite")
        self.genome = genome
        self.neural_ms = float(neural_ms)
        self.bin_ms = 10.0
        self.pulse_ms = min(50.0, self.neural_ms)
        self.brain = VisualMemoryBrain(
            eta=genome.eta,
            kc_rest=genome.kc_rest,
            adaptation_jump=genome.adaptation_jump,
            adaptation_tau=genome.adaptation_tau,
        )
        if inherited_memory is not None:
            inherited_memory.apply(self.brain)
        self.decoder = Decoder(
            self.brain.ids,
            annotations(self.brain.ids),
            genome.decoder_threshold_hz,
        )

    def observe(self, rgb, kind: str):
        if kind not in ("none", "reward", "aversive"):
            raise ValueError("unknown reinforcement")
        b = self.brain
        counts = np.zeros(b.n, dtype=np.int32)
        remaining = round(self.neural_ms / b.dt)
        pulse = round(self.pulse_ms / b.dt) if kind != "none" else 0
        current = (
            self.genome.reward_current
            if kind == "reward"
            else self.genome.aversive_current
        )
        while remaining:
            n = min(remaining, round(self.bin_ms / b.dt))
            if pulse:
                n = min(n, pulse)
            stimulus = (b.circuit[kind], current) if pulse else None
            c, _ = b.rgb_step(
                rgb,
                n * b.dt,
                learning=True,
                stimulation=stimulus,
            )
            counts += c
            remaining -= n
            if pulse:
                pulse -= n
        b.counts[:] = counts
        return self.decoder.decode(counts, self.neural_ms / 1000.0)


class PaperAccount:
    """Small deterministic assay account; never talks to Coinbase or AgentKit."""

    def __init__(self, capital=100.0, order_usdc=10.0, fee_rate=0.006):
        capital = D(capital)
        order_usdc = D(order_usdc)
        fee_rate = D(fee_rate)
        if not 0 < order_usdc <= capital:
            raise ValueError("order_usdc must be within assay capital")
        if not 0 <= fee_rate <= D("0.05"):
            raise ValueError("invalid paper fee")
        self.cash = capital
        self.position = D(0)
        self.order_usdc = order_usdc
        self.fee_rate = fee_rate
        self.trade_count = 0

    def equity(self, quote):
        return self.cash + self.position * quote.bid

    def trade(self, side: str, quote) -> bool:
        if side == "HOLD":
            return False
        if side == "BUY":
            ask = quote.ask
            notional = min(self.order_usdc, self.cash / (1.0 + self.fee_rate))
            if notional <= 0:
                return False
            base = notional / ask
            fee = notional * self.fee_rate
            self.cash -= notional + fee
            self.position += base
        elif side == "SELL":
            bid = quote.bid
            base = min(self.position, self.order_usdc / bid)
            if base <= 0:
                return False
            notional = base * bid
            fee = notional * self.fee_rate
            self.position -= base
            self.cash += notional - fee
        else:
            raise ValueError(f"unknown side: {side}")
        self.trade_count += 1
        return True


def _evaluate_individual(
    individual: Individual,
    *,
    product: str,
    steps: int,
    neural_ms: float,
    order_usdc: float,
    paper_fee: float,
    reward_deadband: str,
    replay_path: Path | None = None,
):
    """Evaluate one individual and return its acquired memory separately."""

    started = time.perf_counter()
    market = (
        ReplayMarket(replay_path, product=product)
        if replay_path is not None
        else FixtureMarket((product,))
    )
    controller = EvolutionController(
        individual.genome,
        neural_ms,
        inherited_memory=individual.inherited_memory,
    )
    account = PaperAccount(order_usdc=order_usdc, fee_rate=paper_fee)
    anchor = D("100")
    equity_curve = [anchor]
    market_input = hashlib.sha256()
    reward_events = 0
    aversive_events = 0
    last_quote = None

    for _ in range(steps):
        quotes = market.snapshot()
        market.record(quotes)
        quote = quotes[product]
        last_quote = quote
        before = account.equity(quote)
        kind, _ = reinforcement(before, anchor, reward_deadband)
        reward_events += int(kind == "reward")
        aversive_events += int(kind == "aversive")
        frame = market_frame(product, market.history[product], quote.bid, quote.ask)
        market_input.update(frame.tobytes())
        neural = controller.observe(frame, kind)
        anchor = before
        account.trade(neural["side"], quote)
        equity_curve.append(account.equity(quote))

    if last_quote is None:
        raise RuntimeError("evaluation produced no market observations")
    fitness = score_equity_curve(equity_curve, account.trade_count)
    learned = LearnedMemory.from_brain(controller.brain)
    return (
        {
            "genome": individual.genome.to_dict(),
            "fingerprint": individual.genome.fingerprint(),
            "fitness": dataclasses.asdict(fitness),
            "final_equity": float(equity_curve[-1]),
            "evaluation_seconds": time.perf_counter() - started,
            "market_input_sha256": market_input.hexdigest(),
            "reward_events": reward_events,
            "aversive_events": aversive_events,
            "memory_inherited": individual.inherited_memory is not None,
            "learned_memory_sha256": learned.fingerprint(),
            "memory": controller.brain.memory(),
        },
        learned,
    )


def evaluate_genome(
    genome: Genome,
    *,
    product: str,
    steps: int,
    neural_ms: float,
    order_usdc: float,
    paper_fee: float,
    reward_deadband: str,
    replay_path: Path | None = None,
):
    """Backward-compatible one-off evaluation with no inherited memory."""

    result, _ = _evaluate_individual(
        Individual(genome),
        product=product,
        steps=steps,
        neural_ms=neural_ms,
        order_usdc=order_usdc,
        paper_fee=paper_fee,
        reward_deadband=reward_deadband,
        replay_path=replay_path,
    )
    return result


def _public_row(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}


def run_evolution(
    *,
    out: Path,
    population_size: int,
    generations: int,
    steps: int,
    elite_count: int,
    mutation_sigma: float,
    seed: int,
    product: str,
    neural_ms: float,
    order_usdc: float,
    paper_fee: float,
    reward_deadband: str,
    replay_path: Path | None = None,
    inheritance: str = "darwinian",
    overwrite: bool = False,
):
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if generations < 1:
        raise ValueError("need at least one generation")
    if not 1 <= elite_count < population_size:
        raise ValueError("elite_count must be between 1 and population_size - 1")
    if inheritance not in ("darwinian", "lamarckian"):
        raise ValueError("inheritance must be darwinian or lamarckian")

    market_config: dict
    if replay_path is not None:
        replay_path = Path(replay_path)
        info = recording_info(replay_path)
        if info["product"] != product:
            raise ValueError(
                f"replay contains {info['product']} but experiment requested {product}"
            )
        if steps == 0:
            steps = info["observations"]
        if steps < 2 or steps > info["observations"]:
            raise ValueError(
                f"steps must be 2..{info['observations']} for this replay, or 0 for all"
            )
        market_config = {
            "source": "coinbase-public-recording",
            "path": str(replay_path),
            **info,
        }
    else:
        if steps < 2:
            raise ValueError("synthetic fixture needs at least two market steps")
        market_config = {"source": "synthetic-fixture"}

    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; pass --overwrite or choose another path")
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "population_size": population_size,
        "generations": generations,
        "steps": steps,
        "elite_count": elite_count,
        "mutation_sigma": mutation_sigma,
        "seed": seed,
        "product": product,
        "neural_ms": neural_ms,
        "order_usdc": order_usdc,
        "paper_fee": paper_fee,
        "reward_deadband": reward_deadband,
        "market": market_config,
        "network_execution": False,
        "inheritance": inheritance,
        "lamarckian_scope": (
            "memory_u/memory_w only; transient membrane/spike/rate state is reset"
            if inheritance == "lamarckian"
            else None
        ),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    rng = np.random.default_rng(seed)
    baseline = Genome()
    population = [Individual(baseline)]
    while len(population) < population_size:
        population.append(Individual(baseline.mutate(rng, mutation_sigma)))

    best_overall = None
    for generation in range(generations):
        ranked = []
        for index, individual in enumerate(population):
            result, learned = _evaluate_individual(
                individual,
                product=product,
                steps=steps,
                neural_ms=neural_ms,
                order_usdc=order_usdc,
                paper_fee=paper_fee,
                reward_deadband=reward_deadband,
                replay_path=replay_path,
            )
            result["generation"] = generation
            result["index"] = index
            result["_learned_memory"] = learned
            ranked.append(result)
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "individual": index,
                        "fingerprint": result["fingerprint"],
                        "score": result["fitness"]["score"],
                        "return_pct": result["fitness"]["return_pct"],
                        "memory_inherited": result["memory_inherited"],
                    }
                ),
                flush=True,
            )

        ranked.sort(key=lambda row: row["fitness"]["score"], reverse=True)
        public_ranked = [_public_row(row) for row in ranked]
        payload = {
            "generation": generation,
            "inheritance": inheritance,
            "ranked": public_ranked,
        }
        (out / f"generation-{generation:04d}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        if (
            best_overall is None
            or ranked[0]["fitness"]["score"] > best_overall["fitness"]["score"]
        ):
            best_overall = _public_row(ranked[0])

        if generation + 1 < generations:
            parents = ranked[:elite_count]
            next_population = []
            for row in parents:
                seed_memory = (
                    row["_learned_memory"].copy()
                    if inheritance == "lamarckian"
                    else None
                )
                next_population.append(
                    Individual(Genome(**row["genome"]), seed_memory)
                )
            while len(next_population) < population_size:
                row = parents[int(rng.integers(0, len(parents)))]
                child = Genome(**row["genome"]).mutate(rng, mutation_sigma)
                seed_memory = (
                    row["_learned_memory"].copy()
                    if inheritance == "lamarckian"
                    else None
                )
                next_population.append(Individual(child, seed_memory))
            population = next_population

    (out / "champion.json").write_text(json.dumps(best_overall, indent=2) + "\n")
    return best_overall
