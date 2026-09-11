from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
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
            notional = min(self.order_usdc, self.cash / (D(1) + self.fee_rate))
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


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _write_progress(out: Path, **values) -> None:
    path = out / "progress.json"
    progress = {}
    if path.exists():
        try:
            progress = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    progress.update(values)
    progress["pid"] = os.getpid()
    progress["updated_at_unix"] = time.time()
    _write_json(path, progress)


def _replay_schedule(count: int, generations: int, seed: int):
    if count < 1:
        return []
    schedule_rng = np.random.default_rng(seed ^ 0x5F3759DF)
    schedule = []
    while len(schedule) < generations:
        schedule.extend(int(index) for index in schedule_rng.permutation(count))
    return schedule[:generations]


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
    replay_paths=None,
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

    if replay_path is not None and replay_paths is not None:
        raise ValueError("use replay_path or replay_paths, not both")
    resolved_replays = (
        [Path(path) for path in replay_paths]
        if replay_paths is not None
        else ([Path(replay_path)] if replay_path is not None else [])
    )
    if replay_paths is not None and not resolved_replays:
        raise ValueError("replay_paths must not be empty")

    market_config: dict
    replay_infos = []
    replay_schedule = []
    if resolved_replays:
        replay_infos = [recording_info(path) for path in resolved_replays]
        for path, info in zip(resolved_replays, replay_infos):
            if info["product"] != product:
                raise ValueError(
                    f"replay {path} contains {info['product']} but experiment "
                    f"requested {product}"
                )
            if steps != 0 and (steps < 2 or steps > info["observations"]):
                raise ValueError(
                    f"steps must be 2..{info['observations']} for replay {path}, "
                    "or 0 for all"
                )
        replay_schedule = _replay_schedule(len(resolved_replays), generations, seed)
        market_config = {
            "source": "coinbase-public-recordings",
            "schedule": "seeded-shuffle-per-cycle",
            "recordings": [
                {"path": str(path), **info}
                for path, info in zip(resolved_replays, replay_infos)
            ],
            "generation_replay_indexes": replay_schedule,
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
    _write_json(out / "config.json", config)
    _write_progress(
        out,
        status="running",
        inheritance=inheritance,
        completed_generations=0,
        total_generations=generations,
        current_generation=0,
        completed_individuals=0,
        total_individuals=population_size,
    )

    rng = np.random.default_rng(seed)
    baseline = Genome()
    population = [Individual(baseline)]
    while len(population) < population_size:
        population.append(Individual(baseline.mutate(rng, mutation_sigma)))

    best_overall = None
    for generation in range(generations):
        if resolved_replays:
            replay_index = replay_schedule[generation]
            generation_replay = resolved_replays[replay_index]
            generation_info = replay_infos[replay_index]
            generation_steps = (
                generation_info["observations"] if steps == 0 else steps
            )
            generation_market = {
                "replay_index": replay_index,
                "path": str(generation_replay),
                "sha256": generation_info["sha256"],
                "observations": generation_steps,
            }
        else:
            generation_replay = None
            generation_steps = steps
            generation_market = {"source": "synthetic-fixture"}
        ranked = []
        for index, individual in enumerate(population):
            _write_progress(
                out,
                status="running",
                current_generation=generation,
                completed_generations=generation,
                current_individual=index,
                completed_individuals=index,
                market=generation_market,
            )
            result, learned = _evaluate_individual(
                individual,
                product=product,
                steps=generation_steps,
                neural_ms=neural_ms,
                order_usdc=order_usdc,
                paper_fee=paper_fee,
                reward_deadband=reward_deadband,
                replay_path=generation_replay,
            )
            result["generation"] = generation
            result["index"] = index
            result["_learned_memory"] = learned
            ranked.append(result)
            _write_progress(
                out,
                current_generation=generation,
                current_individual=index,
                completed_individuals=index + 1,
                latest={
                    "fingerprint": result["fingerprint"],
                    "score": result["fitness"]["score"],
                    "return_pct": result["fitness"]["return_pct"],
                },
            )
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
            "market": generation_market,
            "ranked": public_ranked,
        }
        _write_json(out / f"generation-{generation:04d}.json", payload)
        if (
            best_overall is None
            or ranked[0]["fitness"]["score"] > best_overall["fitness"]["score"]
        ):
            best_overall = _public_row(ranked[0])

        _write_progress(
            out,
            completed_generations=generation + 1,
            current_generation=(
                generation + 1 if generation + 1 < generations else generation
            ),
            current_individual=None,
            completed_individuals=(
                0 if generation + 1 < generations else population_size
            ),
            best={
                "fingerprint": best_overall["fingerprint"],
                "score": best_overall["fitness"]["score"],
                "return_pct": best_overall["fitness"]["return_pct"],
            },
        )

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

    _write_json(out / "champion.json", best_overall)
    _write_progress(out, status="completed", current_individual=None)
    return best_overall
