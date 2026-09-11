from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Genome:
    """Small, bounded set of heritable neural parameters.

    The MaleCNS graph itself is intentionally not mutated in the MVP. Variation
    is limited to learning, reinforcement sensitivity, decoder threshold and a
    few KC physiology parameters so the connectome remains identifiable.
    """

    eta: float = 0.001
    reward_current: float = 20.0
    aversive_current: float = 20.0
    decoder_threshold_hz: float = 2.0
    kc_rest: float = -60.0
    adaptation_jump: float = 8.0
    adaptation_tau: float = 200.0

    BOUNDS = {
        "eta": (1e-5, 0.05),
        "reward_current": (1.0, 60.0),
        "aversive_current": (1.0, 60.0),
        "decoder_threshold_hz": (0.1, 20.0),
        "kc_rest": (-75.0, -50.0),
        "adaptation_jump": (0.1, 20.0),
        "adaptation_tau": (30.0, 1000.0),
    }

    def __post_init__(self):
        for key, (lo, hi) in self.BOUNDS.items():
            value = getattr(self, key)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key} outside evolutionary bounds: {value}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def mutate(self, rng: np.random.Generator, sigma: float = 0.15) -> "Genome":
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("mutation sigma must be positive and finite")

        def log_mutate(name: str, value: float) -> float:
            lo, hi = self.BOUNDS[name]
            mutated = value * math.exp(float(rng.normal(0.0, sigma)))
            return float(np.clip(mutated, lo, hi))

        lo, hi = self.BOUNDS["kc_rest"]
        kc_rest = float(np.clip(self.kc_rest + rng.normal(0.0, 2.0 * sigma), lo, hi))

        return Genome(
            eta=log_mutate("eta", self.eta),
            reward_current=log_mutate("reward_current", self.reward_current),
            aversive_current=log_mutate("aversive_current", self.aversive_current),
            decoder_threshold_hz=log_mutate(
                "decoder_threshold_hz", self.decoder_threshold_hz
            ),
            kc_rest=kc_rest,
            adaptation_jump=log_mutate("adaptation_jump", self.adaptation_jump),
            adaptation_tau=log_mutate("adaptation_tau", self.adaptation_tau),
        )
