from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .experiment import run_evolution


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m stonkfly.evolution",
        description="Paper-only neuroevolution assay for Stonkfly.",
    )
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--elite", type=int, default=2)
    p.add_argument("--mutation-sigma", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--product",
        default="BTC-USDC",
        choices=["BTC-USDC", "ETH-USDC", "SOL-USDC"],
    )
    p.add_argument("--neural-ms", type=float, default=100.0)
    p.add_argument("--order-usdc", type=float, default=10.0)
    p.add_argument("--paper-fee", type=float, default=0.006)
    p.add_argument("--reward-deadband", default="0.01")
    p.add_argument("--out", type=Path, default=Path("runs/evolution"))
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    champion = run_evolution(
        out=a.out,
        population_size=a.population,
        generations=a.generations,
        steps=a.steps,
        elite_count=a.elite,
        mutation_sigma=a.mutation_sigma,
        seed=a.seed,
        product=a.product,
        neural_ms=a.neural_ms,
        order_usdc=a.order_usdc,
        paper_fee=a.paper_fee,
        reward_deadband=a.reward_deadband,
        overwrite=a.overwrite,
    )
    print(
        json.dumps(
            {
                "champion": champion["fingerprint"],
                "fitness": champion["fitness"],
                "genome": champion["genome"],
                "out": str(a.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
