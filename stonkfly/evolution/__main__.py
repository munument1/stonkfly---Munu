from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_evolution


def _run_mode(a, mode: str, out: Path):
    elite_count = min(2, a.population - 1) if a.elite is None else a.elite
    return run_evolution(
        out=out,
        population_size=a.population,
        generations=a.generations,
        steps=a.steps,
        elite_count=elite_count,
        mutation_sigma=a.mutation_sigma,
        seed=a.seed,
        product=a.product,
        neural_ms=a.neural_ms,
        order_usdc=a.order_usdc,
        paper_fee=a.paper_fee,
        reward_deadband=a.reward_deadband,
        replay_path=a.replay,
        inheritance=mode,
        overwrite=a.overwrite,
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m stonkfly.evolution",
        description="Paper-only neuroevolution assay for Stonkfly.",
    )
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--generations", type=int, default=1)
    p.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Market observations per individual; with --replay, 0 uses the full recording",
    )
    p.add_argument(
        "--elite",
        type=int,
        default=None,
        help="Elites retained per generation; defaults to min(2, population - 1)",
    )
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
    p.add_argument(
        "--replay",
        type=Path,
        help="Recorded Coinbase-public JSONL. No credentials or live orders are used.",
    )
    p.add_argument(
        "--inheritance",
        choices=["darwinian", "lamarckian", "both"],
        default="darwinian",
        help="Heritable physiology only, acquired KC/MBON memory too, or a matched comparison.",
    )
    p.add_argument("--out", type=Path, default=Path("runs/evolution"))
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    if a.inheritance == "both":
        champions = {
            mode: _run_mode(a, mode, a.out / mode)
            for mode in ("darwinian", "lamarckian")
        }
        print(
            json.dumps(
                {
                    "comparison": {
                        mode: {
                            "champion": row["fingerprint"],
                            "fitness": row["fitness"],
                            "genome": row["genome"],
                        }
                        for mode, row in champions.items()
                    },
                    "out": str(a.out),
                },
                indent=2,
            )
        )
        return

    champion = _run_mode(a, a.inheritance, a.out)
    print(
        json.dumps(
            {
                "inheritance": a.inheritance,
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
