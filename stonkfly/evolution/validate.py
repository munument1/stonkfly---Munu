from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .experiment import Individual, PaperAccount, _evaluate_individual
from .fitness import score_equity_curve
from .genome import Genome
from .replay import ReplayMarket, recording_info


def evaluate_baseline(
    replay_path: Path,
    strategy: str,
    *,
    seed: int = 0,
    order_usdc: float = 10.0,
    paper_fee: float = 0.006,
):
    if strategy not in {"hold", "random", "momentum"}:
        raise ValueError("unknown baseline strategy")
    market = ReplayMarket(replay_path)
    account = PaperAccount(order_usdc=order_usdc, fee_rate=paper_fee)
    equity_curve = [account.cash]
    rng = np.random.default_rng(seed)
    previous_mid = None

    for _ in range(market.length):
        quotes = market.snapshot()
        quote = quotes[market.product]
        mid = float((quote.bid + quote.ask) / 2)
        if strategy == "hold":
            side = "HOLD"
        elif strategy == "random":
            side = str(rng.choice(("BUY", "SELL", "HOLD")))
        elif previous_mid is None or mid == previous_mid:
            side = "HOLD"
        else:
            side = "BUY" if mid > previous_mid else "SELL"
        account.trade(side, quote)
        equity_curve.append(account.equity(quote))
        market.record(quotes)
        previous_mid = mid

    fitness = score_equity_curve(equity_curve, account.trade_count)
    return {
        "strategy": strategy,
        "fitness": {
            "score": fitness.score,
            "return_pct": fitness.return_pct,
            "max_drawdown_pct": fitness.max_drawdown_pct,
            "trade_count": fitness.trade_count,
        },
        "final_equity": float(equity_curve[-1]),
    }


def _champions(run: Path):
    direct = run / "champion.json"
    if direct.is_file():
        config = json.loads((run / "config.json").read_text())
        return [(config.get("inheritance", "evolution"), direct)]
    return [
        (mode, run / mode / "champion.json")
        for mode in ("darwinian", "lamarckian")
        if (run / mode / "champion.json").is_file()
    ]


def _aggregate(rows):
    returns = [row["fitness"]["return_pct"] for row in rows]
    drawdowns = [row["fitness"]["max_drawdown_pct"] for row in rows]
    return {
        "windows": len(rows),
        "mean_return_pct": statistics.fmean(returns),
        "median_return_pct": statistics.median(returns),
        "worst_return_pct": min(returns),
        "positive_windows": sum(value > 0 for value in returns),
        "mean_max_drawdown_pct": statistics.fmean(drawdowns),
        "worst_max_drawdown_pct": max(drawdowns),
    }


def validate_run(
    *,
    run: Path,
    replays,
    out: Path,
    neural_ms: float = 50.0,
    order_usdc: float = 10.0,
    paper_fee: float = 0.006,
    reward_deadband: str = "0.01",
    seed: int = 7,
):
    run = Path(run)
    replays = [Path(path) for path in replays]
    if not replays:
        raise ValueError("at least one out-of-sample replay is required")
    champions = _champions(run)
    if not champions:
        raise FileNotFoundError("run does not contain a champion")
    replay_info = [recording_info(path) for path in replays]
    products = {info["product"] for info in replay_info}
    if len(products) != 1:
        raise ValueError("all validation recordings must use the same product")
    product = products.pop()

    baselines = {}
    for strategy in ("hold", "random", "momentum"):
        rows = [
            {
                "replay": str(path),
                **evaluate_baseline(
                    path,
                    strategy,
                    seed=seed + index,
                    order_usdc=order_usdc,
                    paper_fee=paper_fee,
                ),
            }
            for index, path in enumerate(replays)
        ]
        baselines[strategy] = {"aggregate": _aggregate(rows), "results": rows}

    modes = {}
    for mode, champion_path in champions:
        champion = json.loads(champion_path.read_text())
        genome = Genome(**champion["genome"])
        rows = []
        for path, info in zip(replays, replay_info):
            result, _ = _evaluate_individual(
                Individual(genome),
                product=product,
                steps=info["observations"],
                neural_ms=neural_ms,
                order_usdc=order_usdc,
                paper_fee=paper_fee,
                reward_deadband=reward_deadband,
                replay_path=path,
            )
            rows.append({"replay": str(path), **result})
        aggregate = _aggregate(rows)
        benchmark = max(
            baseline["aggregate"]["mean_return_pct"]
            for baseline in baselines.values()
        )
        modes[mode] = {
            "champion": champion["fingerprint"],
            "aggregate": aggregate,
            "results": rows,
            "beats_best_baseline_mean": aggregate["mean_return_pct"] > benchmark,
        }

    payload = {
        "format": "evostonkfly-validation-v1",
        "source_run": str(run),
        "product": product,
        "out_of_sample_recordings": replay_info,
        "generalization_ready": len(replays) >= 3,
        "minimum_recommended_windows": 3,
        "baselines": baselines,
        "modes": modes,
        "network_execution": False,
        "live_trading": False,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Out-of-sample profitability validation for Stonkfly champions"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--neural-ms", type=float, default=50.0)
    parser.add_argument("--order-usdc", type=float, default=10.0)
    parser.add_argument("--paper-fee", type=float, default=0.006)
    parser.add_argument("--reward-deadband", default="0.01")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    payload = validate_run(
        run=args.run,
        replays=args.replay,
        out=args.out,
        neural_ms=args.neural_ms,
        order_usdc=args.order_usdc,
        paper_fee=args.paper_fee,
        reward_deadband=args.reward_deadband,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
