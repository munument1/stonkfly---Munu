from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ..market import CoinbaseMarket
from .replay import FORMAT


def record_public_market(
    *,
    out: Path,
    product: str,
    steps: int,
    interval_seconds: float,
    overwrite: bool = False,
):
    if product not in {"BTC-USDC", "ETH-USDC", "SOL-USDC"}:
        raise ValueError("unsupported product")
    if steps < 2:
        raise ValueError("record at least two observations")
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be at least 1")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    if (out.exists() or partial.exists()) and not overwrite:
        raise FileExistsError("recording target already exists; use --overwrite")
    if overwrite:
        for path in (out, partial):
            if path.exists():
                path.unlink()

    market = CoinbaseMarket((product,))
    first = market.snapshot()
    header = {
        "format": FORMAT,
        "product": product,
        "recorded_at": time.time(),
        "interval_seconds": interval_seconds,
        "initial_history": list(market.history[product]),
        "source": "coinbase-public",
        "network_execution": False,
    }

    with partial.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        for tick in range(steps):
            started = time.monotonic()
            quotes = first if tick == 0 else market.snapshot()
            quote = quotes[product]
            row = {"tick": tick, "quote": quote.json()}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            market.record(quotes)
            print(
                json.dumps(
                    {
                        "tick": tick,
                        "product": product,
                        "bid": str(quote.bid),
                        "ask": str(quote.ask),
                    }
                ),
                flush=True,
            )
            if tick + 1 < steps:
                remaining = interval_seconds - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)

    partial.replace(out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m stonkfly.evolution.record",
        description="Record Coinbase public market observations for deterministic EvoStonkFly replay.",
    )
    p.add_argument(
        "--product",
        default="BTC-USDC",
        choices=["BTC-USDC", "ETH-USDC", "SOL-USDC"],
    )
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    path = record_public_market(
        out=a.out,
        product=a.product,
        steps=a.steps,
        interval_seconds=a.interval,
        overwrite=a.overwrite,
    )
    print(json.dumps({"recording": str(path), "network_execution": False}, indent=2))


if __name__ == "__main__":
    main()
