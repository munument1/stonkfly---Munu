from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import D
from ..market import Quote

FORMAT = "evostonkfly-market-v1"


def _decode_quote(payload: dict) -> Quote:
    required = {
        "product",
        "bid",
        "ask",
        "timestamp",
        "base_increment",
        "quote_increment",
        "price_increment",
        "minimum_quote",
        "minimum_base",
    }
    if set(payload) != required:
        raise ValueError("recorded quote schema mismatch")
    return Quote(
        payload["product"],
        D(payload["bid"]),
        D(payload["ask"]),
        float(payload["timestamp"]),
        D(payload["base_increment"]),
        D(payload["quote_increment"]),
        D(payload["price_increment"]),
        D(payload["minimum_quote"]),
        D(payload["minimum_base"]),
    )


def load_recording(path: Path):
    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("recording must contain a header and at least two observations")
    header = json.loads(lines[0])
    if header.get("format") != FORMAT:
        raise ValueError("unsupported market recording format")
    product = header.get("product")
    history = header.get("initial_history")
    if product not in {"BTC-USDC", "ETH-USDC", "SOL-USDC"}:
        raise ValueError("unsupported recorded product")
    if not isinstance(history, list) or len(history) < 2:
        raise ValueError("recording lacks initial market history")
    history = [float(v) for v in history]
    if any(v <= 0 for v in history):
        raise ValueError("invalid initial market history")

    quotes = []
    for expected_tick, line in enumerate(lines[1:]):
        row = json.loads(line)
        if row.get("tick") != expected_tick or set(row) != {"tick", "quote"}:
            raise ValueError("recording tick sequence mismatch")
        quote = _decode_quote(row["quote"])
        if quote.product != product:
            raise ValueError("recording contains a different product")
        quotes.append(quote)
    return {**header, "initial_history": history}, quotes


def recording_info(path: Path):
    path = Path(path)
    header, quotes = load_recording(path)
    return {
        "format": header["format"],
        "product": header["product"],
        "observations": len(quotes),
        "recorded_at": header.get("recorded_at"),
        "interval_seconds": header.get("interval_seconds"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class ReplayMarket:
    """Read-only deterministic replay of recorded Coinbase public observations."""

    def __init__(self, path: Path, product: str | None = None):
        self.path = Path(path)
        header, self._quotes = load_recording(self.path)
        self.product = header["product"]
        if product is not None and product != self.product:
            raise ValueError(
                f"replay product is {self.product}, requested experiment product is {product}"
            )
        self.products = (self.product,)
        self.history = {self.product: list(header["initial_history"])}
        self.tick = 0

    @property
    def length(self):
        return len(self._quotes)

    def snapshot(self):
        if self.tick >= len(self._quotes):
            raise RuntimeError("market replay exhausted")
        quote = self._quotes[self.tick]
        self.tick += 1
        return {self.product: quote}

    def record(self, quotes):
        if set(quotes) != {self.product}:
            raise ValueError("replay record product mismatch")
        q = quotes[self.product]
        self.history[self.product].append(float((q.bid + q.ask) / 2))
        self.history[self.product] = self.history[self.product][-120:]
