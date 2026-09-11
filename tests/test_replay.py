import json
from pathlib import Path

import pytest

from stonkfly.evolution.replay import FORMAT, ReplayMarket, load_recording, recording_info


def _write_recording(path: Path):
    header = {
        "format": FORMAT,
        "product": "BTC-USDC",
        "recorded_at": 1.0,
        "interval_seconds": 60.0,
        "initial_history": [59000.0, 59500.0, 60000.0],
        "source": "coinbase-public",
        "network_execution": False,
    }
    rows = [
        {
            "tick": 0,
            "quote": {
                "product": "BTC-USDC",
                "bid": "60000",
                "ask": "60010",
                "timestamp": 10.0,
                "base_increment": "0.00000001",
                "quote_increment": "0.01",
                "price_increment": "0.01",
                "minimum_quote": "1",
                "minimum_base": "0.00000001",
            },
        },
        {
            "tick": 1,
            "quote": {
                "product": "BTC-USDC",
                "bid": "60100",
                "ask": "60110",
                "timestamp": 70.0,
                "base_increment": "0.00000001",
                "quote_increment": "0.01",
                "price_increment": "0.01",
                "minimum_quote": "1",
                "minimum_base": "0.00000001",
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(x) for x in [header, *rows]) + "\n",
        encoding="utf-8",
    )


def test_replay_is_repeatable(tmp_path):
    path = tmp_path / "market.jsonl"
    _write_recording(path)

    first = ReplayMarket(path, product="BTC-USDC")
    second = ReplayMarket(path, product="BTC-USDC")
    a = first.snapshot()["BTC-USDC"]
    b = second.snapshot()["BTC-USDC"]
    assert a.bid == b.bid
    assert a.ask == b.ask
    first.record({"BTC-USDC": a})
    second.record({"BTC-USDC": b})
    assert first.history == second.history

    info = recording_info(path)
    assert info["product"] == "BTC-USDC"
    assert info["observations"] == 2
    assert len(info["sha256"]) == 64


def test_replay_rejects_wrong_product(tmp_path):
    path = tmp_path / "market.jsonl"
    _write_recording(path)
    try:
        ReplayMarket(path, product="ETH-USDC")
    except ValueError as exc:
        assert "replay product" in str(exc)
    else:
        raise AssertionError("wrong replay product should fail")

def test_replay_rejects_corrupted_tick_sequence(tmp_path):
    path = tmp_path / "market.jsonl"
    _write_recording(path)
    rows = path.read_text().splitlines()
    payload = json.loads(rows[2])
    payload["tick"] = 7
    rows[2] = json.dumps(payload)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="tick sequence"):
        load_recording(path)


def test_replay_rejects_nonfinite_history(tmp_path):
    path = tmp_path / "market.jsonl"
    _write_recording(path)
    rows = path.read_text().splitlines()
    header = json.loads(rows[0])
    header["initial_history"][0] = float("nan")
    rows[0] = json.dumps(header)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="initial market history"):
        load_recording(path)


def test_replay_rejects_nonincreasing_timestamps(tmp_path):
    path = tmp_path / "market.jsonl"
    _write_recording(path)
    rows = path.read_text().splitlines()
    payload = json.loads(rows[2])
    payload["quote"]["timestamp"] = 10.0
    rows[2] = json.dumps(payload)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="timestamps"):
        load_recording(path)

