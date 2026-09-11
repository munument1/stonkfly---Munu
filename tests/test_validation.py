import json
from pathlib import Path

from stonkfly.evolution.dataset import split_recording
from stonkfly.evolution.replay import FORMAT, load_recording
from stonkfly.evolution.validate import evaluate_baseline


def _write_recording(path: Path, count=10):
    header = {
        "format": FORMAT,
        "product": "BTC-USDC",
        "recorded_at": 1.0,
        "interval_seconds": 1.0,
        "initial_history": [99.0, 100.0],
        "source": "test",
        "network_execution": False,
    }
    rows = []
    for tick in range(count):
        price = 100 + tick
        rows.append(
            {
                "tick": tick,
                "quote": {
                    "product": "BTC-USDC",
                    "bid": str(price),
                    "ask": str(price + 0.1),
                    "timestamp": float(tick + 1),
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "price_increment": "0.01",
                    "minimum_quote": "1",
                    "minimum_base": "0.00000001",
                },
            }
        )
    path.write_text(
        "\n".join(json.dumps(row) for row in [header, *rows]) + "\n"
    )


def test_split_recording_is_chronological_without_future_history(tmp_path):
    replay = tmp_path / "source.jsonl"
    _write_recording(replay)
    manifest = split_recording(replay, tmp_path / "split")

    assert [manifest["splits"][name]["observations"] for name in (
        "train", "validation", "test"
    )] == [6, 2, 2]
    train_header, train_quotes = load_recording(tmp_path / "split" / "train.jsonl")
    validation_header, validation_quotes = load_recording(
        tmp_path / "split" / "validation.jsonl"
    )
    assert train_header["initial_history"] == [99.0, 100.0]
    assert validation_header["initial_history"][-1] == 105.05
    assert validation_quotes[0].bid == 106
    assert len(train_quotes) == 6


def test_split_rejects_splits_too_small(tmp_path):
    replay = tmp_path / "source.jsonl"
    _write_recording(replay, count=5)
    try:
        split_recording(replay, tmp_path / "split")
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("undersized holdout splits should fail")


def test_baselines_are_deterministic_and_hold_is_flat(tmp_path):
    replay = tmp_path / "source.jsonl"
    _write_recording(replay)
    hold = evaluate_baseline(replay, "hold")
    first = evaluate_baseline(replay, "random", seed=42)
    second = evaluate_baseline(replay, "random", seed=42)

    assert hold["fitness"]["return_pct"] == 0.0
    assert hold["fitness"]["trade_count"] == 0
    assert first == second
    assert evaluate_baseline(replay, "momentum")["fitness"]["trade_count"] > 0
