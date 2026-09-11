import json

from stonkfly.evolution.dashboard import build_snapshot


def _row(fingerprint, score, return_pct, drawdown, seconds=1.0):
    return {
        "fingerprint": fingerprint,
        "fitness": {
            "score": score,
            "return_pct": return_pct,
            "max_drawdown_pct": drawdown,
            "trade_count": 2,
        },
        "evaluation_seconds": seconds,
        "genome": {"eta": 0.001},
    }


def test_snapshot_summarizes_generations_and_progress(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"inheritance": "darwinian", "generations": 1000})
    )
    (tmp_path / "progress.json").write_text(
        json.dumps({"status": "running", "completed_generations": 1})
    )
    (tmp_path / "generation-0000.json").write_text(
        json.dumps(
            {
                "generation": 0,
                "ranked": [
                    _row("best", 3.0, 2.0, 0.5, 2.0),
                    _row("other", 1.0, -1.0, 2.0, 3.0),
                ],
            }
        )
    )

    mode = build_snapshot(tmp_path)["modes"]["darwinian"]
    assert mode["recorded_generations"] == 1
    assert mode["series"][0]["best_score"] == 3.0
    assert mode["series"][0]["mean_score"] == 2.0
    assert mode["series"][0]["evaluation_seconds"] == 5.0
    assert mode["latest_ranked"][0]["fingerprint"] == "best"
    assert mode["progress"]["status"] == "running"


def test_snapshot_detects_both_modes_and_skips_partial_json(tmp_path):
    for mode in ("darwinian", "lamarckian"):
        path = tmp_path / mode
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps({"inheritance": mode, "generations": 10})
        )
        (path / "generation-0000.json").write_text(
            json.dumps({"generation": 0, "ranked": [_row(mode, 1, 0, 0)]})
        )
        (path / "generation-0001.json").write_text("{")

    snapshot = build_snapshot(tmp_path)
    assert set(snapshot["modes"]) == {"darwinian", "lamarckian"}
    assert all(
        mode["recorded_generations"] == 1
        for mode in snapshot["modes"].values()
    )
