from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .replay import FORMAT, load_recording


def _write_recording(path: Path, header: dict, quotes) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, allow_nan=False) + "\n")
        for tick, quote in enumerate(quotes):
            handle.write(
                json.dumps({"tick": tick, "quote": quote.json()}, allow_nan=False)
                + "\n"
            )
    temporary.replace(path)


def split_recording(
    replay: Path,
    out: Path,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    overwrite: bool = False,
):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")

    replay = Path(replay)
    out = Path(out)
    header, quotes = load_recording(replay)
    train_end = int(len(quotes) * train_fraction)
    validation_end = train_end + int(len(quotes) * validation_fraction)
    ranges = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(quotes)),
    }
    if any(end - start < 2 for start, end in ranges.values()):
        raise ValueError("each chronological split must contain at least two observations")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    original_history = list(header["initial_history"])
    preceding = []
    manifest = {
        "format": "evostonkfly-split-v1",
        "source": str(replay),
        "source_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
        "product": header["product"],
        "chronological": True,
        "network_execution": False,
        "splits": {},
    }
    for name, (start, end) in ranges.items():
        path = out / f"{name}.jsonl"
        history = (original_history + preceding)[-120:]
        split_header = {
            **header,
            "format": FORMAT,
            "initial_history": history,
            "source": "chronological-split",
            "parent_recording": str(replay),
            "split": name,
            "split_start": start,
            "split_end": end,
            "network_execution": False,
        }
        selected = quotes[start:end]
        _write_recording(path, split_header, selected)
        manifest["splits"][name] = {
            "path": str(path),
            "start": start,
            "end": end,
            "observations": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        preceding.extend(float((quote.bid + quote.ask) / 2) for quote in selected)

    temporary = out / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(out / "manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chronologically split a Stonkfly market recording"
    )
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = split_recording(
        args.replay,
        args.out,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
