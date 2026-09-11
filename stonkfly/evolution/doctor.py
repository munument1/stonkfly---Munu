from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from ..data import verify
from .replay import recording_info


def _result(name: str, status: str, detail: str):
    return {"check": name, "status": status, "detail": detail}


def run_doctor(replay: Path | None = None):
    rows = []

    if sys.version_info >= (3, 11):
        rows.append(_result("python", "ok", platform.python_version()))
    else:
        rows.append(_result("python", "fail", f"Python {platform.python_version()} detected; need >=3.11"))

    if os.name == "nt":
        rows.append(
            _result(
                "platform",
                "warn",
                "Native Windows detected. The upstream neural kernel build uses Unix-style c++/-fPIC/.so conventions; WSL2/Linux is recommended for the first end-to-end run.",
            )
        )
    else:
        rows.append(_result("platform", "ok", platform.platform()))

    compilers = [name for name in ("c++", "g++", "clang++") if shutil.which(name)]
    if compilers:
        rows.append(_result("cxx", "ok", ", ".join(compilers)))
    else:
        rows.append(
            _result(
                "cxx",
                "fail",
                "No c++/g++/clang++ executable found on PATH; the native neural kernel cannot build.",
            )
        )

    try:
        info = verify()
        rows.append(
            _result(
                "dataset",
                "ok",
                f"{info['release']}: {info['neurons']} neurons, {info['directed_edges']} directed edges",
            )
        )
    except Exception as exc:
        rows.append(
            _result(
                "dataset",
                "fail",
                f"{type(exc).__name__}: {exc}. Run `stonkfly prepare` then `stonkfly verify`.",
            )
        )

    if replay is not None:
        try:
            info = recording_info(Path(replay))
            rows.append(
                _result(
                    "replay",
                    "ok",
                    f"{info['product']}, {info['observations']} observations, sha256={info['sha256']}",
                )
            )
        except Exception as exc:
            rows.append(_result("replay", "fail", f"{type(exc).__name__}: {exc}"))

    failed = [r for r in rows if r["status"] == "fail"]
    warned = [r for r in rows if r["status"] == "warn"]
    return {
        "ready": not failed,
        "warnings": len(warned),
        "checks": rows,
        "recommended_first_run": (
            "python -m stonkfly.evolution --inheritance both --population 2 "
            "--generations 2 --steps 6 --neural-ms 50 --out runs/smoke"
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m stonkfly.evolution.doctor",
        description="Check whether the local machine is ready for an EvoStonkFly end-to-end assay.",
    )
    p.add_argument("--replay", type=Path, help="Optional recorded Coinbase-public JSONL to validate")
    a = p.parse_args(argv)
    result = run_doctor(a.replay)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
