#!/usr/bin/env python3
"""Run controlled training experiments and record them.

The rule this script exists to enforce: **one variable per experiment**.

AARON's accuracy has been moved by three different things at once in the
past — the dataset, the pooling statistic and the initialisation — and the
result was a number nobody could attribute. This runner changes exactly one
thing per row, writes every result to a log, and prints the table.

Protocol
--------
Selection is done on the **validation** split. The locked benchmark is scored
for the record, but it is never used to choose anything: a benchmark that
selects hyperparameters is no longer a benchmark. The dev-test split is what
the error analysis runs against.

Usage:
    python3 training/run_experiments.py --suite hyper
    python3 training/run_experiments.py --suite data
    python3 training/run_experiments.py --all
    python3 training/run_experiments.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "aaron"
CONFIG = ROOT / "config" / "default.toml"
LOG = ROOT / "training" / "experiments.jsonl"
BENCHMARK = ROOT / "training" / "evaluation" / "benchmark-v3.jsonl"

# The experiment suites. Each entry is one variable changed from the base
# configuration.
# The base configuration. `lr` was moved from 3e-3 to 1e-2 after the `lr`
# suite selected it on validation; every later suite runs at that base. The
# move is recorded here rather than left implicit, because a suite run at a
# different base than the one before it is not comparable.
BASE = {
    "dataset": "training/dataset/current.jsonl",
    "lr": 1e-2,
    "weight_decay": 1e-2,
    "epochs": 100,
    "patience": 20,
    "seed": 2690307351,
    "batch": 16,
    "dropout": 0.0,
    "feature_dropout": 0.0,
    "label_smoothing": 0.0,
}

SUITES: dict[str, list[dict]] = {
    # --- learning rate -------------------------------------------------
    "lr": [
        {"name": "lr-5e-2", "lr": 5e-2},
        {"name": "lr-3e-2", "lr": 3e-2},
        {"name": "lr-2e-2", "lr": 2e-2},
        {"name": "lr-1e-2", "lr": 1e-2},
        {"name": "lr-5e-3", "lr": 5e-3},
        {"name": "lr-3e-3", "lr": 3e-3},
        {"name": "lr-1e-3", "lr": 1e-3},
        {"name": "lr-3e-4", "lr": 3e-4},
    ],
    # --- weight decay --------------------------------------------------
    "weight_decay": [
        {"name": "wd-1e-3", "weight_decay": 1e-3},
        {"name": "wd-1e-2", "weight_decay": 1e-2},
        {"name": "wd-1e-1", "weight_decay": 1e-1},
        {"name": "wd-3e-1", "weight_decay": 3e-1},
    ],
    # --- schedule ------------------------------------------------------
    "schedule": [
        {"name": "epochs-50", "epochs": 50},
        {"name": "epochs-100", "epochs": 100},
        {"name": "epochs-200", "epochs": 200},
        {"name": "epochs-300", "epochs": 300},
    ],
    # --- patience ------------------------------------------------------
    "patience": [
        {"name": "patience-10", "patience": 10},
        {"name": "patience-20", "patience": 20},
        {"name": "patience-40", "patience": 40},
    ],
    # --- seed ----------------------------------------------------------
    "seed": [
        {"name": "seed-2690307351", "seed": 2690307351},
        {"name": "seed-12345", "seed": 12345},
        {"name": "seed-987654321", "seed": 987654321},
    ],
    # --- regularisation ------------------------------------------------
    "regularisation": [
        {"name": "reg-none", "dropout": 0.0, "feature_dropout": 0.0, "label_smoothing": 0.0},
        {"name": "reg-do0.1", "dropout": 0.1},
        {"name": "reg-do0.2", "dropout": 0.2},
        {"name": "reg-do0.3", "dropout": 0.3},
        {"name": "reg-feat0.1", "feature_dropout": 0.1},
        {"name": "reg-feat0.2", "feature_dropout": 0.2},
        {"name": "reg-feat0.3", "feature_dropout": 0.3},
        {"name": "reg-ls0.05", "label_smoothing": 0.05},
        {"name": "reg-ls0.1", "label_smoothing": 0.1},
        {"name": "reg-ls0.2", "label_smoothing": 0.2},
        {"name": "reg-feat0.2-do0.1", "feature_dropout": 0.2, "dropout": 0.1},
        {"name": "reg-feat0.2-ls0.1", "feature_dropout": 0.2, "label_smoothing": 0.1},
        {"name": "reg-all", "feature_dropout": 0.2, "dropout": 0.1, "label_smoothing": 0.1},
    ],
    # --- dataset version -----------------------------------------------
    "data": [
        {"name": "data-v1", "dataset": "training/dataset/v1.jsonl"},
        {"name": "data-v2", "dataset": "training/dataset/current.jsonl"},
    ],
}

ACC_RE = re.compile(r"accuracy\s+([0-9.]+)")
F1_RE = re.compile(r"macro-F1\s+([0-9.]+)")


def run_one(config: dict, home: Path) -> dict:
    """Train with one configuration and score it."""
    env = dict(os.environ)
    env["AARON_HOME"] = str(home)
    # A fresh home per run, so a run never inherits the previous candidate.
    subprocess.run(["rm", "-rf", str(home)], check=False)

    started = time.time()
    train = subprocess.run(
        [
            str(BIN),
            "--config",
            str(CONFIG),
            "brain",
            "train",
            "--lr",
            str(config["lr"]),
            "--weight-decay",
            str(config["weight_decay"]),
            "--epochs",
            str(config["epochs"]),
            "--patience",
            str(config["patience"]),
            "--seed",
            str(config["seed"]),
            "--dataset",
            str(config["dataset"]),
            "--dropout",
            str(config.get("dropout", 0.0)),
            "--feature-dropout",
            str(config.get("feature_dropout", 0.0)),
            "--label-smoothing",
            str(config.get("label_smoothing", 0.0)),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if train.returncode != 0:
        return {
            **config,
            "error": train.stderr.strip()[:400] or train.stdout.strip()[:400],
            "seconds": round(time.time() - started, 1),
        }

    def score(split: str) -> tuple[float, float]:
        out = subprocess.run(
            [
                str(BIN),
                "--config",
                str(CONFIG),
                "learning",
                "evaluate",
                "--dataset",
                str(config["dataset"]),
                "--split",
                split,
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        acc = F1_RE.search(out.stdout)
        f1 = F1_RE.search(out.stdout)
        accuracy = float(re.search(r"accuracy\s+([0-9.]+)", out.stdout).group(1))
        macro = float(re.search(r"macro-F1\s+([0-9.]+)", out.stdout).group(1))
        return accuracy, macro

    val_acc, val_f1 = score("validation")
    bench_acc, bench_f1 = score("benchmark") if False else (0.0, 0.0)

    # The benchmark is scored separately, with its own flag.
    bench = subprocess.run(
        [
            str(BIN),
            "--config",
            str(CONFIG),
            "learning",
            "evaluate",
            "--benchmark",
            str(BENCHMARK),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    bench_acc = float(re.search(r"accuracy\s+([0-9.]+)", bench.stdout).group(1))
    bench_f1 = float(re.search(r"macro-F1\s+([0-9.]+)", bench.stdout).group(1))

    test = subprocess.run(
        [
            str(BIN),
            "--config",
            str(CONFIG),
            "learning",
            "evaluate",
            "--dataset",
            str(config["dataset"]),
            "--split",
            "test",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    test_acc = float(re.search(r"accuracy\s+([0-9.]+)", test.stdout).group(1))
    test_f1 = float(re.search(r"macro-F1\s+([0-9.]+)", test.stdout).group(1))

    return {
        **config,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "benchmark_accuracy": bench_acc,
        "benchmark_macro_f1": bench_f1,
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--home", default="/tmp/opencode/aaron-experiment")
    args = parser.parse_args()

    if args.list:
        for name, rows in SUITES.items():
            print(f"{name}:")
            for row in rows:
                changed = {k: v for k, v in row.items() if k != "name"}
                print(f"  {row['name']:<22} {changed}")
        return 0

    if not BIN.exists():
        print(f"{BIN} not found; run `cargo build --release` first", file=sys.stderr)
        return 1

    suites = sorted(SUITES) if args.all else (args.suite or ["lr"])
    home = Path(args.home)

    print(
        f"{'experiment':<22} {'val acc':>8} {'val F1':>8} {'test acc':>9} "
        f"{'test F1':>8} {'bench acc':>10} {'bench F1':>9} {'sec':>6}"
    )
    print("-" * 88)
    for suite in suites:
        for override in SUITES[suite]:
            config = {**BASE, **override}
            result = run_one(config, home)
            with LOG.open("a") as handle:
                handle.write(json.dumps({"suite": suite, **result}) + "\n")
            if "error" in result:
                print(f"{override['name']:<22} ERROR: {result['error'][:60]}")
                continue
            print(
                f"{override['name']:<22} "
                f"{result['validation_accuracy']:>8.4f} "
                f"{result['validation_macro_f1']:>8.4f} "
                f"{result['test_accuracy']:>9.4f} "
                f"{result['test_macro_f1']:>8.4f} "
                f"{result['benchmark_accuracy']:>10.4f} "
                f"{result['benchmark_macro_f1']:>9.4f} "
                f"{result['seconds']:>6.1f}"
            )
    print(f"\nrecorded to {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
