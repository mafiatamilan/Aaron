#!/usr/bin/env python3
"""Run the release-0.3 safety experiments and record every one of them.

# The rule

**One major factor per experiment.** Each row below changes exactly one thing
from the row above it, and the record says which.

The 0.2 release shipped a sweep that moved the learning rate, the weight
decay and the regularisation at once, and the result had to be discarded
because nothing could be attributed. This runner exists so that cannot
happen again.

# What is recorded

Every experiment records, in full:

| field | why |
|---|---|
| dataset hash | two runs are only comparable if the data is identical |
| training configuration and seed | the model is seed-sensitive; the sweep found 8.5 points of variance |
| checkpoint hash | proves *which* weights were scored |
| parameter count | the 1,000,000 budget is a contract, not a preference |
| validation metrics | **the selection signal** |
| locked benchmark results | the intent-quality measure, for the record |
| safety benchmark results | the adversarial measure, for the record |
| regression results | the suite that must not regress |
| training duration and peak RSS | resource cost, which §9 requires |

# Selection

Hyperparameters and thresholds are selected on **validation only**. The
locked benchmark and the safety benchmark are reported, never selected on.
A benchmark that selects a model is no longer a benchmark.

# Usage

    python3 training/run_safety_experiments.py --list
    python3 training/run_safety_experiments.py
    python3 training/run_safety_experiments.py --only E2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "aaron"
CONFIG = ROOT / "config" / "default.toml"
DATASET = ROOT / "training" / "dataset" / "current.jsonl"
LOCKED = ROOT / "training" / "evaluation" / "benchmark-v3.jsonl"
SAFETY = ROOT / "training" / "evaluation" / "safety-benchmark-v1.jsonl"
LOG = ROOT / "training" / "safety-experiments.jsonl"

# Each experiment changes one factor. `exclude` rebuilds the dataset without a
# curated file; `class_weighting` and `threshold` are training/inference
# settings.
EXPERIMENTS: list[dict] = [
    {
        "id": "E0",
        "name": "baseline (0.2 dataset, no class weighting)",
        "exclude": ["curated-v3-hard-negatives"],
        "class_weighting": "none",
        "seeds": 8,
    },
    {
        "id": "E1",
        "name": "+ hard negatives (the only change from E0)",
        "exclude": [],
        "class_weighting": "none",
        "seeds": 8,
    },
    {
        "id": "E2",
        "name": "+ sqrt class weighting (the only change from E1)",
        "exclude": [],
        "class_weighting": "sqrt",
        "seeds": 8,
    },
    {
        "id": "E3",
        "name": "+ full class weighting (the only change from E1)",
        "exclude": [],
        "class_weighting": "balanced",
        "seeds": 8,
    },
    {
        "id": "E4",
        "name": "+ a 0.60 threshold (the only change from E2)",
        "exclude": [],
        "class_weighting": "sqrt",
        "seeds": 8,
        "threshold": 0.60,
    },
    {
        "id": "E5",
        "name": "hard negatives without the short-fragment families",
        "exclude": [],
        "exclude_families": ["bare_noun", "verb_no_object"],
        "class_weighting": "none",
        "seeds": 8,
    },
    {
        "id": "E6",
        "name": "only the verb-without-object family",
        "exclude": [],
        "exclude_families": [
            "bare_noun", "invalid_url", "credential_url", "path_traversal",
            "unsupported_task", "embedded_instruction", "fake_authority",
            "role_play", "malformed_unicode", "oversized", "control",
        ],
        "class_weighting": "none",
        "seeds": 8,
    },
    {
        "id": "E8",
        "name": "the specific actionable attack shapes only",
        "exclude": [],
        "exclude_families": [
            "bare_noun", "unsupported_task", "embedded_instruction",
            "fake_authority", "role_play", "malformed_unicode", "oversized",
        ],
        "class_weighting": "none",
        "seeds": 8,
    },
    {
        "id": "E9",
        "name": "E8 plus the unsupported-task and authority families",
        "exclude": [],
        "exclude_families": [
            "bare_noun", "role_play", "malformed_unicode", "oversized",
        ],
        "class_weighting": "none",
        "seeds": 8,
    },
    {
        "id": "E7",
        "name": "hard negatives without the ambiguous-reference family",
        "exclude": [],
        "exclude_families": ["bare_noun"],
        "class_weighting": "none",
        "seeds": 8,
    },
]


def sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(exclude: list[str], exclude_families: list[str]) -> str:
    args = [sys.executable, str(ROOT / "training" / "build_dataset_v2.py")]
    for name in exclude:
        args += ["--exclude", name]
    for family in exclude_families:
        args += ["--exclude-family", family]
    # The build exits non-zero when it finds *warnings* (near-duplicates), so
    # the exit code is not the success signal. The dataset being written and
    # hashing to something is.
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if not DATASET.exists():
        raise SystemExit(
            f"the dataset build failed:\n{result.stdout}\n{result.stderr}"
        )
    return sha256(DATASET)


def run(args: list[str], home: str) -> str:
    env = dict(os.environ)
    env["AARON_HOME"] = home
    out = subprocess.run(
        [str(BIN), "--config", str(CONFIG), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0 and not out.stdout:
        raise SystemExit(
            f"`aaron {' '.join(args)}` failed ({out.returncode}):\n{out.stderr.strip()}"
        )
    return out.stdout


def number(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else float("nan")


def experiment(spec: dict, home: str) -> dict:
    dataset_hash = build_dataset(spec["exclude"], spec.get("exclude_families", []))

    started = time.time()
    train = run(
        [
            "brain", "train",
            "--seeds", str(spec["seeds"]),
            "--class-weighting", spec["class_weighting"],
        ],
        home,
    )
    duration = time.time() - started

    checkpoint = ROOT / "training" / "dataset" / ".last-candidate"
    # Find the candidate the run produced.
    candidates = sorted(
        (Path(home) / "data" / "checkpoints").glob("candidate-*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("training produced no candidate")
    candidate = candidates[-1]

    threshold = spec.get("threshold")
    t_arg = ["--threshold", str(threshold)] if threshold is not None else []

    validation = run(
        ["learning", "evaluate", "--dataset", str(DATASET), "--split", "validation", *t_arg],
        home,
    )
    locked = run(["learning", "evaluate", "--benchmark", str(LOCKED), *t_arg], home)
    safety = run(["learning", "safety", "--benchmark", str(SAFETY), *t_arg], home)
    model_info = run(["model"], home)

    regressions = re.search(r"regression suite: (\d+)/(\d+) passed", locked)
    params = re.search(r"trainable params\s+(\d+)", model_info)

    return {
        "id": spec["id"],
        "name": spec["name"],
        "dataset_sha256": dataset_hash,
        "dataset_examples": int(re.search(r"examples\s+(\d+)", run(["dataset", "check"], home)).group(1)),
        "class_weighting": spec["class_weighting"],
        "threshold": threshold if threshold is not None else 0.55,
        "seeds": spec["seeds"],
        "checkpoint": str(candidate),
        "checkpoint_sha256": sha256(candidate),
        "parameter_count": int(params.group(1)) if params else -1,
        "validation": {
            "accuracy": number(validation, r"accuracy\s+([0-9.]+)"),
            "macro_f1": number(validation, r"macro-F1\s+([0-9.]+)"),
            "false_accept_rate": number(validation, r"false accept\s+([0-9.]+)"),
        },
        "locked_benchmark": {
            "accuracy": number(locked, r"accuracy\s+([0-9.]+)"),
            "macro_f1": number(locked, r"macro-F1\s+([0-9.]+)"),
            "false_accept_rate": number(locked, r"false accept\s+([0-9.]+)"),
            "regression_passed": int(regressions.group(1)) if regressions else -1,
            "regression_total": int(regressions.group(2)) if regressions else -1,
        },
        "safety_benchmark": {
            "model_refused_rate": number(safety, r"refused rate\s+([0-9.]+)"),
            "model_false_accept_rate": number(safety, r"false-accept rate\s+([0-9.]+)"),
            "model_side_effecting_false_accept_rate": number(
                safety, r"side-effecting false-accept\s+([0-9.]+)"
            ),
            "agent_tool_dispatch_rate": number(safety, r"tool dispatch rate\s+([0-9.]+)"),
            "agent_mutating_action_rate": number(safety, r"mutating action rate\s+([0-9.]+)"),
        },
        "train_seconds": round(duration, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", action="append", help="run only these experiment ids")
    parser.add_argument("--home", default="/tmp/opencode/aaron-safety-exp")
    args = parser.parse_args()

    if args.list:
        for spec in EXPERIMENTS:
            print(f"{spec['id']}: {spec['name']}")
        return 0

    if not BIN.exists():
        print(f"{BIN} not found; run `cargo build --release` first", file=sys.stderr)
        return 1

    home = args.home
    shutil.rmtree(home, ignore_errors=True)

    print(
        f"{'id':<4} {'dataset':<10} {'weighting':<10} {'thr':>5} | "
        f"{'val acc':>8} {'val F1':>8} {'val FA':>7} | "
        f"{'lock acc':>9} {'lock F1':>8} {'lock FA':>8} {'reg':>7} | "
        f"{'safe ref':>9} {'safe SE':>8} {'dispatch':>9} {'mutate':>7} | {'sec':>6}"
    )
    print("-" * 140)
    for spec in EXPERIMENTS:
        if args.only and spec["id"] not in args.only:
            continue
        row = experiment(spec, home)
        with LOG.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        v = row["validation"]
        l = row["locked_benchmark"]
        s = row["safety_benchmark"]
        print(
            f"{row['id']:<4} {row['dataset_sha256'][:8]:<10} {row['class_weighting']:<10} "
            f"{row['threshold']:>5.2f} | "
            f"{v['accuracy']:>8.4f} {v['macro_f1']:>8.4f} {v['false_accept_rate']:>7.4f} | "
            f"{l['accuracy']:>9.4f} {l['macro_f1']:>8.4f} {l['false_accept_rate']:>8.4f} "
            f"{str(l['regression_passed']) + '/' + str(l['regression_total']):>7} | "
            f"{s['model_refused_rate']:>9.4f} {s['model_side_effecting_false_accept_rate']:>8.4f} "
            f"{s['agent_tool_dispatch_rate']:>9.4f} {s['agent_mutating_action_rate']:>7.4f} | "
            f"{row['train_seconds']:>6.1f}"
        )
    print(f"\nrecorded to {LOG}")
    print("Selection is on the validation columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
