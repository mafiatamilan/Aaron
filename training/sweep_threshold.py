#!/usr/bin/env python3
"""Sweep the confidence threshold and record the trade-off.

Why the threshold needs a sweep rather than a guess
----------------------------------------------------
The threshold is the model's abstention point: below it, the answer is
`unknown` and nothing is dispatched. Moving it trades two errors against each
other:

* **too low** — out-of-schema requests are accepted as commands. A false
  accept can run the wrong tool.
* **too high** — real requests are rejected. A false reject is merely
  unhelpful, but it makes the assistant frustrating and the user stops asking.

A threshold is only meaningful with both numbers attached. This script
records, for one checkpoint, at every candidate threshold:

* the validation false-accept and false-reject rates (**the selection
  signal**);
* the locked intent benchmark's accuracy, macro-F1 and false-accept rate;
* the adversarial safety benchmark's model-level false-accept rate and the
  agent-level tool-dispatch rate, measured on the **v1 development** benchmark.
  The held-out v2 benchmark is deliberately not used here.

Selection is on **validation only**. The locked benchmark and the safety
benchmark are reported, never used to choose. A threshold picked on either of
them would stop being a measurement and become a fit.

The threshold is an inference-time setting, so the sweep re-scores one
checkpoint rather than retraining. That is what makes it cheap enough to run
on every candidate.

Usage:
    python3 training/sweep_threshold.py
    python3 training/sweep_threshold.py --thresholds 0.3,0.4,0.5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "aaron"
CONFIG = ROOT / "config" / "default.toml"
LOG = ROOT / "training" / "threshold-sweep.jsonl"

# The sweep uses the **v1** adversarial benchmark, which is the *development*
# set: it was used to select the release-0.3 dataset, so it is no longer
# held out. The v2 benchmark is the held-out measure and is scored once, at the
# end, by the final evaluation. Using v1 here is what keeps v2 clean.
SAFETY = ROOT / "training" / "evaluation" / "safety-benchmark-v1.jsonl"
LOCKED = ROOT / "training" / "evaluation" / "benchmark-v3.jsonl"
DATASET = ROOT / "training" / "dataset" / "current.jsonl"

DEFAULT_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


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
        # A silent failure here would produce a table of NaNs, which reads as
        # a measurement and is not one. Fail loudly instead.
        raise SystemExit(
            f"`aaron {' '.join(args)}` failed with exit {out.returncode}:\n"
            f"{out.stderr.strip()}"
        )
    return out.stdout


def number(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else float("nan")


def evaluate(threshold: float, candidate: str | None, home: str) -> dict:
    def maybe_candidate() -> list[str]:
        return ["--candidate", candidate] if candidate else []

    # --- validation: the selection signal ---------------------------
    val = run(
        ["learning", "evaluate", "--dataset", str(DATASET), "--split", "validation",
         "--threshold", str(threshold), *maybe_candidate()],
        home,
    )
    validation = {
        "accuracy": number(val, r"accuracy\s+([0-9.]+)"),
        "macro_f1": number(val, r"macro-F1\s+([0-9.]+)"),
        "false_accept_rate": number(val, r"false accept\s+([0-9.]+)"),
    }

    # --- the locked intent benchmark, for the record ------------------
    locked = run(
        ["learning", "evaluate", "--benchmark", str(LOCKED),
         "--threshold", str(threshold), *maybe_candidate()],
        home,
    )
    locked_metrics = {
        "accuracy": number(locked, r"accuracy\s+([0-9.]+)"),
        "macro_f1": number(locked, r"macro-F1\s+([0-9.]+)"),
        "false_accept_rate": number(locked, r"false accept\s+([0-9.]+)"),
    }

    # --- the adversarial safety benchmark -----------------------------
    safety = run(
        ["learning", "safety", "--benchmark", str(SAFETY),
         "--threshold", str(threshold), *maybe_candidate()],
        home,
    )
    safety_metrics = {
        "model_refused_rate": number(safety, r"refused rate\s+([0-9.]+)"),
        "model_false_accept_rate": number(safety, r"false-accept rate\s+([0-9.]+)"),
        "model_side_effecting_false_accept_rate": number(
            safety, r"side-effecting false-accept\s+([0-9.]+)"
        ),
        "agent_tool_dispatch_rate": number(safety, r"tool dispatch rate\s+([0-9.]+)"),
        "agent_mutating_action_rate": number(safety, r"mutating action rate\s+([0-9.]+)"),
    }
    return {
        "threshold": threshold,
        "validation": validation,
        "locked_benchmark": locked_metrics,
        "safety_benchmark": safety_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", help="comma-separated list")
    parser.add_argument("--candidate", help="checkpoint to score")
    parser.add_argument(
        "--home",
        default=os.environ.get("AARON_HOME", "/tmp/opencode/aaron-sweep"),
        help="state directory holding the candidate checkpoint",
    )
    args = parser.parse_args()

    if not BIN.exists():
        print(f"{BIN} not found; run `cargo build --release` first", file=sys.stderr)
        return 1

    thresholds = (
        [float(t) for t in args.thresholds.split(",")]
        if args.thresholds
        else DEFAULT_THRESHOLDS
    )

    print(
        f"{'thr':>5} | {'val acc':>8} {'val F1':>8} {'val FA':>7} | "
        f"{'lock acc':>8} {'lock F1':>8} {'lock FA':>8} | "
        f"{'safe ref':>8} {'safe FA':>8} {'safe SE':>8} | "
        f"{'agent disp':>10} {'agent mut':>9}"
    )
    print("-" * 118)
    rows = []
    for threshold in thresholds:
        row = evaluate(threshold, args.candidate, args.home)
        rows.append(row)
        with LOG.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        v = row["validation"]
        l = row["locked_benchmark"]
        s = row["safety_benchmark"]
        print(
            f"{threshold:>5.2f} | {v['accuracy']:>8.4f} {v['macro_f1']:>8.4f} "
            f"{v['false_accept_rate']:>7.4f} | "
            f"{l['accuracy']:>8.4f} {l['macro_f1']:>8.4f} {l['false_accept_rate']:>8.4f} | "
            f"{s['model_refused_rate']:>8.4f} {s['model_false_accept_rate']:>8.4f} "
            f"{s['model_side_effecting_false_accept_rate']:>8.4f} | "
            f"{s['agent_tool_dispatch_rate']:>10.4f} {s['agent_mutating_action_rate']:>9.4f}"
        )
    print(f"\nrecorded to {LOG}")
    print("Selection is on the validation columns. The benchmark columns are for")
    print("the record and must not be used to pick a threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
