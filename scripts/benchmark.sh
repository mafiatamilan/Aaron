#!/usr/bin/env bash
#
# Reproducible benchmark run.
#
# Produces the numbers in `docs/` and prints everything needed to judge them:
# the machine, the configuration, the sample counts and the tolerances.
#
# Usage:
#   ./scripts/benchmark.sh                 # default workload
#   ./scripts/benchmark.sh --iterations 500
#   ./scripts/benchmark.sh --json out.json
#   ./scripts/benchmark.sh --quick         # skip the long stress run

set -uo pipefail

RED=$'\033[31m'; GREEN=$'\033[32m'; BLUE=$'\033[34m'; DIM=$'\033[2m'; RESET=$'\033[0m'

ITERATIONS=200
STRESS_ITERATIONS=20000
JSON_OUT=""
QUICK=0
PROFILE="debug"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --stress-iterations) STRESS_ITERATIONS="$2"; shift 2 ;;
        --json) JSON_OUT="$2"; shift 2 ;;
        --release) PROFILE="release"; shift ;;
        --quick) QUICK=1; shift ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf '%sunknown option: %s%s\n' "$RED" "$1" "$RESET" >&2; exit 64 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

printf '%s%sAARON benchmark%s\n' "$BLUE" "$1" "$RESET" 2>/dev/null
printf 'build profile   %s\n' "$PROFILE"
printf 'iterations      %s\n' "$ITERATIONS"

if [[ "$PROFILE" == "release" ]]; then
    cargo build --release >/dev/null 2>&1 || { echo "release build failed"; exit 1; }
    AARON="$ROOT/target/release/aaron"
else
    cargo build >/dev/null 2>&1 || { echo "build failed"; exit 1; }
    AARON="$ROOT/target/debug/aaron"
fi

if [[ ! -x "$AARON" ]]; then
    echo "no aaron binary at $AARON"; exit 1
fi

CONFIG="$ROOT/config/default.toml"

# --------------------------------------------------------------- environment
printf '\n%s== environment ==%s\n' "$BLUE" "$RESET"
"$ROOT/scripts/check-environment.sh" 2>/dev/null | sed -n '1,20p' || true

# ------------------------------------------------------------------- latency
printf '\n%s== latency and memory ==%s\n' "$BLUE" "$RESET"
if [[ -n "$JSON_OUT" ]]; then
    "$AARON" --config "$CONFIG" --json benchmark --iterations "$ITERATIONS" \
        | tee "$JSON_OUT"
    printf '\nwrote %s\n' "$JSON_OUT"
else
    "$AARON" --config "$CONFIG" benchmark --iterations "$ITERATIONS"
fi

# ------------------------------------------------------------------- stress
if (( ! QUICK )); then
    printf '\n%s== stability ==%s\n' "$BLUE" "$RESET"
    printf '%sRSS tolerance: max(2 MiB, 1%% of baseline) between the first and last%s\n' "$DIM"
    printf '%sthirds of the post-warm-up samples. See docs/resources.md.%s\n' "$DIM" "$RESET"
    "$AARON" --config "$CONFIG" resources stress --iterations "$STRESS_ITERATIONS"
    stress_rc=$?
else
    printf '\n%s== stability ==%s (skipped: --quick)\n' "$BLUE" "$RESET"
    stress_rc=0
fi

# ------------------------------------------------------------ resource status
printf '\n%s== resource status ==%s\n' "$BLUE" "$RESET"
"$AARON" --config "$CONFIG" resources status

printf '\n%s== model ==%s\n' "$BLUE" "$RESET"
"$AARON" --config "$CONFIG" model

if (( stress_rc != 0 )); then
    printf '\n%sA workload showed unbounded growth. See `aaron resources stress`.%s\n' \
        "$RED" "$RESET"
    exit 1
fi

printf '\n%sDone. Numbers above were measured on this machine, just now.%s\n' \
    "$GREEN" "$RESET"
printf '%sDo not quote them for a different machine; re-run there.%s\n' "$DIM" "$RESET"