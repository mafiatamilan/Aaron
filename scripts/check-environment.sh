#!/usr/bin/env bash
#
# AARON environment check.
#
# Reports what is installed, what is missing, and what to do about it. Never
# installs anything and never modifies the system: run it before
# `install-arch.sh` to see what that script would do.
#
# Exit codes:
#   0  the toolchain is ready to build
#   1  something required is missing (the report says what)
#   2  optional pieces are missing (AARON still builds and runs in text mode)

set -uo pipefail

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; DIM=$'\033[2m'; RESET=$'\033[0m'

# Counters for the verdict. `required_ok` used to sit here and was never
# read: shellcheck caught it, and it was genuinely dead.
required_missing=0
optional_missing=0

section() { printf '\n%s== %s ==%s\n' "$BLUE" "$1" "$RESET"; }

have() { command -v "$1" >/dev/null 2>&1; }

# check <label> <required|optional> <hint>
check() {
    local label="$1" kind="$2" hint="$3"
    local marker colour
    if have "$label"; then
        marker="ok"; colour="$GREEN"
    elif [[ "$kind" == required ]]; then
        marker="MISSING"; colour="$RED"; required_missing=$((required_missing + 1))
    else
        marker="not installed"; colour="$YELLOW"; optional_missing=$((optional_missing + 1))
    fi
    printf '  %-28s %s%-24s%s\n' "$label" "$colour" "$marker" "$RESET"
    if [[ "$marker" != "ok" ]]; then
        printf '  %-28s %s%s%s\n' "" "$DIM" "$hint" "$RESET"
    fi
}

section "system"
printf '  %-28s %s\n' "kernel" "$(uname -sr)"
if [[ -r /etc/os-release ]]; then
    printf '  %-28s %s\n' "distribution" "$(. /etc/os-release && echo "$PRETTY_NAME")"
fi
printf '  %-28s %s\n' "cpus" "$(nproc 2>/dev/null || echo '?')"
if [[ -r /proc/meminfo ]]; then
    printf '  %-28s %s\n' "memory" \
        "$(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
fi

section "toolchain (required to build)"
check rustc required      "install with: pacman -S rust"
check cargo required      "install with: pacman -S rust"
check pkg-config required "install with: pacman -S pkgconf"
if have rustc; then
    printf '  %-28s %s\n' "" "${DIM}$(rustc --version)${RESET}"
fi
if have cargo; then
    printf '  %-28s %s\n' "" "${DIM}$(cargo --version)${RESET}"
fi

section "build dependencies"
# ALSA headers are needed by cpal; without them, build with
# --no-default-features to get a text-only binary.
if have pkg-config && pkg-config --exists alsa 2>/dev/null; then
    printf '  %-28s %sok%s\n' "alsa (dev)" "$GREEN" "$RESET"
else
    printf '  %-28s %sno headers%s\n' "alsa (dev)" "$YELLOW" "$RESET"
    printf '  %-28s %s%s%s\n' "" "$DIM" \
        "install alsa-lib, or build text-only: cargo build --no-default-features" "$RESET"
    optional_missing=$((optional_missing + 1))
fi
printf '  %-28s %s%s%s\n' "c toolchain" \
    "$(have cc && echo "${GREEN}ok${RESET}" || echo "${RED}missing${RESET}")" "" "$RESET"
if ! have cc; then
    printf '  %-28s %s%s%s\n' "" "$DIM" "install with: pacman -S base-devel" "$RESET"
    required_missing=$((required_missing + 1))
fi

section "audio (optional — needed for voice input)"
check pactl     optional "install with: pacman -S pipewire-pulse (or pulseaudio)"
check arecord   optional "install with: pacman -S alsa-utils"
if [[ -d /dev/snd ]]; then
    printf '  %-28s %sok%s\n' "/dev/snd" "$GREEN" "$RESET"
else
    printf '  %-28s %sno sound devices%s\n' "/dev/snd" "$YELLOW" "$RESET"
fi

section "speech (optional — each model is separate and user-installed)"
check whisper-cli optional "install with: pacman -S whisper.cpp, then set asr.model_path"
check espeak-ng   optional "install with: pacman -S espeak-ng"
check spd-say     optional "install with: pacman -S speech-dispatcher"

section "nvidia (optional — AARON's own model runs on CPU)"
check nvidia-smi optional "install with: pacman -S nvidia"
check nvcc       optional "install with: pacman -S cuda"
if have nvidia-smi; then
    printf '  %-28s %s\n' "" "${DIM}$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)${RESET}"
    printf '  %-28s %s\n' "" \
        "${DIM}AARON does not use CUDA for its own 1M-parameter model;${RESET}"
    printf '  %-28s %s\n' "" \
        "${DIM}whisper.cpp may use it if you build it with CUDA.${RESET}"
fi

section "source tree"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '  %-28s %s\n' "workspace" "$root"
for f in Cargo.toml config/default.toml config/permissions.toml; do
    if [[ -f "$root/$f" ]]; then
        printf '  %-28s %sok%s\n' "$f" "$GREEN" "$RESET"
    else
        printf '  %-28s %smissing%s\n' "$f" "$RED" "$RESET"
        required_missing=$((required_missing + 1))
    fi
done
if [[ -f "$root/models/brain/aaron-intent-1m.ckpt" ]]; then
    printf '  %-28s %sinstalled%s\n' "intent checkpoint" "$GREEN" "$RESET"
else
    printf '  %-28s %snot installed%s\n' "intent checkpoint" "$YELLOW" "$RESET"
    printf '  %-28s %s%s%s\n' "" "$DIM" \
        "AARON uses the deterministic router until you run: aaron brain train" "$RESET"
fi

section "verdict"
if (( required_missing > 0 )); then
    printf '  %s%d required item(s) missing. AARON cannot be built yet.%s\n' \
        "$RED" "$required_missing" "$RESET"
    exit 1
fi
printf '  %sToolchain is ready.%s\n' "$GREEN" "$RESET"
if (( optional_missing > 0 )); then
    printf '  %s%d optional item(s) missing.%s\n' "$YELLOW" "$optional_missing" "$RESET"
    printf '  Text-only AARON works now; voice needs the items listed above.\n'
    exit 2
fi
printf '  Everything is installed.\n'
exit 0