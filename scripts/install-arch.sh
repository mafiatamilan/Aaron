#!/usr/bin/env bash
#
# Install AARON's dependencies and build it.
#
# Safety rules this script obeys, without exception:
#   * It prints every package before installing anything.
#   * It asks before touching pacman. `--yes` is the only way to skip the prompt.
#   * It never uses an AUR helper. AUR packages are offered as instructions, not
#     installed automatically.
#   * It never runs as root. If it detects uid 0 it stops and tells you to run
#     it as your normal user.
#   * It never changes system configuration: no sudoers edits, no firewall
#     rules, no boot settings, no kernel modules, no service enablement.
#
# Usage:
#   ./scripts/install-arch.sh                 # check, explain, ask, install
#   ./scripts/install-arch.sh --check-only    # print the plan and stop
#   ./scripts/install-arch.sh --yes           # non-interactive (for CI)
#   ./scripts/install-arch.sh --no-install    # build only, install nothing

set -uo pipefail

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

CHECK_ONLY=0
ASSUME_YES=0
DO_INSTALL=1

for arg in "$@"; do
    case "$arg" in
        --check-only) CHECK_ONLY=1 ;;
        --yes|-y)     ASSUME_YES=1 ;;
        --no-install) DO_INSTALL=0 ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            printf '%sunknown option: %s%s\n' "$RED" "$arg" "$RESET" >&2
            exit 64
            ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { printf '%s%s%s\n' "$RED" "$1" "$RESET" >&2; exit 1; }
note() { printf '%s%s%s\n' "$BLUE" "$1" "$RESET"; }
warn() { printf '%s%s%s\n' "$YELLOW" "$1" "$RESET"; }
ok()   { printf '%s%s%s\n' "$GREEN" "$1" "$RESET"; }

# --------------------------------------------------------------- preconditions

if [[ ${EUID} -eq 0 ]]; then
    die "Do not run this script as root.

  AARON refuses to execute tools as root, and an installer that needs root
  would defeat that. Run it as your normal user; it will call sudo itself if
  and only if you approve the package list."
fi

if [[ "$(uname -s)" != Linux ]]; then
    die "This installer is for Arch Linux. Other distributions are not supported."
fi

if [[ ! -r /etc/os-release ]]; then
    die "/etc/os-release is unreadable; cannot confirm this is Arch Linux."
fi
# shellcheck disable=SC1091
. /etc/os-release
if [[ "${ID:-}" != arch && "${ID_LIKE:-}" != *arch* ]]; then
    die "This looks like ${PRETTY_NAME:-an unknown distribution}, not Arch Linux."
fi

have() { command -v "$1" >/dev/null 2>&1; }

# ------------------------------------------------------------------ the plan

# Packages split by whether AARON can build and run without them.
REQUIRED_PKGS=(rust pkgconf base-devel)
# Optional pacman packages. AARON builds and runs in text mode without any of
# them; each one only enables a voice or capture feature.
OPTIONAL_PKGS=(
    alsa-lib            # ALSA headers and library (needed by cpal even for text builds' hardware probing)
    alsa-utils          # aplay, for playing rendered WAVs
    pipewire-pulse      # PulseAudio compatibility for PipeWire desktops
    libpipewire         # PipeWire development headers (libpipewire-0.3, libspa-0.2) for the capture-pipewire feature
    espeak-ng           # formant TTS, no model file
    speech-dispatcher   # spd-say TTS backend
)

# Packages commonly available from the AUR. Never installed automatically:
# there is no AUR helper invocation anywhere in this script.
AUR_PKGS=("whisper.cpp  (voice input; also available as 'whisper.cpp' in some repos)")

missing_required=()
for p in "${REQUIRED_PKGS[@]}"; do
    if ! pacman -Qq "$p" >/dev/null 2>&1; then
        missing_required+=("$p")
    fi
done
missing_optional=()
for p in "${OPTIONAL_PKGS[@]}"; do
    if ! pacman -Qq "$p" >/dev/null 2>&1; then
        missing_optional+=("$p")
    fi
done

printf '\n%s%sAARON installer%s\n' "$BOLD" "$BLUE" "$RESET"
printf '  distribution   %s\n' "${PRETTY_NAME:-unknown}"
printf '  source tree    %s\n\n' "$ROOT"

note "Packages that would be installed:"
if (( ${#missing_required[@]} )); then
    for p in "${missing_required[@]}"; do printf '    - %s   %s(required)%s\n' "$p" "$BOLD" "$RESET"; done
else
    printf '    none — everything required is already installed\n'
fi
if (( ${#missing_optional[@]} )); then
    for p in "${missing_optional[@]}"; do printf '    - %s   %s(optional, for voice)%s\n' "$p" "$DIM" "$RESET"; done
else
    printf '    none — every optional package is already installed\n'
fi

printf '\n%sPackages AARON will NOT install for you:%s\n' "$YELLOW" "$RESET"
printf '  Speech-recognition models (ggml Whisper weights) and neural TTS voices.\n'
printf '  They are large, separately licensed, and downloaded from third parties.\n'
printf '  Download one yourself, then point AARON at it:\n'
printf '    asr.model_path  = "/path/to/ggml-base.en.bin"\n'
printf '    tts.model_path  = "/path/to/en_US-*.onnx"\n'
printf '\n'
printf '  Optional Python engines (installed by you, not by this script):\n'
printf '    Speech recognition (Phonon-2, Fermion Research):\n'
printf '      pip install --no-deps torch --index-url https://download.pytorch.org/whl/cpu\n'
printf '      pip install fermion-research\n'
printf '    The engine fetches the 164 MB model on first use into ~/.cache/fermion/speech/.\n'
printf '    Neural TTS (piper):\n'
printf '      pip install piper-tts\n'
printf '    Then download a voice (e.g. en_US-lessac-medium, 63 MB) from\n'
printf '    huggingface.co/rhasspy/piper-voices and set tts.model_path.\n'
for p in "${AUR_PKGS[@]}"; do printf '  AUR: %s%s\n' "$p" "$RESET"; done
printf '  No AUR helper is used by this script, by design.\n'

if (( CHECK_ONLY )); then
    printf '\n--check-only: stopping before any change.\n'
    exit 0
fi

if (( DO_INSTALL )) && (( ${#missing_required[@]} + ${#missing_optional[@]} > 0 )); then
    if (( ASSUME_YES )); then
        printf '\nProceeding because --yes was given.\n'
    else
        printf '\nInstall these with pacman? [y/N] '
        read -r reply
        case "$reply" in
            [yY]|[yY][eE][sS]) ;;
            *) printf 'Nothing was installed.\n'; exit 0 ;;
        esac
    fi
fi

# ------------------------------------------------------------------- install

if (( DO_INSTALL )) && (( ${#missing_required[@]} )); then
    note "Installing required packages with pacman..."
    sudo pacman -S --needed "${missing_required[@]}" || die "pacman failed"
fi
if (( DO_INSTALL )) && (( ${#missing_optional[@]} )); then
    note "Installing optional packages with pacman..."
    sudo pacman -S --needed "${missing_optional[@]}" || \
        warn "Optional packages failed to install; text-only AARON still works."
fi

# --------------------------------------------------------------------- build

if ! have cargo; then
    die "cargo is still not on PATH after installation. Is '~/.cargo/bin' in your PATH?"
fi

printf '\n%sBuilding AARON (this takes a few minutes the first time)...%s\n' "$BOLD" "$RESET"
if ! cargo build --release --manifest-path "$ROOT/Cargo.toml"; then
    die "Build failed. Run 'cargo build --release' in $ROOT to see the errors."
fi
ok "Built $(readlink -f "$ROOT/target/release/aaron")"

# --------------------------------------------------------------- verify

printf '\n%sVerifying the build...%s\n' "$BOLD" "$RESET"
"$ROOT/target/release/aaron" --version || die "the binary does not run"
"$ROOT/target/release/aaron" --config "$ROOT/config/default.toml" doctor \
    || warn "doctor reported problems (exit $?); see the output above"

if command -v "$ROOT/scripts/check-environment.sh" >/dev/null 2>&1; then
    "$ROOT/scripts/check-environment.sh" || true
fi

# -------------------------------------------------------------- instructions

printf '\n%s%sNext steps%s\n' "$BOLD" "$GREEN" "$RESET"
cat <<EOF
  1. Install the binary:
       sudo install -Dm755 "$ROOT/target/release/aaron" /usr/local/bin/aaron

  2. Create your configuration:
       aaron setup
     This writes ~/.config/aaron/config.toml and permissions.toml.

  3. Set up your own filesystem allow-list:
       \$EDITOR ~/.config/aaron/config.toml
     security.allowed_roots decides what read_file and search_files may touch.
     Empty means they refuse everything, which is the safe default.

  4. Check what works on this machine:
       aaron doctor

  5. Train the intent model (optional; ~6 s on a modern laptop):
       aaron brain train
     AARON works without it, using a deterministic keyword router.

  6. Enable voice (optional, after installing espeak-ng and a Whisper model):
       \$EDITOR ~/.config/aaron/config.toml   # set asr.model_path
       aaron listen

  Microphone access: nothing is installed as a service and no device is opened
  at boot. Run 'aaron listen' when you want to talk to it.
EOF

if [[ ! -e /etc/systemd/system/aaron.service ]]; then
    printf '\n%sA systemd user unit is available at systemd/aaron.service.%s\n' "$DIM" "$RESET"
    printf 'It is NOT enabled automatically. See docs/troubleshooting.md before enabling it.\n'
fi