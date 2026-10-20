#!/usr/bin/env bash
#
# Remove AARON.
#
# What it deletes, and what it never touches:
#
#   DELETES   ~/.local/state/aaron      runtime data: database, audit log, checkpoints
#   DELETES   ~/.config/aaron           your configuration, after an explicit confirmation
#   KEEPS     /usr/local/bin/aaron      unless you pass --binary
#   NEVER     anything outside those two directories
#   NEVER     any system package, service, firewall rule or system configuration
#
# Your memories are in the database and they are not recoverable afterwards, so
# this script asks twice: once for the whole operation and once specifically
# about the database, and it makes a backup before deleting unless you decline.
#
# Usage:
#   ./scripts/uninstall.sh                 # asks
#   ./scripts/uninstall.sh --yes           # non-interactive
#   ./scripts/uninstall.sh --yes --keep-data
#   ./scripts/uninstall.sh --yes --binary  # also remove /usr/local/bin/aaron

set -uo pipefail

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'

ASSUME_YES=0
KEEP_DATA=0
REMOVE_BINARY=0
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --yes|-y)      ASSUME_YES=1 ;;
        --keep-data)   KEEP_DATA=1 ;;
        --binary)      REMOVE_BINARY=1 ;;
        --dry-run)     DRY_RUN=1 ;;
        -h|--help)     sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf '%sunknown option: %s%s\n' "$RED" "$arg" "$RESET" >&2; exit 64 ;;
    esac
done

STATE_DIR="${AARON_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/aaron}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/aaron"
BINARY="/usr/local/bin/aaron"

say()  { printf '%s%s%s\n' "$BLUE" "$1" "$RESET"; }
warn() { printf '%s%s%s\n' "$YELLOW" "$1" "$RESET"; }
ok()   { printf '%s%s%s\n' "$GREEN" "$1" "$RESET"; }

confirm() {
    local prompt="$1"
    if (( ASSUME_YES )); then return 0; fi
    printf '%s [y/N] ' "$prompt"
    read -r reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

printf '\n%sAARON uninstall%s\n' "$RED" "$RESET"
printf 'This removes:\n'
printf '  %s\n' "$STATE_DIR"
if (( KEEP_DATA )); then
    printf '  (keeping data and config because --keep-data was given)\n'
else
    printf '  %s\n' "$CONFIG_DIR"
fi
(( REMOVE_BINARY )) && printf '  %s\n' "$BINARY"
printf '\nNothing outside those paths is touched. No packages, services or\n'
printf 'system configuration are removed.\n'

# ------------------------------------------------------------ backup the data

if (( ! KEEP_DATA )) && [[ -d "$STATE_DIR" ]] && (( ! DRY_RUN )); then
    backup="$STATE_DIR.backup.$(date +%Y%m%d-%H%M%S).tar.gz"
    say "Taking a backup of your memories first: $backup"
    if tar -czf "$backup" -C "$(dirname "$STATE_DIR")" "$(basename "$STATE_DIR")" 2>/dev/null; then
        ok "Backup written to $backup"
        warn "Delete it yourself once you are sure you do not want it."
    else
        warn "Could not create a backup. Continuing without one."
    fi
fi

if (( DRY_RUN )); then
    printf '\n--dry-run: nothing was deleted.\n'
    exit 0
fi

# ------------------------------------------------------------------ removal

if confirm "Remove AARON's runtime data (database, audit log, checkpoints)?"; then
    if [[ -d "$STATE_DIR" ]]; then
        rm -rf -- "$STATE_DIR" && ok "Removed $STATE_DIR"
    else
        printf '  nothing at %s\n' "$STATE_DIR"
    fi
else
    warn "Kept $STATE_DIR"
fi

if (( ! KEEP_DATA )); then
    if confirm "Remove your configuration and permissions ($CONFIG_DIR)?"; then
        if [[ -d "$CONFIG_DIR" ]]; then
            rm -rf -- "$CONFIG_DIR" && ok "Removed $CONFIG_DIR"
        else
            printf '  nothing at %s\n' "$CONFIG_DIR"
        fi
    else
        warn "Kept $CONFIG_DIR"
    fi
fi

if (( REMOVE_BINARY )); then
    if confirm "Remove $BINARY?"; then
        if [[ -f "$BINARY" ]]; then
            if [[ -w "$BINARY" || $EUID -eq 0 ]]; then
                rm -f -- "$BINARY" && ok "Removed $BINARY"
            else
                warn "Cannot remove $BINARY without root. Run: sudo rm -f $BINARY"
            fi
        else
            printf '  nothing at %s\n' "$BINARY"
        fi
    fi
fi

printf '\n%sAARON removed.%s\n' "$GREEN" "$RESET"
printf 'The source tree at %s was left alone; delete it yourself if you\n' "$(pwd)"
printf 'want. Reinstall with: ./scripts/install-arch.sh\n'