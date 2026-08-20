#!/bin/bash
# Install doc-search as a macOS user LaunchAgent (auto-start at login, keep-alive).
# Uninstall:  launchctl bootout gui/$(id -u)/com.sodashikenn.docsearch \
#             && rm ~/Library/LaunchAgents/com.sodashikenn.docsearch.plist
#
# NOTE (macOS TCC): if this repo lives under ~/Desktop, ~/Documents, or
# ~/Downloads, macOS may deny file access to the launchd-spawned python the
# first time. If the service loops without serving, grant the python binary
# access in System Settings > Privacy & Security > Files and Folders, or move
# the repo outside those folders. Check logs/docsearch.err.log.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$HERE/deploy/launchd/com.sodashikenn.docsearch.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.sodashikenn.docsearch.plist"
SERVICE="gui/$(id -u)/com.sodashikenn.docsearch"

[ -x "$HERE/.venv/bin/python" ] || { echo "error: $HERE/.venv missing — create it first (see README)" >&2; exit 1; }
[ -f "$HERE/index/meta.json" ]  || { echo "error: no index at $HERE/index — run: .venv/bin/python -m docsearch index <docs_dir>" >&2; exit 1; }

mkdir -p "$HERE/logs" "$HOME/Library/LaunchAgents"
sed "s|@HERE@|$HERE|g" "$PLIST_SRC" > "$PLIST_DST"

launchctl bootout "$SERVICE" 2>/dev/null || true
# bootout is asynchronous — retry bootstrap until launchd finishes the removal
for attempt in 1 2 3 4 5; do
    if launchctl bootstrap "gui/$(id -u)" "$PLIST_DST" 2>/dev/null; then
        break
    fi
    [ "$attempt" = 5 ] && { echo "error: launchctl bootstrap kept failing" >&2; exit 1; }
    sleep 1
done
launchctl kickstart -k "$SERVICE"

sleep 2
if curl -sf -m 5 "http://127.0.0.1:8765/api/stats" > /dev/null; then
    echo "installed and serving: http://127.0.0.1:8765"
else
    echo "installed, but the service is not answering yet — check $HERE/logs/docsearch.err.log" >&2
    launchctl print "$SERVICE" | grep -E "state|pid" | head -3 || true
    exit 1
fi
