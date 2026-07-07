#!/usr/bin/env bash
#
# deploy-restart.sh — test, (re)install, and restart whisper-daemon for local testing.
#
# The Python package is installed editable (pip install -e .), so a daemon restart
# picks up source changes. The Chrome extension (MV3) has no build step — it must be
# reloaded manually in chrome://extensions after this script prints the reminder.
#
# Usage:
#   scripts/deploy-restart.sh              # run tests, restart daemon
#   scripts/deploy-restart.sh --no-tests   # skip tests, just restart
#   scripts/deploy-restart.sh --no-restart # run tests only
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
DAEMON="$REPO/.venv/bin/whisper-daemon"

RUN_TESTS=1
RUN_RESTART=1
for arg in "$@"; do
  case "$arg" in
    --no-tests) RUN_TESTS=0 ;;
    --no-restart) RUN_RESTART=0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }
red() { printf "\033[31m%s\033[0m\n" "$1"; }
yellow() { printf "\033[33m%s\033[0m\n" "$1"; }

TESTS_OK=1

if [[ "$RUN_TESTS" == "1" ]]; then
  bold "── Python tests (pytest) ─────────────────────────────"
  if [[ -d tests ]]; then
    "$PY" -m pytest tests/ -q || TESTS_OK=0
  else
    yellow "no tests/ directory — skipping"
  fi

  bold "── Extension tests (node --test) ─────────────────────"
  # NB: use the glob form, not a bare `extension/test/` dir — Node v22 treats a
  # bare directory arg as a module to require and throws MODULE_NOT_FOUND.
  if compgen -G "extension/test/*.test.js" >/dev/null && command -v node >/dev/null 2>&1; then
    node --test extension/test/*.test.js || TESTS_OK=0
  else
    yellow "no extension/test/*.test.js or node missing — skipping"
  fi

  if [[ "$TESTS_OK" == "1" ]]; then
    green "✓ all tests passed"
  else
    red "✗ some tests FAILED (see above)"
  fi
fi

if [[ "$RUN_RESTART" == "1" ]]; then
  bold "── Restart daemon ────────────────────────────────────"
  if [[ "$TESTS_OK" != "1" ]]; then
    yellow "restarting despite test failures (you asked to test the build)"
  fi
  "$DAEMON" restart || { "$DAEMON" stop || true; "$DAEMON" start; }
  sleep 1
  "$DAEMON" status || true

  bold "── Reload the Chrome extension ───────────────────────"
  echo "  1. open  chrome://extensions"
  echo "  2. find 'Whisper Daemon Tab Capture' → click the ⟳ reload icon"
  echo "  3. reload your Google Meet tab so the new content scripts inject"
  echo ""
  echo "  Tail daemon logs:  $DAEMON logs -f"
fi

green "done."
