#!/usr/bin/env bash
# One command to run the face-recognition web app on macOS / Linux:   ./run_app.sh   (optional: PORT=8080 NO_BROWSER=1 ./run_app.sh)
# Creates a private virtual environment (.venv-app), installs the 6 runtime packages, starts the app and opens the browser.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

PY=""
for c in python3 python py; do   # first one that really runs (skips the Microsoft Store stub on Windows)
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(sys.version_info < (3, 11))" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "Python 3.11 or newer was not found. Install it from https://www.python.org/downloads/ and run this again." >&2; exit 1; fi

if [ ! -d .venv-app ]; then echo "Creating virtual environment (first run only)..."; "$PY" -m venv .venv-app; fi
if [ -x .venv-app/bin/python ]; then VPY=.venv-app/bin/python; else VPY=.venv-app/Scripts/python; fi   # Scripts/ = Git Bash on Windows

echo "Installing packages (first run takes about a minute)..."
"$VPY" -m pip install --quiet --disable-pip-version-check -r requirements-app.txt

URL="http://localhost:${PORT}"
echo "Starting Aperture at ${URL}  (press Ctrl+C to stop)"
if [ -z "${NO_BROWSER:-}" ]; then ( sleep 4; { command -v xdg-open >/dev/null && xdg-open "$URL"; } || { command -v open >/dev/null && open "$URL"; } || { command -v start >/dev/null && start "$URL"; } || true ) >/dev/null 2>&1 & fi
exec "$VPY" -m uvicorn api.app:app --host 127.0.0.1 --port "$PORT"
