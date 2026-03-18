#!/usr/bin/env bash
# upgrade_deps.sh — Upgrade project dependencies, with special handling for mlx-audio
#
# Usage:
#   bash scripts/upgrade_deps.sh            # upgrade all deps
#   bash scripts/upgrade_deps.sh --mlx-only # upgrade mlx-audio only
#
# mlx-audio is installed from git (https://github.com/Blaizzy/mlx-audio),
# so a plain `uv sync --upgrade` won't re-fetch the latest commit.
# This script forces a fresh pull by removing the cached source and re-locking.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

MLX_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--mlx-only" ]] && MLX_ONLY=true
done

echo "=== TTS Inference API — Dependency Upgrade ==="
echo "    Project root: $PROJECT_ROOT"
echo ""

# ── Verify uv is available ────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "ERROR: 'uv' not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "uv version: $(uv --version)"
echo ""

# ── Force-upgrade mlx-audio from git ─────────────────────────────────────────
# uv caches git sources under .uv/cache. The only reliable way to pull the
# latest HEAD is to pass --upgrade-package, which re-resolves the git ref.
echo ">>> Upgrading mlx-audio (git HEAD)..."
uv lock --upgrade-package mlx-audio
echo "    mlx-audio lock updated."
echo ""

if [[ "$MLX_ONLY" == true ]]; then
  echo ">>> --mlx-only: syncing mlx-audio only..."
  uv sync --no-dev
  echo ""
  echo "=== Done (mlx-audio only) ==="
  exit 0
fi

# ── Upgrade all other dependencies ───────────────────────────────────────────
echo ">>> Upgrading all dependencies..."
uv lock --upgrade
echo "    Lock file updated."
echo ""

echo ">>> Syncing environment..."
uv sync --all-extras
echo "    Environment synced."
echo ""

# ── Print summary of what changed ────────────────────────────────────────────
echo ">>> Changed packages (vs git HEAD~1):"
if git diff --quiet HEAD -- uv.lock 2>/dev/null; then
  echo "    No changes in uv.lock (already up to date)."
else
  # Show package lines that changed — filter to name+version lines only
  git diff HEAD -- uv.lock 2>/dev/null \
    | grep -E '^[+-]name\s*=' \
    | grep -v '^---\|^+++' \
    | sed 's/^+/  UPDATED: /; s/^-/  WAS:     /' \
    || echo "    (diff unavailable)"
fi

echo ""
echo "=== Upgrade complete ==="
echo ""
echo "Next steps:"
echo "  1. Test the server:  uv run python -m src --model-path ./models/<your-model>"
echo "  2. Run tests:        uv run python -m pytest tests/ -q"
echo "  3. Commit lockfile:  git add uv.lock && git commit -m 'chore: upgrade dependencies'"
