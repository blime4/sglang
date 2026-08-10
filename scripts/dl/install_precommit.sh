#!/usr/bin/env bash
# DL begin
# Install a git pre-commit hook that runs the DLIN marker checker
# (scripts/dl/check_dl_markers.py) on every commit. Use this if you don't run
# the `pre-commit` framework; otherwise `pre-commit install` already wires the
# `dl-markers` hook from .pre-commit-config.yaml. Safe to run repeatedly.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
HOOK="$REPO/.git/hooks/pre-commit"

if [ -f "$HOOK" ] && ! grep -q "check_dl_markers.py" "$HOOK"; then
  echo "ERROR: $HOOK already exists and is not managed by this script." >&2
  echo "       Inspect it; if safe, remove it and re-run." >&2
  exit 1
fi

cat > "$HOOK" <<'EOF'
#!/bin/bash
# Managed by scripts/dl/install_precommit.sh — DLIN marker check on commit.
exec python3 scripts/dl/check_dl_markers.py
EOF
chmod +x "$HOOK"

echo "Installed pre-commit hook -> $HOOK"
echo "Every 'git commit' now runs scripts/dl/check_dl_markers.py (blocks on violation)."
echo "See .claude/skills/sglang-modify/SKILL.md for the convention."
# DL end
