#!/bin/bash
# =============================================================================
# deploy-docs.sh — GitLab Pages docs deploy (PLACEHOLDER)
# =============================================================================
# sglang's docs live under docs_new/ and use Mintlify (docs.json), NOT mkdocs.
# The upstream vLLM deploy-docs.sh drives mkdocs + mike, which does not apply
# here. This job is therefore a deliberate placeholder:
#   - It produces a minimal public/ so the `pages` artifact is valid.
#   - Wiring a real Mintlify build (npm/mintlify install + build → public/) is
#     a follow-up (spec E4). Uncomment/implement the MINTLIFY BUILD block below.
#
# Triggered only on tags (see .gitlab-ci.yml `pages` job rule).
# =============================================================================
set -e

export PATH="$PATH:$HOME/.local/bin"
echo "Hostname: $(hostname)"
echo "Git tag: ${CI_COMMIT_TAG:-<none>}"

PUBLIC_DIR="${CI_PROJECT_DIR}/public"
rm -rf "${PUBLIC_DIR}"
mkdir -p "${PUBLIC_DIR}"

# ── TODO: real Mintlify build (spec E4) ──────────────────────────────────────
# if [ -f "${CI_PROJECT_DIR}/docs_new/docs.json" ]; then
#     echo "Building Mintlify docs from docs_new/ ..."
#     cd "${CI_PROJECT_DIR}/docs_new"
#     npm install -g mintlify
#     mintlify build --out "${PUBLIC_DIR}"
# fi
# ── end TODO ─────────────────────────────────────────────────────────────────

# Placeholder landing page so the pages artifact is non-empty.
cat > "${PUBLIC_DIR}/index.html" <<HTML
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>sglang (DengLin) docs</title></head>
<body>
<h1>sglang (DengLin build) — docs placeholder</h1>
<p>The GitLab Pages <code>pages</code> job is a placeholder.
Real Mintlify docs build (from <code>docs_new/</code>) is pending — see
<code>docs/superpowers/specs/2026-08-06-sglang-gitlab-ci-design.md</code> (E4).</p>
<p>Built from tag: <code>${CI_COMMIT_TAG:-n/a}</code></p>
</body>
</html>
HTML

echo "Placeholder public/ created at ${PUBLIC_DIR}"
ls -la "${PUBLIC_DIR}"
