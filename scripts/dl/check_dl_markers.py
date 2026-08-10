#!/usr/bin/env python3
# DL begin
"""
DLIN modification marker checker for sglang (mirrors Denglin vLLM's patch style).

Convention (see .claude/skills/sglang-modify/SKILL.md):
  * Inline DLIN edits inside an UPSTREAM (non-DL-only) file MUST be wrapped in
        # DL begin   ...   # DL end      (Python / TOML / CMake / shell)
        // DL begin  ...   // DL end     (C / C++ / CUDA / headers)
    or tagged inline with  # DL:  /  // DL:.
  * New DL-only files (dl_*.py, *_dl.*, setup_dl.py, pyproject_dl.toml,
    common_extension_dl.cc, ...) are EXEMPT — the whole file is DL.

This script checks staged files (default) or files passed on the CLI, and exits
non-zero if any DL edit in an upstream file falls outside the markers.

Used as a pre-commit hook (see scripts/dl/install_precommit / .pre-commit-config.yaml).
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

# Files that are entirely DLIN-owned -> exempt from inline-marker checks.
EXEMPT_GLOBS = [
    "**/dl_*.py", "**/dl_*.cc", "**/dl_*.cu",
    "**/dlin.py", "**/dlin_*.py",
    "**/*_dl.py", "**/*_dl.cc", "**/*_dl.cu", "**/*_dl.toml",
    "**/setup_dl.py", "**/pyproject_dl.toml", "**/common_extension_dl.cc",
    "**/requirements/dl.txt",
    "sgl-kernel/3rdparty/**",
    "scripts/dl/**",
    ".claude/skills/sglang-modify/**",
    "DLIN_INTEGRATION_PLAN.md", "dlin_environment.yml", "run_sglang.sh",
]

# Source files whose inline edits must carry DL markers.
# C/C++ headers (.h/.cuh/.hpp) use #if defined(SGL_ON_DLIN) preprocessor guards
# (greppable) instead of comment markers — excluded from per-line checks.
CHECKABLE_EXT = {".py", ".cc", ".cu", ".cpp", ".c",
                 ".toml", ".cmake", ".in", ".sh", ".yaml", ".yml"}
CMAKEFILE_RE = re.compile(r"(^|/)CMakeLists\.txt$")

# DL marker lines (both comment styles).
DL_BEGIN = re.compile(r"^\s*(#|//)\s*DL\s+begin\b")
DL_END = re.compile(r"^\s*(#|//)\s*DL\s+end\b")
DL_INLINE = re.compile(r"(#|//)\s*DL(:|\b)")  # '# DL:' / '// DL:' / '# DL end'
# C/C++ preprocessor DL guards also count as DL markers.
DL_GUARD = re.compile(r"SGL_ON_DLIN|USE_DLIN")
# Preprocessor directives are structural, not content — tolerate like blank lines.
PREPROC = re.compile(r"^\s*#(if|elif|else|endif|ifdef|ifndef|pragma)\b")


def is_exempt(path: str) -> bool:
    for g in EXEMPT_GLOBS:
        if fnmatch.fnmatch(path, g) or fnmatch.fnmatch("/" + path, "/" + g):
            return True
    return False


def is_checkable(path: str) -> bool:
    if CMAKEFILE_RE.search(path):
        return True
    return Path(path).suffix in CHECKABLE_EXT


def dl_marked_lines(text: str) -> set[int]:
    """Line numbers (1-based) that are within a DL begin..end span or carry a # DL marker."""
    marked, in_dl = set(), False
    for i, line in enumerate(text.splitlines(), start=1):
        if DL_BEGIN.search(line):
            in_dl = True
            marked.add(i)
            continue
        if DL_END.search(line):
            marked.add(i)
            in_dl = False
            continue
        if in_dl or DL_INLINE.search(line) or DL_GUARD.search(line):
            marked.add(i)
    return marked


def staged_added_lines(path: str) -> tuple[list[int], set[str]]:
    """Added line numbers (1-based, in the new file) from `git diff --cached -U0`,
    plus the set of removed-line contents (to detect pure moves)."""
    out = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", path],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent.parent,
    ).stdout
    added, removed, new_ln = [], set(), 0
    for line in out.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                new_ln = int(m.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(new_ln)
            new_ln += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed.add(line[1:].strip())
        else:
            new_ln += 1
    return added, removed


def check_file(path: str) -> list[str]:
    """Return a list of violation strings for the given path."""
    if is_exempt(path) or not is_checkable(path):
        return []
    abs_path = Path(__file__).resolve().parent.parent.parent / path
    if not abs_path.exists():
        return []
    text = abs_path.read_text(errors="replace")
    marked = dl_marked_lines(text)
    added, removed = staged_added_lines(path)
    violations = []
    lines = text.splitlines()
    for ln in added:
        if ln in marked:
            continue
        content = lines[ln - 1] if 0 < ln <= len(lines) else ""
        if not content.strip():  # blank lines are tolerated
            continue
        if PREPROC.search(content):  # preprocessor directives are structural
            continue
        if content.strip() in removed:  # pure move (line shifted, not a DL edit)
            continue
        violations.append(f"{path}:{ln}: DL edit outside `# DL begin/end`: {content.strip()[:90]}")
    return violations


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent.parent,
    ).stdout
    return [f for f in out.splitlines() if f]


def main() -> int:
    ap = argparse.ArgumentParser(description="Check DLIN edits carry # DL markers.")
    ap.add_argument("files", nargs="*", help="specific files: balance-check only (no diff)")
    args = ap.parse_args()

    files = args.files if args.files else staged_files()
    problems = []

    # 1) staged added-line check (only for the staged set)
    if not args.files:
        for f in files:
            problems.extend(check_file(f))

    # 2) # DL begin/end balance check (always)
    for f in files:
        if is_exempt(f) or not is_checkable(f):
            continue
        abs_path = Path(__file__).resolve().parent.parent.parent / f
        if not abs_path.exists():
            continue
        body = abs_path.read_text(errors="replace").splitlines()
        begins = sum(1 for l in body if DL_BEGIN.search(l))
        ends = sum(1 for l in body if DL_END.search(l))
        if begins != ends:
            problems.append(f"{f}: unbalanced DL begin/end markers (begins={begins}, ends={ends})")

    if problems:
        print("DL marker check FAILED - wrap DLIN edits in `# DL begin`/`# DL end` (or `# DL:`):")
        for p in problems:
            print("  " + p)
        print(f"\n{len(problems)} violation(s). See .claude/skills/sglang-modify/SKILL.md")
        return 1
    print("DL marker check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
# DL end
