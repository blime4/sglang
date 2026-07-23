---
name: sglang-modify
description: Convention for Denglin (DLIN/DLIN) modifications to sglang — how to mark and place DL changes so the DL diff stays greppable and survives upstream syncs. Read before editing sglang source for DLIN. Enforced by scripts/dl/check_dl_markers.py at commit time.
---

# sglang-modify — DLIN modification convention

**Goal:** every Denglin (DLIN/DLIN) change to sglang must be trivially findable
(`grep -rn "# DL"`) and cleanly re-applicable when upstream sglang is updated.
This mirrors how Denglin's vLLM fork marks its patches (~240 `# DL` markers).

## The rule

There are exactly two kinds of DL change. Both must be identifiable.

### 1. Inline edits to an UPSTREAM file → wrap in markers

Any change inside a file that exists in upstream sglang MUST be wrapped:

```python
# Python / TOML / CMake / shell (.py .toml CMakeLists.txt .sh .yaml)
# DL begin
<your DLIN change>
# DL end
```
```cpp
// C / C++ / CUDA / headers (.cc .cu .cpp .h .cuh)
// DL begin
<your DLIN change>
// DL end
```

For a one-liner, use the inline form on the same line:
```python
return not _is_dl()  # DL: DLIN also counts as CUDA
```
```cmake
if(NOT USE_DLIN) # DL: disable DeepSeek V3 router GEMM kernel
```

Rules:
- Every `# DL begin` MUST have a matching `# DL end` (balanced).
- Put a one-line `# DL:` comment on the `begin` line if the reason isn't obvious.
- Do NOT mark non-DL edits. If you touch a line for a non-DL reason, it stays unmarked.
- Markers use the file's native comment prefix: `#` for py/toml/cmake/sh, `//` for C/C++/CUDA.

### 2. New DL-only files → use DL naming (no internal markers needed)

Put brand-new, entirely-DLIN code in files whose names scream "DL":
- Python modules: `dl_*.py`, `*_dl.py`, `dlin.py`, `dlin_*.py`
- Build: `setup_dl.py`, `pyproject_dl.toml`, `common_extension_dl.cc`
- CMake gate: `if(USE_DLIN)` / `if(NOT USE_DLIN)` (mirrors vLLM)
- Kernel sources: `csrc/dl/*.cu`, vendored headers under `sgl-kernel/3rdparty/`
- Dep list: `requirements/dl.txt` (mirror vLLM)

These whole-file-DL paths are exempt from the inline-marker check. It's still
fine (encouraged) to put a single `# DL begin`/`# DL end` around the file body
for clarity, but not required.

## When to apply

Read this skill BEFORE editing any of these for DLIN:
- `python/sglang/srt/**` (runtime: platforms, attention, layers, sampler, utils)
- `python/sglang/jit_kernel/**`, `sgl-kernel/csrc/**`, `sgl-kernel/*.py`
- build files: `sgl-kernel/CMakeLists.txt`, `sgl-kernel/setup_*.py`, `pyproject*.toml`

If you're editing one of the large classes (`Scheduler`, `TokenizerManager`,
`ModelRunner` `__init__`), also read `large-class-init-style`. If touching env
vars, read `env-var-conventions`.

## Enforcement — commit-time check

`scripts/dl/check_dl_markers.py` scans staged files:
- DL-only files (by name/path) → exempt.
- Every other staged file → any added line NOT inside `# DL begin..end` (and not
  a `# DL:` line, not blank) is reported as a violation. Also flags unbalanced
  begin/end.

Run it manually any time:
```bash
python scripts/dl/check_dl_markers.py            # checks staged files
python scripts/dl/check_dl_markers.py <file>     # balance-check a file
```

It is wired as a `pre-commit` hook (`.pre-commit-config.yaml`, repo `local`,
id `dl-markers`). After `pre-commit install`, every `git commit` is checked and
blocked on violation. Install once:
```bash
pre-commit install --hook-type pre-commit
```

## Quick audit

```bash
# the entire DL diff, at a glance:
grep -rn "# DL" python/ sgl-kernel/csrc sgl-kernel/setup_dl.py scripts/dl
```

## Why this matters

When upstream sglang releases a new version, you'll re-apply the DL layer onto a
fresh checkout. Greppable `# DL` markers + DL-only files make the rebase a
mechanical search-and-port instead of an archaeology dig. This is exactly how
Denglin's vLLM fork stays current.
