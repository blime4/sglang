#!/usr/bin/env python3
"""compare_results.py — persistent sglang-vs-vLLM results store.

Tracks every `run_sglang.sh compare` run as a JSON record keyed by sglang
commit, so the sglang-vs-vLLM gap can be followed across commits. vLLM is
recorded in two groups, MRV1 and MRV2 (MRV1 frequently fails on DLIN — that is
recorded too, so you can see when it starts working).

Store (default docs/dl/compare_results.json):
  {
    "schema_version": 1,
    "runs": [
      {
        "id": "r001",
        "timestamp": "2026-07-22T16:24:42",
        "sglang_commit": "275ba016c8", "sglang_branch": "dl-main",
        "sglang_dirty": false,
        "model": "...", "tp": 4, "sdk": "...", "mem_frac": 0.55,
        "scenarios": ["SC1","SC2","SC3"],
        "engines": {
          "sglang":    {"status":"ok", "SC1_warm_ms":5628, ...},
          "vllm_mrv2": {"status":"ok", ...},
          "vllm_mrv1": {"status":"fail", "error":"<last log lines>"}
        }
      }, ...
    ]
  }

Subcommands:
  append   read per-engine metrics (or fail logs), add metadata, append a run.
  history  print all runs (commit + headline metrics per engine).
  diff     compare the latest run vs a baseline (previous, by id, or by commit).
  csv      export the history to CSV (open in Excel).

Metric files are the `METRIC KEY=VALUE` blocks emitted by
scripts/dl/showcase_prefix_sharing.py (cached by run_sglang.sh compare).
"""
from __future__ import annotations
import argparse, json, os, sys, datetime

DEFAULT_STORE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "dl", "compare_results.json",
)

# Headline metrics shown in the compact history/diff tables, in display order.
# (key, header, better) — better="high" means higher is better (sglang-favored).
HEADLINE = [
    ("SC1_warm_ms",       "SC1 warm ms",  "low"),
    ("SC1_speedup_x",     "SC1 speedup",  "high"),
    ("SC2_turn5_ms",      "SC2 turn5 ms", "low"),
    ("SC3_throughput_tps","SC3 tok/s",    "high"),
    # DL: new candidate scenarios (render as FAIL/blank if not run).
    ("SC5_total_ms",      "SC5 fork ms",  "low"),
    ("SC7_throughput_tps","SC7 RAG t/s",  "high"),
    ("SC8_tps",           "SC8 samp t/s", "high"),
    ("SC9_tps",           "SC9 decode t/s","high"),
    ("SC10_throughput_tps","SC10 sysprompt t/s", "high"),
    ("SC11_throughput_tps","SC11 conc t/s", "high"),
]
ENGINE_ORDER = ["sglang", "vllm_mrv2", "vllm_mrv1"]
ENGINE_LABEL = {"sglang": "sglang", "vllm_mrv2": "vLLM-MRV2", "vllm_mrv1": "vLLM-MRV1"}


def _load_store(path: str) -> dict:
    if not os.path.exists(path):
        return {"schema_version": 1, "runs": []}
    with open(path) as f:
        data = json.load(f)
    data.setdefault("schema_version", 1)
    data.setdefault("runs", [])
    return data


def _save_store(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _parse_metrics_file(path: str) -> dict:
    """Read a `METRIC KEY=VALUE` file into {KEY: float-or-str}."""
    out: dict = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("METRIC "):
                continue
            body = line[len("METRIC "):]
            if "=" not in body:
                continue
            k, v = body.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _next_id(runs: list) -> str:
    n = len(runs) + 1
    return f"r{n:03d}"


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------
def cmd_append(args) -> int:
    data = _load_store(args.store)
    engines: dict = {}
    # --metrics NAME=FILE  (status=ok)
    for spec in args.metrics:
        if "=" not in spec:
            print(f"[compare_results] ignoring bad --metrics {spec!r}", file=sys.stderr)
            continue
        name, fname = spec.split("=", 1)
        m = _parse_metrics_file(fname)
        if m:
            engines[name] = {"status": "ok", **m}
        else:
            engines[name] = {"status": "empty", "file": fname}
    # --fail NAME=LOGFILE  (status=fail, capture tail)
    for spec in args.fail:
        if "=" not in spec:
            continue
        name, logfile = spec.split("=", 1)
        tail = ""
        if logfile and os.path.exists(logfile):
            try:
                with open(logfile, errors="replace") as f:
                    tail = "".join(f.readlines()[-12:]).strip()
            except OSError:
                tail = ""
        engines[name] = {"status": "fail", "error": tail}

    # Also stash the METRICS header (model/runner) from each ok engine's log if given.
    record = {
        "id": _next_id(data["runs"]),
        "timestamp": args.timestamp or datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "sglang_commit": args.commit,
        "sglang_branch": args.branch,
        "sglang_dirty": bool(args.dirty),
        "model": args.model,
        "tp": int(args.tp) if args.tp else None,
        "sdk": args.sdk,
        "mem_frac": float(args.mem_frac) if args.mem_frac else None,
        "scenarios": [s.strip() for s in args.scenarios.split(",") if s.strip()],
        "engines": engines,
    }
    data["runs"].append(record)
    _save_store(args.store, data)
    n_eng_ok = sum(1 for e in engines.values() if e.get("status") == "ok")
    print(f"[compare_results] appended {record['id']} (commit {args.commit[:12]}): "
          f"{n_eng_ok}/{len(engines)} engines ok -> {args.store}")
    print(f"[compare_results] runs total: {len(data['runs'])}")
    return 0


# ---------------------------------------------------------------------------
# helpers for history / diff
# ---------------------------------------------------------------------------
def _val(engines: dict, engine: str, key: str):
    e = engines.get(engine)
    if not e or e.get("status") != "ok":
        return None
    return e.get(key)


def _fmt(v, w=8):
    if v is None:
        return "FAIL".rjust(w) if w else "FAIL"
    if isinstance(v, float):
        return f"{v:.1f}".rjust(w)
    return str(v).rjust(w)


def _resolve_baseline(data: dict, baseline: str) -> dict | None:
    runs = data["runs"]
    if not runs:
        return None
    if baseline in (None, "", "prev", "previous"):
        return runs[-2] if len(runs) >= 2 else None  # diff caller handles latest
    # by id (r001) or commit prefix
    for r in runs:
        if r.get("id") == baseline or r.get("sglang_commit", "").startswith(baseline):
            return r
    return None


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------
def cmd_history(args) -> int:
    data = _load_store(args.store)
    runs = data["runs"]
    if not runs:
        print(f"[compare_results] no runs yet in {args.store}")
        return 1
    print(f"[compare_results] {len(runs)} run(s) in {args.store}\n")
    # Header: id | date | commit | for each headline metric: sglang/MRV2/MRV1
    parts = ["id", "date", "commit"]
    for key, label, _ in HEADLINE:
        parts.append(label)
    cw = [6, 11, 10] + [max(len(HEADLINE[i][1]), 22) for i in range(len(HEADLINE))]
    hdr = "  ".join(p.ljust(cw[i]) for i, p in enumerate(parts))
    print(hdr)
    print("  ".join("-" * c for c in cw))
    for r in runs:
        cells = [r["id"], r["timestamp"][:10], (r.get("sglang_commit") or "?")[:10]]
        eng = r.get("engines", {})
        for key, _, _b in HEADLINE:
            triple = "/".join(
                _fmt(_val(eng, e, key), 0) for e in ENGINE_ORDER
            )  # sglang/MRV2/MRV1
            cells.append(triple)
        # left-align the triple cells, right-align the rest
        out = []
        for i, c in enumerate(cells):
            out.append(c.ljust(cw[i]) if i >= 3 else c.ljust(cw[i]))
        print("  ".join(out))
    print(f"\n  (each metric cell = sglang / vLLM-MRV2 / vLLM-MRV1; FAIL = engine crashed)")
    return 0


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def cmd_diff(args) -> int:
    data = _load_store(args.store)
    runs = data["runs"]
    if not runs:
        print(f"[compare_results] no runs in {args.store}")
        return 1
    cur = runs[-1]
    base = _resolve_baseline(data, args.baseline)
    if base is None:
        print(f"[compare_results] only {len(runs)} run(s); need >=2 to diff "
              f"(or pass --baseline <id|commit>). Latest:")
        cmd_history(args)
        return 0
    if base is cur:
        # baseline resolved to the latest itself (e.g. only run); fall back
        base = runs[-2] if len(runs) >= 2 else None
        if base is None:
            print("[compare_results] need >=2 runs to diff.")
            return 0

    def _tag(r):
        d = "*" if r.get("sglang_dirty") else ""
        return f"{r['id']} {r.get('sglang_commit','?')[:10]}{d} ({r['timestamp'][:10]})"

    print(f"[compare_results] diff: current {_tag(cur)}  vs  baseline {_tag(base)}\n")
    print(f"  {'metric':<20}{'engine':<12}{'base':>10}{'current':>10}{'delta':>12}  verdict")
    print(f"  {'-'*20}{'-'*12}{'-'*10}{'-'*10}{'-'*12}  -------")
    any_row = False
    for key, label, better in HEADLINE:
        for engine in ENGINE_ORDER:
            b = _val(base.get("engines", {}), engine, key)
            c = _val(cur.get("engines", {}), engine, key)
            if b is None and c is None:
                continue
            any_row = True
            if b is None or c is None:
                verdict = "new/lost"
                delta_s = "N/A"
            else:
                delta = c - b
                delta_s = f"{delta:+.1f}"
                if better == "high":
                    verdict = "BETTER" if delta > 0 else ("worse" if delta < 0 else "same")
                else:  # low better
                    verdict = "BETTER" if delta < 0 else ("worse" if delta > 0 else "same")
            print(f"  {label:<20}{ENGINE_LABEL[engine]:<12}"
                  f"{_fmt(b,10):>10}{_fmt(c,10):>10}{delta_s:>12}  {verdict}")
    if not any_row:
        print("  (no comparable metrics between these runs)")
    return 0


# ---------------------------------------------------------------------------
# csv
# ---------------------------------------------------------------------------
def cmd_csv(args) -> int:
    import csv
    data = _load_store(args.store)
    runs = data["runs"]
    keys = sorted({k for r in runs for e in r.get("engines", {}).values()
                   if e.get("status") == "ok" for k in e if k != "status"})
    cols = ["id", "timestamp", "sglang_commit", "sglang_branch", "model", "tp",
            "sdk", "mem_frac", "scenarios"]
    for engine in ENGINE_ORDER:
        cols.append(f"{engine}__status")
        for k in keys:
            cols.append(f"{engine}__{k}")
    out = sys.stdout if not args.out else open(args.out, "w", newline="")
    try:
        w = csv.writer(out)
        w.writerow(cols)
        for r in runs:
            row = {c: "" for c in cols}
            row["id"] = r["id"]
            row["timestamp"] = r.get("timestamp", "")
            row["sglang_commit"] = r.get("sglang_commit", "")
            row["sglang_branch"] = r.get("sglang_branch", "")
            row["model"] = r.get("model", "")
            row["tp"] = r.get("tp", "")
            row["sdk"] = r.get("sdk", "")
            row["mem_frac"] = r.get("mem_frac", "")
            row["scenarios"] = ",".join(r.get("scenarios", []))
            for engine in ENGINE_ORDER:
                e = r.get("engines", {}).get(engine, {})
                row[f"{engine}__status"] = e.get("status", "")
                for k in keys:
                    if k in e:
                        row[f"{engine}__{k}"] = e[k]
            w.writerow([row[c] for c in cols])
    finally:
        if args.out:
            out.close()
            print(f"[compare_results] wrote {len(runs)} run(s) -> {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add_store(p):
        p.add_argument("--store", default=DEFAULT_STORE,
                       help=f"JSON store path (default: {DEFAULT_STORE})")

    a = sub.add_parser("append", help="append a run record")
    _add_store(a)
    a.add_argument("--commit", required=True)
    a.add_argument("--branch", default="")
    a.add_argument("--dirty", action="store_true")
    a.add_argument("--model", default="")
    a.add_argument("--tp", default="")
    a.add_argument("--sdk", default="")
    a.add_argument("--mem-frac", default="")
    a.add_argument("--scenarios", default="")
    a.add_argument("--timestamp", default="", help="override (else now())")
    a.add_argument("--metrics", action="append", default=[],
                   help="NAME=metrics_file (status=ok). Repeatable.")
    a.add_argument("--fail", action="append", default=[],
                   help="NAME=log_file (status=fail; tail captured). Repeatable.")
    a.set_defaults(func=cmd_append)

    h = sub.add_parser("history", help="print all runs")
    _add_store(h)
    h.set_defaults(func=cmd_history)

    d = sub.add_parser("diff", help="diff latest vs baseline (prev|id|commit)")
    _add_store(d)
    d.add_argument("--baseline", default="prev", help="prev (default) | run id (r001) | commit prefix")
    d.set_defaults(func=cmd_diff)

    c = sub.add_parser("csv", help="export history to CSV")
    _add_store(c)
    c.add_argument("--out", default="", help="output file (default: stdout)")
    c.set_defaults(func=cmd_csv)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
