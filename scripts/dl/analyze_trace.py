#!/usr/bin/env python3
"""Analyze a sglang chrome trace: CPU dispatch vs GPU launch breakdown.
Separates prefill from decode using scheduler annotations."""
import gzip, json, sys, glob, os
from collections import Counter, defaultdict

TRACE_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sglang_profile_decode"
N_STEPS = int(os.environ.get("N_PROFILE", "20"))

# newest sgl_decode TP-0 trace
traces = sorted(glob.glob(f"{TRACE_DIR}/sgl_decode-*.trace.json.gz"), key=os.path.getmtime)
if not traces:
    traces = sorted(glob.glob(f"{TRACE_DIR}/*.trace.json.gz"), key=os.path.getmtime)
p = [t for t in traces if "TP-0" in os.path.basename(t)] or traces
p = p[-1]
print(f"[analyze] {os.path.basename(p)}")

with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
    tr = json.load(f)
evts = tr["traceEvents"]

# find decode step boundaries via scheduler annotations
sched = [e for e in evts if e.get("cat") == "user_annotation" and "get_next_batch_to_run" in e.get("name", "")]
sched.sort(key=lambda e: e["ts"])
print(f"[analyze] {len(sched)} scheduler steps recorded")

# Step windows: each scheduler step annotation marks a forward.
# Decode steps are the ones we want. Prefill = first, decodes = rest.
# Use ts ranges to bucket cpu_ops into steps.
step_windows = [(s["ts"], s["ts"] + s.get("dur", 0)) for s in sched]
# extend each window to the start of the next
for i in range(len(step_windows) - 1):
    step_windows[i] = (step_windows[i][0], step_windows[i + 1][0])

cpu_ops = [e for e in evts if e.get("cat") == "cpu_op" and "dur" in e]
cuda_rt = [e for e in evts if e.get("cat") == "cuda_runtime" and "dur" in e]

def bucket(evts_list, windows):
    per = [[] for _ in windows]
    for e in evts_list:
        ts = e.get("ts", 0)
        for i, (lo, hi) in enumerate(windows):
            if lo <= ts < hi:
                per[i].append(e); break
    return per

cpu_per = bucket(cpu_ops, step_windows)
cuda_per = bucket(cuda_rt, step_windows)

# decode steps = indices 1..end (skip first prefill). If only decode visible, use all.
n_prefill = 1
decode_idx = list(range(n_prefill, len(step_windows)))
if len(decode_idx) < 2:  # maybe no prefill captured
    decode_idx = list(range(len(step_windows)))

def agg(evts_list):
    by_name = defaultdict(lambda: [0, 0.0])
    tot = 0.0
    for e in evts_list:
        n = e.get("name", "?")
        d = e.get("dur", 0)
        by_name[n][0] += 1
        by_name[n][1] += d
        tot += d
    return by_name, tot

# decode aggregate
dec_cpu = [e for i in decode_idx for e in cpu_per[i]]
dec_cuda = [e for i in decode_idx for e in cuda_per[i]]
n_dec = len(decode_idx)

cpu_agg, cpu_tot = agg(dec_cpu)
cuda_agg, cuda_tot = agg(dec_cuda)
cpu_cnt = sum(v[0] for v in cpu_agg.values())
cuda_cnt = sum(v[0] for v in cuda_agg.values())

print("\n" + "=" * 76)
print(f"DECODE-ONLY ANALYSIS  ({n_dec} decode steps, prefill excluded)")
print("=" * 76)
print(f"  CPU op (dispatch) launches:  {cpu_cnt:7d}   ({cpu_cnt/n_dec:,.0f}/step)")
print(f"  CPU op total wall time:      {cpu_tot/1000:8.1f} ms  ({cpu_tot/n_dec/1000:.1f} ms/step)")
print(f"  cuda_runtime launches:       {cuda_cnt:7d}   ({cuda_cnt/n_dec:,.0f}/step)")
print(f"  cuda_runtime total wall:     {cuda_tot/1000:8.1f} ms  ({cuda_tot/n_dec/1000:.1f} ms/step)")
print(f"  => if CPU op time/step ~ decode latency, you are DISPATCH-BOUND")
print("=" * 76)

print(f"\nTOP 25 CPU OPS BY COUNT (decode, {n_dec} steps):")
print(f"{'count':>7} {'cnt/step':>9} {'tot_ms':>9}  name")
for name, (cnt, tot) in sorted(cpu_agg.items(), key=lambda x: -x[1][0])[:25]:
    print(f"{cnt:7d} {cnt/n_dec:9.1f} {tot/1000:9.1f}  {name[:55]}")

print(f"\nTOP 20 CPU OPS BY TIME (decode, {n_dec} steps):")
print(f"{'count':>7} {'tot_ms':>9} {'us/op':>8}  name")
for name, (cnt, tot) in sorted(cpu_agg.items(), key=lambda x: -x[1][1])[:20]:
    print(f"{cnt:7d} {tot/1000:9.1f} {tot/max(cnt,1):8.1f}  {name[:55]}")

print(f"\nTOP 15 cuda_runtime BY COUNT (decode, {n_dec} steps):")
print(f"{'count':>7} {'cnt/step':>9}  name")
for name, (cnt, tot) in sorted(cuda_agg.items(), key=lambda x: -x[1][0])[:15]:
    print(f"{cnt:7d} {cnt/n_dec:9.1f}  {name[:55]}")
