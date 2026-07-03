#!/usr/bin/env python3
"""Profile sglang decode via HTTP /start_profile + /stop_profile, then parse the
Chrome trace to get ground-truth kernel counts and times.

Answers: is the 405ms/token from kernel TIME (slow kernels) or kernel COUNT
(dispatch overhead of many small launches)?
"""
import os, sys, json, time, glob, subprocess, requests, signal

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
TP = os.environ.get("TP", "2")
PORT = os.environ.get("PORT", "31055")
TRACE_DIR = "/tmp/sglang_profile_decode"
os.makedirs(TRACE_DIR, exist_ok=True)
# clean old traces
for f in glob.glob(f"{TRACE_DIR}/*.json"):
    os.remove(f)

# launch server
cmd = [
    sys.executable, "-m", "sglang.launch_server",
    "--model-path", MODEL,
    "--tp", str(TP),
    "--dtype", "bfloat16",
    "--context-length", os.environ.get("CONTEXT_LEN", "4096"),
    "--mem-fraction-static", os.environ.get("MEM_FRAC", "0.82"),
    "--max-running-requests", os.environ.get("MAX_RUNNING_REQUESTS", "16"),
    "--disable-cuda-graph",
    "--attention-backend", "fa3",
    "--page-size", "16",
    "--port", PORT,
    "--host", "0.0.0.0",
]
env = dict(os.environ)
env["SGLANG_TORCH_PROFILER_DIR"] = TRACE_DIR
env["SGLANG_DL_MOE_DLBLAS"] = "1"
print("[launch]", " ".join(cmd[:6]), "...", flush=True)
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# wait for ready
base = f"http://127.0.0.1:{PORT}"
deadline = time.time() + 900
ready = False
while time.time() < deadline:
    line = proc.stdout.readline()
    if line:
        sys.stdout.write(line); sys.stdout.flush()
        if "The server is fired up and ready to roll" in line:
            ready = True; break
    if proc.poll() is not None:
        print("[ERR] server died early"); sys.exit(1)
if not ready:
    print("[ERR] timeout waiting for server"); proc.kill(); sys.exit(1)

print("\n[profile] server ready, warming up...", flush=True)
s = requests.Session()
# warmup
for _ in range(2):
    s.post(f"{base}/generate", json={"text": "Explain relativity.", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}})

# Profile N decode steps. num_steps auto-stops after N scheduler steps.
N = int(os.environ.get("N_PROFILE", "20"))
print(f"[profile] starting profiler (num_steps={N}, will auto-stop)...", flush=True)
r = s.post(f"{base}/start_profile", json={
    "num_steps": N,
    "activities": ["CPU", "GPU"],
    "output_dir": TRACE_DIR,
    "profile_prefix": "sgl_decode",
    "record_shapes": False,
    "with_stack": False,
})
print(f"[profile] start_profile resp: {r.status_code} {r.text[:120]}", flush=True)

# Short prompt -> prefill is 1 tiny step, remaining N-1 are decode
gen = s.post(f"{base}/generate", json={
    "text": "Write a long detailed essay about the history and future of computing.",
    "sampling_params": {"max_new_tokens": N + 5, "temperature": 0.0},
})
print(f"[profile] generate done (status {gen.status_code})", flush=True)

# give the auto-stop flush time to write the trace
for _ in range(60):
    if glob.glob(f"{TRACE_DIR}/*.trace.json.gz") or glob.glob(f"{TRACE_DIR}/*.trace.json"):
        break
    time.sleep(1)

# shutdown server
proc.send_signal(signal.SIGTERM)
try: proc.wait(timeout=30)
except: proc.kill()

# find trace (written as .trace.json.gz, one per TP rank)
traces = sorted(glob.glob(f"{TRACE_DIR}/*.trace.json.gz") + glob.glob(f"{TRACE_DIR}/*.trace.json"), key=os.path.getmtime)
if not traces:
    print("[ERR] no trace file found in", TRACE_DIR); sys.exit(1)
print(f"\n[profile] found {len(traces)} trace file(s): {[os.path.basename(t) for t in traces]}", flush=True)

import gzip
def load_trace(p):
    if p.endswith(".gz"):
        with gzip.open(p, "rt") as f:
            return json.load(f)
    with open(p) as f:
        return json.load(f)

# Use TP0's trace (rank 0); if multiple, pick the one with TP-0 in name else newest
tp0 = [t for t in traces if "TP-0" in os.path.basename(t)] or traces
trace_path = tp0[0]
print(f"[profile] parsing {os.path.basename(trace_path)}", flush=True)
trace = load_trace(trace_path)
events = trace["traceEvents"]

# separate CPU ops vs GPU kernels by pid/category
from collections import defaultdict
cpu_ops = [e for e in events if e.get("cat") == "operator"]
gpu_kernels = [e for e in events if e.get("cat") == "kernel"]

# count + time by name
def aggregate(evts, dur_key="dur"):
    by_name = defaultdict(lambda: [0, 0.0])  # [count, total_dur_us]
    for e in evts:
        n = e.get("name", "?")
        d = e.get(dur_key, 0)
        by_name[n][0] += 1
        by_name[n][1] += d
    return by_name

cpu_agg = aggregate(cpu_ops)
gpu_agg = aggregate(gpu_kernels)

total_cpu_us = sum(v[1] for v in cpu_agg.values())
total_gpu_us = sum(v[1] for v in gpu_agg.values())
cpu_count = sum(v[0] for v in cpu_agg.values())
gpu_count = sum(v[0] for v in gpu_agg.values())

print("\n" + "=" * 75)
print(f"PROFILE SUMMARY (over {N} decode steps)")
print("=" * 75)
print(f"  CPU op launches:   {cpu_count:6d}   ({cpu_count/N:.0f}/step)")
print(f"  GPU kernel launches:{gpu_count:6d}   ({gpu_count/N:.0f}/step)")
print(f"  CPU op wall time:   {total_cpu_us/1000:8.1f} ms  ({total_cpu_us/N/1000:.1f} ms/step)")
print(f"  GPU kernel time:    {total_gpu_us/1000:8.1f} ms  ({total_gpu_us/N/1000:.1f} ms/step)")
print(f"  CPU/GPU ratio:      {total_cpu_us/max(total_gpu_us,1):.2f}x  (>1.5 = dispatch-bound)")
print("=" * 75)

print(f"\nTOP 25 GPU KERNELS BY TIME ({N} steps):")
print(f"{'count':>6} {'tot_ms':>9} {'us/launch':>10}  name")
for name, (cnt, tot) in sorted(gpu_agg.items(), key=lambda x: -x[1][1])[:25]:
    print(f"{cnt:6d} {tot/1000:9.2f} {tot/max(cnt,1):10.2f}  {name[:60]}")

print(f"\nTOP 25 CPU OPS BY COUNT ({N} steps):")
print(f"{'count':>6} {'tot_ms':>9}  name")
for name, (cnt, tot) in sorted(cpu_agg.items(), key=lambda x: -x[1][0])[:25]:
    print(f"{cnt:6d} {tot/1000:9.2f}  {name[:60]}")
