"""DL: deferred-sync per-component decode profiler for V4-Flash on DLIN.

Spawn-safe (module-level accumulators, one per TP worker process) and sync-light:
records CUDA event pairs per call, syncs only every FLUSH_EVERY calls to avoid the
per-call-sync pipeline-serialization artifact (the blog's indexer 111->22ms = 5x
inflation came from per-call sync; MoE's 87ms is suspected to be the same artifact,
true value ~17ms per the 1-GPU microbench).

Usage (gated, zero overhead when off):
    from sglang.srt.layers.quantization.dl_moe_profile import dl_timer
    with dl_timer("moe_w13"):
        _G(...)            # timed region
    dl_timer.maybe_flush()  # call once per decode step

Env:
    SGLANG_DL_DECODE_PROFILE=1   enable
    SGLANG_DL_DECODE_FLUSH=50    sync+print every N calls (default 50)
"""
import os
import threading

_ENABLED = os.environ.get("SGLANG_DL_DECODE_PROFILE", "0") == "1"
_FLUSH_EVERY = int(os.environ.get("SGLANG_DL_DECODE_FLUSH", "50"))
import torch

_lock = threading.Lock()
# name -> {"starts":[Event], "ends":[Event], "total_us":float, "n":int}
_acc: dict = {}


class _Region:
    __slots__ = ("name", "_s")

    def __init__(self, name: str):
        self.name = name
        self._s = None

    def __enter__(self):
        if _ENABLED:
            self._s = torch.cuda.Event(enable_timing=True)
            self._s.record()
        return self

    def __exit__(self, *exc):
        if not _ENABLED:
            return False
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        with _lock:
            a = _acc.setdefault(self.name, {"starts": [], "ends": [], "total_us": 0.0, "n": 0})
            a["starts"].append(self._s)
            a["ends"].append(e)
        return False


def dl_timer(name: str):
    """Context manager timing a GPU region (deferred sync)."""
    return _Region(name)


def maybe_flush():
    """Sync pending events every _FLUSH_EVERY calls and print running averages."""
    if not _ENABLED:
        return
    # A host sync (torch.cuda.synchronize) is illegal inside CUDA-graph capture
    # and invalidates the graph (cudaErrorStreamCaptureInvalidated). Skip while
    # capturing; events accumulate and flush on the next eager (non-capture) call.
    if torch.cuda.is_current_stream_capturing():
        return
    with _lock:
        total_pending = sum(len(a["starts"]) for a in _acc.values())
        if total_pending < _FLUSH_EVERY:
            return
        torch.cuda.synchronize()
        lines = []
        for name in sorted(_acc):
            a = _acc[name]
            if not a["starts"]:
                continue
            elapsed_ms = sum(s.elapsed_time(e) for s, e in zip(a["starts"], a["ends"]))
            a["total_us"] += elapsed_ms * 1e3
            a["n"] += len(a["starts"])
            a["starts"].clear()
            a["ends"].clear()
            avg_us = a["total_us"] / max(a["n"], 1)
            lines.append(f"{name}={avg_us:8.1f}us/call (n={a['n']})")
        if lines:
            print(f"[DL_DECODE_PROF] " + "  ".join(lines), flush=True)
