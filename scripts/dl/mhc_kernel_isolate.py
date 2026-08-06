#!/usr/bin/env python3
"""DL: isolate-compile the MHC Triton kernel (1 card, dummy inputs).
If this compiles fast -> the model-context hang is fixable.
If this hangs -> the kernel is incompatible with DLIN's Triton compiler."""
import os, time, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")

def main():
    import torch
    from sglang.jit_kernel.dsv4.dl_mhc_triton import dl_mhc_pre_triton
    d = torch.device("cuda")
    torch.zeros(1, device=d)
    # V4-Flash MHC pre: residual [M, hc_mult=4, hidden=4096], fn [hc_mult3=24, hc_mult*hidden=16384]
    hc_mult = 4
    hidden = 4096
    hc_mult3 = hc_mult * (2 + hc_mult)  # 24
    hc_hidden = hc_mult * hidden  # 16384
    residual = torch.randn(1, hc_mult, hidden, dtype=torch.bfloat16, device=d)
    fn = torch.randn(hc_mult3, hc_hidden, dtype=torch.float32, device=d)
    hc_scale = torch.randn(3, dtype=torch.float32, device=d)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=d)
    print(f"[isolate] calling dl_mhc_pre_triton (M=1, hc={hc_mult}, H={hidden})...", flush=True)
    t0 = time.perf_counter()
    post, comb, y = dl_mhc_pre_triton(
        residual, fn, hc_scale, hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20, n_splits=1,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[isolate] COMPILED + RAN in {dt:.1f}s", flush=True)
    print(f"[isolate] post={tuple(post.shape)} comb={tuple(comb.shape)} y={tuple(y.shape)}", flush=True)
    print(f"[isolate] y[0,:5]={y.flatten()[:5].tolist()}", flush=True)
    print("[isolate] SUCCESS", flush=True)

if __name__ == "__main__":
    main()
