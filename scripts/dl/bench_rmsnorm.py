#!/usr/bin/env python3
# DL begin — microbench sglang rmsnorm_dl.cu (M=1 decode vs M=4096 prefill).
# Decides if vectorization (half2 pair load + single-pass) is worth doing:
#   M=4096 is throughput-bound (vectorize helps a lot);
#   M=1 (decode) is latency-bound — one block, time dominated by launch +
#   block-reduce, so vectorizing the element loop may give little.
# Decision rule: if M=1 rmsnorm > ~3% of decode step (~22ms => >~600us) it's
# worth it; if it's a few tens of us (<0.3%), skip — not worth the kernel churn.
#
# Run: source $SDK_DIR/env.sh; CUDA_VISIBLE_DEVICES=<free> python scripts/dl/bench_rmsnorm.py
import torch
import sgl_kernel  # noqa: F401  # loads torch.ops.sgl_kernel.* (bench doesn't import sglang)

dev = "cuda"
H = 2048            # Qwen3.6-35B-A3B hidden_size (rmsnorm input dim)
Ms = [1, 4, 16, 64, 4096]   # 1 = decode, 4096 = prefill
eps = 1e-6
iters = 200


def bench(fn, args, warmup=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def main():
    rms = torch.ops.sgl_kernel.rmsnorm
    fused = torch.ops.sgl_kernel.fused_add_rmsnorm
    print(f"rmsnorm microbench: H={H}, iters={iters} (us/call)")
    print(f"{'M':>6} {'rmsnorm(us)':>14} {'fused_add(us)':>16} {'M*H':>10}")
    for M in Ms:
        x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
        w = torch.randn(H, dtype=torch.bfloat16, device=dev) * 0.1
        out = torch.empty_like(x)
        res = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
        t_rms = bench(rms, (out, x, w, eps, False))
        # fused_add_rmsnorm mutates input+residual in place -> clone each iter
        t_fused = bench(fused, (x.clone(), res.clone(), w, eps, False))
        print(f"{M:>6} {t_rms:>14.2f} {t_fused:>16.2f} {M*H:>10}")


if __name__ == "__main__":
    main()
# DL end
