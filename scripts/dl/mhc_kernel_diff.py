#!/usr/bin/env python3
"""DL: compare Triton MHC pre output vs torch formula on identical input."""
import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")
import torch, time

def main():
    from sglang.jit_kernel.dsv4.dl_mhc_triton import dl_mhc_pre_triton
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm
    d = torch.device("cuda"); torch.zeros(1, device=d)
    hc_mult, hidden, sinkhorn_repeat = 4, 4096, 20
    hc_mult3 = hc_mult * (2 + hc_mult)  # 24
    hc_hidden = hc_mult * hidden  # 16384
    torch.manual_seed(42)
    x = torch.randn(1, hc_mult, hidden, dtype=torch.bfloat16, device=d)
    fn = torch.randn(hc_mult3, hc_hidden, dtype=torch.float32, device=d)
    scale = torch.randn(3, dtype=torch.float32, device=d)
    base = torch.randn(hc_mult3, dtype=torch.float32, device=d)

    # --- Triton path ---
    t0 = time.perf_counter()
    post_t, comb_t, y_t = dl_mhc_pre_triton(
        x.clone(), fn.clone(), scale.clone(), base.clone(),
        1e-6, 1e-6, 1e-6, 2.0, sinkhorn_repeat, 1)
    torch.cuda.synchronize(); dt_t = time.perf_counter() - t0

    # --- Torch reference (same as sglang DeepGEMM path) ---
    t0 = time.perf_counter()
    x_flat = x.flatten(1).bfloat16()  # [1, hc_mult*hidden]
    m, k = x_flat.shape
    d_out = torch.empty((m, hc_mult3), dtype=torch.float32, device=d)
    s_out = torch.empty((m,), dtype=torch.float32, device=d)
    tf32_hc_prenorm_gemm(x_flat, fn.float().contiguous(), d_out, s_out, num_splits=None)
    rsqrt = torch.rsqrt(s_out / k + 1e-6)
    mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)  # [1,1,hc_mult3]
    # Sinkhorn via sgl_kernel
    from sglang.kernels.ops.layernorm.mhc import hc_split_sinkhorn
    pre_ref, post_ref, comb_ref = hc_split_sinkhorn(
        mixes, scale, base, hc_mult, sinkhorn_repeat, 1e-6)
    shape = x.size()  # [1, hc_mult, hidden]
    y_ref = (pre_ref.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1).to(torch.bfloat16)
    torch.cuda.synchronize(); dt_r = time.perf_counter() - t0

    # --- Compare ---
    print(f"Triton: {dt_t*1000:.1f}ms  Torch: {dt_r*1000:.1f}ms", flush=True)
    print(f"y_t  first5: {y_t.flatten()[:5].tolist()}", flush=True)
    print(f"y_ref first5: {y_ref.flatten()[:5].tolist()}", flush=True)
    y_diff = (y_t.float() - y_ref.float()).abs()
    print(f"y  max_diff={y_diff.max():.4f} mean_diff={y_diff.mean():.4f}", flush=True)
    print(f"post_t  first4: {post_t.flatten()[:4].tolist()}", flush=True)
    print(f"post_ref first4: {post_ref.flatten()[:4].tolist()}", flush=True)
    post_diff = (post_t.float() - post_ref.float().unsqueeze(-1)).abs()
    print(f"post max_diff={post_diff.max():.4f}", flush=True)
    print(f"comb_t  [0,:4]: {comb_t[0,:4].tolist()}", flush=True)
    print(f"comb_ref[0,:4]: {comb_ref[0,:4].tolist()}", flush=True)
    comb_diff = (comb_t.float() - comb_ref.float()).abs()
    print(f"comb max_diff={comb_diff.max():.4f}", flush=True)

if __name__ == "__main__":
    main()
