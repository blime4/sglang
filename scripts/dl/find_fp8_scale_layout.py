#!/usr/bin/env python3
"""Find correct blockwise-FP8 scale layout for gptq_dlblas_gemmex with K != N
(matches real MoE w13: K=hidden=2048, N=2*inter=1024). Lets the GEMM's shape
check reject invalid layouts; reports which valid layout matches the bf16 ref."""
import torch, os
for p in ["../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so"]:
    if os.path.exists(p): torch.ops.load_library(p)

torch.manual_seed(0)
BN, BK = 128, 128
K, N = 2048, 1024   # K != N, matches MoE w13 (hidden=2048, 2*inter=1024)
M = 2
dev = "cuda"
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
w_bf = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.3

nb, kb = N // BN, K // BK   # nb=8, kb=16
w_bf_blk = w_bf.view(nb, BN, kb, BK)
scale = w_bf_blk.float().abs().view(nb, kb, BN, BK).amax(dim=(2, 3)).clamp(min=1e-6)  # [nb,kb]=[8,16]
scale = scale * torch.arange(1, kb + 1, device=dev).float().expand(nb, kb)  # vary per k-block

# reference
sc_full = scale.repeat_interleave(BN, 0).repeat_interleave(BK, 1)
w_deq = (w_bf.view(nb, BN, kb, BK).float() / scale.view(nb, 1, kb, 1)).clamp(-1,1).to(torch.float8_e4m3fn)
w_fp8 = w_deq.view(N, K)
ref = (x @ (w_fp8.float() * sc_full).to(torch.bfloat16).t().contiguous()).float()

wt = w_fp8.t().contiguous()  # [K,N]
print(f"K={K} N={N} scale={tuple(scale.shape)} (nb={nb},kb={kb})  ref|.|={ref.abs().mean():.3f}")

def tryit(name, sc):
    try:
        o = torch.ops._dl_C.gptq_dlblas_gemmex(x, wt, sc, sc, 2, 8)
        rel = (o.float() - ref).abs().mean().item() / (ref.abs().mean().item() + 1e-9)
        flag = "  <<< MATCH" if rel < 0.03 else ""
        print(f"  {name:38s} shape={tuple(sc.shape)} rel_err={rel:.4f}{flag}")
    except Exception as e:
        print(f"  {name:38s} shape={tuple(sc.shape)} REJECTED: {str(e)[:45]}")

tryit("natural [nb,kb]=[N/128,K/128]", scale)                       # size(1)=kb=K/128 ✓check
tryit("transposed .t() [kb,nb]", scale.t().contiguous())            # size(1)=nb=N/128
tryit("values transposed, shape [nb,kb]", scale.t().contiguous().t().contiguous())  # == natural
# per-channel over N: [nb, kb] but each row = its max (collapse k-block)
sc_pc = scale.amax(dim=1, keepdim=True).expand(nb, kb)
tryit("per-channel-N (row=max) [nb,kb]", sc_pc)
# maybe scale should be [K/128, N/128] VALUES in [nb,kb] shape: scatter
sc_kbn_in_nkb = scale.t().reshape(nb, kb)  # reshapes [kb,nb]->[nb,kb] (different values)
tryit("scale.t().reshape(nb,kb)", sc_kbn_in_nkb.contiguous())
