#!/usr/bin/env python3
"""Test quant_type=0 (per-tensor) and =1 (per-channel) for gptq_dlblas_gemmex.
vLLM uses these; sglang wrongly used quant_type=2 (GPTQ-blockwise, expects packed)."""
import torch, os
for p in ["../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so"]:
    if os.path.exists(p): torch.ops.load_library(p)
torch.manual_seed(0)
K, N = 2048, 1024
M = 2
dev = "cuda"
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
w_bf = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.3

# --- per-tensor reference ---
pt = w_bf.float().abs().max().clamp(min=1e-6)
w_pt = (w_bf.float() / pt).clamp(-1, 1).to(torch.float8_e4m3fn)  # [N,K]
ref_pt = (x @ (w_pt.float() * pt).to(torch.bfloat16).t().contiguous()).float()

# --- per-channel (over N) reference ---
pc = w_bf.float().abs().view(N, K).amax(dim=1).clamp(min=1e-6)  # [N]
w_pc = (w_bf.float() / pc.view(N, 1)).clamp(-1, 1).to(torch.float8_e4m3fn)
ref_pc = (x @ (w_pc.float() * pc.view(N, 1)).to(torch.bfloat16).t().contiguous()).float()

G = torch.ops._dl_C.gptq_dlblas_gemmex
print(f"K={K} N={N}  ref_pt|.|={ref_pt.abs().mean():.3f}  ref_pc|.|={ref_pc.abs().mean():.3f}")
def t(name, qt, w, zeros, sc, ref):
    try:
        o = G(x, w, zeros, sc, qt, 8)
        rel = (o.float() - ref).abs().mean().item() / (ref.abs().mean().item() + 1e-9)
        print(f"  {name:50s} qt={qt} w={tuple(w.shape)} sc={tuple(sc.shape)} rel={rel:.4f}{'  <<<MATCH' if rel<0.05 else ('  ~close' if rel<0.15 else '')}")
    except Exception as e:
        print(f"  {name:50s} qt={qt} REJECTED: {str(e)[:40]}")

print("PER-TENSOR (quant_type=0, vLLM TENSOR):")
empty = torch.empty(0, 0, dtype=torch.float32, device=dev)
t("w_pt.t()[K,N] scale[1,1] zeros=empty", 0, w_pt.t().contiguous(), empty, pt.to(torch.float32).view(1, 1), ref_pt)
t("w_pt.t()[K,N] scale[1,1] zeros=scale", 0, w_pt.t().contiguous(), pt.to(torch.float32).view(1, 1), pt.to(torch.float32).view(1, 1), ref_pt)
print("PER-CHANNEL (quant_type=1, vLLM CHANNEL):")
t("w_pc.t()[K,N] scale[N,1] zeros=scale", 1, w_pc.t().contiguous(), pc.to(torch.float32).view(N, 1), pc.to(torch.float32).view(N, 1), ref_pc)
t("w_pc.t()[K,N] scale[1,N] zeros=scale", 1, w_pc.t().contiguous(), pc.to(torch.float32).view(1, N), pc.to(torch.float32).view(1, N), ref_pc)
print("quant_type=2 (sglang current, blockwise) for reference:")
BN = BK = 128
nb, kb = N // BN, K // BK
sc_blk = w_bf.view(nb, BN, kb, BK).float().abs().view(nb, kb, BN, BK).amax(dim=(2, 3)).clamp(min=1e-6)
w_blk = (w_bf.view(nb, BN, kb, BK).float() / sc_blk.view(nb, 1, kb, 1)).clamp(-1, 1).to(torch.float8_e4m3fn).view(N, K)
sc_full = sc_blk.repeat_interleave(BN, 0).repeat_interleave(BK, 1)
ref_blk = (x @ (w_blk.float() * sc_full).to(torch.bfloat16).t().contiguous()).float()
t("w_blk.t()[K,N] scale[nb,kb]", 2, w_blk.t().contiguous(), sc_blk.contiguous(), sc_blk.contiguous(), ref_blk)
