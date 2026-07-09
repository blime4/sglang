#!/usr/bin/env python3
"""Microbench invoke_fused_moe_opt for M>1 (prefill) — correctness vs bf16 reference."""
import torch, os, time
for p in ["../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so"]:
    if os.path.exists(p): torch.ops.load_library(p)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

torch.manual_seed(42)
dev = "cuda"
hidden, inter = 2048, 256
E, top_k = 8, 8
M = 3  # prefill (M>1)
BN = BK = 128

x = torch.randn(M, hidden, dtype=torch.bfloat16, device=dev)

def make_fp8_blockwise(shape):
    N, K = shape[-2], shape[-1]
    w = torch.randn(*shape, dtype=torch.bfloat16, device=dev) * 0.3
    nb, kb = N // BN, K // BK
    sc = w.float().view(*shape[:-2], nb, BN, kb, BK).abs().amax(dim=(-3, -1)).clamp(min=1e-6)
    wf = (w.float().view(*shape[:-2], nb, BN, kb, BK) / sc.view(*shape[:-2], nb, 1, kb, 1)).clamp(-1, 1).to(torch.float8_e4m3fn).view(*shape)
    return wf, sc

w13, sc13 = make_fp8_blockwise((E, 2 * inter, hidden))
w2, sc2 = make_fp8_blockwise((E, hidden, inter))

# Each token routes to 8 different experts
topk_ids = torch.randint(0, E, (M, top_k), device=dev, dtype=torch.int32)
topk_weights = torch.rand(M, top_k, device=dev, dtype=torch.float32)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

# bf16 reference (per-token, per-expert)
def ref_moe():
    out = torch.zeros(M, hidden, dtype=torch.bfloat16, device=dev)
    for t in range(M):
        for k in range(top_k):
            eid = topk_ids[t, k].item()
            sc13f = sc13[eid].repeat_interleave(BN, 0).repeat_interleave(BK, 1)
            w13_bf = (w13[eid].float() * sc13f).to(torch.bfloat16)
            gu = x[t] @ w13_bf.t()
            g, u = gu[:inter], gu[inter:]
            he = torch.nn.functional.silu(g) * u
            sc2f = sc2[eid].repeat_interleave(BN, 0).repeat_interleave(BK, 1)
            w2_bf = (w2[eid].float() * sc2f).to(torch.bfloat16)
            de = he @ w2_bf.t()
            out[t] += de * topk_weights[t, k]
    return out

ref = ref_moe()
print(f"ref |.| = {ref.abs().mean():.4f}, shape={ref.shape}")

# DLIN fused MoE
G = torch.ops._dl_C.invoke_fused_moe_opt
srt, eid_m, npp = moe_align_block_size(topk_ids, 16, E)
print(f"moe_align: srt={srt.shape}, eid={eid_m.shape}, npp={npp.item()}")

c13 = torch.empty(M, top_k, 2 * inter, dtype=torch.bfloat16, device=dev)
G(x, w13, c13, None, sc13.contiguous(), None,
  topk_weights.contiguous(), topk_ids.contiguous(), srt, eid_m, npp,
  False, top_k, 16, 128, 128, True, False, False, False, [128, 128], M)
print(f"c13 |.| = {c13.abs().mean():.4f}")

gate, up = c13[:, :, :inter], c13[:, :, inter:]
he = (torch.nn.functional.silu(gate) * up).contiguous()
c2 = torch.empty(M, top_k, hidden, dtype=torch.bfloat16, device=dev)
G(he, w2, c2, None, sc2.contiguous(), None,
  topk_weights.contiguous(), topk_ids.contiguous(), srt, eid_m, npp,
  True, top_k, 16, 128, 128, True, False, False, False, [128, 128], M)

out = c2.sum(dim=1)
print(f"out |.| = {out.abs().mean():.4f}")
rel = (out.float() - ref.float()).abs().mean().item() / (ref.abs().mean() + 1e-9)
print(f"rel_err = {rel:.4f}  {'<<< MATCH' if rel < 0.1 else 'MISMATCH'}")
