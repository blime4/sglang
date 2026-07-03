#!/usr/bin/env python3
"""Microbench invoke_fused_moe_opt (DLIN fused blockwise FP8 MoE) for M=1 decode.
Tests correctness vs bf16 reference + speed vs the bf16-dequant+bmm path."""
import torch, os, time
for p in ["../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so"]:
    if os.path.exists(p):
        torch.ops.load_library(p)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)

torch.manual_seed(0)
dev = "cuda"
hidden, inter = 2048, 256          # real Qwen3.5-35B MoE dims
E = 8                              # small expert count for test
top_k = 8
BN = BK = 128

x = torch.randn(1, hidden, dtype=torch.bfloat16, device=dev)

def make_fp8_blockwise(shape):
    N, K = shape[-2], shape[-1]
    w = torch.randn(*shape, dtype=torch.bfloat16, device=dev) * 0.3
    nb, kb = N // BN, K // BK
    sc = w.float().view(*shape[:-2], nb, BN, kb, BK).abs().amax(dim=(-3, -1)).clamp(min=1e-6)
    sc = sc * (torch.arange(1, kb + 1, device=dev).float().view(*([1] * (len(shape) - 2)), kb))
    wf = (w.float().view(*shape[:-2], nb, BN, kb, BK) / sc.view(*shape[:-2], nb, 1, kb, 1)).clamp(-1, 1).to(torch.float8_e4m3fn).view(*shape)
    return wf, sc

w13, sc13 = make_fp8_blockwise((E, 2 * inter, hidden))
w2, sc2 = make_fp8_blockwise((E, hidden, inter))

topk_ids = torch.arange(E, device=dev).int().view(1, top_k)
topk_weights = torch.full((1, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)

def ref_moe():
    sc13f = sc13.repeat_interleave(BN, 1).repeat_interleave(BK, 2)
    w13_bf = (w13.float() * sc13f).to(torch.bfloat16)
    sc2f = sc2.repeat_interleave(BN, 1).repeat_interleave(BK, 2)
    w2_bf = (w2.float() * sc2f).to(torch.bfloat16)
    x_exp = x.unsqueeze(0).expand(E, 1, hidden)
    gu = torch.bmm(x_exp, w13_bf.transpose(1, 2)).squeeze(1)
    g, u = gu[:, :inter], gu[:, inter:]
    he = (torch.nn.functional.silu(g) * u).unsqueeze(1)
    de = torch.bmm(he, w2_bf.transpose(1, 2)).squeeze(1)
    return (de * topk_weights.view(E, 1).to(torch.bfloat16)).sum(0, keepdim=True)

ref = ref_moe().float()
print(f"ref |.| = {ref.abs().mean():.4f}")

G = torch.ops._dl_C.invoke_fused_moe_opt
sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
    topk_ids, block_size=16, num_experts=E
)
def dl_moe():
    c13 = torch.empty(1, top_k, 2 * inter, dtype=torch.bfloat16, device=dev)
    G(x, w13, c13, None, sc13.contiguous(), None,
      topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
      False, top_k, 16, 128, 128, True, False, False, False, [128, 128], 1)
    g, u = c13[0, :, :inter], c13[0, :, inter:]
    he = (torch.nn.functional.silu(g) * u).contiguous().view(1, top_k, inter)
    # w2 output is [M, top_k, hidden]; mul_routed_weight applies routing weights;
    # sum over top_k -> [M, hidden] (matches vLLM's moe_sum_reduce)
    c2 = torch.empty(1, top_k, hidden, dtype=torch.bfloat16, device=dev)
    G(he, w2, c2, None, sc2.contiguous(), None,
      topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
      True, top_k, 16, 128, 128, True, False, False, False, [128, 128], 1)
    return c2.sum(dim=1)  # [1, hidden]

try:
    # DEBUG: check w13 GEMM output vs reference intermediate
    c13 = torch.empty(1, top_k, 2 * inter, dtype=torch.bfloat16, device=dev)
    G(x, w13, c13, None, sc13.contiguous(), None,
      topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
      False, top_k, 16, 128, 128, True, False, False, False, [128, 128], 1)
    # reference w13 output per expert
    sc13f = sc13.repeat_interleave(BN, 1).repeat_interleave(BK, 2)
    w13_bf = (w13.float() * sc13f).to(torch.bfloat16)
    gu_ref = torch.bmm(x.unsqueeze(0).expand(E,1,hidden), w13_bf.transpose(1,2)).squeeze(1)  # [E,2*inter]
    print(f"w13 c13[0,0,:5] = {c13[0,0,:5].float().tolist()}")
    print(f"w13 ref[0,:5]   = {gu_ref[0,:5].tolist()}")
    print(f"w13 rel_err per expert0 = {(c13[0,0].float()-gu_ref[0]).abs().mean().item()/(gu_ref[0].abs().mean()+1e-9):.4f}")
    print(f"topk_ids={topk_ids.tolist()}, expert_ids={expert_ids[:16].tolist()}, sorted={sorted_token_ids[:16].tolist()}, postpad={num_tokens_post_padded.tolist()}")
    out = dl_moe()
    rel = (out.float() - ref).abs().mean().item() / (ref.abs().mean() + 1e-9)
    print(f"invoke_fused_moe_opt full: rel_err = {rel:.4f}  {'<<< MATCH' if rel < 0.05 else 'MISMATCH'}")
except Exception:
    import traceback; traceback.print_exc()
