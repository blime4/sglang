#!/usr/bin/env python3
# Minimal repro for the DLIN dleol paged-DECODE flash-attention crash.
#
# Single `flash_attn_with_kvcache` call (paged KV cache, decode shape) crashes
# the process inside libdleol.so. NOT model-dependent, NOT sglang-dependent:
# reproduces with a standalone random-tensor call.
#
# Required env (clean DLIN runtime):
#   export SDK_DIR=/LocalRun/.../sdk            # MRrc-4.2.0-202606161052
#   export CUDA_HOME=$SDK_DIR DLI_V2=ON
#   export LD_LIBRARY_PATH=$SDK_DIR/lib          # ONLY this dir (no sdk-0401 leak)
#   export CUDA_VISIBLE_DEVICES=<free GPU>
#   source <venv>/bin/activate                   # torch==2.9.1+dl24.sdk202606031721
# Run:   python repro_fa2_decode_crash.py
# Expect: process dies (exit 255 / SIGSEGV) printing:
#          "to bc failed."
#          "CUDA Driver API error = 0001 from file <.../dleol/src/op/flash_attn_kvcache_mha_op.cc>, line 867."
import sys, torch
from flash_attn import flash_attn_with_kvcache

dev = "cuda"
# Qwen3-1.7B decode shape: GQA 16 q-heads / 8 kv-heads, head_dim 128, page_size 16.
B, Hq, Hkv, D, Pg, nblk = 2, 16, 8, 128, 16, 4
q = torch.randn(B, 1, Hq, D, dtype=torch.bfloat16, device=dev)
k_cache = torch.randn(nblk, Pg, Hkv, D, dtype=torch.bfloat16, device=dev)
v_cache = torch.randn(nblk, Pg, Hkv, D, dtype=torch.bfloat16, device=dev)
block_table = torch.zeros(B, nblk, dtype=torch.int32, device=dev)
cache_seqlens = torch.full((B,), Pg * 2, dtype=torch.int32, device=dev)

out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_seqlens, block_table=block_table, causal=True,
)
torch.cuda.synchronize()
print("OK:", tuple(out.shape))   # never reached on affected SDK/torch combo
sys.exit(0)
