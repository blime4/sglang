"""DL: Triton replacement for the V4 indexer's fp8_fp4_paged_mqa_logits DL op.

ROOT CAUSE (see memory dsv4-indexer-bottleneck): the vendored DL op processes
max_c4_seq_len PADDED entries (~16384 at the model's default 1M context), not the
valid seq_lens entries (~3-68 at decode). At M=1 this is ~1700us/call of wasted
work (34ms = 29% of CG TPOT over ~21 indexer layers).

This kernel processes ONLY the valid pages: grid (M,), each program loops over
num_valid_pages = ceil(seq_len / 64) pages (runtime bound, CG-safe static grid).
Runtime therefore depends on seq_len, NOT max_c4_seq_len — fast at any context.
Matches the verified torch reference `fp8_paged_mqa_logits_torch` (indexer.py:56):

    scores[e,h] = relu(kv_e . q_h)
    logits[e]   = (sum_h weight[h] * scores[e,h]) * kv_scale[e]
    output [M, max_model_len] fp32, e >= seq_len -> 0

KV buffer layout (confirmed from SetKAndS.triton, index_buf_accessor.py:373):
  SPLIT per page [8448 bytes] = [64*128 data fp8 bytes][64*4 scale fp32 bytes].
  - data  entry off: buf_fp8 [page*8448 + off*128 + 0:128]
  - scale entry off: buf_fp32[page*2112 + 2048 + off]   (8448/4=2112, 8192/4=2048)
"""
import torch
import triton
import triton.language as tl

_NUM_HEADS = 64
_HEAD_DIM = 128
_BLOCK_SIZE = 64  # entries per page
_PAGE_BYTES = 8448  # 64 * (128 + 4)
_SCALE_FP32_PER_PAGE = _PAGE_BYTES // 4  # 2112
_SCALE_OFFSET_FP32 = (_BLOCK_SIZE * _HEAD_DIM) // 4  # 2048


@triton.jit
def _mqa_logits_kernel(
    q_ptr,            # [M, NUM_HEADS, HEAD_DIM] fp8
    kv_data_ptr,      # [num_pages * 8448] fp8  (buf.view(fp8))
    kv_scale_ptr,     # [num_pages * 2112] fp32 (buf.view(fp32))
    weight_ptr,       # [M, NUM_HEADS] fp32
    block_table_ptr,  # [M, max_num_pages] int
    seq_lens_ptr,     # [M] int
    logits_ptr,       # [M, max_model_len] fp32 (pre-zeroed)
    stride_qm, stride_qh,
    stride_wm,
    stride_btm, stride_btp,
    stride_lm,
    MAX_NUM_PAGES,
    NUM_PAGES,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    SCALE_FP32_PER_PAGE: tl.constexpr,
    SCALE_OFFSET_FP32: tl.constexpr,
):
    m = tl.program_id(0)

    sl = tl.load(seq_lens_ptr + m)
    # ceil(sl / BLOCK_SIZE) — number of pages holding valid entries (runtime loop bound)
    num_valid_pages = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE
    # clamp to available pages (block_table padding may carry out-of-range indices)
    num_valid_pages = tl.minimum(num_valid_pages, NUM_PAGES)

    # q[m]: [NUM_HEADS, HEAD_DIM] fp8 -> fp32 (loaded once, reused across pages)
    q_offs = (
        m * stride_qm
        + tl.arange(0, NUM_HEADS)[:, None] * stride_qh
        + tl.arange(0, HEAD_DIM)[None, :]
    )
    q = tl.load(q_ptr + q_offs).to(tl.float32)  # [NUM_HEADS, HEAD_DIM]
    w = tl.load(weight_ptr + m * stride_wm + tl.arange(0, NUM_HEADS))  # [NUM_HEADS]
    q_t = tl.trans(q)  # [HEAD_DIM, NUM_HEADS]

    off = tl.arange(0, BLOCK_SIZE)
    for page_slot in range(num_valid_pages):
        base_e = page_slot * BLOCK_SIZE
        page = tl.load(block_table_ptr + m * stride_btm + page_slot * stride_btp).to(tl.int64)
        # clamp page index to valid range (padding entries may be -1 or >= NUM_PAGES)
        page = tl.maximum(page, 0)
        page = tl.minimum(page, NUM_PAGES - 1)
        data_base = page * PAGE_BYTES
        scale_base = page * SCALE_FP32_PER_PAGE + SCALE_OFFSET_FP32

        # kv data page: [BLOCK_SIZE, HEAD_DIM] fp8 -> fp32
        d_offs = data_base + off[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        kv = tl.load(kv_data_ptr + d_offs).to(tl.float32)  # [BLOCK_SIZE, HEAD_DIM]

        # scores[e,h] = relu(kv_e . q_h)
        dots = tl.dot(kv, q_t).to(tl.float32)       # [BLOCK_SIZE, NUM_HEADS]
        dots = tl.maximum(dots, 0.0)                 # relu
        per_entry = tl.sum(dots * w[None, :], axis=1)  # [BLOCK_SIZE]

        s_offs = scale_base + off
        kv_scale = tl.load(kv_scale_ptr + s_offs)    # [BLOCK_SIZE]
        logit_page = per_entry * kv_scale            # [BLOCK_SIZE]

        e = base_e + off
        valid = e < sl
        tl.store(logits_ptr + m * stride_lm + e, logit_page, mask=valid)


def dl_mqa_logits_triton(
    q_fp8: torch.Tensor,         # [M, 1, NUM_HEADS, HEAD_DIM] fp8
    kv_buf_u8: torch.Tensor,     # [num_pages, 8448] uint8 (the c4 indexer kv buffer)
    weight: torch.Tensor,        # [M, NUM_HEADS] fp32
    block_table: torch.Tensor,   # [M, max_num_pages] int
    seq_lens: torch.Tensor,      # [M] int
    max_model_len: int,
) -> torch.Tensor:
    """Drop-in replacement returning [M, max_model_len] fp32 logits.

    The output buffer is cached + reused (allocated on first call, typically
    during eager warmup, then reused under CUDA-graph capture). Allocating a
    fresh tensor on the indexer's alt stream DURING capture invalidates the
    graph (cudaErrorStreamCaptureInvalidated); reuse avoids that.
    """
    M = q_fp8.shape[0]
    q2 = q_fp8.view(M, _NUM_HEADS, _HEAD_DIM)
    kv_data = kv_buf_u8.view(torch.float8_e4m3fn)   # [num_pages*8448] fp8
    kv_scale = kv_buf_u8.view(torch.float32)         # [num_pages*2112] fp32

    key = (M, max_model_len, q_fp8.device.index)
    buf = _logits_buf_cache.get(key)
    if buf is None or buf.shape != (M, max_model_len):
        buf = torch.empty((M, max_model_len), dtype=torch.float32, device=q_fp8.device)
        _logits_buf_cache[key] = buf
    # No buf.zero_(): topk_transform_512_pytorch_vectorized masks entries >= seq_len
    # with -inf (idx.py:288), so garbage in unwritten (invalid) slots is ignored.

    grid = (M,)
    _mqa_logits_kernel[grid](
        q2, kv_data, kv_scale, weight.float(), block_table, seq_lens, buf,
        q2.stride(0), q2.stride(1),
        weight.stride(0),
        block_table.stride(0), block_table.stride(1),
        buf.stride(0),
        block_table.shape[1],
        kv_buf_u8.shape[0],
        NUM_HEADS=_NUM_HEADS, HEAD_DIM=_HEAD_DIM, BLOCK_SIZE=_BLOCK_SIZE,
        PAGE_BYTES=_PAGE_BYTES, SCALE_FP32_PER_PAGE=_SCALE_FP32_PER_PAGE,
        SCALE_OFFSET_FP32=_SCALE_OFFSET_FP32,
        num_warps=4,
    )
    return buf


_logits_buf_cache: dict = {}
