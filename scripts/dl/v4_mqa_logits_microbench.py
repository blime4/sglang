#!/usr/bin/env python3
"""DL: isolate-time the Triton MQA logits kernel vs the DL op on realistic inputs.
Pinpoints whether slowness is the kernel, the fp8/dot, the dynamic loop, or the
torch.zeros wrapper allocation."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")
import torch, time

NUM_HEADS, HEAD_DIM, BLOCK_SIZE = 64, 128, 64
PAGE_BYTES = 8448


def main():
    from sglang.jit_kernel.dsv4.dl_mqa_logits_triton import dl_mqa_logits_triton
    d = torch.device("cuda")
    torch.zeros(1, device=d)  # init ctx

    # realistic decode: M=1, a few pages of KV, sl small
    for (num_pages, sl, max_model_len, label) in [
        (256, 3, 16384, "default-ctx sl=3 (256 pages cap)"),
        (256, 66, 16384, "default-ctx sl=66"),
        (4, 3, 192, "small-ctx sl=3 (4 pages)"),
    ]:
        torch.manual_seed(0)
        kv_buf = torch.randint(0, 200, (num_pages, PAGE_BYTES), dtype=torch.uint8, device=d)
        q = torch.randint(0, 200, (1, 1, NUM_HEADS, HEAD_DIM), dtype=torch.uint8, device=d).view(torch.float8_e4m3fn)
        weight = torch.randn(1, NUM_HEADS, dtype=torch.float32, device=d)
        bt = torch.arange(num_pages, dtype=torch.int32, device=d).view(1, num_pages)
        seq_lens = torch.tensor([sl], dtype=torch.int32, device=d)

        # warmup triton
        for _ in range(5):
            dl_mqa_logits_triton(q, kv_buf, weight, bt, seq_lens, max_model_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            out_t = dl_mqa_logits_triton(q, kv_buf, weight, bt, seq_lens, max_model_len)
        torch.cuda.synchronize()
        dt_t = (time.perf_counter() - t0) / 50 * 1e3

        # DL op comparison (may fail on synthetic inputs — that's OK)
        dt_d = -1.0
        out_d = None
        try:
            _sm = torch.empty(0, dtype=torch.int32, device=d)
            kv_view = kv_buf.view(num_pages, BLOCK_SIZE, 1, HEAD_DIM + 4)
            for _ in range(5):
                torch.ops.sgl_kernel.fp8_fp4_paged_mqa_logits(
                    q.contiguous(), None, kv_view, weight.float(), seq_lens, bt, _sm, max_model_len, False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                out_d = torch.ops.sgl_kernel.fp8_fp4_paged_mqa_logits(
                    q.contiguous(), None, kv_view, weight.float(), seq_lens, bt, _sm, max_model_len, False)
            torch.cuda.synchronize()
            dt_d = (time.perf_counter() - t0) / 50 * 1e3
        except Exception as e:
            print(f"  [DL op skipped: {type(e).__name__}]", flush=True)

        # correctness (only if DL op produced output)
        if out_d is not None and out_t.shape == out_d.shape:
            mdv = (out_t.float() - out_d.float()).abs().max().item()
        else:
            mdv = "n/a"
        dl_str = f"DL_op={dt_d:.2f}ms" if dt_d > 0 else "DL_op=skipped"
        print(f"[{label}] triton={dt_t:.3f}ms  {dl_str}  max_diff={mdv}  "
              f"triton_out={tuple(out_t.shape)}", flush=True)


if __name__ == "__main__":
    main()
