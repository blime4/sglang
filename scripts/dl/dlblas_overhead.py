import os, torch, time
torch.ops.load_library("../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so")

for M in [1, 8, 64]:
    K, N = 2048, 512
    BK, BN = 128, 128
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.float8_e4m3fn, device="cuda")
    sc = torch.ones(N//BN, K//BK, dtype=torch.float32, device="cuda")
    # warmup
    for _ in range(10):
        torch.ops._dl_C.gptq_dlblas_gemmex(x, w.t(), sc, sc, quant_type=2, bit=8)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        torch.ops._dl_C.gptq_dlblas_gemmex(x, w.t(), sc, sc, quant_type=2, bit=8)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 1000
    print(f"M={M:3d} [{M}x{K}]x[{K}x{N}]: {dt*1000:.3f} ms/call")
