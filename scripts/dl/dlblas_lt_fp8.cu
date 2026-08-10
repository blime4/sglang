// dlblasLtMatmul FP8 blockwise GEMM — the optimization kernel for sglang/DLIN.
// C = (A_fp8 * a_scale) @ (W_fp8 * w_scale)^T, blockwise 128x128 weight quant.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include "dlblasLt_ext.h"
#include <vector>

torch::Tensor dlblas_lt_fp8_gemm(
    torch::Tensor A,      // [M,K] fp8 e4m3
    torch::Tensor W,      // [K,N] fp8 e4m3
    torch::Tensor a_scales, // [M,1] fp32 (per-token)
    torch::Tensor w_scales, // [K/bk, N/bn] fp32 (blockwise)
    int64_t bk, int64_t bn  // block sizes (128, 128)
) {
    int M = A.size(0), K = A.size(1), N = W.size(1);
    auto C = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 1. cublasLt handle
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // 2. Matmul descriptor
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    int32_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    // 3. Matrix layouts (cublasLt is col-major; A_row[M,K] → col[K,M], W_row[K,N] → col[N,K])
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, M, K);  // [K,M] col, ld=K
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, N, K, N);  // [N,K] col, ld=N
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, N);     // [N,M] col, ld=N

    // 4. Scale layouts for quant params
    cublasLtMatrixLayout_t aScaleDesc, wScaleDesc;
    cublasLtMatrixLayoutCreate(&aScaleDesc, CUDA_R_32F, M, 1, M);          // [M,1]
    cublasLtMatrixLayoutCreate(&wScaleDesc, CUDA_R_32F, K/bk, N/bn, K/bk); // [K/bk, N/bn]

    // 5. dlblasLt quant params
    dlblasLtQuantParamsConfig_t aqConfig, bqConfig;
    dlblasLtQuantParamsConfigCreate(&aqConfig);
    dlblasLtQuantParamsConfigSet(aqConfig, aScaleDesc, a_scales.data_ptr(), 0, nullptr,
                                 DLBLASLT_QUANT_OP_DEQUANTIZE);
    dlblasLtQuantParamsConfigSetGroupSize(aqConfig, 1, K);  // per-token

    dlblasLtQuantParamsConfigCreate(&bqConfig);
    dlblasLtQuantParamsConfigSet(bqConfig, wScaleDesc, w_scales.data_ptr(), 0, nullptr,
                                 DLBLASLT_QUANT_OP_DEQUANTIZE);
    dlblasLtQuantParamsConfigSetGroupSize(bqConfig, (int32_t)bk, (int32_t)bn);  // blockwise

    // 6. Workspace
    size_t wsSize = 32 * 1024 * 1024;  // 32MB default
    auto workspace = torch::empty({(int64_t)wsSize}, torch::dtype(torch::kUInt8).device(torch::kCUDA));

    // 7. Matmul
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t rc = dlblasLtMatmul(
        ltHandle, opDesc, &alpha,
        A.data_ptr(), Adesc, aqConfig,
        W.data_ptr(), Bdesc, bqConfig,
        &beta,
        C.data_ptr(), Cdesc, nullptr,
        C.data_ptr(), Cdesc, nullptr,
        nullptr, workspace.data_ptr(), wsSize, stream
    );

    // Cleanup
    dlblasLtQuantParamsConfigDestroy(aqConfig);
    dlblasLtQuantParamsConfigDestroy(bqConfig);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(aScaleDesc);
    cublasLtMatrixLayoutDestroy(wScaleDesc);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(ltHandle);

    TORCH_CHECK(rc == CUBLAS_STATUS_SUCCESS, "dlblasLtMatmul FP8 failed, rc=", rc);
    return C;
}

// bf16 via dlblasLtMatmul (no quant) — validates cublasLt descriptor setup.
torch::Tensor dlblas_lt_bf16_gemm(torch::Tensor A, torch::Tensor W) {
    int M = A.size(0), K = A.size(1), N = W.size(1);
    auto C = torch::empty({M, N}, A.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasLtHandle_t h; cublasLtCreate(&h);
    cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    int32_t t = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &t, sizeof(t));
    cublasLtMatrixLayout_t aD, bD, cD;
    cublasLtMatrixLayoutCreate(&aD, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&bD, CUDA_R_16BF, N, K, N);
    cublasLtMatrixLayoutCreate(&cD, CUDA_R_16BF, N, M, N);
    auto ws = torch::empty({32*1024*1024}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t rc = dlblasLtMatmul(h, d, &alpha,
        A.data_ptr(), aD, nullptr,
        W.data_ptr(), bD, nullptr,
        &beta, C.data_ptr(), cD, nullptr,
        C.data_ptr(), cD, nullptr,
        nullptr, ws.data_ptr(), ws.numel(), stream);
    cublasLtMatrixLayoutDestroy(aD); cublasLtMatrixLayoutDestroy(bD); cublasLtMatrixLayoutDestroy(cD);
    cublasLtMatmulDescDestroy(d); cublasLtDestroy(h);
    TORCH_CHECK(rc == CUBLAS_STATUS_SUCCESS, "dlblasLtMatmul bf16 failed, rc=", rc);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlblas_lt_fp8_gemm", &dlblas_lt_fp8_gemm, "dlblasLtMatmul FP8 blockwise GEMM");
    m.def("dlblas_lt_bf16_gemm", &dlblas_lt_bf16_gemm, "dlblasLtMatmul bf16 (no quant)");
}
