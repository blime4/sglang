// dlblas FP8 blockwise GEMM — the optimization kernel for sglang/DLIN.
// Wraps dlblasGemmExV2 (cublasGemmEx-style + dlblasExtQuantParametersV2) for
// blockwise FP8 W8A8 GEMM: C[M,N] = (A_fp8 * a_scale) @ (W_fp8 * w_scale)^T.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include "dlblas_ext.h"

// C[M,N] = A[M,K]_fp8 @ W[K,N]_fp8, with per-token A scale + blockwise W scale.
// A: [M,K] fp8_e4m3, W: [K,N] fp8_e4m3, a_scales: [M,1] fp32, w_scales: [K/block_k, N/block_n] fp32.
torch::Tensor dlblas_fp8_blockwise_gemm(
    torch::Tensor A, torch::Tensor W,
    torch::Tensor a_scales, torch::Tensor w_scales,
    int64_t block_k, int64_t block_n
) {
    int M = A.size(0), K = A.size(1), N = W.size(1);
    auto C = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    float alpha = 1.0f, beta = 0.0f;

    dlblasExtQuantParametersV2_t qp;
    memset(&qp, 0, sizeof(qp));
    qp.a_group_size_m = 1;
    qp.a_group_size_k = K;  // per-token (whole row)
    qp.a_scales = a_scales.data_ptr();
    qp.a_scales_type = CUDA_R_32F;
    qp.b_group_size_k = block_k;
    qp.b_group_size_n = block_n;
    qp.b_scales = w_scales.data_ptr();
    qp.b_scales_type = CUDA_R_32F;

    // Row-major A[M,K], W[K,N], C[M,N]. cublas is col-major:
    // A_col[K,M] ld=K, transa=T → [M,K]. W_col[N,K] ld=N, transb=T → [K,N].
    cublasStatus_t rc = dlblasGemmExV2(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        M, N, K,
        &alpha,
        A.data_ptr(), CUDA_R_8F_E4M3, K,
        W.data_ptr(), CUDA_R_8F_E4M3, N,
        &beta,
        C.data_ptr(), CUDA_R_16BF, N,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT, &qp
    );
    TORCH_CHECK(rc == CUBLAS_STATUS_SUCCESS, "dlblasGemmExV2 failed, rc=", rc);
    return C;
}

// Sweep: try per-tensor FP8 with various compute/output types + algos to find a working config.
std::vector<int> dlblas_fp8_probe(torch::Tensor A, torch::Tensor W,
                                  torch::Tensor a_scale, torch::Tensor w_scale) {
    int M = A.size(0), K = A.size(1), N = W.size(1);
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    float alpha = 1.0f, beta = 0.0f;

    // per-tensor quant params
    dlblasExtQuantParametersV2_t qp;
    memset(&qp, 0, sizeof(qp));
    qp.a_group_size_m = M; qp.a_group_size_k = K;
    qp.a_scales = a_scale.data_ptr(); qp.a_scales_type = CUDA_R_32F;
    qp.b_group_size_k = K; qp.b_group_size_n = N;
    qp.b_scales = w_scale.data_ptr(); qp.b_scales_type = CUDA_R_32F;

    std::vector<int> results;
    int compute_types[] = {CUDA_R_32F};  // 0
    int out_types[] = {CUDA_R_16BF, CUDA_R_32F};  // 14, 0
    int algos[] = {0, 1, 99};  // DEFAULT, ALGO0, DEFAULT_TENSOR_OP

    for (int ct : compute_types) {
      for (int ot : out_types) {
        for (int algo : algos) {
          auto C = torch::zeros({M, N}, A.options().dtype(
            ot == CUDA_R_16BF ? torch::kBFloat16 : torch::kFloat32));
          cublasStatus_t rc = dlblasGemmExV2(
              handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha,
              A.data_ptr(), CUDA_R_8F_E4M3, K,
              W.data_ptr(), CUDA_R_8F_E4M3, N,
              &beta, C.data_ptr(), (cudaDataType)ot, N,
              (cudaDataType)ct, (cublasGemmAlgo_t)algo, &qp
          );
          results.push_back(rc);
        }
      }
    }
    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlblas_fp8_blockwise_gemm", &dlblas_fp8_blockwise_gemm,
          "dlblas FP8 blockwise GEMM (W8A8, 2D block quant)");
    m.def("dlblas_fp8_probe", &dlblas_fp8_probe,
          "sweep FP8 configs (compute/out types x algos) to find a working one");
}
