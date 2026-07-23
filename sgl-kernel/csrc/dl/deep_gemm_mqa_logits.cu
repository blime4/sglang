// DL begin: vendored from Denglin vLLM fork csrc/dl/deep_gemm_mqa_logits.cu (port plan 4b/4d).
#include <cstdint>
#include <cstdio>
#include <memory>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Descriptors.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <dldnn_ext.h>
#include <cublasLt.h>
#include <dlblasLt_ext.h>

namespace {

template <typename A, typename B>
static constexpr auto align(A a, B b) -> decltype(a) {
    return ((a + b - 1) / b) * b;
}

// RAII wrappers for cublasLt / dlblasLt descriptors (following PyTorch pattern)
template <typename T, cublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDABLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, cublasStatus_t (*destructor)(T*)>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const { return descriptor_.get(); }
  T* descriptor() { return descriptor_.get(); }

 protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};

class CuBlasLtMatrixLayout : public CuBlasLtDescriptor<
                                 cublasLtMatrixLayoutOpaque_t,
                                 &cublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtMatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols,
                        int64_t ld) {
    cublasLtMatrixLayout_t raw = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&raw, type, rows, cols, ld));
    descriptor_.reset(raw);
  }
  template <typename T>
  void setAttribute(cublasLtMatrixLayoutAttribute_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatmulDescriptor : public CuBlasLtDescriptor<
                                     cublasLtMatmulDescOpaque_t,
                                     &cublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(cublasComputeType_t compute_type,
                            cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&raw, compute_type, scale_type));
    descriptor_.reset(raw);
  }
  template <typename T>
  void setAttribute(cublasLtMatmulDescAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

// dlblasLt uses pointer-to-incomplete-struct (not opaque struct like cublasLt),
// so it needs a separate RAII wrapper.
class DlBlasLtQuantParamsConfig {
 public:
  DlBlasLtQuantParamsConfig() : config_(nullptr) {
    TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigCreate(&config_));
  }
  ~DlBlasLtQuantParamsConfig() {
    if (config_) dlblasLtQuantParamsConfigDestroy(config_);
  }
  DlBlasLtQuantParamsConfig(const DlBlasLtQuantParamsConfig&) = delete;
  DlBlasLtQuantParamsConfig& operator=(const DlBlasLtQuantParamsConfig&) = delete;
  DlBlasLtQuantParamsConfig(DlBlasLtQuantParamsConfig&& o) noexcept
      : config_(o.config_) { o.config_ = nullptr; }
  DlBlasLtQuantParamsConfig& operator=(DlBlasLtQuantParamsConfig&& o) noexcept {
    if (this != &o) {
      if (config_) dlblasLtQuantParamsConfigDestroy(config_);
      config_ = o.config_;
      o.config_ = nullptr;
    }
    return *this;
  }
  dlblasLtQuantParamsConfig_t get() const { return config_; }
  void set(cublasLtMatrixLayout_t scale_layout, const void* scale_data,
            cublasLtMatrixLayout_t zp_layout, const void* zp_data,
            dlblasLtQuantOp_t direction) {
    TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigSet(
        config_, scale_layout, scale_data, zp_layout, zp_data, direction));
  }
  void setGroupSize(int32_t row, int32_t col) {
    TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigSetGroupSize(config_, row, col));
  }
 private:
  dlblasLtQuantParamsConfig_t config_;
};

inline cudnnDataType_t getDataType(const at::Tensor& t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (scalar_type == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  } else if (scalar_type == at::kQInt8) {
    return CUDNN_DATA_INT8;
  } else if (scalar_type == at::kInt) {
    return CUDNN_DATA_INT32;
  } else if (scalar_type == at::kFloat8_e4m3fn) {
    return CUDNN_DATA_FP8_E4M3;
  } else if (scalar_type == at::kFloat8_e5m2) {
    return CUDNN_DATA_FP8_E5M2;
  } else if (scalar_type == at::kByte) {
    return CUDNN_DATA_UINT8;
  } else if (scalar_type == at::kChar) {
    return CUDNN_DATA_INT8;
  }
  TORCH_CHECK(false, "TensorDescriptor does not support ", scalar_type);
}

struct TensorInfo {
  void* ptr = nullptr;
  at::native::TensorDescriptor desc;

  TensorInfo() {
    desc.set(CUDNN_DATA_FLOAT, {1, 1, 1}, {1, 1, 1});
  }

  TensorInfo(const torch::Tensor& tensor)
      : ptr(tensor.data_ptr()) {
    desc.set(getDataType(tensor), tensor.sizes(), tensor.strides());
  }

  TensorInfo(const c10::optional<torch::Tensor>& tensor) {
    if (tensor) {
      torch::Tensor t = tensor.value();
      ptr = t.data_ptr();
      desc.set(getDataType(t), t.sizes(), t.strides());
    } else {
      desc.set(CUDNN_DATA_FLOAT, {1, 1, 1}, {1, 1, 1});
    }
  }
};

} // namespace

torch::Tensor fp8_fp4_mqa_logits(
    const torch::Tensor& q_values, const c10::optional<torch::Tensor>& q_scale,
    const torch::Tensor& kv_packed, const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seqlen_ks, const torch::Tensor& cu_seqlen_ke,
    bool clean_logits,
    int64_t max_seqlen_k,
    at::ScalarType logits_type) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_values));
  auto handle = at::native::getCudnnHandle();

  TensorInfo q_info(q_values);
  TensorInfo q_scales_info(q_scale);
  TensorInfo kv_info(kv_packed);
  TensorInfo kv_scales_info(kv_scales);
  TensorInfo weights_info(weights);
  TensorInfo ks_info(cu_seqlen_ks);
  TensorInfo ke_info(cu_seqlen_ke);

  //auto print_info_desc = [](const char* name, const torch::Tensor& t) {
  //  torch::Tensor expanded = t;
  //  printf("[fp8_fp4_mqa_logits] %s info: dim=%ld, sizes=[", name, expanded.dim());
  //  for (int64_t i = 0; i < expanded.dim(); i++) printf("%s%ld", i ? ", " : "", expanded.size(i));
  //  printf("], strides=[");
  //  for (int64_t i = 0; i < expanded.dim(); i++) printf("%s%ld", i ? ", " : "", expanded.stride(i));
  //  printf("]\n");
  //};

  //print_info_desc("q_values", q_values);
  //if (q_scale.has_value()) print_info_desc("q_scale", q_scale.value());
  //print_info_desc("kv_packed", kv_packed);
  //print_info_desc("kv_scales", kv_scales);
  //print_info_desc("weights", weights);
  //print_info_desc("cu_seqlen_ks", cu_seqlen_ks);
  //print_info_desc("cu_seqlen_ke", cu_seqlen_ke);

  const int64_t M = q_values.size(0);
  const int64_t num_heads = q_values.size(1);
  const int64_t N = kv_packed.size(0);
  //printf("[fp8_fp4_mqa_logits] M=%ld, N=%ld, max_seqlen_k=%ld, clean_logits=%d, logits_type=%s\n",
  //       M, N, max_seqlen_k, clean_logits, toString(logits_type));

  constexpr int block_qh = 128;
  constexpr int block_kv = 256;
  const int block_q = block_qh / num_heads;
  TORCH_CHECK(block_qh % num_heads == 0,
              "block_qh (", block_qh, ") must be divisible by num_heads (", num_heads, ")");
  int64_t aligned_seq_len = align(M, block_q);
  int64_t stride_logits = align(N, block_kv);

  auto logits = at::empty({aligned_seq_len, stride_logits}, q_values.options().dtype(logits_type));
  logits = logits.index({torch::indexing::Slice(0, M), torch::indexing::Slice(0, N)});
  TensorInfo logits_info(logits);

  AT_CUDNN_CHECK(cudnnDeepGemmMqaLogits(
      handle,
      q_info.desc.desc(), q_info.ptr,
      q_scales_info.desc.desc(), q_scales_info.ptr,
      kv_info.desc.desc(), kv_info.ptr,
      kv_scales_info.desc.desc(), kv_scales_info.ptr,
      weights_info.desc.desc(), weights_info.ptr,
      ks_info.desc.desc(), ks_info.ptr,
      ke_info.desc.desc(), ke_info.ptr,
      clean_logits,
      max_seqlen_k,
      logits_info.desc.desc(), logits_info.ptr));

  return logits;
}

torch::Tensor fp8_fp4_paged_mqa_logits(
    const torch::Tensor& q_values, const c10::optional<torch::Tensor>& q_scale,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_tables,
    const torch::Tensor& schedule_metadata,
    int64_t max_model_len, bool clean_logits = false,
    at::ScalarType logits_type = at::kFloat,
    const c10::optional<torch::Tensor>& indices = c10::nullopt) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_values));
  auto handle = at::native::getCudnnHandle();

  //auto print_tensor_info = [](const char* name, const torch::Tensor& t) {
  //  printf("[fp8_fp4_paged_mqa_logits] %s: dtype=%s, dims=%ld, sizes=[", name, toString(t.scalar_type()), t.dim());
  //  for (int64_t i = 0; i < t.dim(); i++) printf("%s%ld", i ? ", " : "", t.size(i));
  //  printf("], strides=[");
  //  for (int64_t i = 0; i < t.dim(); i++) printf("%s%ld", i ? ", " : "", t.stride(i));
  //  printf("]\n");
  //};

  //print_tensor_info("q_values", q_values);
  //if (q_scale.has_value()) print_tensor_info("q_scale", q_scale.value());
  //else printf("[fp8_fp4_paged_mqa_logits] q_scale: nullopt\n");
  //print_tensor_info("kv_cache", kv_cache);
  //print_tensor_info("weights", weights);
  //print_tensor_info("context_lens", context_lens);
  //print_tensor_info("block_tables", block_tables);
  //print_tensor_info("schedule_metadata", schedule_metadata);
  //if (indices.has_value()) print_tensor_info("indices", indices.value());
  //else printf("[fp8_fp4_paged_mqa_logits] indices: nullopt\n");

  TensorInfo q_info(q_values);
  TensorInfo q_scales_info(q_scale);
  TensorInfo kv_cache_info(kv_cache);
  TensorInfo weights_info(weights);
  TensorInfo context_lens_info(context_lens);
  TensorInfo block_tables_info(block_tables);
  TensorInfo schedule_meta_info(schedule_metadata);
  TensorInfo indices_info(indices);

  // q_values: [B, next_n, H, D] -> output: [B * next_n, max_model_len]
  const int64_t B = q_values.size(0);
  const int64_t next_n = q_values.size(1);
  //printf("[fp8_fp4_paged_mqa_logits] B=%ld, next_n=%ld, max_model_len=%ld, clean_logits=%d, logits_type=%s\n",
  //       B, next_n, max_model_len, clean_logits, toString(logits_type));


  constexpr int split_kv = 256;
  const auto aligned_max_context_len = align(max_model_len, split_kv);
  auto logits =
      at::empty({B * next_n, aligned_max_context_len},
                q_values.options().dtype(logits_type));
  logits = logits.slice(-1, 0, max_model_len);
  TensorInfo logits_info(logits);
  //print_tensor_info("logits", logits);

  AT_CUDNN_CHECK(cudnnDeepGemmPagedMqaLogits(
      handle,
      q_info.desc.desc(), q_info.ptr,
      q_scales_info.desc.desc(), q_scales_info.ptr,
      kv_cache_info.desc.desc(), kv_cache_info.ptr,
      weights_info.desc.desc(), weights_info.ptr,
      context_lens_info.desc.desc(), context_lens_info.ptr,
      block_tables_info.desc.desc(), block_tables_info.ptr,
      schedule_meta_info.desc.desc(), schedule_meta_info.ptr,
      static_cast<int32_t>(max_model_len),
      clean_logits,
      indices_info.desc.desc(), indices_info.ptr,
      logits_info.desc.desc(), logits_info.ptr));

  return logits;
}

void fp8_einsum(
    const torch::Tensor& a_values,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_values,
    const torch::Tensor& b_scale,
    torch::Tensor& out,
    const std::string& equation,
    const std::vector<int64_t>& recipe) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a_values));
    // Dimensions: a=[B,H,R], b=[H,D,R], out=[B,H,D]
  const int64_t B = a_values.size(0);
  const int64_t H = a_values.size(1);
  const int64_t R = a_values.size(2);
  const int64_t D = b_values.size(1);

  // GEMM mapping [mbk] * [bnk] -> [mbn]
  const int64_t M = B;
  const int64_t N = D;
  const int64_t K = R;
  const int64_t batch = H;

  const int64_t gs_a_col = (K + a_scale.size(2) - 1) / a_scale.size(2);
  const int64_t gs_b_row = (N + b_scale.size(1) - 1) / b_scale.size(1);
  const int64_t gs_b_col = (K + b_scale.size(2) - 1) / b_scale.size(2);

  cublasLtHandle_t ltHandle =
  reinterpret_cast<cublasLtHandle_t>(at::cuda::getCurrentCUDABlasHandle());

  const int64_t lda = batch * K;
  const int64_t ldb = K;
  const int64_t ldc = batch * N;
  const int64_t ldd = ldc;
  const int64_t strideA = K;
  const int64_t strideB = ldb * N;
  const int64_t strideC = N;
  const int64_t strideD = N;

  CuBlasLtMatrixLayout Adesc(CUDA_R_8F_E4M3, M, K, lda);
  CuBlasLtMatrixLayout Bdesc(CUDA_R_8F_E4M3, N, K, ldb);
  CuBlasLtMatrixLayout Cdesc(CUDA_R_16BF, M, N, ldc);
  CuBlasLtMatrixLayout Ddesc(CUDA_R_16BF, M, N, ldd);

  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, row_order);
  Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, row_order);
  Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, row_order);
  Ddesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, row_order);

  Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch);
  Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideA);
  Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch);
  Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideB);
  Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch);
  Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideC);
  Ddesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch);
  Ddesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideD);

  // Scale A layout
  const int64_t num_groups_a = a_scale.size(2);
  CuBlasLtMatrixLayout scaleALayout(CUDA_R_32F, M * batch, num_groups_a, num_groups_a);
  scaleALayout.setAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, row_order);

  DlBlasLtQuantParamsConfig AqConfig;
  AqConfig.set(scaleALayout.descriptor(), a_scale.data_ptr(),
               nullptr, nullptr, DLBLASLT_QUANT_OP_DEQUANTIZE);
  AqConfig.setGroupSize(1, gs_a_col);

  // Scale B layout
  const int64_t num_groups_b_row = b_scale.size(1);
  const int64_t num_groups_b_col = b_scale.size(2);
  CuBlasLtMatrixLayout scaleBLayout(CUDA_R_32F, batch * num_groups_b_row, num_groups_b_col, num_groups_b_col);
  scaleBLayout.setAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, row_order);

  DlBlasLtQuantParamsConfig BqConfig;
  BqConfig.set(scaleBLayout.descriptor(), b_scale.data_ptr(),
               nullptr, nullptr, DLBLASLT_QUANT_OP_DEQUANTIZE);
  BqConfig.setGroupSize(gs_b_row, gs_b_col);

  // Matmul descriptor
  CuBlasLtMatmulDescriptor matmulDesc(CUBLAS_COMPUTE_32F, CUDA_R_32F);
  matmulDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_N);
  matmulDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_T);

  float alpha = 1.0f;
  float beta = 0.0f;
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CUDABLAS_CHECK(dlblasLtMatmul(
      ltHandle, matmulDesc.descriptor(), &alpha,
      a_values.data_ptr(), Adesc.descriptor(), AqConfig.get(),
      b_values.data_ptr(), Bdesc.descriptor(), BqConfig.get(),
      &beta,
      nullptr, Cdesc.descriptor(), nullptr,
      out.data_ptr(), Ddesc.descriptor(), nullptr,
      nullptr, nullptr, 0, stream));
}
// DL end
