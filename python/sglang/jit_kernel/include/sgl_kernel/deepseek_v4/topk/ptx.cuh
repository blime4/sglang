#pragma once
#include <sgl_kernel/utils.cuh>

#include <cstdint>

// DL begin: cuda/ptx (Hopper TMA/mbarrier/elect_sync) not available on DLIN's
// dlcc. Provide no-op stubs in the #else branch so all topk headers compile.
// The TMA/streaming paths are never called at runtime (kClusterSize=1 routes
// everything through the small single-block path for decode).
#ifndef SGL_ON_DLIN
#include <cuda/ptx>
#endif

namespace device::top512 {

namespace ptx {

#ifndef SGL_ON_DLIN
SGL_DEVICE void mbarrier_wait(uint64_t* addr, uint32_t phase) {
  while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, addr, phase))
    ;
}

SGL_DEVICE void mbarrier_init(uint64_t* addr, uint32_t arrives) {
  cuda::ptx::mbarrier_init(addr, arrives);
}

SGL_DEVICE void mbarrier_arrive_expect_tx(uint64_t* addr, uint32_t tx) {
  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::space_shared, addr, tx);
}

SGL_DEVICE void mbarrier_arrive(uint64_t* addr) {
  cuda::ptx::mbarrier_arrive(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::space_shared, addr);
}

SGL_DEVICE void tma_load(void* dst, const void* src, uint32_t num_bytes, uint64_t* mbar) {
  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global, dst, src, num_bytes, mbar);
}

SGL_DEVICE uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

SGL_DEVICE bool elect_sync_cta(uint32_t tx) {
  const auto warp_id = tx / 32;
  const auto uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
  return (uniform_warp_id == 0 && elect_sync());
}
#else
// DL stubs: no-op implementations so streaming.cuh + register.cuh compile on DLIN.
// Never called at runtime (kClusterSize=1, decode uses small path only).
SGL_DEVICE void mbarrier_wait(uint64_t* addr, uint32_t phase) {}
SGL_DEVICE void mbarrier_init(uint64_t* addr, uint32_t arrives) {}
SGL_DEVICE void mbarrier_arrive_expect_tx(uint64_t* addr, uint32_t tx) {}
SGL_DEVICE void mbarrier_arrive(uint64_t* addr) {}
SGL_DEVICE void tma_load(void* dst, const void* src, uint32_t num_bytes, uint64_t* mbar) {}
SGL_DEVICE uint32_t elect_sync() { return threadIdx.x == 0 ? 1u : 0u; }
SGL_DEVICE bool elect_sync_cta(uint32_t tx) { return tx == 0; }
#endif

}  // namespace ptx

}  // namespace device::top512
