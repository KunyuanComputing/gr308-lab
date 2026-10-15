#pragma once

#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {

__global__ void bf16_copy_kernel(
    const __hip_bfloat16* input,
    __hip_bfloat16* output,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index];
  }
}

hipError_t launch_bf16_copy(
    const void* input,
    void* output,
    int64_t count,
    hipStream_t stream) {
  if (count == 0) {
    return hipSuccess;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(
      bf16_copy_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      static_cast<const __hip_bfloat16*>(input),
      static_cast<__hip_bfloat16*>(output),
      count);
  return hipGetLastError();
}

}  // namespace
