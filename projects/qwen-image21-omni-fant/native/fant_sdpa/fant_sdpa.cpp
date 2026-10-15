#include <ATen/ATen.h>
#include <torch/library.h>

#include "torch_fant/csrc/core/HIPStream.h"

#include "fant_sdpa_kernel.hip"

namespace {

at::Tensor copy_smoke(const at::Tensor& input) {
  TORCH_CHECK(
      input.device().type() == c10::DeviceType::PrivateUse1,
      "fant_sdpa::copy_smoke expects a FANT tensor");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16,
      "fant_sdpa::copy_smoke expects BF16 input");
  TORCH_CHECK(
      input.is_contiguous(),
      "fant_sdpa::copy_smoke expects contiguous input");

  auto output = at::empty_like(input);
  auto stream = c10::fant::getCurrentHIPStream(input.device().index());
  const hipError_t status = launch_bf16_copy(
      input.data_ptr(), output.data_ptr(), input.numel(), stream.stream());
  TORCH_CHECK(
      status == hipSuccess,
      "fant_sdpa::copy_smoke kernel launch failed: ",
      hipGetErrorString(status));
  return output;
}

at::Tensor sdpa_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value) {
  TORCH_CHECK(false, "fant_sdpa::forward is not implemented in the SDK probe");
}

}  // namespace

TORCH_LIBRARY(fant_sdpa, m) {
  m.def("copy_smoke(Tensor input) -> Tensor");
  m.def("forward(Tensor query, Tensor key, Tensor value) -> Tensor");
}

TORCH_LIBRARY_IMPL(fant_sdpa, PrivateUse1, m) {
  m.impl("copy_smoke", copy_smoke);
  m.impl("forward", sdpa_forward);
}
