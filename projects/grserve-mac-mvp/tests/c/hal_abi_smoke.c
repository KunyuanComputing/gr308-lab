#include "grserve/hal.h"

#include <stdint.h>

int main(void) {
  grserve_hal_api_request_t request = {0};
  grserve_hal_api_t api = {0};
  request.struct_size = (uint32_t)sizeof(request);
  request.abi_major = GRSERVE_HAL_ABI_MAJOR;
  request.min_abi_minor = GRSERVE_HAL_ABI_MINOR;
  api.struct_size = (uint32_t)sizeof(api);
  return request.struct_size == 0u || api.struct_size == 0u;
}
