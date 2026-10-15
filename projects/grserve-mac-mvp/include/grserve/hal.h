#ifndef GRSERVE_HAL_H_
#define GRSERVE_HAL_H_

#include <stddef.h>
#include <stdint.h>

/*
 * This is the C ABI boundary between GRServe and a HAL backend.  It is not a
 * C++ interface: implementations must not allow C++ exceptions to cross any
 * callback or grserve_hal_get_api boundary.
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(GRSERVE_HAL_BUILDING_LIBRARY)
#define GRSERVE_HAL_EXPORT __declspec(dllexport)
#else
#define GRSERVE_HAL_EXPORT __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define GRSERVE_HAL_EXPORT __attribute__((visibility("default")))
#else
#define GRSERVE_HAL_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Kept in lockstep with grserve::kHalAbiVersion without including C++ headers. */
#define GRSERVE_HAL_ABI_MAJOR UINT32_C(1)
#define GRSERVE_HAL_ABI_MINOR UINT32_C(0)

typedef uint32_t grserve_hal_status_t;
#define GRSERVE_HAL_STATUS_OK UINT32_C(0)
#define GRSERVE_HAL_STATUS_INVALID_ARGUMENT UINT32_C(1)
#define GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI UINT32_C(2)
#define GRSERVE_HAL_STATUS_UNSUPPORTED UINT32_C(3)
#define GRSERVE_HAL_STATUS_OUT_OF_MEMORY UINT32_C(4)
#define GRSERVE_HAL_STATUS_TIMEOUT UINT32_C(5)
#define GRSERVE_HAL_STATUS_INTERNAL UINT32_C(6)
#define GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL UINT32_C(7)

typedef uint32_t grserve_hal_event_state_t;
#define GRSERVE_HAL_EVENT_STATE_PENDING UINT32_C(0)
#define GRSERVE_HAL_EVENT_STATE_COMPLETE UINT32_C(1)
#define GRSERVE_HAL_EVENT_STATE_FAILED UINT32_C(2)

typedef uint64_t grserve_hal_capabilities_t;
#define GRSERVE_HAL_CAPABILITY_STREAMS UINT64_C(0x0000000000000001)
#define GRSERVE_HAL_CAPABILITY_EVENTS UINT64_C(0x0000000000000002)
#define GRSERVE_HAL_CAPABILITY_ASYNC_COPY UINT64_C(0x0000000000000004)
#define GRSERVE_HAL_CAPABILITY_EXECUTABLES UINT64_C(0x0000000000000008)

/*
 * Opaque handles are owned by the backend which created them. A successful
 * create call transfers responsibility for the handle's lifetime to the
 * caller, which releases it through the matching destroy callback.
 */
typedef struct grserve_hal_runtime_s* grserve_hal_runtime_t;
typedef struct grserve_hal_device_s* grserve_hal_device_t;
typedef struct grserve_hal_stream_s* grserve_hal_stream_t;
typedef struct grserve_hal_event_s* grserve_hal_event_t;
typedef struct grserve_hal_buffer_s* grserve_hal_buffer_t;
typedef struct grserve_hal_executable_s* grserve_hal_executable_t;

/* Caller-owned, extensible input to grserve_hal_get_api. */
typedef struct grserve_hal_api_request_t {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t min_abi_minor;
  grserve_hal_capabilities_t required_capabilities;
} grserve_hal_api_request_t;

/* Caller-owned, extensible input to stream_submit. */
typedef struct grserve_hal_submit_info_t {
  uint32_t struct_size;
  const grserve_hal_event_t* wait_events;
  uint32_t wait_event_count;
  const grserve_hal_event_t* signal_events;
  uint32_t signal_event_count;
} grserve_hal_submit_info_t;

/* Caller-owned, extensible input to executable_launch. */
typedef struct grserve_hal_launch_info_t {
  uint32_t struct_size;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
  const void* argument_data;
  uint64_t argument_bytes;
} grserve_hal_launch_info_t;

/*
 * String copy protocol:
 * - out_required_bytes is required. For every valid call, it receives the
 *   complete required byte count including the terminating NUL (one for an
 *   empty string).
 * - out_bytes == NULL and out_capacity == 0 is a size query and returns OK.
 * - out_bytes must otherwise be non-NULL and out_capacity must be nonzero.
 *   Any invalid pointer/capacity combination returns INVALID_ARGUMENT and
 *   leaves output bytes and *out_required_bytes unchanged.
 * - If out_capacity is sufficient, output is the full NUL-terminated string
 *   and the callback returns OK. If nonzero out_capacity is insufficient,
 *   output is a NUL-terminated prefix, *out_required_bytes still reports the
 *   full count, and the callback returns BUFFER_TOO_SMALL.
 */
typedef grserve_hal_status_t (*grserve_hal_string_copy_fn)(
    char* out_bytes, uint32_t out_capacity, uint32_t* out_required_bytes);

typedef grserve_hal_status_t (*grserve_hal_runtime_create_fn)(
    uint32_t flags, grserve_hal_runtime_t* out_runtime);
typedef grserve_hal_status_t (*grserve_hal_runtime_destroy_fn)(
    grserve_hal_runtime_t* runtime);
typedef grserve_hal_status_t (*grserve_hal_device_create_fn)(
    grserve_hal_runtime_t runtime, uint32_t device_ordinal,
    grserve_hal_device_t* out_device);
typedef grserve_hal_status_t (*grserve_hal_device_destroy_fn)(
    grserve_hal_device_t* device);
typedef grserve_hal_status_t (*grserve_hal_stream_create_fn)(
    grserve_hal_device_t device, uint32_t flags, grserve_hal_stream_t* out_stream);
typedef grserve_hal_status_t (*grserve_hal_stream_destroy_fn)(
    grserve_hal_stream_t* stream);
typedef grserve_hal_status_t (*grserve_hal_event_create_fn)(
    grserve_hal_device_t device, uint32_t flags, grserve_hal_event_t* out_event);
typedef grserve_hal_status_t (*grserve_hal_event_destroy_fn)(
    grserve_hal_event_t* event);
typedef grserve_hal_status_t (*grserve_hal_buffer_create_fn)(
    grserve_hal_device_t device, uint64_t byte_length, uint64_t alignment,
    uint32_t flags, grserve_hal_buffer_t* out_buffer);
typedef grserve_hal_status_t (*grserve_hal_buffer_destroy_fn)(
    grserve_hal_buffer_t* buffer);
typedef grserve_hal_status_t (*grserve_hal_executable_create_fn)(
    grserve_hal_device_t device, grserve_hal_executable_t* out_executable);
typedef grserve_hal_status_t (*grserve_hal_executable_destroy_fn)(
    grserve_hal_executable_t* executable);

typedef grserve_hal_status_t (*grserve_hal_stream_copy_fn)(
    grserve_hal_stream_t stream, grserve_hal_buffer_t destination,
    uint64_t destination_offset, grserve_hal_buffer_t source,
    uint64_t source_offset, uint64_t byte_count);
typedef grserve_hal_status_t (*grserve_hal_stream_submit_fn)(
    grserve_hal_stream_t stream, const grserve_hal_submit_info_t* submit_info);
typedef grserve_hal_status_t (*grserve_hal_event_record_fn)(
    grserve_hal_event_t event, grserve_hal_stream_t stream);
typedef grserve_hal_status_t (*grserve_hal_event_wait_fn)(
    grserve_hal_event_t event, uint64_t timeout_ns);
typedef grserve_hal_status_t (*grserve_hal_event_query_fn)(
    grserve_hal_event_t event, grserve_hal_event_state_t* out_state);
typedef grserve_hal_status_t (*grserve_hal_stream_synchronize_fn)(
    grserve_hal_stream_t stream, uint64_t timeout_ns);
typedef grserve_hal_status_t (*grserve_hal_executable_load_fn)(
    grserve_hal_executable_t executable, const uint8_t* image_bytes,
    uint64_t image_byte_count);
typedef grserve_hal_status_t (*grserve_hal_executable_launch_fn)(
    grserve_hal_executable_t executable, grserve_hal_stream_t stream,
    const uint8_t* entry_name, uint64_t entry_name_byte_count,
    const grserve_hal_launch_info_t* launch_info);

/*
 * Stable API-table prefix.  A backend writes at most the caller's
 * struct_size.  A caller must provide at least GRSERVE_HAL_API_REQUIRED_SIZE;
 * later ABI minors may append fields only after last_error_copy.
 *
 * All destroy callbacks accept a pointer to a handle. destroy(NULL) and
 * destroy(pointer-to-null-handle) succeed. On first successful destruction the
 * callback nulls *handle. A raw address that was freed must never be passed to
 * a destroy callback again.
 *
 * last_error_copy reports diagnostic text for the calling thread using the
 * string copy protocol above. It is non-mutating: a last_error_copy call never
 * clears or replaces that thread's diagnostic, including size queries,
 * insufficient-capacity copies, and invalid-argument calls. A separate HAL
 * call on the same thread may replace the diagnostic.
 */
typedef struct grserve_hal_api_t {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  grserve_hal_capabilities_t capabilities;

  grserve_hal_string_copy_fn backend_name_copy;
  grserve_hal_string_copy_fn build_id_copy;

  grserve_hal_runtime_create_fn runtime_create;
  grserve_hal_runtime_destroy_fn runtime_destroy;
  grserve_hal_device_create_fn device_create;
  grserve_hal_device_destroy_fn device_destroy;
  grserve_hal_stream_create_fn stream_create;
  grserve_hal_stream_destroy_fn stream_destroy;
  grserve_hal_event_create_fn event_create;
  grserve_hal_event_destroy_fn event_destroy;
  grserve_hal_buffer_create_fn buffer_create;
  grserve_hal_buffer_destroy_fn buffer_destroy;
  grserve_hal_executable_create_fn executable_create;
  grserve_hal_executable_destroy_fn executable_destroy;

  grserve_hal_stream_copy_fn stream_copy;
  grserve_hal_stream_submit_fn stream_submit;
  grserve_hal_event_record_fn event_record;
  grserve_hal_event_wait_fn event_wait;
  grserve_hal_event_query_fn event_query;
  grserve_hal_stream_synchronize_fn stream_synchronize;
  grserve_hal_executable_load_fn executable_load;
  grserve_hal_executable_launch_fn executable_launch;
  grserve_hal_string_copy_fn last_error_copy;
} grserve_hal_api_t;

/* Required bytes use offsetof so they do not assume packed layout. */
#define GRSERVE_HAL_API_REQUEST_REQUIRED_SIZE \
  ((uint32_t)(offsetof(grserve_hal_api_request_t, required_capabilities) + \
              sizeof(((grserve_hal_api_request_t*)0)->required_capabilities)))
#define GRSERVE_HAL_API_REQUIRED_SIZE \
  ((uint32_t)(offsetof(grserve_hal_api_t, last_error_copy) + \
              sizeof(((grserve_hal_api_t*)0)->last_error_copy)))

/* The sole exported ABI entry point. The caller owns both request and out_api. */
GRSERVE_HAL_EXPORT grserve_hal_status_t grserve_hal_get_api(
    const grserve_hal_api_request_t* request, grserve_hal_api_t* out_api);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* GRSERVE_HAL_H_ */
