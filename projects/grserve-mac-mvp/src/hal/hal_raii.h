#pragma once

#include <cstdint>
#include <memory>

#include "grserve/error.h"
#include "grserve/hal.h"
#include "hal/hal_library.h"

namespace grserve {

class HalDevice;
class HalStream;
class HalEvent;
class HalBuffer;
class HalExecutable;

class HalRuntime {
 public:
  static StatusOr<HalRuntime> Create(std::shared_ptr<HalLibrary> library, uint32_t flags = 0U);
  HalRuntime() = default;
  HalRuntime(HalRuntime&& other) noexcept;
  HalRuntime& operator=(HalRuntime&& other) noexcept;
  HalRuntime(const HalRuntime&) = delete;
  HalRuntime& operator=(const HalRuntime&) = delete;
  ~HalRuntime() noexcept;
  Status reset();
  grserve_hal_runtime_t get() const noexcept { return handle_; }
  explicit operator bool() const noexcept { return handle_ != nullptr; }
  StatusOr<HalDevice> CreateDevice(uint32_t ordinal) const;
 private:
  bool TryResetNoexcept() noexcept;
  HalRuntime(std::shared_ptr<HalLibrary> library, grserve_hal_runtime_t handle) : library_(std::move(library)), handle_(handle) {}
  std::shared_ptr<HalLibrary> library_; grserve_hal_runtime_t handle_ = nullptr;
};

class HalDevice {
 public:
  HalDevice() = default; HalDevice(HalDevice&&) noexcept; HalDevice& operator=(HalDevice&&) noexcept;
  HalDevice(const HalDevice&) = delete; HalDevice& operator=(const HalDevice&) = delete; ~HalDevice() noexcept;
  Status reset(); grserve_hal_device_t get() const noexcept { return handle_; } explicit operator bool() const noexcept { return handle_ != nullptr; }
  StatusOr<HalStream> CreateStream(uint32_t flags) const;
  StatusOr<HalEvent> CreateEvent(uint32_t flags) const;
  StatusOr<HalBuffer> CreateBuffer(uint64_t bytes, uint64_t alignment, uint32_t flags) const;
  StatusOr<HalExecutable> CreateExecutable() const;
 private:
  bool TryResetNoexcept() noexcept;
  friend class HalRuntime; HalDevice(std::shared_ptr<HalLibrary> library, grserve_hal_device_t handle) : library_(std::move(library)), handle_(handle) {}
  std::shared_ptr<HalLibrary> library_; grserve_hal_device_t handle_ = nullptr;
};

class HalStream { public: HalStream() = default; HalStream(HalStream&&) noexcept; HalStream& operator=(HalStream&&) noexcept; HalStream(const HalStream&) = delete; HalStream& operator=(const HalStream&) = delete; ~HalStream() noexcept; Status reset(); grserve_hal_stream_t get() const noexcept { return handle_; } explicit operator bool() const noexcept { return handle_ != nullptr; } private: bool TryResetNoexcept() noexcept; friend class HalDevice; HalStream(std::shared_ptr<HalLibrary> l, grserve_hal_stream_t h) : library_(std::move(l)), handle_(h) {} std::shared_ptr<HalLibrary> library_; grserve_hal_stream_t handle_ = nullptr; };
class HalEvent { public: HalEvent() = default; HalEvent(HalEvent&&) noexcept; HalEvent& operator=(HalEvent&&) noexcept; HalEvent(const HalEvent&) = delete; HalEvent& operator=(const HalEvent&) = delete; ~HalEvent() noexcept; Status reset(); grserve_hal_event_t get() const noexcept { return handle_; } explicit operator bool() const noexcept { return handle_ != nullptr; } private: bool TryResetNoexcept() noexcept; friend class HalDevice; HalEvent(std::shared_ptr<HalLibrary> l, grserve_hal_event_t h) : library_(std::move(l)), handle_(h) {} std::shared_ptr<HalLibrary> library_; grserve_hal_event_t handle_ = nullptr; };
class HalBuffer { public: HalBuffer() = default; HalBuffer(HalBuffer&&) noexcept; HalBuffer& operator=(HalBuffer&&) noexcept; HalBuffer(const HalBuffer&) = delete; HalBuffer& operator=(const HalBuffer&) = delete; ~HalBuffer() noexcept; Status reset(); grserve_hal_buffer_t get() const noexcept { return handle_; } explicit operator bool() const noexcept { return handle_ != nullptr; } private: bool TryResetNoexcept() noexcept; friend class HalDevice; HalBuffer(std::shared_ptr<HalLibrary> l, grserve_hal_buffer_t h) : library_(std::move(l)), handle_(h) {} std::shared_ptr<HalLibrary> library_; grserve_hal_buffer_t handle_ = nullptr; };
class HalExecutable { public: HalExecutable() = default; HalExecutable(HalExecutable&&) noexcept; HalExecutable& operator=(HalExecutable&&) noexcept; HalExecutable(const HalExecutable&) = delete; HalExecutable& operator=(const HalExecutable&) = delete; ~HalExecutable() noexcept; Status reset(); grserve_hal_executable_t get() const noexcept { return handle_; } explicit operator bool() const noexcept { return handle_ != nullptr; } private: bool TryResetNoexcept() noexcept; friend class HalDevice; HalExecutable(std::shared_ptr<HalLibrary> l, grserve_hal_executable_t h) : library_(std::move(l)), handle_(h) {} std::shared_ptr<HalLibrary> library_; grserve_hal_executable_t handle_ = nullptr; };

}  // namespace grserve
