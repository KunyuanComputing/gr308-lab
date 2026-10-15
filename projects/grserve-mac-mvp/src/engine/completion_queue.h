#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace grserve {

// The owner must join every thread and release every Producer before destroying
// a queue. Close rejects ordinary pushes and new producers, while existing
// producers may drain values until they release their token.
template <class T>
class CompletionQueue {
 public:
  class Producer {
   public:
    Producer() = default;
    Producer(const Producer&) = delete;
    Producer& operator=(const Producer&) = delete;
    Producer(Producer&& other) noexcept { MoveFrom(std::move(other)); }
    Producer& operator=(Producer&& other) noexcept {
      if (this != &other) {
        Release();
        MoveFrom(std::move(other));
      }
      return *this;
    }
    ~Producer() { Release(); }

    [[nodiscard]] bool active() const noexcept { return queue_ != nullptr; }
    [[nodiscard]] bool CanAccept() const noexcept {
      if (queue_ == nullptr) return false;
      std::lock_guard lock(queue_->mutex_);
      return !queue_->close_requested_ && !queue_->abort_requested_;
    }
    [[nodiscard]] bool Push(T value) { return Publish(std::move(value), []() noexcept {}); }

    template <class Commit>
    [[nodiscard]] bool Publish(T value, Commit&& commit) {
      static_assert(std::is_nothrow_invocable_v<Commit&>);
      if (queue_ == nullptr) return false;
      std::unique_lock lock(queue_->mutex_);
      if (queue_->abort_requested_ || queue_->values_.size() >= queue_->capacity_) {
        return false;
      }
      // deque insertion is the only throwing work before state commit. An
      // active registered producer remains valid after Close for draining.
      queue_->values_.push_back(std::move(value));
      commit();
      lock.unlock();
      queue_->cv_.notify_one();
      return true;
    }

    void Release() noexcept {
      if (queue_ == nullptr) return;
      CompletionQueue* queue = std::exchange(queue_, nullptr);
      {
        std::lock_guard lock(queue->mutex_);
        --queue->producer_count_;
      }
      queue->cv_.notify_all();
    }

   private:
    friend class CompletionQueue;
    explicit Producer(CompletionQueue* queue) noexcept : queue_(queue) {}
    void MoveFrom(Producer&& other) noexcept {
      queue_ = std::exchange(other.queue_, nullptr);
    }
    CompletionQueue* queue_{nullptr};
  };

  explicit CompletionQueue(std::size_t capacity = 4096) noexcept
      : capacity_(capacity) {}
  CompletionQueue(const CompletionQueue&) = delete;
  CompletionQueue& operator=(const CompletionQueue&) = delete;
  ~CompletionQueue() { Close(); }

  [[nodiscard]] Producer RegisterProducer() noexcept {
    std::lock_guard lock(mutex_);
    if (close_requested_ || abort_requested_) return Producer{};
    ++producer_count_;
    return Producer{this};
  }

  [[nodiscard]] bool Push(T value) {
    std::unique_lock lock(mutex_);
    if (close_requested_ || abort_requested_ || values_.size() >= capacity_) {
      return false;
    }
    values_.push_back(std::move(value));
    lock.unlock();
    cv_.notify_one();
    return true;
  }

  template <class... Args>
  [[nodiscard]] bool Emplace(Args&&... args) {
    std::unique_lock lock(mutex_);
    if (close_requested_ || abort_requested_ || values_.size() >= capacity_) {
      return false;
    }
    values_.emplace_back(std::forward<Args>(args)...);
    lock.unlock();
    cv_.notify_one();
    return true;
  }

  [[nodiscard]] std::optional<T> WaitPop() {
    std::unique_lock lock(mutex_);
    ++wait_generation_;
    waiter_cv_.notify_all();
    cv_.wait(lock, [this] {
      return abort_requested_ || !values_.empty() ||
             (close_requested_ && producer_count_ == 0);
    });
    if (abort_requested_ || values_.empty()) return std::nullopt;
    T value = std::move(values_.front());
    values_.pop_front();
    return value;
  }

  // Narrow deterministic test seam: waits until a consumer has entered
  // WaitPop after the caller observed the supplied generation.
  [[nodiscard]] std::size_t wait_generation_for_testing() const noexcept {
    std::lock_guard lock(mutex_);
    return wait_generation_;
  }

  void WaitForConsumerWaitForTesting(std::size_t observed_generation) {
    std::unique_lock lock(mutex_);
    waiter_cv_.wait(lock, [this, observed_generation] {
      return wait_generation_ > observed_generation;
    });
  }

  [[nodiscard]] std::optional<T> TryPop() {
    std::lock_guard lock(mutex_);
    if (values_.empty()) return std::nullopt;
    T value = std::move(values_.front());
    values_.pop_front();
    return value;
  }

  void WaitForValue() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] {
      return abort_requested_ || !values_.empty() ||
             (close_requested_ && producer_count_ == 0);
    });
  }

  void WaitForAbort() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return abort_requested_; });
  }

  void Close() noexcept {
    {
      std::lock_guard lock(mutex_);
      close_requested_ = true;
    }
    cv_.notify_all();
  }

  // Fatal abort discards queued values, rejects every future publication, and
  // wakes consumers even when other registered producers have not shut down.
  void Abort() noexcept {
    {
      std::lock_guard lock(mutex_);
      abort_requested_ = true;
      close_requested_ = true;
      values_.clear();
    }
    cv_.notify_all();
  }

  [[nodiscard]] bool closed() const noexcept {
    std::lock_guard lock(mutex_);
    return close_requested_;
  }

  [[nodiscard]] bool aborted() const noexcept {
    std::lock_guard lock(mutex_);
    return abort_requested_;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    std::lock_guard lock(mutex_);
    return values_.size();
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable waiter_cv_;
  std::deque<T> values_;
  const std::size_t capacity_;
  std::size_t producer_count_{0};
  std::size_t wait_generation_{0};
  bool close_requested_{false};
  bool abort_requested_{false};
};

}  // namespace grserve
