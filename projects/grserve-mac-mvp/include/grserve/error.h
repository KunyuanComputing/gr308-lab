#pragma once

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "grserve/types.h"

namespace grserve {

enum class ErrorCode {
  kInvalidArgument,
  kIncompatibleVersion,
  kUnsupportedCapability,
  kResourceExhausted,
  kDeadlineExceeded,
  kRankFailure,
  kReplicaFatal,
  kInternal,
};

class Error {
 public:
  [[nodiscard]] static Error From(
      ErrorCode code, std::string message,
      std::optional<RequestId> request_id = std::nullopt,
      std::optional<TransactionId> transaction_id = std::nullopt);

  Error(const Error&) noexcept = default;
  Error& operator=(const Error&) noexcept = default;
  Error(Error&&) noexcept = default;
  Error& operator=(Error&&) noexcept = default;

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return *message_; }
  [[nodiscard]] bool retryable() const noexcept;
  [[nodiscard]] Error WithRequestId(RequestId request_id) const noexcept {
    return Error{code_, message_, request_id, transaction_id_};
  }
  [[nodiscard]] Error RecodeForRequest(ErrorCode code,
                                       RequestId request_id) const noexcept {
    return Error{code, message_, request_id, std::nullopt};
  }
  [[nodiscard]] std::optional<RequestId> request_id() const noexcept {
    return request_id_;
  }
  [[nodiscard]] std::optional<TransactionId> transaction_id() const noexcept {
    return transaction_id_;
  }

 private:
  Error(ErrorCode code, std::shared_ptr<const std::string> message,
        std::optional<RequestId> request_id,
        std::optional<TransactionId> transaction_id) noexcept
      : code_(code),
        message_(std::move(message)),
        request_id_(request_id),
        transaction_id_(transaction_id) {}

  ErrorCode code_;
  std::shared_ptr<const std::string> message_;
  std::optional<RequestId> request_id_;
  std::optional<TransactionId> transaction_id_;
};

class Status {
 public:
  Status() noexcept = default;
  explicit Status(Error error) : error_(std::move(error)) {}

  [[nodiscard]] static Status Ok() noexcept { return Status{}; }
  [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] const Error* error() const noexcept {
    return error_ ? &*error_ : nullptr;
  }

 private:
  std::optional<Error> error_;
};

template <class T>
class StatusOr {
 private:
  struct ValueState {
    explicit ValueState(std::unique_ptr<T> value) noexcept
        : value(std::move(value)) {}

    std::unique_ptr<T> value;
  };
  struct ErrorState {
    explicit ErrorState(Error error) noexcept : error(std::move(error)) {}

    Error error;
  };
  using State = std::variant<ValueState, ErrorState>;

  static_assert(std::is_nothrow_move_constructible_v<State>);
  static_assert(std::is_nothrow_move_assignable_v<State>);

  [[nodiscard]] static State MakeValueState(T&& value) {
    return State{std::in_place_type<ValueState>,
                 std::make_unique<T>(std::move(value))};
  }

  [[nodiscard]] static State CopyState(const State& state)
    requires std::is_copy_constructible_v<T>
  {
    if (const auto* value = std::get_if<ValueState>(&state)) {
      return State{std::in_place_type<ValueState>,
                   std::make_unique<T>(std::as_const(*value->value))};
    }
    return State{std::in_place_type<ErrorState>,
                 std::get<ErrorState>(state).error};
  }

  [[nodiscard]] static State MoveState(State& state) {
    if (auto* value = std::get_if<ValueState>(&state)) {
      return MakeValueState(std::move(*value->value));
    }
    return State{std::in_place_type<ErrorState>,
                 std::move(std::get<ErrorState>(state).error)};
  }

 public:
  explicit StatusOr(T value)
    requires std::is_move_constructible_v<T>
      : state_(MakeValueState(std::move(value))) {}
  explicit StatusOr(Error error)
      : state_(std::in_place_type<ErrorState>, std::move(error)) {}

  StatusOr(const StatusOr& other)
    requires std::is_copy_constructible_v<T>
      : state_(CopyState(other.state_)) {}
  StatusOr(const StatusOr&)
    requires(!std::is_copy_constructible_v<T>) = delete;

  StatusOr(StatusOr&& other)
    requires std::is_move_constructible_v<T>
      : state_(MoveState(other.state_)) {}
  StatusOr(StatusOr&&)
    requires(!std::is_move_constructible_v<T>) = delete;

  StatusOr& operator=(const StatusOr& other)
    requires std::is_copy_constructible_v<T>
  {
    if (this != &other) {
      State replacement = CopyState(other.state_);
      state_ = std::move(replacement);
    }
    return *this;
  }
  StatusOr& operator=(const StatusOr&)
    requires(!std::is_copy_constructible_v<T>) = delete;

  StatusOr& operator=(StatusOr&& other)
    requires std::is_move_constructible_v<T>
  {
    if (this != &other) {
      State replacement = MoveState(other.state_);
      state_ = std::move(replacement);
    }
    return *this;
  }
  StatusOr& operator=(StatusOr&&)
    requires(!std::is_move_constructible_v<T>) = delete;

  [[nodiscard]] bool ok() const noexcept {
    return std::holds_alternative<ValueState>(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }

  [[nodiscard]] T* value_if() noexcept {
    auto* value = std::get_if<ValueState>(&state_);
    return value ? value->value.get() : nullptr;
  }
  [[nodiscard]] const T* value_if() const noexcept {
    const auto* value = std::get_if<ValueState>(&state_);
    return value ? value->value.get() : nullptr;
  }

  [[nodiscard]] T& value() & noexcept {
    assert(ok());
    return *std::get<ValueState>(state_).value;
  }
  [[nodiscard]] const T& value() const& noexcept {
    assert(ok());
    return *std::get<ValueState>(state_).value;
  }
  [[nodiscard]] T&& value() && noexcept {
    assert(ok());
    return std::move(*std::get<ValueState>(state_).value);
  }

  [[nodiscard]] const Error* error() const noexcept {
    const auto* error = std::get_if<ErrorState>(&state_);
    return error ? &error->error : nullptr;
  }
  [[nodiscard]] Status status() const {
    const Error* failure = error();
    return failure ? Status{*failure} : Status::Ok();
  }

 private:
  State state_;
};

}  // namespace grserve
