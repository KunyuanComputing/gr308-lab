#include <array>
#include <string_view>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <gtest/gtest.h>

#include "engine.pb.h"

namespace grserve::rpc::v1 {
namespace {

using Field = google::protobuf::FieldDescriptor;

struct ExpectedField {
  std::string_view name;
  int number;
  Field::Type type;
  Field::Label label;
  std::string_view type_name;
  std::string_view oneof_name;
  bool explicitly_packed;
};

template <std::size_t Size>
void ExpectMessage(const google::protobuf::Descriptor* message,
                   std::string_view full_name,
                   const std::array<ExpectedField, Size>& expected) {
  ASSERT_NE(message, nullptr);
  EXPECT_EQ(message->full_name(), full_name);
  EXPECT_EQ(message->field_count(), static_cast<int>(Size));
  EXPECT_EQ(message->oneof_decl_count(),
            full_name.ends_with("ClientEnvelope") || full_name.ends_with("ServerEnvelope") ? 1 : 0);
  EXPECT_EQ(message->reserved_range_count(), 0);
  EXPECT_EQ(message->reserved_name_count(), 0);
  EXPECT_EQ(message->extension_range_count(), 0);
  for (std::size_t index = 0; index < Size; ++index) {
    const auto* field = message->field(static_cast<int>(index));
    ASSERT_NE(field, nullptr);
    EXPECT_EQ(field->name(), expected[index].name);
    EXPECT_EQ(field->number(), expected[index].number);
    EXPECT_EQ(field->type(), expected[index].type);
    EXPECT_EQ(field->label(), expected[index].label);
    EXPECT_EQ(field->containing_oneof() == nullptr ? "" : field->containing_oneof()->name(),
              expected[index].oneof_name);
    const std::string_view actual_type_name = field->message_type() != nullptr
                                                  ? field->message_type()->full_name()
                                              : field->enum_type() != nullptr
                                                  ? field->enum_type()->full_name()
                                                  : "";
    EXPECT_EQ(actual_type_name, expected[index].type_name);
    EXPECT_EQ(field->options().has_packed(), expected[index].explicitly_packed);
    if (expected[index].explicitly_packed) {
      EXPECT_TRUE(field->options().packed());
    }
  }
}

struct ExpectedEnumValue {
  std::string_view name;
  int number;
};

template <std::size_t Size>
void ExpectEnum(const google::protobuf::EnumDescriptor* enumeration,
                std::string_view full_name,
                const std::array<ExpectedEnumValue, Size>& expected) {
  ASSERT_NE(enumeration, nullptr);
  EXPECT_EQ(enumeration->full_name(), full_name);
  EXPECT_EQ(enumeration->value_count(), static_cast<int>(Size));
  EXPECT_EQ(enumeration->reserved_range_count(), 0);
  EXPECT_EQ(enumeration->reserved_name_count(), 0);
  for (std::size_t index = 0; index < Size; ++index) {
    EXPECT_EQ(enumeration->value(static_cast<int>(index))->name(), expected[index].name);
    EXPECT_EQ(enumeration->value(static_cast<int>(index))->number(), expected[index].number);
  }
}

TEST(EngineProtoContract, FreezesFileAndEnums) {
  const auto* file = ProtocolVersion::descriptor()->file();
  ASSERT_NE(file, nullptr);
  EXPECT_EQ(file->name(), "engine.proto");
  EXPECT_EQ(file->package(), "grserve.rpc.v1");
  google::protobuf::FileDescriptorProto file_proto;
  file->CopyTo(&file_proto);
  EXPECT_EQ(file_proto.syntax(), "proto3");
  EXPECT_EQ(file->dependency_count(), 0);
  EXPECT_EQ(file->message_type_count(), 16);
  EXPECT_EQ(file->enum_type_count(), 6);
  EXPECT_EQ(file->service_count(), 0);
  EXPECT_EQ(file->extension_count(), 0);

  ExpectEnum(ErrorCode_descriptor(), "grserve.rpc.v1.ErrorCode",
             std::to_array<ExpectedEnumValue>({
                 {"ERROR_CODE_UNSPECIFIED", 0}, {"ERROR_CODE_INVALID_ARGUMENT", 1},
                 {"ERROR_CODE_INCOMPATIBLE_VERSION", 2},
                 {"ERROR_CODE_UNSUPPORTED_CAPABILITY", 3},
                 {"ERROR_CODE_RESOURCE_EXHAUSTED", 4},
                 {"ERROR_CODE_DEADLINE_EXCEEDED", 5},
                 {"ERROR_CODE_RANK_FAILURE", 6}, {"ERROR_CODE_REPLICA_FATAL", 7},
                 {"ERROR_CODE_INTERNAL", 8},
             }));
  ExpectEnum(ReplicaState_descriptor(), "grserve.rpc.v1.ReplicaState",
             std::to_array<ExpectedEnumValue>({
                 {"REPLICA_STATE_UNSPECIFIED", 0}, {"REPLICA_STATE_STARTING", 1},
                 {"REPLICA_STATE_HEALTHY", 2}, {"REPLICA_STATE_FATAL", 3},
                 {"REPLICA_STATE_STOPPING", 4},
             }));
  ExpectEnum(DecodeMode_descriptor(), "grserve.rpc.v1.DecodeMode",
             std::to_array<ExpectedEnumValue>({
                 {"DECODE_MODE_UNSPECIFIED", 0}, {"DECODE_MODE_GREEDY", 1},
             }));
  ExpectEnum(CancelReason_descriptor(), "grserve.rpc.v1.CancelReason",
             std::to_array<ExpectedEnumValue>({
                 {"CANCEL_REASON_UNSPECIFIED", 0},
                 {"CANCEL_REASON_CLIENT_DISCONNECTED", 1},
                 {"CANCEL_REASON_DEADLINE", 2}, {"CANCEL_REASON_SERVER_SHUTDOWN", 3},
             }));
  ExpectEnum(FinishReason_descriptor(), "grserve.rpc.v1.FinishReason",
             std::to_array<ExpectedEnumValue>({
                 {"FINISH_REASON_UNSPECIFIED", 0}, {"FINISH_REASON_LENGTH", 1},
                 {"FINISH_REASON_CANCELLED", 2},
             }));
  ExpectEnum(CancelDisposition_descriptor(), "grserve.rpc.v1.CancelDisposition",
             std::to_array<ExpectedEnumValue>({
                 {"CANCEL_DISPOSITION_UNSPECIFIED", 0},
                 {"CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION", 1},
                 {"CANCEL_DISPOSITION_COMMITTING_DETACHED", 2},
                 {"CANCEL_DISPOSITION_ALREADY_TERMINAL", 3},
                 {"CANCEL_DISPOSITION_NOT_FOUND", 4},
             }));
}

TEST(EngineProtoContract, FreezesMessagesAndFields) {
  constexpr auto optional = Field::LABEL_OPTIONAL;
  constexpr auto repeated = Field::LABEL_REPEATED;
  ExpectMessage(ProtocolVersion::descriptor(), "grserve.rpc.v1.ProtocolVersion",
                std::to_array<ExpectedField>({
                    {"major", 1, Field::TYPE_UINT32, optional, "", "", false},
                    {"minor", 2, Field::TYPE_UINT32, optional, "", "", false},
                }));
  ExpectMessage(RpcError::descriptor(), "grserve.rpc.v1.RpcError",
                std::to_array<ExpectedField>({
                    {"code", 1, Field::TYPE_ENUM, optional, "grserve.rpc.v1.ErrorCode", "", false},
                    {"message", 2, Field::TYPE_STRING, optional, "", "", false},
                    {"request_id", 3, Field::TYPE_STRING, optional, "", "", false},
                    {"transaction_id", 4, Field::TYPE_STRING, optional, "", "", false},
                    {"retryable", 5, Field::TYPE_BOOL, optional, "", "", false},
                }));
  ExpectMessage(HealthRequest::descriptor(), "grserve.rpc.v1.HealthRequest",
                std::array<ExpectedField, 0>{});
  ExpectMessage(HealthResponse::descriptor(), "grserve.rpc.v1.HealthResponse",
                std::to_array<ExpectedField>({
                    {"engine_build_id", 1, Field::TYPE_STRING, optional, "", "", false},
                    {"accepting_requests", 2, Field::TYPE_BOOL, optional, "", "", false},
                    {"replica_state", 3, Field::TYPE_ENUM, optional, "grserve.rpc.v1.ReplicaState", "", false},
                    {"fatal_error", 4, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.RpcError", "", false},
                }));
  ExpectMessage(ListModelsRequest::descriptor(), "grserve.rpc.v1.ListModelsRequest",
                std::array<ExpectedField, 0>{});
  ExpectMessage(ModelInfo::descriptor(), "grserve.rpc.v1.ModelInfo",
                std::to_array<ExpectedField>({
                    {"id", 1, Field::TYPE_STRING, optional, "", "", false},
                    {"manifest_sha256", 2, Field::TYPE_STRING, optional, "", "", false},
                    {"max_context_tokens", 3, Field::TYPE_UINT32, optional, "", "", false},
                }));
  ExpectMessage(ListModelsResponse::descriptor(), "grserve.rpc.v1.ListModelsResponse",
                std::to_array<ExpectedField>({
                    {"models", 1, Field::TYPE_MESSAGE, repeated, "grserve.rpc.v1.ModelInfo", "", false},
                }));
  ExpectMessage(CompletionRequest::descriptor(), "grserve.rpc.v1.CompletionRequest",
                std::to_array<ExpectedField>({
                    {"request_id", 1, Field::TYPE_STRING, optional, "", "", false},
                    {"model_id", 2, Field::TYPE_STRING, optional, "", "", false},
                    {"prompt_token_ids", 3, Field::TYPE_UINT32, repeated, "", "", true},
                    {"max_tokens", 4, Field::TYPE_UINT32, optional, "", "", false},
                    {"timeout_ms", 5, Field::TYPE_UINT64, optional, "", "", false},
                    {"decode_mode", 6, Field::TYPE_ENUM, optional, "grserve.rpc.v1.DecodeMode", "", false},
                }));
  ExpectMessage(CancelRequest::descriptor(), "grserve.rpc.v1.CancelRequest",
                std::to_array<ExpectedField>({
                    {"request_id", 1, Field::TYPE_STRING, optional, "", "", false},
                    {"reason", 2, Field::TYPE_ENUM, optional, "grserve.rpc.v1.CancelReason", "", false},
                }));
  ExpectMessage(CompletionAccepted::descriptor(), "grserve.rpc.v1.CompletionAccepted",
                std::to_array<ExpectedField>({
                    {"request_id", 1, Field::TYPE_STRING, optional, "", "", false},
                }));
  ExpectMessage(TokenEvent::descriptor(), "grserve.rpc.v1.TokenEvent",
                std::to_array<ExpectedField>({
                    {"request_id", 1, Field::TYPE_STRING, optional, "", "", false},
                    {"transaction_id", 2, Field::TYPE_STRING, optional, "", "", false},
                    {"sequence_number", 3, Field::TYPE_UINT64, optional, "", "", false},
                    {"token_id", 4, Field::TYPE_UINT32, optional, "", "", false},
                }));
  ExpectMessage(Usage::descriptor(), "grserve.rpc.v1.Usage",
                std::to_array<ExpectedField>({
                    {"prompt_tokens", 1, Field::TYPE_UINT32, optional, "", "", false},
                    {"completion_tokens", 2, Field::TYPE_UINT32, optional, "", "", false},
                    {"total_tokens", 3, Field::TYPE_UINT32, optional, "", "", false},
                }));
  ExpectMessage(CompletionFinished::descriptor(), "grserve.rpc.v1.CompletionFinished",
                std::to_array<ExpectedField>({
                    {"request_id", 1, Field::TYPE_STRING, optional, "", "", false},
                    {"finish_reason", 2, Field::TYPE_ENUM, optional, "grserve.rpc.v1.FinishReason", "", false},
                    {"usage", 3, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.Usage", "", false},
                    {"last_sequence_number", 4, Field::TYPE_UINT64, optional, "", "", false},
                }));
  ExpectMessage(CancelAck::descriptor(), "grserve.rpc.v1.CancelAck",
                std::to_array<ExpectedField>({
                    {"request_id", 1, Field::TYPE_STRING, optional, "", "", false},
                    {"disposition", 2, Field::TYPE_ENUM, optional, "grserve.rpc.v1.CancelDisposition", "", false},
                }));
  ExpectMessage(ClientEnvelope::descriptor(), "grserve.rpc.v1.ClientEnvelope",
                std::to_array<ExpectedField>({
                    {"protocol_version", 1, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.ProtocolVersion", "", false},
                    {"health", 10, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.HealthRequest", "payload", false},
                    {"list_models", 11, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.ListModelsRequest", "payload", false},
                    {"completion", 12, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.CompletionRequest", "payload", false},
                    {"cancel", 13, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.CancelRequest", "payload", false},
                }));
  ExpectMessage(ServerEnvelope::descriptor(), "grserve.rpc.v1.ServerEnvelope",
                std::to_array<ExpectedField>({
                    {"protocol_version", 1, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.ProtocolVersion", "", false},
                    {"health", 10, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.HealthResponse", "payload", false},
                    {"list_models", 11, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.ListModelsResponse", "payload", false},
                    {"accepted", 12, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.CompletionAccepted", "payload", false},
                    {"token", 13, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.TokenEvent", "payload", false},
                    {"finished", 14, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.CompletionFinished", "payload", false},
                    {"cancel_ack", 15, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.CancelAck", "payload", false},
                    {"error", 16, Field::TYPE_MESSAGE, optional, "grserve.rpc.v1.RpcError", "payload", false},
                }));
}

}  // namespace
}  // namespace grserve::rpc::v1
