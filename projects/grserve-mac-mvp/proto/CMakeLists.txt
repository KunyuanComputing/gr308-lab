set(GRSERVE_ENGINE_PROTO ${CMAKE_CURRENT_SOURCE_DIR}/engine.proto)
set(GRSERVE_EVENT_PROTO ${CMAKE_CURRENT_SOURCE_DIR}/event.proto)
set(GRSERVE_ENGINE_PROTO_OUTPUTS
  ${GRSERVE_GENERATED_DIR}/engine.pb.cc
  ${GRSERVE_GENERATED_DIR}/engine.pb.h)
set(GRSERVE_PROTOC_ATTESTATION
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/protoc.attestation)

add_custom_command(
  OUTPUT ${GRSERVE_ENGINE_PROTO_OUTPUTS}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GRSERVE_GENERATED_DIR}
  COMMAND protobuf::protoc
    --cpp_out=${GRSERVE_GENERATED_DIR}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
    ${GRSERVE_ENGINE_PROTO}
  DEPENDS ${GRSERVE_ENGINE_PROTO} protobuf::protoc
  COMMENT "Generating Engine RPC v1 C++ protocol"
  VERBATIM)

add_custom_target(grserve_rpc_proto_generated DEPENDS ${GRSERVE_ENGINE_PROTO_OUTPUTS})

add_library(grserve_rpc_proto STATIC ${GRSERVE_ENGINE_PROTO_OUTPUTS})
add_dependencies(grserve_rpc_proto grserve_rpc_proto_generated)
target_include_directories(grserve_rpc_proto PUBLIC ${GRSERVE_GENERATED_DIR})
target_link_libraries(grserve_rpc_proto PUBLIC protobuf::libprotobuf)
set_target_properties(grserve_rpc_proto PROPERTIES POSITION_INDEPENDENT_CODE YES)

set(GRSERVE_EVENT_PROTO_OUTPUTS
  ${GRSERVE_GENERATED_DIR}/event.pb.cc
  ${GRSERVE_GENERATED_DIR}/event.pb.h)
add_custom_command(
  OUTPUT ${GRSERVE_EVENT_PROTO_OUTPUTS}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GRSERVE_GENERATED_DIR}
  COMMAND protobuf::protoc
    --cpp_out=${GRSERVE_GENERATED_DIR}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
    ${GRSERVE_EVENT_PROTO}
  DEPENDS ${GRSERVE_EVENT_PROTO} protobuf::protoc
  COMMENT "Generating structured event v1 C++ protocol"
  VERBATIM)
add_custom_target(grserve_event_proto_generated DEPENDS ${GRSERVE_EVENT_PROTO_OUTPUTS})
add_library(grserve_event_proto STATIC ${GRSERVE_EVENT_PROTO_OUTPUTS})
add_dependencies(grserve_event_proto grserve_event_proto_generated)
target_include_directories(grserve_event_proto PUBLIC ${GRSERVE_GENERATED_DIR})
target_link_libraries(grserve_event_proto PUBLIC protobuf::libprotobuf)
set_target_properties(grserve_event_proto PROPERTIES POSITION_INDEPENDENT_CODE YES)

add_custom_command(
  OUTPUT ${GRSERVE_PROTOC_ATTESTATION}
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/proto_codegen.py
    --write-attestation ${GRSERVE_PROTOC_ATTESTATION}
    --protoc $<TARGET_FILE:protobuf::protoc>
    --config $<CONFIG>
  DEPENDS
    ${CMAKE_SOURCE_DIR}/tools/proto_codegen.py
    ${CMAKE_SOURCE_DIR}/CMakePresets.json
    ${CMAKE_BINARY_DIR}/CMakeCache.txt
    protobuf::protoc
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Attesting pinned protobuf compiler"
  VERBATIM)
add_custom_target(grserve_protoc_attestation DEPENDS ${GRSERVE_PROTOC_ATTESTATION})

add_custom_target(check_proto_generated
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/proto_codegen.py
    --check
    --protoc $<TARGET_FILE:protobuf::protoc>
    --attestation ${GRSERVE_PROTOC_ATTESTATION}
    --config $<CONFIG>
  DEPENDS
    ${CMAKE_SOURCE_DIR}/tools/proto_codegen.py
    ${GRSERVE_ENGINE_PROTO}
    ${GRSERVE_EVENT_PROTO}
    ${GRSERVE_PROTOC_ATTESTATION}
    grserve_event_proto_generated
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Checking committed Python protobuf output"
  VERBATIM)
