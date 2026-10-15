cmake_minimum_required(VERSION 3.28)
project(grserve LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
find_package(Threads REQUIRED)
find_package(Python3 3.11 EXACT COMPONENTS Interpreter REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/Sanitizers.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)
grserve_enable_project_sanitizers()

enable_testing()
add_executable(hal_abi_smoke tests/c/hal_abi_smoke.c)
target_include_directories(hal_abi_smoke PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME c.hal_header COMMAND hal_abi_smoke)

add_executable(hal_abi_test tests/cpp/hal_abi_test.cc)
target_include_directories(hal_abi_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hal_abi_test PRIVATE GTest::gtest_main Threads::Threads)
add_test(NAME cpp.hal_abi COMMAND hal_abi_test)

add_executable(grserve_smoke_test tests/cpp/smoke_test.cc)
target_link_libraries(grserve_smoke_test PRIVATE GTest::gtest_main Threads::Threads)
add_test(NAME cpp.smoke COMMAND grserve_smoke_test)

add_library(grserve_common STATIC src/common/error.cc)
target_include_directories(grserve_common PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(grserve_common PROPERTIES
  POSITION_INDEPENDENT_CODE YES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

set(GRSERVE_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
add_subdirectory(proto)
add_library(grserve_event_log STATIC src/common/event_log.cc)
target_include_directories(grserve_event_log PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_event_log PUBLIC
  grserve_common grserve_event_proto protobuf::libprotobuf)
set_target_properties(grserve_event_log PROPERTIES POSITION_INDEPENDENT_CODE YES)
add_custom_command(
  OUTPUT
    ${GRSERVE_GENERATED_DIR}/model_manifest.pb.cc
    ${GRSERVE_GENERATED_DIR}/model_manifest.pb.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GRSERVE_GENERATED_DIR}
  COMMAND protobuf::protoc
    --cpp_out=${GRSERVE_GENERATED_DIR}
    --proto_path=${CMAKE_SOURCE_DIR}/proto
    ${CMAKE_SOURCE_DIR}/proto/model_manifest.proto
  DEPENDS proto/model_manifest.proto protobuf::protoc
  VERBATIM)
add_library(grserve_state_codec STATIC
  src/common/sha256.cc
  src/model/state_codec.cc)
target_include_directories(grserve_state_codec PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_state_codec PUBLIC grserve_common)
set_target_properties(grserve_state_codec PROPERTIES POSITION_INDEPENDENT_CODE YES)

add_library(grserve_model STATIC
  src/model/manifest.cc
  src/model/mini_qwen.cc
  ${GRSERVE_GENERATED_DIR}/model_manifest.pb.cc)
target_include_directories(grserve_model PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  ${GRSERVE_GENERATED_DIR})
target_link_libraries(grserve_model PUBLIC
  grserve_state_codec grserve_common protobuf::libprotobuf)
set_target_properties(grserve_model PROPERTIES POSITION_INDEPENDENT_CODE YES)

add_executable(grserve_sha256_test tests/cpp/sha256_test.cc)
target_include_directories(grserve_sha256_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_sha256_test PRIVATE grserve_model GTest::gtest_main Threads::Threads)
add_test(NAME cpp.sha256 COMMAND grserve_sha256_test)

add_executable(grserve_event_log_test tests/cpp/event_log_test.cc)
target_include_directories(grserve_event_log_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_event_log_test PRIVATE
  grserve_event_log GTest::gtest_main Threads::Threads)
add_test(NAME cpp.event_log COMMAND grserve_event_log_test)

add_executable(grserve_manifest_test tests/cpp/manifest_test.cc)
target_include_directories(grserve_manifest_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_manifest_test PRIVATE grserve_model GTest::gtest_main Threads::Threads)
target_compile_definitions(grserve_manifest_test PRIVATE
  GRSERVE_MODEL_DIR="${CMAKE_SOURCE_DIR}/models/mini-qwen38-hybrid")
add_test(NAME cpp.manifest COMMAND grserve_manifest_test)

add_executable(grserve_mini_qwen_test tests/cpp/mini_qwen_test.cc)
target_include_directories(grserve_mini_qwen_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_mini_qwen_test PRIVATE
  grserve_model GTest::gtest_main Threads::Threads)
target_compile_definitions(grserve_mini_qwen_test PRIVATE
  GRSERVE_MODEL_DIR="${CMAKE_SOURCE_DIR}/models/mini-qwen38-hybrid")
add_test(NAME cpp.mini_qwen COMMAND grserve_mini_qwen_test)

add_library(grserve_mock_scheduler STATIC
  src/hal_mock/mock_scheduler.cc
  src/hal_mock/scripted_result.cc)
target_include_directories(grserve_mock_scheduler PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_mock_scheduler PUBLIC
  grserve_state_codec grserve_common protobuf::libprotobuf Threads::Threads)
target_compile_definitions(grserve_mock_scheduler PRIVATE GRSERVE_MOCK_SCHEDULER_TESTING)

add_library(grserve_hal_loader STATIC
  src/hal/hal_library.cc
  src/hal/hal_raii.cc)
target_include_directories(grserve_hal_loader PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_hal_loader PUBLIC grserve_common ${CMAKE_DL_LIBS})

add_library(grserve_mock_hal SHARED
  src/hal_mock/mock_hal.cc
  src/hal_mock/scripted_result.cc)
target_include_directories(grserve_mock_hal PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_mock_hal PRIVATE
  grserve_state_codec grserve_common protobuf::libprotobuf)
target_compile_definitions(grserve_mock_hal PRIVATE GRSERVE_HAL_BUILDING_LIBRARY)
set_target_properties(grserve_mock_hal PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

add_library(grserve_fault_plan STATIC src/hal_fault/fault_plan.cc)
set_target_properties(grserve_fault_plan PROPERTIES
  POSITION_INDEPENDENT_CODE YES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
target_include_directories(grserve_fault_plan PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_fault_plan PUBLIC grserve_common Threads::Threads)

add_library(grserve_transaction STATIC src/engine/transaction.cc)
target_include_directories(grserve_transaction PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_transaction PUBLIC grserve_common)

add_library(grserve_rank_worker STATIC
  src/engine/completion_queue.cc
  src/engine/rank_worker.cc)
target_include_directories(grserve_rank_worker PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_rank_worker PUBLIC grserve_common Threads::Threads)

add_library(grserve_admission STATIC src/engine/admission.cc)
target_include_directories(grserve_admission PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_admission PUBLIC grserve_common Threads::Threads)

add_library(grserve_controller STATIC
  src/engine/controller.cc
  src/engine/replica_gate.cc)
target_include_directories(grserve_controller PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_controller PUBLIC
  grserve_admission grserve_rank_worker grserve_transaction grserve_common
  Threads::Threads)

add_library(grserve_fault_hal SHARED src/hal_fault/fault_hal.cc)
target_include_directories(grserve_fault_hal PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_fault_hal PRIVATE grserve_fault_plan ${CMAKE_DL_LIBS})
target_compile_definitions(grserve_fault_hal PRIVATE
  GRSERVE_HAL_BUILDING_LIBRARY
  GRSERVE_MOCK_HAL_PATH="$<TARGET_FILE:grserve_mock_hal>")
set_target_properties(grserve_fault_hal PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
add_dependencies(grserve_fault_hal grserve_mock_hal)

add_library(grserve_cpu_hal SHARED src/hal_cpu/cpu_hal.cc)
target_include_directories(grserve_cpu_hal PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_cpu_hal PRIVATE grserve_model Threads::Threads)
target_compile_definitions(grserve_cpu_hal PRIVATE GRSERVE_HAL_BUILDING_LIBRARY)
set_target_properties(grserve_cpu_hal PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

add_executable(grserve_backend_parity_test tests/cpp/backend_parity_test.cc)
target_include_directories(grserve_backend_parity_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_backend_parity_test PRIVATE
  grserve_hal_loader grserve_model GTest::gtest_main Threads::Threads ${CMAKE_DL_LIBS})
target_compile_definitions(grserve_backend_parity_test PRIVATE
  CPU_HAL_PATH="$<TARGET_FILE:grserve_cpu_hal>"
  MOCK_HAL_PATH="$<TARGET_FILE:grserve_mock_hal>"
  FAULT_HAL_PATH="$<TARGET_FILE:grserve_fault_hal>"
  GRSERVE_MODEL_DIR="${CMAKE_SOURCE_DIR}/models/mini-qwen38-hybrid")
add_dependencies(grserve_backend_parity_test
  grserve_cpu_hal grserve_mock_hal grserve_fault_hal)
add_test(NAME cpp.backend_parity COMMAND grserve_backend_parity_test)

add_library(hal_good SHARED tests/fixtures/hal_good.cc)
add_library(hal_bad_major SHARED tests/fixtures/hal_bad_major.cc)
add_library(hal_missing_capability SHARED tests/fixtures/hal_missing_capability.cc)
add_library(hal_missing_entry SHARED tests/fixtures/hal_good.cc)
target_compile_definitions(hal_missing_entry PRIVATE GRSERVE_FIXTURE_MODE=3)
add_library(hal_null_callback SHARED tests/fixtures/hal_good.cc)
target_compile_definitions(hal_null_callback PRIVATE GRSERVE_FIXTURE_MODE=4)
add_library(hal_malformed_negotiation SHARED tests/fixtures/hal_good.cc)
target_compile_definitions(hal_malformed_negotiation PRIVATE GRSERVE_FIXTURE_MODE=5)
add_library(hal_destroy_failure SHARED tests/fixtures/hal_good.cc)
target_compile_definitions(hal_destroy_failure PRIVATE GRSERVE_FIXTURE_MODE=6)
foreach(hal_fixture IN ITEMS hal_good hal_bad_major hal_missing_capability
    hal_missing_entry hal_null_callback hal_malformed_negotiation hal_destroy_failure)
  target_include_directories(${hal_fixture} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(${hal_fixture} PRIVATE GRSERVE_HAL_BUILDING_LIBRARY)
endforeach()

add_executable(grserve_hal_loader_test tests/cpp/hal_loader_test.cc)
target_include_directories(grserve_hal_loader_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_hal_loader_test PRIVATE grserve_hal_loader GTest::gtest_main Threads::Threads ${CMAKE_DL_LIBS})
target_compile_definitions(grserve_hal_loader_test PRIVATE
  HAL_GOOD_PATH="$<TARGET_FILE:hal_good>"
  HAL_BAD_MAJOR_PATH="$<TARGET_FILE:hal_bad_major>"
  HAL_MISSING_CAPABILITY_PATH="$<TARGET_FILE:hal_missing_capability>"
  HAL_MISSING_ENTRY_PATH="$<TARGET_FILE:hal_missing_entry>"
  HAL_NULL_CALLBACK_PATH="$<TARGET_FILE:hal_null_callback>"
  HAL_MALFORMED_NEGOTIATION_PATH="$<TARGET_FILE:hal_malformed_negotiation>")
add_dependencies(grserve_hal_loader_test hal_good hal_bad_major hal_missing_capability
  hal_missing_entry hal_null_callback hal_malformed_negotiation)
add_test(NAME cpp.hal_loader COMMAND grserve_hal_loader_test)

add_executable(grserve_hal_raii_test tests/cpp/hal_raii_test.cc)
target_include_directories(grserve_hal_raii_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_hal_raii_test PRIVATE grserve_hal_loader GTest::gtest_main Threads::Threads ${CMAKE_DL_LIBS})
target_compile_definitions(grserve_hal_raii_test PRIVATE
  HAL_GOOD_PATH="$<TARGET_FILE:hal_good>"
  HAL_DESTROY_FAILURE_PATH="$<TARGET_FILE:hal_destroy_failure>")
add_dependencies(grserve_hal_raii_test hal_good hal_destroy_failure)
add_test(NAME cpp.hal_raii COMMAND grserve_hal_raii_test)

add_executable(grserve_types_test tests/cpp/types_test.cc)
target_include_directories(grserve_types_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(grserve_types_test PRIVATE grserve_common GTest::gtest_main Threads::Threads)
add_test(NAME cpp.types COMMAND grserve_types_test)

add_executable(grserve_error_test tests/cpp/error_test.cc)
target_include_directories(grserve_error_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(grserve_error_test PRIVATE grserve_common GTest::gtest_main Threads::Threads)
add_test(NAME cpp.error COMMAND grserve_error_test)

add_executable(grserve_mock_async_test tests/cpp/mock_async_test.cc)
target_include_directories(grserve_mock_async_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_mock_async_test PRIVATE
  grserve_mock_scheduler grserve_hal_loader GTest::gtest_main Threads::Threads
  ${CMAKE_DL_LIBS})
target_compile_definitions(grserve_mock_async_test PRIVATE
  MOCK_HAL_PATH="$<TARGET_FILE:grserve_mock_hal>"
  GRSERVE_MOCK_SCHEDULER_TESTING)
add_dependencies(grserve_mock_async_test grserve_mock_hal)
add_test(NAME cpp.mock_async COMMAND grserve_mock_async_test)

add_executable(grserve_fault_plan_test tests/cpp/fault_plan_test.cc)
target_include_directories(grserve_fault_plan_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_fault_plan_test PRIVATE
  grserve_fault_plan grserve_hal_loader GTest::gtest_main Threads::Threads ${CMAKE_DL_LIBS})
target_compile_definitions(grserve_fault_plan_test PRIVATE
  FAULT_HAL_PATH="$<TARGET_FILE:grserve_fault_hal>")
add_dependencies(grserve_fault_plan_test grserve_fault_hal)
add_test(NAME cpp.fault_plan COMMAND grserve_fault_plan_test)

add_executable(grserve_transaction_test tests/cpp/transaction_test.cc)
target_include_directories(grserve_transaction_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_transaction_test PRIVATE
  grserve_transaction GTest::gtest_main Threads::Threads)
add_test(NAME cpp.transaction COMMAND grserve_transaction_test)

add_executable(grserve_rank_worker_test tests/cpp/rank_worker_test.cc)
target_include_directories(grserve_rank_worker_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_rank_worker_test PRIVATE
  grserve_rank_worker GTest::gtest_main Threads::Threads)
add_test(NAME cpp.rank_worker COMMAND grserve_rank_worker_test)

add_executable(grserve_admission_test tests/cpp/admission_test.cc)
target_include_directories(grserve_admission_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_admission_test PRIVATE
  grserve_admission GTest::gtest_main Threads::Threads)
add_test(NAME cpp.admission COMMAND grserve_admission_test)

add_executable(grserve_controller_success_test tests/cpp/controller_success_test.cc)
target_include_directories(grserve_controller_success_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_controller_success_test PRIVATE
  grserve_controller GTest::gtest_main Threads::Threads)
add_test(NAME cpp.controller_success COMMAND grserve_controller_success_test)

add_executable(grserve_controller_faults_test tests/cpp/controller_faults_test.cc)
target_include_directories(grserve_controller_faults_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_controller_faults_test PRIVATE
  grserve_controller GTest::gtest_main Threads::Threads)
add_test(NAME cpp.controller_faults COMMAND grserve_controller_faults_test)

add_executable(grserve_controller_cancel_test tests/cpp/controller_cancel_test.cc)
target_include_directories(grserve_controller_cancel_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_controller_cancel_test PRIVATE
  grserve_controller GTest::gtest_main Threads::Threads)
add_test(NAME cpp.controller_cancel COMMAND grserve_controller_cancel_test)

add_executable(grserve_proto_contract_test tests/cpp/rpc/proto_contract_test.cc)
target_link_libraries(grserve_proto_contract_test PRIVATE
  grserve_rpc_proto GTest::gtest_main Threads::Threads)
add_test(NAME cpp.rpc.proto_contract COMMAND grserve_proto_contract_test)

add_library(grserve_rpc STATIC
  src/rpc/frame_codec.cc
  src/rpc/unix_socket.cc
  src/rpc/rpc_session.cc
  src/rpc/engine_rpc_server.cc)
target_include_directories(grserve_rpc PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve_rpc PUBLIC
  grserve_common grserve_rpc_proto Threads::Threads)
set_target_properties(grserve_rpc PROPERTIES POSITION_INDEPENDENT_CODE YES)

add_executable(grserve-engine src/rpc/engine_main.cc)
set_target_properties(grserve-engine PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
target_include_directories(grserve-engine PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(grserve-engine PRIVATE
  grserve_rpc grserve_controller grserve_hal_loader grserve_model
  Threads::Threads ${CMAKE_DL_LIBS})

if(BUILD_TESTING)
  add_library(grserve_blocking_hal SHARED
    src/hal_mock/mock_hal.cc
    src/hal_mock/scripted_result.cc)
  target_include_directories(grserve_blocking_hal PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(grserve_blocking_hal PRIVATE
    grserve_state_codec grserve_common protobuf::libprotobuf)
  target_compile_definitions(grserve_blocking_hal PRIVATE
    GRSERVE_HAL_BUILDING_LIBRARY GRSERVE_HAL_BLOCKING_TEST)
  set_target_properties(grserve_blocking_hal PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

  add_library(grserve_rpc_test STATIC
    src/rpc/frame_codec.cc
    src/rpc/unix_socket.cc
    src/rpc/rpc_session.cc
    src/rpc/engine_rpc_server.cc)
  target_include_directories(grserve_rpc_test PUBLIC
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(grserve_rpc_test PUBLIC
    grserve_common grserve_rpc_proto Threads::Threads)
  target_compile_definitions(grserve_rpc_test PRIVATE GRSERVE_RPC_TESTING)
  add_executable(grserve_frame_sigpipe_helper
    tests/cpp/rpc/frame_codec_sigpipe_helper.cc)
  target_include_directories(grserve_frame_sigpipe_helper PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(grserve_frame_sigpipe_helper PRIVATE
    grserve_rpc grserve_common grserve_rpc_proto Threads::Threads)

  add_executable(grserve_frame_codec_test tests/cpp/rpc/frame_codec_test.cc)
  target_include_directories(grserve_frame_codec_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(grserve_frame_codec_test PRIVATE
    grserve_rpc grserve_common grserve_rpc_proto GTest::gtest_main
    Threads::Threads)
  target_compile_definitions(grserve_frame_codec_test PRIVATE
    GRSERVE_FRAME_SIGPIPE_HELPER_PATH="$<TARGET_FILE:grserve_frame_sigpipe_helper>")
  add_dependencies(grserve_frame_codec_test grserve_frame_sigpipe_helper)
  add_test(NAME cpp.rpc.frame_codec COMMAND grserve_frame_codec_test)

  add_executable(grserve_unix_socket_test tests/cpp/rpc/unix_socket_test.cc)
  target_include_directories(grserve_unix_socket_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(grserve_unix_socket_test PRIVATE
    grserve_rpc_test grserve_common GTest::gtest_main Threads::Threads)
  target_compile_definitions(grserve_unix_socket_test PRIVATE
    GRSERVE_RPC_TESTING)
  add_test(NAME cpp.rpc.unix_socket COMMAND grserve_unix_socket_test)

  add_executable(grserve_rpc_session_test tests/cpp/rpc/rpc_session_test.cc)
  target_include_directories(grserve_rpc_session_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(grserve_rpc_session_test PRIVATE
    grserve_rpc_test grserve_controller GTest::gtest_main Threads::Threads)
  target_compile_definitions(grserve_rpc_session_test PRIVATE GRSERVE_RPC_TESTING)
  add_test(NAME cpp.rpc.session COMMAND grserve_rpc_session_test)

  add_executable(grserve_rpc_disconnect_test tests/cpp/rpc/disconnect_test.cc)
  target_include_directories(grserve_rpc_disconnect_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(grserve_rpc_disconnect_test PRIVATE
    grserve_rpc_test grserve_controller GTest::gtest_main Threads::Threads)
  target_compile_definitions(grserve_rpc_disconnect_test PRIVATE GRSERVE_RPC_TESTING)
  add_test(NAME cpp.rpc.disconnect COMMAND grserve_rpc_disconnect_test)

  add_test(NAME cpp.rpc.engine_help COMMAND $<TARGET_FILE:grserve-engine> --help)

  add_executable(grserve_engine_process_test
    tests/cpp/rpc/engine_process_test.cc)
  target_include_directories(grserve_engine_process_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(grserve_engine_process_test PRIVATE
    grserve_rpc_proto GTest::gtest_main Threads::Threads)
  target_compile_definitions(grserve_engine_process_test PRIVATE
    GRSERVE_ENGINE_PATH="$<TARGET_FILE:grserve-engine>"
    GRSERVE_MOCK_HAL_PATH="$<TARGET_FILE:grserve_mock_hal>"
    GRSERVE_BLOCKING_HAL_PATH="$<TARGET_FILE:grserve_blocking_hal>"
    GRSERVE_MODEL_DIR="${CMAKE_SOURCE_DIR}/models/mini-qwen38-hybrid")
  add_dependencies(grserve_engine_process_test grserve-engine grserve_mock_hal
    grserve_blocking_hal)
  add_test(NAME cpp.rpc.engine_process COMMAND grserve_engine_process_test)

  add_test(NAME cmake.rpc_build_testing_off
    COMMAND ${CMAKE_COMMAND}
      "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DBINARY_DIR=${CMAKE_SOURCE_DIR}/build/build-testing-off-test"
      "-DGENERATOR=${CMAKE_GENERATOR}"
      "-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
      "-DOSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
      -P ${CMAKE_SOURCE_DIR}/tests/cmake/build_testing_off_test.cmake)
endif()
