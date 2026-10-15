if(NOT DEFINED SOURCE_DIR OR NOT DEFINED BINARY_DIR OR NOT DEFINED GENERATOR
    OR NOT DEFINED PYTHON_EXECUTABLE)
  message(FATAL_ERROR
    "SOURCE_DIR, BINARY_DIR, GENERATOR, and PYTHON_EXECUTABLE are required")
endif()

file(MAKE_DIRECTORY "${BINARY_DIR}/.cmake/api/v1/query")
file(REMOVE_RECURSE "${BINARY_DIR}/.cmake/api/v1/reply")
file(WRITE "${BINARY_DIR}/.cmake/api/v1/query/codemodel-v2" "")

set(configure_command
  "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${BINARY_DIR}"
  -G "${GENERATOR}" -DBUILD_TESTING=OFF
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  "-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}")
if(DEFINED OSX_ARCHITECTURES AND NOT OSX_ARCHITECTURES STREQUAL "")
  list(APPEND configure_command "-DCMAKE_OSX_ARCHITECTURES=${OSX_ARCHITECTURES}")
endif()

execute_process(
  COMMAND ${configure_command}
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR
    "BUILD_TESTING=OFF configure failed:\n${configure_output}\n${configure_error}")
endif()

file(GLOB rpc_targets
  "${BINARY_DIR}/.cmake/api/v1/reply/target-grserve_rpc-*.json")
if(NOT rpc_targets)
  message(FATAL_ERROR "BUILD_TESTING=OFF did not generate grserve_rpc")
endif()

foreach(forbidden_target IN ITEMS
    grserve_frame_codec_test grserve_frame_sigpipe_helper
    grserve_unix_socket_test grserve_rpc_session_test
    grserve_rpc_disconnect_test grserve_engine_process_test grserve_rpc_test
    grserve_blocking_hal)
  file(GLOB forbidden_targets
    "${BINARY_DIR}/.cmake/api/v1/reply/target-${forbidden_target}-*.json")
  if(forbidden_targets)
    message(FATAL_ERROR
      "BUILD_TESTING=OFF unexpectedly generated ${forbidden_target}")
  endif()
endforeach()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${BINARY_DIR}" --target
    grserve_rpc grserve-engine grserve_mock_hal
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR
    "BUILD_TESTING=OFF production build failed:\n${build_output}\n${build_error}")
endif()

file(STRINGS "${BINARY_DIR}/compile_commands.json" testing_definitions
  REGEX "GRSERVE_RPC_TESTING")
if(testing_definitions)
  message(FATAL_ERROR
    "BUILD_TESTING=OFF production compile commands contain GRSERVE_RPC_TESTING")
endif()
