set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(protobuf_FORCE_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)

# FetchContent uses the first declaration for a dependency name. Declare Abseil
# before protobuf so its moving release tag can never select different source.
include(FetchContent)
FetchContent_Declare(absl
  GIT_REPOSITORY "https://github.com/abseil/abseil-cpp.git"
  GIT_TAG "9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f"
  GIT_SHALLOW FALSE)

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest ${CMAKE_BINARY_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/protobuf ${CMAKE_BINARY_DIR}/third_party/protobuf EXCLUDE_FROM_ALL)
