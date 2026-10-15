cmake_minimum_required(VERSION 3.18)

# FTUCA's PyTorch wheel reports neither CUDA_HOME nor a stock ROCm extension.
# Compile both translation units directly with the SDK compiler instead.
set(CMAKE_CXX_COMPILER "/usr/local/fant/bin/hipcc" CACHE FILEPATH "" FORCE)
project(fant_sdpa LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import pathlib, torch_fant; print(pathlib.Path(torch_fant.__file__).parent.parent)"
  OUTPUT_VARIABLE TORCH_FANT_PACKAGE_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

find_library(
  TORCH_FANT_LIBRARY
  NAMES fant_python
  PATHS "${TORCH_FANT_PACKAGE_ROOT}/torch_fant/lib"
  REQUIRED
  NO_DEFAULT_PATH)
find_library(
  FANT_HIP_RUNTIME
  NAMES fanthip amdhip64
  PATHS
    /usr/local/fant/lib
    /usr/lib/x86_64-linux-gnu/fantgpu-fh3
  REQUIRED
  NO_DEFAULT_PATH)

set_source_files_properties(fant_sdpa_kernel.hip PROPERTIES HEADER_FILE_ONLY ON)
add_library(fant_sdpa SHARED fant_sdpa.cpp fant_sdpa_kernel.hip)
set_target_properties(
  fant_sdpa
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "${TORCH_FANT_PACKAGE_ROOT}/torch/lib;${TORCH_FANT_PACKAGE_ROOT}/torch_fant/lib;/usr/local/fant/lib;/usr/lib/x86_64-linux-gnu/fantgpu-fh3")
target_compile_definitions(
  fant_sdpa PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_compile_options(fant_sdpa PRIVATE ${TORCH_CXX_FLAGS})
target_include_directories(
  fant_sdpa
  PRIVATE
    ${TORCH_INCLUDE_DIRS}
    "${TORCH_FANT_PACKAGE_ROOT}"
    /usr/local/fant/include)
target_link_libraries(
  fant_sdpa
  PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_FANT_LIBRARY}
    ${FANT_HIP_RUNTIME}
    /usr/local/fant/lib/libfhblas.so.4.0)
