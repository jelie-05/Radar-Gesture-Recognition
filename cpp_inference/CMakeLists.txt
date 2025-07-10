cmake_minimum_required(VERSION 3.13)
project(tvm_runtime_only C CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CMake utilities
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils")
include(Utils OPTIONAL)
include(Summary OPTIONAL)
include(Linker OPTIONAL)

# Optional external config
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/config.cmake")
endif()

# === Include paths ===
include_directories(
    include
    third_party/tvm
    third_party/tvm/runtime
    third_party/tvm/runtime/threading
    third_party/tvm/support
    3rdparty/dmlc-core/include
    3rdparty/dlpack/include
)

# === TVM runtime sources ===
set(RUNTIME_SRCS
    third_party/tvm/runtime/c_runtime_api.cc
    third_party/tvm/runtime/registry.cc
    third_party/tvm/runtime/module.cc
    third_party/tvm/runtime/ndarray.cc
    third_party/tvm/runtime/object.cc
    third_party/tvm/runtime/logging.cc
    third_party/tvm/runtime/source_utils.cc
    third_party/tvm/runtime/library_module.cc
    third_party/tvm/runtime/file_utils.cc
    third_party/tvm/runtime/thread_pool.cc
    third_party/tvm/runtime/threading_backend.cc
    third_party/tvm/runtime/graph_executor/graph_executor_factory.cc
    third_party/tvm/runtime/graph_executor/graph_executor.cc
    third_party/tvm/runtime/cpu_device_api.cc
    third_party/tvm/runtime/workspace_pool.cc
)

# === Object libraries ===
add_library(tvm_runtime_obj OBJECT ${RUNTIME_SRCS})
target_compile_definitions(tvm_runtime_obj PRIVATE TVM_RUNTIME_EXPORT_RUNTIME=1 DMLC_LOG_CUSTOMIZE=1)

add_library(tvm_runtime_dso_loader_obj OBJECT third_party/tvm/runtime/dso_library.cc)
target_compile_definitions(tvm_runtime_dso_loader_obj PRIVATE TVM_RUNTIME_EXPORT_RUNTIME=1)

# === Final static runtime lib ===
add_library(tvm_runtime STATIC
    $<TARGET_OBJECTS:tvm_runtime_obj>
    $<TARGET_OBJECTS:tvm_runtime_dso_loader_obj>
)

set_target_properties(tvm_runtime PROPERTIES LINKER_LANGUAGE CXX)

# Create archive alias for whole-archive use
add_custom_command(TARGET tvm_runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tvm_runtime> ${CMAKE_BINARY_DIR}/libtvm_runtime.a
)

# === Executable ===
add_executable(run_so main_so.cpp)

# Link with forced whole-archive around the actual .a file
target_link_options(run_so PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/libtvm_runtime.a"
    "-Wl,--no-whole-archive"
)

# Link additional system libs
target_link_libraries(run_so PRIVATE pthread m)

# === C Code Generation ===
file(GLOB MODEL_MICRO_SRCS "model-micro/codegen/host/src/*.c")

add_executable(run_c main_c.cpp ${MODEL_MICRO_SRCS})
target_include_directories(run_c PRIVATE
    model-micro/codegen/host/include
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(run_inference main_inference.cpp ${MODEL_MICRO_SRCS})
target_include_directories(run_inference PRIVATE
    model-micro/codegen/host/include
    ${PROJECT_SOURCE_DIR}/include
)