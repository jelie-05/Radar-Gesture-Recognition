# Minimal CMakeLists.txt for standalone TVM Runtime build
cmake_minimum_required(VERSION 3.13)
project(tvm_runtime_only C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include TVM CMake utilities
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils")
include(Utils)
include(Summary)
include(Linker)

# Load config
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/config.cmake")
endif()

# Set visibility and logging macros (minimal)
add_definitions(-DTVM_RUNTIME_EXPORT_RUNTIME=1 -DDMLC_LOG_CUSTOMIZE=1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

# Include paths
include_directories(include)  
include_directories(third_party/tvm)  # Runtime headers
include_directories(3rdparty/dmlc-core/include)
include_directories(3rdparty/dlpack/include)

# Source files (minimal runtime subset)
file(GLOB RUNTIME_SRCS
    third_party/tvm/runtime/c_runtime_api.cc
    third_party/tvm/runtime/registry.cc
    third_party/tvm/runtime/module.cc
    third_party/tvm/runtime/ndarray.cc
    third_party/tvm/runtime/object.cc
    third_party/tvm/runtime/logging.cc
    third_party/tvm/runtime/source_utils.cc  
    third_party/tvm/runtime/library_module.cc
)

# Runtime static library
add_library(tvm_runtime STATIC ${RUNTIME_SRCS})

# # Build main inference binary
# add_executable(run_model main.cpp)
# target_link_libraries(run_model PRIVATE tvm_runtime)
