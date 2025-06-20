cmake_minimum_required(VERSION 3.10)
project(tvm_cpp_inference)

set(CMAKE_CXX_STANDARD 17)
set(TVM_HOME ${CMAKE_SOURCE_DIR}/../tvm)

# Include TVM headers
include_directories(${TVM_HOME}/include)
include_directories(${TVM_HOME}/3rdparty/dmlc-core/include)
include_directories(${TVM_HOME}/3rdparty/dlpack/include)

# Link to TVM runtime
link_directories(${TVM_HOME}/build)

# Source file
add_executable(infer main.cpp)

# Link with TVM runtime
target_link_libraries(infer tvm_runtime)
