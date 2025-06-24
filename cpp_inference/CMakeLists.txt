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


# === Extract sample headers from TAR archive ===
set(SAMPLES_TAR ${CMAKE_SOURCE_DIR}/../data/samples/test_samples.tar)
set(EXTRACT_DIR ${CMAKE_SOURCE_DIR}/include/data/samples)

# Ensure the output directory exists
file(MAKE_DIRECTORY ${EXTRACT_DIR})

# Custom command to extract header files
add_custom_command(
    OUTPUT ${EXTRACT_DIR}/class0_input.h  # Dummy file to trigger command
    COMMAND ${CMAKE_COMMAND} -E tar xvf ${SAMPLES_TAR} --strip-components=1 -C ${EXTRACT_DIR}
    DEPENDS ${SAMPLES_TAR}
    COMMENT "Extracting test_samples.tar to include/data/samples/"
)

# Custom target to run the extraction step
add_custom_target(extract_samples ALL
    DEPENDS ${EXTRACT_DIR}/class0_input.h
)

# Add include directory for extracted files
include_directories(${CMAKE_SOURCE_DIR}/include)
