cmake_minimum_required(VERSION 3.13)
project(tvm_c_inference C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# === Path to the generated C model ===
set(TVM_MODEL_DIR "${CMAKE_SOURCE_DIR}/tvm_model_c_extracted")

# === Automatically include crt_config.h in all compilation units ===
add_compile_definitions(__STDC_CONSTANT_MACROS)
add_compile_options("-include${TVM_MODEL_DIR}/runtime/include/crt_config.h")

# === Include directories ===
include_directories(
    ${TVM_MODEL_DIR}/runtime/include         # FIRST: contains crt_config.h
    include                                  # THEN other includes
    ${TVM_MODEL_DIR}/codegen/host/include
)

# === Source files from TVM runtime and model ===
file(GLOB_RECURSE RUNTIME_SRCS
    ${TVM_MODEL_DIR}/runtime/src/*.c
)

file(GLOB MODEL_SRCS
    ${TVM_MODEL_DIR}/codegen/host/src/*.c
)

# === Add executable ===
add_executable(main
    main.cpp
    ${RUNTIME_SRCS}
    ${MODEL_SRCS}
)

# === Link math library if needed ===
target_link_libraries(main m)
