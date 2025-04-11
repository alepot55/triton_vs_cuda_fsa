cmake_minimum_required(VERSION 3.10)
project(MatrixTests LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA and cuBLAS component again in this scope
find_package(CUDA REQUIRED COMPONENTS cublas)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70) # Equivalent to sm_70
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

# Define project paths
set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR}/../..)
set(COMMON_DIR ${PROJECT_ROOT_DIR}/common)
set(CUDA_DIR ${PROJECT_ROOT_DIR}/cuda)

# Include directories
include_directories(
    ${COMMON_DIR}/include
    ${COMMON_DIR}/benchmark
    ${CUDA_DIR}/include
    ${CUDA_DIR}/src
)

# Define the executable target for the CUDA matrix test runner
add_executable(cuda_matrix_test_runner matrix_test_runner.cpp)

# Link the test runner against the necessary libraries using keyword signature
target_link_libraries(cuda_matrix_test_runner PRIVATE
    ${CUDA_LIB}                 # Link against our cuda_fsa library
    ${CUDA_CUBLAS_LIBRARIES}    # Use the variable populated by FindCUDA
    ${CUDA_LIBRARIES}           # General CUDA runtime libraries (might include cudart)
    Threads::Threads            # Link against Threads
)

# Add include directories if needed (though likely inherited)
target_include_directories(cuda_matrix_test_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/../common/include
    ${CMAKE_SOURCE_DIR}/../cuda/include
    ${CMAKE_SOURCE_DIR}/../cuda/src # Include src for cuda_utils.h etc.
)

# Ensure the executable is placed in the correct output directory
set_target_properties(cuda_matrix_test_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/matrix
)

# Additional compiler options
target_compile_options(cuda_matrix_test_runner PRIVATE -Wall -Wextra)
