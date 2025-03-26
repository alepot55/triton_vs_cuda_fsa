cmake_minimum_required(VERSION 3.10)
project(MatrixTests LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
find_package(CUDA REQUIRED)
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

# Add CUDA matrix test runner
add_executable(cuda_matrix_test_runner matrix_test_runner.cpp)

# Link against CUDA libraries
target_link_libraries(cuda_matrix_test_runner
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    nvidia-ml
)

# Make sure it can find the CUDA implementation
target_include_directories(cuda_matrix_test_runner PRIVATE
    ${CUDA_DIR}/src
)

# If cuda_fsa is already defined in parent scope, add dependency
if(TARGET cuda_fsa)
    add_dependencies(cuda_matrix_test_runner cuda_fsa)
    target_link_libraries(cuda_matrix_test_runner cuda_fsa)
endif()

# Additional compiler options
target_compile_options(cuda_matrix_test_runner PRIVATE -Wall -Wextra)
