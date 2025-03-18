cmake_minimum_required(VERSION 3.10)
project(FSATests LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Threads package
find_package(Threads REQUIRED)

# CUDA settings (replacing Makefile functionality)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70) # Equivalent to sm_70 in the Makefile
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

# Define project paths
set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR}/..)
set(COMMON_DIR ${PROJECT_ROOT_DIR}/common)
set(CUDA_DIR ${PROJECT_ROOT_DIR}/cuda)

# Include directories (fixed paths for CUDA source)
include_directories(
    ${COMMON_DIR}/include
    ${COMMON_DIR}/test
    ${COMMON_DIR}/src
    ${CUDA_DIR}/include
    ${CUDA_DIR}/src
    ${CUDA_DIR}/benchmarks
)

# Set the common library path
set(COMMON_LIB ${COMMON_DIR}/build/libcommon.a)
if(NOT EXISTS ${COMMON_LIB})
    message(FATAL_ERROR "Common library not found at ${COMMON_LIB}. Run the script with proper setup.")
endif()

# Build CUDA library directly here instead of expecting a pre-built library
file(GLOB CUDA_SOURCES 
    "${CUDA_DIR}/src/*.cu" 
    "${CUDA_DIR}/src/*.cpp"
    "${CUDA_DIR}/benchmarks/*.cu"
    "${CUDA_DIR}/benchmarks/*.cpp"
)

# Create CUDA library target
add_library(cuda_fsa STATIC ${CUDA_SOURCES})
target_link_libraries(cuda_fsa PRIVATE ${COMMON_LIB} ${CUDA_LIBRARIES} nvidia-ml)
set(CUDA_LIB cuda_fsa)

# Add subdirectories for different test categories
add_subdirectory(cuda)
add_subdirectory(regex)

# Create the fsa_engine_cuda executable (what the original Makefile built)
file(GLOB COMMON_BENCHMARK_SOURCES "${COMMON_DIR}/benchmark/*.cpp")
file(GLOB COMMON_TEST_SOURCES "${COMMON_DIR}/test/*.cpp")
file(GLOB COMMON_SRC_SOURCES "${COMMON_DIR}/src/*.cpp")

add_executable(fsa_engine_cuda 
    ${COMMON_BENCHMARK_SOURCES}
    ${COMMON_TEST_SOURCES}
    ${COMMON_SRC_SOURCES}
)

target_link_libraries(fsa_engine_cuda PRIVATE 
    ${CUDA_LIB}
    ${COMMON_LIB}
    ${CUDA_LIBRARIES}
    nvidia-ml
    Threads::Threads
)
