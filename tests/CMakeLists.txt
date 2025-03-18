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

# Aggiungere il target common (precedentemente in common/CMakeLists.txt)
add_library(common STATIC
    ${COMMON_DIR}/src/regex_conversion.cpp
)
target_include_directories(common PRIVATE
    ${COMMON_DIR}/include
    ${COMMON_DIR}/test
)
target_compile_options(common PRIVATE -Wall -Wextra)
option(DEBUG_FSA "Enable debug output for FSA engine" OFF)
if(DEBUG_FSA)
    target_compile_definitions(common PRIVATE -DDEBUG_FSA)
endif()
option(CPU_ONLY "Build without CUDA dependencies" OFF)
if(CPU_ONLY)
    target_compile_definitions(common PRIVATE -DCPU_ONLY)
else()
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA found, version: ${CUDA_VERSION}")
        target_include_directories(common PRIVATE ${CUDA_INCLUDE_DIRS})
    else()
        message(WARNING "CUDA not found, building in CPU-only mode")
        target_compile_definitions(common PRIVATE -DCPU_ONLY)
    endif()
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
target_link_libraries(cuda_fsa PRIVATE common ${CUDA_LIBRARIES} nvidia-ml)
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
    common
    ${CUDA_LIBRARIES}
    nvidia-ml
    Threads::Threads
)
