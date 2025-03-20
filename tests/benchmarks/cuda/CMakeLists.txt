cmake_minimum_required(VERSION 3.10)
project(cuda_benchmarks LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Print some debug info
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "Project source dir: ${CMAKE_CURRENT_SOURCE_DIR}")

# Include project directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/cases
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../cuda/src
)

# NVML library is required for GPU metrics
find_library(NVML_LIBRARY nvidia-ml
    HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/nvidia/ /usr/local/cuda/lib64/ /usr/lib/
)

# If NVML is not found, provide a fallback
if(NOT NVML_LIBRARY)
    message(WARNING "NVML library not found, using stub implementation")
    add_definitions(-DNO_NVML)
else()
    message(STATUS "Found NVML library: ${NVML_LIBRARY}")
endif()

# Define source files
set(COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/benchmark/cmdline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/benchmark/benchmark_metrics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/cases/test_case.cpp
)

# Find the FSA base engine implementation
set(FSA_ENGINE_SOURCE "")
foreach(dir IN ITEMS 
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../cuda/src")
    if(EXISTS "${dir}/fsa_engine.cpp")
        set(FSA_ENGINE_SOURCE "${dir}/fsa_engine.cpp")
        message(STATUS "Found FSA engine source: ${FSA_ENGINE_SOURCE}")
        break()
    endif()
    if(EXISTS "${dir}/fsa_engine.cu")
        set(FSA_ENGINE_SOURCE "${dir}/fsa_engine.cu")
        message(STATUS "Found FSA engine source: ${FSA_ENGINE_SOURCE}")
        break()
    endif()
endforeach()

set(CUDA_ENGINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../cuda/src/cuda_fsa_engine.cu
    ${FSA_ENGINE_SOURCE}
)

# Error out if the FSA engine isn't found
if("${FSA_ENGINE_SOURCE}" STREQUAL "")
    message(FATAL_ERROR "FSA engine source not found. Build cannot continue.")
endif()

# Build the benchmark executable
add_executable(benchmark_fsa 
    ${COMMON_SOURCES}
    ${CUDA_ENGINE_SOURCES}
)

# Link libraries
target_link_libraries(benchmark_fsa ${CUDA_LIBRARIES})
if(NVML_LIBRARY)
    target_link_libraries(benchmark_fsa ${NVML_LIBRARY})
endif()

# Copy the executable to the benchmarks directory
add_custom_command(TARGET benchmark_fsa POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:benchmark_fsa> ${CMAKE_CURRENT_SOURCE_DIR}/
)
