cmake_minimum_required(VERSION 3.10)
project(FSATests LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Threads package
find_package(Threads REQUIRED)

# CUDA settings (replacing Makefile functionality)
# Request cublas component here
find_package(CUDA REQUIRED COMPONENTS cublas)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70) # Equivalent to sm_70 in the Makefile
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

# Find NVML (required by cuda_utils)
find_library(NVML_LIBRARY NAMES nvidia-ml nvml PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib) # Added common paths
if(NOT NVML_LIBRARY)
    message(WARNING "NVML library not found. GPU utilization metrics will be unavailable in CUDA tests.")
    # Define NO_NVML preprocessor flag if NVML is not found
    add_compile_definitions(NO_NVML)
else()
    message(STATUS "Found NVML library: ${NVML_LIBRARY}")
endif()

# Define project paths
set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR}/..)
set(COMMON_DIR ${PROJECT_ROOT_DIR}/common)
set(CUDA_DIR ${PROJECT_ROOT_DIR}/cuda)

# Include directories
include_directories(
    ${PROJECT_ROOT_DIR}/include # General project includes
    ${COMMON_DIR}/include
    ${CUDA_DIR}/include
    ${CUDA_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cases # Include for test_case.h
)

# Common library (now includes cmdline)
add_library(common STATIC
    ${COMMON_DIR}/src/regex_conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cases/test_case.cpp
    ${COMMON_DIR}/src/cmdline.cpp # Added cmdline source
)
target_include_directories(common PRIVATE
    ${COMMON_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cases
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

# Build CUDA library
file(GLOB CUDA_SOURCES 
    "${CUDA_DIR}/src/*.cu" 
    "${CUDA_DIR}/src/*.cpp"
)
# Add the new utility files
list(APPEND CUDA_SOURCES "${CUDA_DIR}/src/cuda_utils.cu")
# Add the new kernel file to the list of sources
list(APPEND CUDA_SOURCES "${CUDA_DIR}/src/cuda_fsa_table_kernels.cu")

# Create CUDA library target
add_library(cuda_fsa STATIC ${CUDA_SOURCES})
# Link common library, CUDA libraries, and NVML (conditionally)
target_link_libraries(cuda_fsa PRIVATE common ${CUDA_LIBRARIES} Threads::Threads)
if(NVML_LIBRARY)
    target_link_libraries(cuda_fsa PRIVATE ${NVML_LIBRARY}) # Link only if found
endif()
set(CUDA_LIB cuda_fsa)

# Add subdirectories for different test categories
add_subdirectory(cuda)
add_subdirectory(regex)
add_subdirectory(matrix)

