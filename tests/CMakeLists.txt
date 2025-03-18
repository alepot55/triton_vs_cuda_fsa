cmake_minimum_required(VERSION 3.10)
project(FSATests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Threads package
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../common/include
    ${CMAKE_SOURCE_DIR}/../common/test
    ${CMAKE_SOURCE_DIR}/../cuda/include
)

# Set the common library path
set(COMMON_LIB ${CMAKE_SOURCE_DIR}/../common/build/libcommon.a)
if(NOT EXISTS ${COMMON_LIB})
    message(FATAL_ERROR "Common library not found at ${COMMON_LIB}. Run the script with proper setup.")
endif()

# Set the CUDA library path
set(CUDA_LIB ${CMAKE_SOURCE_DIR}/../cuda/build/libcuda_fsa.a)
if(NOT EXISTS ${CUDA_LIB})
    message(WARNING "CUDA library not found at ${CUDA_LIB}. CUDA tests will be skipped.")
    set(CUDA_LIB "")
endif()

# Add subdirectories for different test categories
add_subdirectory(cuda)
add_subdirectory(regex)
