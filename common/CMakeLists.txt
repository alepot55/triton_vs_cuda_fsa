cmake_minimum_required(VERSION 3.10)
project(FSACommon)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files for the common library
set(SOURCES
    src/regex_conversion.cpp
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create static library
add_library(common STATIC ${SOURCES})

# Set compiler flags
target_compile_options(common PRIVATE -Wall -Wextra)

# Optionally enable debug output
option(DEBUG_FSA "Enable debug output for FSA engine" OFF)
if(DEBUG_FSA)
    target_compile_definitions(common PRIVATE -DDEBUG_FSA)
endif()

# For CPU-only tests, provide a version without CUDA dependencies
option(CPU_ONLY "Build without CUDA dependencies" OFF)
if(CPU_ONLY)
    target_compile_definitions(common PRIVATE -DCPU_ONLY)
else()
    # Find CUDA package
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA found, version: ${CUDA_VERSION}")
        target_include_directories(common PRIVATE ${CUDA_INCLUDE_DIRS})
    else()
        message(WARNING "CUDA not found, building in CPU-only mode")
        target_compile_definitions(common PRIVATE -DCPU_ONLY)
    endif()
endif()
