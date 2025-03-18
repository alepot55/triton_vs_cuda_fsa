cmake_minimum_required(VERSION 3.10)
project(TestRegexConversion)

# Find includes in corresponding directories
include_directories(${CMAKE_SOURCE_DIR}/../common/include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Threads package
find_package(Threads REQUIRED)

# Build the common library first
add_custom_target(build_common
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../common/build
    COMMAND cd ${CMAKE_SOURCE_DIR}/../common/build && cmake .. && make
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building common library"
)

# Find the common library after building it
find_library(COMMON_LIB common PATHS ${CMAKE_SOURCE_DIR}/../common/build NO_DEFAULT_PATH)
if(NOT COMMON_LIB)
    message(FATAL_ERROR "Common library not found at ${CMAKE_SOURCE_DIR}/../common/build")
endif()

# Add the executable
add_executable(test_regex_conversion test_regex_conversion.cpp)

# Make sure the common library is built before building the test executable
add_dependencies(test_regex_conversion build_common)

# Link against the common library containing FSAEngine and getConversionDebugLog
target_link_libraries(test_regex_conversion PRIVATE Threads::Threads)
target_link_libraries(test_regex_conversion PRIVATE ${COMMON_LIB})

# Add gtest for unit testing
include(GoogleTest)
add_test(NAME regex_tests COMMAND test_regex_conversion)
