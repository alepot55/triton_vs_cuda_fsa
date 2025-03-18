# Find CUDA package
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found, version: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(WARNING "CUDA not found, tests may not work properly")
endif()

if(CUDA_LIB STREQUAL "")
    message(WARNING "Skipping CUDA tests since CUDA library is not available.")
else()
    # Add the CUDA test executable
    add_executable(cuda_test_runner cuda_test_runner.cpp)

    # Link against both CUDA and common libraries
    target_link_libraries(cuda_test_runner PRIVATE 
        ${CUDA_LIB}
        common
        ${CUDA_LIBRARIES}
        Threads::Threads
    )

    # Add test
    add_test(NAME cuda_tests COMMAND cuda_test_runner)
endif()
