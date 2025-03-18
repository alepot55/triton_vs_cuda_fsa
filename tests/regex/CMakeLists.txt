# Add the regex conversion test executable
add_executable(test_regex_conversion test_regex_conversion.cpp)

# Link against the common library
target_link_libraries(test_regex_conversion PRIVATE 
    common
    Threads::Threads
)

# Add test
add_test(NAME regex_tests COMMAND test_regex_conversion)
