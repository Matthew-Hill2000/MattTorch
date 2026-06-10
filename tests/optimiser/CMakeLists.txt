add_executable(optimiserAccuracyTest optimiserAccuracyTest.cpp)

target_link_libraries(optimiserAccuracyTest PRIVATE mattTorch::mattTorch)
target_include_directories(optimiserAccuracyTest PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(optimiserAccuracyTest PROPERTIES
    FOLDER "tests/optimiser"
)
