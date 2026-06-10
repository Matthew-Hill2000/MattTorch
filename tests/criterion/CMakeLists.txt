add_executable(criterionAccuracyTests criterionAccuracyTest.cpp)

add_executable(criterionSpeedTests criterionSpeedTest.cpp)

target_link_libraries(criterionAccuracyTests PRIVATE mattTorch::mattTorch)

target_include_directories(criterionAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(criterionSpeedTests PRIVATE mattTorch::mattTorch)
target_include_directories(criterionSpeedTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(criterionAccuracyTests criterionSpeedTests PROPERTIES FOLDER
                                                                            "tests/criterion/")
