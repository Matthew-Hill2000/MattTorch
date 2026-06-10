
add_executable(tensorOverloadSpeedTests
    speedTests.cpp
)

add_executable(tensorOverloadAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorOverloadSpeedTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorOverloadSpeedTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(tensorOverloadAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorOverloadAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(
    tensorOverloadSpeedTests
    tensorOverloadAccuracyTests
    PROPERTIES
        FOLDER "tests/tensor/overloads/tensor"
)
