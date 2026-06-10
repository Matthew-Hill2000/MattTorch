add_executable(tensorScalarOverloadSpeedTests
    speedTests.cpp
)

add_executable(tensorScalarOverloadAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorScalarOverloadSpeedTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorScalarOverloadSpeedTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(tensorScalarOverloadAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorScalarOverloadAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(
    tensorScalarOverloadSpeedTests
    tensorScalarOverloadAccuracyTests
    PROPERTIES
        FOLDER "tests/tensor/overloads/tensor-scalar"
)
