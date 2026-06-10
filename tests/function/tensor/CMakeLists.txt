add_executable(tensorFunctionSpeedTests
    speedTests.cpp
)

add_executable(tensorFunctionAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorFunctionSpeedTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorFunctionSpeedTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(tensorFunctionAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorFunctionAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(
    tensorFunctionSpeedTests
    tensorFunctionAccuracyTests
    PROPERTIES
        FOLDER "tests/function/tensor"
)
