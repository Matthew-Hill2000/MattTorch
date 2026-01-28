add_executable(tensorFunctionSpeedTests
    speedTests.cpp
)

add_executable(tensorFunctionAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorFunctionSpeedTests
    PRIVATE mattTorch::mattTorch
)

target_link_libraries(tensorFunctionAccuracyTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    tensorFunctionSpeedTests
    tensorFunctionAccuracyTests
    PROPERTIES
        FOLDER "tests/function/tensor"
)
