add_executable(tensorTensorFunctionSpeedTests
    speedTests.cpp
)

add_executable(tensorTensorFunctionAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorTensorFunctionSpeedTests
    PRIVATE mattTorch::mattTorch
)

target_link_libraries(tensorTensorFunctionAccuracyTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    tensorTensorFunctionSpeedTests
    tensorTensorFunctionAccuracyTests
    PROPERTIES
        FOLDER "tests/function/tensor-tensor"
)
