
add_executable(tensorScalarfunctionSpeedTests
    speedTests.cpp
)

add_executable(tensorScalarfunctionAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorScalarfunctionSpeedTests
    PRIVATE mattTorch::mattTorch
)

target_link_libraries(tensorScalarfunctionAccuracyTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    tensorScalarfunctionSpeedTests
    tensorScalarfunctionAccuracyTests
    PROPERTIES
        FOLDER "tests/function/tensor-scalar"
)
