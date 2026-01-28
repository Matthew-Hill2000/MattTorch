add_executable(tensorScalarOverloadSpeedTests
    speedTests.cpp
)

add_executable(tensorScalarOverloadAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorScalarOverloadSpeedTests
    PRIVATE mattTorch::mattTorch
)

target_link_libraries(tensorScalarOverloadAccuracyTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    tensorScalarOverloadSpeedTests
    tensorScalarOverloadAccuracyTests
    PROPERTIES
        FOLDER "tests/tensorView/overloads/tensor-scalar"
)
