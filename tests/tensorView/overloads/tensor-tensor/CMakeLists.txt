
add_executable(tensorTensorOverloadSpeedTests
    speedTests.cpp
)

add_executable(tensorTensorOverloadAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorTensorOverloadSpeedTests
    PRIVATE mattTorch::mattTorch
)

target_link_libraries(tensorTensorOverloadAccuracyTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    tensorTensorOverloadSpeedTests
    tensorTensorOverloadAccuracyTests
    PROPERTIES
        FOLDER "tests/tensorView/overloads/tensor-tensor"
)
