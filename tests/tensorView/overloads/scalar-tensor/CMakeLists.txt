add_executable(scalarTensorOverloadSpeedTests
    speedTests.cpp
)

add_executable(scalarTensorOverloadAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(scalarTensorOverloadSpeedTests
    PRIVATE mattTorch::mattTorch
)

target_link_libraries(scalarTensorOverloadAccuracyTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    scalarTensorOverloadSpeedTests
    scalarTensorOverloadAccuracyTests
    PROPERTIES
        FOLDER "tests/tensorView/overloads/scalar-tensor"
)
