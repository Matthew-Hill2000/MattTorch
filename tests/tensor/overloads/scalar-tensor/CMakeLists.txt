add_executable(scalarTensorOverloadSpeedTests
    speedTests.cpp
)

add_executable(scalarTensorOverloadAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(scalarTensorOverloadSpeedTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(scalarTensorOverloadSpeedTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(scalarTensorOverloadAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(scalarTensorOverloadAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(
    scalarTensorOverloadSpeedTests
    scalarTensorOverloadAccuracyTests
    PROPERTIES
        FOLDER "tests/tensor/overloads/scalar-tensor"
)
