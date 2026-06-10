
add_executable(tensorTensorOverloadSpeedTests
    speedTests.cpp
)

add_executable(tensorTensorOverloadAccuracyTests
    accuracyTests.cpp
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)

target_link_libraries(tensorTensorOverloadSpeedTests
    PRIVATE mattTorch::mattTorch PkgConfig::OPENBLAS
)
target_include_directories(tensorTensorOverloadSpeedTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(tensorTensorOverloadAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorTensorOverloadAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(
    tensorTensorOverloadSpeedTests
    tensorTensorOverloadAccuracyTests
    PROPERTIES
        FOLDER "tests/tensor/overloads/tensor-tensor"
)
