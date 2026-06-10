
add_executable(tensorScalarfunctionSpeedTests
    speedTests.cpp
)

add_executable(tensorScalarfunctionAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorScalarfunctionSpeedTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorScalarfunctionSpeedTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(tensorScalarfunctionAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorScalarfunctionAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(
    tensorScalarfunctionSpeedTests
    tensorScalarfunctionAccuracyTests
    PROPERTIES
        FOLDER "tests/function/tensor-scalar"
)
