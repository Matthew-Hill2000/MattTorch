add_executable(tensorTensorFunctionSpeedTests
    speedTests.cpp
)

add_executable(tensorTensorFunctionAccuracyTests
    accuracyTests.cpp
)

add_executable(tensorTensorFunctionMatrixAccuracyTests
    matrixAccuracyTests.cpp
)

add_executable(tensorTensorFunctionMatrixSpeedTests
    matrixSpeedTests.cpp
)

target_link_libraries(tensorTensorFunctionSpeedTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorTensorFunctionSpeedTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(tensorTensorFunctionAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorTensorFunctionAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(tensorTensorFunctionMatrixAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorTensorFunctionMatrixAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(tensorTensorFunctionMatrixSpeedTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(tensorTensorFunctionMatrixSpeedTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(
    tensorTensorFunctionSpeedTests
    tensorTensorFunctionAccuracyTests
    tensorTensorFunctionMatrixAccuracyTests
    tensorTensorFunctionMatrixSpeedTests
    PROPERTIES
        FOLDER "tests/function/tensor-tensor"
)
