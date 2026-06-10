add_executable(tensorTensorKernelElementwiseSpeedTest elementwiseSpeedTest.cpp)
add_executable(tensorTensorKernelElementwiseAccuracyTest elementwiseAccuracyTest.cpp)

set(targets
    tensorTensorKernelElementwiseSpeedTest
    tensorTensorKernelElementwiseAccuracyTest
)

foreach(t ${targets})
    target_link_libraries(${t} PRIVATE mattTorch::mattTorch)
    target_include_directories(${t} PRIVATE ${TESTS_COMMON_DIR})
endforeach()

set_target_properties(${targets} PROPERTIES
    FOLDER "tests/tensor/kernels/cpu/tensor-tensor/elementwise"
)
