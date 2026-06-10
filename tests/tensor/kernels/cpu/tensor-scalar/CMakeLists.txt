add_executable(tensorScalarKernelElementwiseSpeedTest
    tensorScalarElementwiseSpeedTest.cpp)
add_executable(tensorScalarKernelElementwiseAccuracyTest
    tensorScalarElementwiseAccuracyTest.cpp)

set(targets
    tensorScalarKernelElementwiseSpeedTest
    tensorScalarKernelElementwiseAccuracyTest
)

foreach(t ${targets})
    target_link_libraries(${t} PRIVATE mattTorch::mattTorch)
    target_include_directories(${t} PRIVATE ${TESTS_COMMON_DIR})
endforeach()

set_target_properties(${targets} PROPERTIES
    FOLDER "tests/tensor/kernels/cpu/tensor-scalar"
)
