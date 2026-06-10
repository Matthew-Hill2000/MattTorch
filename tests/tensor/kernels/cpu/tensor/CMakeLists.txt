add_executable(tensorKernelElementwiseSpeedTest tensorElementwiseSpeedTest.cpp)
add_executable(tensorKernelElementwiseAccuracyTest tensorElementwiseAccuracyTest.cpp)

set(targets
    tensorKernelElementwiseSpeedTest
    tensorKernelElementwiseAccuracyTest
)

foreach(t ${targets})
    target_link_libraries(${t} PRIVATE mattTorch::mattTorch)
    target_include_directories(${t} PRIVATE ${TESTS_COMMON_DIR})
endforeach()

set_target_properties(${targets} PROPERTIES
    FOLDER "tests/tensor/kernels/cpu/tensor"
)
