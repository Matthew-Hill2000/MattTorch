find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)

add_executable(tensorTensorKernelMatrixSpeedTest matrixSpeedTest.cpp)
add_executable(tensorTensorKernelMatrixAccuracyTest matrixAccuracyTest.cpp)

set(targets tensorTensorKernelMatrixSpeedTest tensorTensorKernelMatrixAccuracyTest)

foreach(t ${targets})
  target_link_libraries(${t} PRIVATE mattTorch::mattTorch)
  target_include_directories(${t} PRIVATE ${TESTS_COMMON_DIR})
endforeach()

target_link_libraries(tensorTensorKernelMatrixSpeedTest PRIVATE PkgConfig::OPENBLAS)

set_target_properties(${targets} PROPERTIES FOLDER
                                            "tests/tensor/kernels/cpu/tensor-tensor/matrixmult")
