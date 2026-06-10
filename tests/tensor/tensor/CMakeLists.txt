add_executable(tensorCoreAccuracyTest tensorAccuracyTest.cpp)
add_executable(tensorCoreAccesssorsAccuracyTest tensorAccesssorsAccuracyTest.cpp)
add_executable(tensorCoreBackPropAccuracyTest tensorBackPropAccuracyTest.cpp)
add_executable(tensorCoreBackPropSpeedTest tensorBackPropSpeedTest.cpp)

set(targets
    tensorCoreAccuracyTest
    tensorCoreAccesssorsAccuracyTest
    tensorCoreBackPropAccuracyTest
    tensorCoreBackPropSpeedTest
)

foreach(t ${targets})
    target_link_libraries(${t} PRIVATE mattTorch::mattTorch)
    target_include_directories(${t} PRIVATE ${TESTS_COMMON_DIR})
endforeach()

set_target_properties(${targets} PROPERTIES
    FOLDER "tests/tensor/tensor"
)
