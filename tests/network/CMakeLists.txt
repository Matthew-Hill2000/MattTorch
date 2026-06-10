add_executable(networkAccuracyTest networkAccuracyTest.cpp)
add_executable(networkSpeedTest networkSpeedTest.cpp)

set(targets networkAccuracyTest networkSpeedTest)

foreach(t ${targets})
    target_link_libraries(${t} PRIVATE mattTorch::mattTorch)
    target_include_directories(${t} PRIVATE ${TESTS_COMMON_DIR})
endforeach()

set_target_properties(${targets} PROPERTIES
    FOLDER "tests/network"
)
