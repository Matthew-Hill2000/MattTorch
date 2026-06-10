add_executable(layerAccuracyTests
    layerAccuracyTest.cpp
)

add_executable(layerSpeedTests
    layerSpeedTest.cpp
)

target_link_libraries(layerAccuracyTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(layerAccuracyTests PRIVATE ${TESTS_COMMON_DIR})

target_link_libraries(layerSpeedTests
    PRIVATE mattTorch::mattTorch
)
target_include_directories(layerSpeedTests PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(
    layerAccuracyTests
    layerSpeedTests
    PROPERTIES
        FOLDER "tests/layer/"
)
