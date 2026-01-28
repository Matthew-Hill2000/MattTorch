
add_executable(layerTests
    layerTest.cpp
)

target_link_libraries(layerTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    layerTests
    PROPERTIES
        FOLDER "tests/criterion/"
)
