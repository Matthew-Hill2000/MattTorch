
add_executable(tensorOverloadSpeedTests
    speedTests.cpp
)

add_executable(tensorOverloadAccuracyTests
    accuracyTests.cpp
)

target_link_libraries(tensorOverloadSpeedTests
    PRIVATE mattTorch::mattTorch
)

target_link_libraries(tensorOverloadAccuracyTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    tensorOverloadSpeedTests
    tensorOverloadAccuracyTests
    PROPERTIES
        FOLDER "tests/tensorView/overloads/tensor"
)
