add_executable(tensorStorageAccuracyTest tensorStorageAccuracyTest.cpp)

target_link_libraries(tensorStorageAccuracyTest PRIVATE mattTorch::mattTorch)
target_include_directories(tensorStorageAccuracyTest PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(tensorStorageAccuracyTest PROPERTIES
    FOLDER "tests/tensor/tensorStorage"
)
