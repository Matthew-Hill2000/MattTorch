add_executable(datasetAccuracyTest datasetAccuracyTest.cpp)

target_link_libraries(datasetAccuracyTest PRIVATE mattTorch::mattTorch)
target_include_directories(datasetAccuracyTest PRIVATE ${TESTS_COMMON_DIR})

set_target_properties(datasetAccuracyTest PROPERTIES
    FOLDER "tests/dataset"
)
