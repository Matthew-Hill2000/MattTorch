add_executable(criterionTests
    criterionTest.cpp
)

target_link_libraries(criterionTests
    PRIVATE mattTorch::mattTorch
)

set_target_properties(
    criterionTests
    PROPERTIES
        FOLDER "tests/criterion/"
)
