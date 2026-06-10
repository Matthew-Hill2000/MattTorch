
add_executable(tensorPrintTest
    printTest.cpp
)


target_link_libraries(tensorPrintTest
    PRIVATE mattTorch::mattTorch
)


set_target_properties(
    tensorPrintTest
    PROPERTIES
        FOLDER "tests/tensor/printing"
)
