add_executable(basicExample basicExample.cpp)
add_executable(tensorExample tensorExample.cpp)
add_executable(datasetExample datasetExample.cpp)
add_executable(autograd autograd.cpp)
add_executable(PINN PINN.cpp)

target_link_libraries(tensorExample
  PRIVATE
    mattTorch::mattTorch
)

target_link_libraries(datasetExample
  PRIVATE
    mattTorch::mattTorch
)

target_link_libraries(basicExample
    PRIVATE
        mattTorch::mattTorch
)

target_link_libraries(autograd
  PRIVATE
    mattTorch::mattTorch
)

target_link_libraries(PINN
  PRIVATE
    mattTorch::mattTorch
)
