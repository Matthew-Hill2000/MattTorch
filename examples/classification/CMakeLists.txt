add_executable(classification classification.cpp)

target_link_libraries(classification PRIVATE mattTorch::mattTorch)
