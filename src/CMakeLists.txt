add_library(mattTorch)
add_library(mattTorch::mattTorch ALIAS mattTorch)

target_include_directories(mattTorch PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mattTorch PUBLIC cxx_std_23)

include(FetchContent)

FetchContent_Declare(plotlypp GIT_REPOSITORY https://github.com/jimmyorourke/plotlypp.git
                     GIT_TAG main)

FetchContent_MakeAvailable(plotlypp)

add_subdirectory(tensor)
add_subdirectory(function)
add_subdirectory(layer)
add_subdirectory(network)
add_subdirectory(optimiser)
add_subdirectory(dataset)
add_subdirectory(criterion)

find_package(OpenMP REQUIRED)
target_link_libraries(mattTorch PUBLIC OpenMP::OpenMP_CXX plotlypp)

find_package(sleef CONFIG REQUIRED)
target_link_libraries(mattTorch PRIVATE sleef::sleef)

target_compile_options(mattTorch PRIVATE -mavx2 -mfma -march=native)

target_compile_options(mattTorch PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -ffast-math>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(mattTorch PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
  target_compile_options(mattTorch PRIVATE /W4)
endif()
