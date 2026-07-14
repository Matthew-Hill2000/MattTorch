cmake_minimum_required(VERSION 3.25)

project(
  mattTorch
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
