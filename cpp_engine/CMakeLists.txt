cmake_minimum_required(VERSION 3.16)
project(cpp_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Never benchmark a debug build. AVX2 + FMA + native tuning.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma")

# Select the matmul inner loop: ON = AVX2 + FMA + OpenMP, OFF = naive triple loop.
# Flip to A/B correctness and speedup between two builds.
option(ENGINE_AVX2 "Use the AVX2 matmul kernels" ON)

find_package(OpenMP)

add_library(engine
  src/weights.cpp
  src/matmul.cpp
  src/ops.cpp
  src/model.cpp
)
target_include_directories(engine PUBLIC include)
if(ENGINE_AVX2)
  target_compile_definitions(engine PUBLIC ENGINE_AVX2)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(load_check tools/load_check.cpp)
target_link_libraries(load_check PRIVATE engine)

add_executable(parity_test tests/parity_test.cpp)
target_link_libraries(parity_test PRIVATE engine)
