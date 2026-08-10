cmake_minimum_required(VERSION 3.16)
project(cpp_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Never benchmark a debug build. AVX2 + FMA + native tuning.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma")

find_package(OpenMP)

# Engine library. New .cpp files (matmul, ops, model) get added here as they land.
add_library(engine
  src/weights.cpp
)
target_include_directories(engine PUBLIC include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX)
endif()

# Loader sanity check (temporary tool; parity_test replaces it later).
add_executable(load_check tools/load_check.cpp)
target_link_libraries(load_check PRIVATE engine)
