cmake_minimum_required(VERSION 3.16)
project(fiber_pfczm_dealii LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(deal.II 9.3 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)

deal_ii_initialize_cached_variables()

add_executable(fiber_pfczm src/fiber_pfczm.cc)
deal_ii_setup_target(fiber_pfczm)
