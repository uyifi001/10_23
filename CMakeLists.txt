cmake_minimum_required(VERSION 3.16)
project(coating_pfczm_dealii LANGUAGES CXX)

find_package(deal.II 9.4 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

deal_ii_initialize_cached_variables()

add_executable(coating_pfczm source/coating_pfczm.cc)
deal_ii_setup_target(coating_pfczm)

set_target_properties(coating_pfczm PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
