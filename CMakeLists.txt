cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(BalancedPermutations LANGUAGES CXX CUDA)

add_executable(BalancedPermutations BalancedPermutations.cu)
target_compile_features(BalancedPermutations PUBLIC cxx_std_14)
target_compile_options(BalancedPermutations PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60 --ftz=true --prec-div=false --prec-sqrt=false --fmad=true>)
set_directory_properties(PROPERTIES VS_STARTUP_PROJECT BalancedPermutations)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_compile_definitions(BalancedPermutations PUBLIC USE_OPENMP)
    target_compile_options(BalancedPermutations PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>)
endif()