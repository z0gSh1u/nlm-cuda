cmake_minimum_required (VERSION 3.8)

project(nlm-cuda CUDA CXX)

find_package(CUDA REQUIRED)

cuda_add_executable (nlm-cuda "nlm-cuda.cu")
