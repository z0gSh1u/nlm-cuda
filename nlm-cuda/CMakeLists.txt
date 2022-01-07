cmake_minimum_required (VERSION 3.8)

set(RapidJSON_DIR "F:/vcpkg/vcpkg-2021.05.12/installed/x86-windows/share/rapidjson")

project(nlm-cuda CUDA CXX)

find_package(CUDA REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)

add_executable (nlm-cuda "nlm-cuda.cu")

set_target_properties(nlm-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nlm-cuda rapidjson)
