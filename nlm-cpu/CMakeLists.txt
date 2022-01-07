cmake_minimum_required(VERSION 3.8)

set(RapidJSON_DIR "F:/vcpkg/vcpkg-2021.05.12/installed/x86-windows/share/rapidjson")

project(nlm-cpu CXX)

find_package(RapidJSON CONFIG REQUIRED)

add_executable(nlm-cpu "nlm-cpu.cpp")

target_link_libraries(nlm-cpu rapidjson)
