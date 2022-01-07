cmake_minimum_required(VERSION 3.8)

project(nlm-cpu CXX)

find_package(RapidJSON CONFIG REQUIRED)

add_executable(nlm-cpu "nlm-cpu.cpp")

target_link_libraries(nlm-cpu rapidjson)
