cmake_minimum_required (VERSION 3.8)

project("nlm-cpu-and-cuda")

# You should modify this according to where your RapidJSON is.
set(RapidJSON_DIR "F:/vcpkg/vcpkg-2021.05.12/installed/x86-windows/share/rapidjson")

add_subdirectory("nlm-cpu")
add_subdirectory("nlm-cuda")
