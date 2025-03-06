cmake_minimum_required(VERSION 3.28)
project(tut)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CXX_STANDARD_LIB "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")

add_executable(infer_water infer_water.cpp)



set_source_files_properties(
        infer_water.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)


set(Python3_INCLUDE_DIRS "/home/adada/anaconda3/include/python3.9")
set(Python3_LIBRARIES "/home/adada/anaconda3/lib/libpython3.9.so")
include_directories(${Python3_INCLUDE_DIRS})
include_directories("/home/adada/cppLocal/libtorch_cpu/include")
include_directories("/home/adada/cppLocal/deep_potential_cpp/include")

set(LIBTORCH_DIR "/home/adada/cppLocal/libtorch_cpu/lib/libtorch.so")
set(dpcppDir "/home/adada/cppLocal/deep_potential_cpp/lib/libdeepmd_cc.so")

set(Boost_NO_SYSTEM_PATHS YES)
if (Boost_NO_SYSTEM_PATHS)
    set(Boost_ROOT "~/cppLocal/boostBuild")
    set(Boost_INCLUDE_DIR "~/cppLocal/boostBuild/include")
    set(Boost_LIBRARY_DIR "~/cppLocal/boostBuild/lib")
endif (Boost_NO_SYSTEM_PATHS)

find_package(Boost  REQUIRED COMPONENTS python numpy filesystem)
if(Boost_FOUND)
    message(STATUS "boost root is: ${Boost_ROOT}")
    message(STATUS "boost include path is : ${Boost_INCLUDE_DIR}")
    message(STATUS "boost library path is : ${Boost_LIBRARY_DIR}")
    message(STATUS "boost libraries is : ${Boost_LIBRARIES}")
    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${Boost_LIBRARY_DIR})
else()
    message(WARNING "boost not found.")
endif()
target_link_directories(infer_water PRIVATE  "/home/adada/cppLocal/deep_potential_cpp/lib")
target_link_libraries(infer_water ${LIBTORCH_DIR})
target_link_libraries(infer_water ${dpcppDir})





