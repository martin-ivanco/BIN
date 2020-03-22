cmake_minimum_required(VERSION 3.0)

project( evobool )

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

find_package( OpenMP )

add_executable( evobool "src/main.cpp" )

if(OpenMP_CXX_FOUND)
    target_link_libraries( evobool PUBLIC OpenMP::OpenMP_CXX )
else()
    target_link_libraries( evobool )
endif()
