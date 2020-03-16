cmake_minimum_required(VERSION 3.0)

project( evobool )

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

add_executable( evobool "src/main.cpp" )

target_link_libraries( evobool )
