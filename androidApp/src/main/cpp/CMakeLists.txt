cmake_minimum_required(VERSION 3.22.1)
project(malaria_crash_handler)
add_library(malaria_crash_handler SHARED crash_handler.cpp)
target_link_libraries(malaria_crash_handler log)
