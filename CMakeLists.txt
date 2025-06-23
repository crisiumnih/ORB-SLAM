cmake_minimum_required(VERSION 3.10)
project(TrackerProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/src/tracker)

add_executable(tracker main.cpp src/tracker/tracker.cpp)
target_link_libraries(tracker ${OpenCV_LIBS})