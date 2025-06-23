cmake_minimum_required(VERSION 3.10)
project(HomogeneousORB CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV REQUIRED)

set(Sophus_DIR "/home/crisi/catkin_ws/src/ORB_SLAM3/Thirdparty/Sophus/build")

find_package(Sophus REQUIRED)
if(Sophus_FOUND)
    message(STATUS "Sophus found: ${Sophus_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Sophus not found. Please set Sophus_DIR or install Sophus.")
endif()

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found.")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS} # Add Sophus include directory
    ${EIGEN3_INCLUDE_DIR}  # Add Eigen3 include directory
    src/tracker
)

add_executable(homogeneous_orb main.cpp src/tracker/tracker.cpp)

target_link_libraries(
    homogeneous_orb
    ${OpenCV_LIBS}
    # Sophus is typically header-only, so no specific library to link.
    # However, if you are using a Sophus build that creates a library (e.g., Sophus_TESTS or other modules),
    # you might need to link them here. For standard usage, this line is often commented out or not needed.
    # ${Sophus_LIBRARIES}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)