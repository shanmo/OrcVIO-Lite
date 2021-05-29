cmake_minimum_required(VERSION 2.8.8)

# Project name
project(sort_ros)

IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
ENDIF()

# Find catkin (the ROS build system)
find_package(catkin_simple REQUIRED)

# Include libraries
find_package(OpenCV 3 REQUIRED)

# display message to user
message(STATUS "OPENCV VERSION: " ${OpenCV_VERSION})

catkin_simple(ALL_DEPS_REQUIRED)

# Try to compile with c++11
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized")


# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${OpenCV_LIBRARIES}
)

##################################################
# Make the core library
##################################################
cs_add_library(sort_core_lib
        src/Hungarian.cpp
        src/KalmanTracker.cpp
        src/sort_tracking.cpp
        src/sort_ros_nodelet.cpp
)
target_link_libraries(sort_core_lib ${thirdparty_libraries})

##################################################
# Make binary files!
##################################################
cs_add_executable(sort_ros src/main.cpp)
target_link_libraries(sort_ros sort_core_lib ${thirdparty_libraries})

cs_add_executable(test_kf src/test_kf.cpp)
target_link_libraries(test_kf sort_core_lib ${thirdparty_libraries})

## Specify libraries to link a library or executable target against
cs_add_executable(sort_ros_node src/sort_ros_node.cpp)
target_link_libraries(sort_ros_node sort_core_lib ${catkin_LIBRARIES})

cs_install()
cs_export()
install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
