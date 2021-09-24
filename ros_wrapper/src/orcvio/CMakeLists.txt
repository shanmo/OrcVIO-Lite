cmake_minimum_required(VERSION 2.8.12)
project(orcvio)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
add_compile_options(-std=c++11)

# Modify cmake module path if new .cmake files are required
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")
set(ORCVIO_SRC "${CMAKE_CURRENT_LIST_DIR}/../../../src")
set(ORCVIO_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/../../../include")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    tf
    nav_msgs
    sensor_msgs
    geometry_msgs
    eigen_conversions
    tf_conversions
    tf2_ros
    nodelet
    image_transport
    cv_bridge
    message_filters
    pcl_conversions
    pcl_ros
    std_srvs
    sort_ros 
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus QUIET)
if (NOT ${Sophus_FOUND})
  find_package(sophus REQUIRED)
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
    INCLUDE_DIRS include ${ORCVIO_INCLUDE}
    LIBRARIES system_manager estimator image_processor
    CATKIN_DEPENDS
        roscpp std_msgs tf nav_msgs sensor_msgs geometry_msgs
        eigen_conversions tf_conversions message_runtime
        image_transport cv_bridge message_filters pcl_conversions
        pcl_ros std_srvs
    DEPENDS Boost EIGEN3 OpenCV SUITESPARSE
)

###########
## Build ##
###########

include_directories(
    include
    ${ORCVIO_INCLUDE}
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${SUITESPARSE_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

# Image processor
add_library(image_processor
    ${ORCVIO_SRC}/image_processor.cpp
    ${ORCVIO_SRC}/ORBDescriptor.cpp
)
target_link_libraries(image_processor
    ${OpenCV_LIBRARIES}
)

# Initializer
# -- Static Initializer
add_library(staticInitializer
    ${ORCVIO_SRC}/StaticInitializer.cpp
)
target_link_libraries(staticInitializer
    ${SUITESPARSE_LIBRARIES}
)
# -- Dynamic Initializer
add_library(dynamicInitializer
    ${ORCVIO_SRC}/StaticInitializer.cpp
    ${ORCVIO_SRC}/DynamicInitializer.cpp
    ${ORCVIO_SRC}/feature_manager.cpp
    ${ORCVIO_SRC}/initial_alignment.cpp
    ${ORCVIO_SRC}/initial_sfm.cpp
    ${ORCVIO_SRC}/solve_5pts.cpp
)
target_link_libraries(dynamicInitializer
    ${SUITESPARSE_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
# -- Flexible Initializer
add_library(flexibleInitializer
    ${ORCVIO_SRC}/FlexibleInitializer.cpp
)
target_link_libraries(flexibleInitializer
    staticInitializer
    dynamicInitializer
)

# Estimator
add_library(estimator
    ${ORCVIO_SRC}/orcvio.cpp
)
target_link_libraries(estimator
    flexibleInitializer
    ${SUITESPARSE_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# System Manger
add_library(system_manager
    src/System.cpp
    ${ORCVIO_SRC}/orcvio.cpp
    ${ORCVIO_SRC}/image_processor.cpp
)
add_dependencies(system_manager
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(system_manager
    image_processor estimator
    ${catkin_LIBRARIES}
)

# Object Mapper 
add_library(object_mapper
    src/ObjectInitNode.cpp 
    ${ORCVIO_SRC}/obj/ObjectFeature.cpp 
    ${ORCVIO_SRC}/obj/ObjectFeatureInitializer.cpp 
    ${ORCVIO_SRC}/obj/ObjectLM.cpp
    ${ORCVIO_SRC}/obj/ObjectLMLite.cpp
    ${ORCVIO_SRC}/obj/ObjectResJacCam.cpp
    ${ORCVIO_SRC}/obj/ObjectState.cpp
    ${ORCVIO_SRC}/feat/kf.cpp
    ${ORCVIO_SRC}/feat/Feature.cpp
    ${ORCVIO_SRC}/feat/FeatureInitializer.cpp
)
add_dependencies(object_mapper
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(object_mapper
  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES}
  )

# System Manager nodelet
add_library(OrcVIO_Nodelets
    src/System_nodelet.cpp
    src/ObjectMapper_nodelet.cpp
)
add_dependencies(OrcVIO_Nodelets
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(OrcVIO_Nodelets
    system_manager
    object_mapper
    ${catkin_LIBRARIES}
)

if (catkin_FOUND)
    
    # gt path publisher
    add_executable(publish_gt_path src/publish_gt_path.cpp src/gt_odom.cpp)
    target_link_libraries(publish_gt_path ${catkin_LIBRARIES})

endif()

#############
## Install ##
#############

install(TARGETS
  image_processor estimator system_manager OrcVIO_Nodelets publish_gt_path
  object_mapper flexibleInitializer dynamicInitializer
  staticInitializer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN "*_nodelet.h" EXCLUDE
)

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
    launch
    ${PROJECT_SOURCE_DIR}/../../../config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
