cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0057 NEW)

project(orcvio)
include(CTest)

option(BUILD_SHARED_LIBS "Whether to build shared libs"
  True)

option(CMAKE_BUILD_TYPE "Release or Debug" "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED filesystem)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${SUITESPARSE_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

# Image processor
add_library(${PROJECT_NAME}_image_processor
    ${PROJECT_SOURCE_DIR}/src/image_processor.cpp
    ${PROJECT_SOURCE_DIR}/src/ORBDescriptor.cpp
)
target_link_libraries(${PROJECT_NAME}_image_processor
    ${OpenCV_LIBRARIES}
)

# Initializer
# -- Static Initializer
add_library(${PROJECT_NAME}_static_initializer
    ${PROJECT_SOURCE_DIR}/src/StaticInitializer.cpp
)
target_link_libraries(${PROJECT_NAME}_static_initializer
    ${SUITESPARSE_LIBRARIES}
)
# -- Dynamic Initializer
add_library(${PROJECT_NAME}_dynamic_initializer
    ${PROJECT_SOURCE_DIR}/src/StaticInitializer.cpp
    ${PROJECT_SOURCE_DIR}/src/DynamicInitializer.cpp
    ${PROJECT_SOURCE_DIR}/src/feature_manager.cpp
    ${PROJECT_SOURCE_DIR}/src/initial_alignment.cpp
    ${PROJECT_SOURCE_DIR}/src/initial_sfm.cpp
    ${PROJECT_SOURCE_DIR}/src/solve_5pts.cpp
)
target_link_libraries(${PROJECT_NAME}_dynamic_initializer
    ${SUITESPARSE_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
# -- Flexible Initializer
add_library(${PROJECT_NAME}_flexible_initializer
    ${PROJECT_SOURCE_DIR}/src/FlexibleInitializer.cpp
)
target_link_libraries(${PROJECT_NAME}_flexible_initializer
    ${PROJECT_NAME}_static_initializer
    ${PROJECT_NAME}_dynamic_initializer
)

# Estimator
add_library(${PROJECT_NAME}_estimator
    ${PROJECT_SOURCE_DIR}/src/orcvio.cpp
)
target_link_libraries(${PROJECT_NAME}_estimator
    ${PROJECT_NAME}_flexible_initializer
    ${SUITESPARSE_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# App
add_executable(orcvio app/orcvioMain.cpp)
target_link_libraries(orcvio
    ${PROJECT_NAME}_image_processor
    ${PROJECT_NAME}_estimator
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
)

file(GLOB mappingFiles
  LIST_DIRECTORIES false
  ${PROJECT_SOURCE_DIR}/src/obj/*.cpp
  ${PROJECT_SOURCE_DIR}/src/feat/*.cpp)

add_library(${PROJECT_NAME}_mapping
  ${mappingFiles})
target_link_libraries(${PROJECT_NAME}_mapping
  ${OpenCV_LIBS})

if(BUILD_TESTING)

  find_package(GTest REQUIRED)
  find_package(PythonLibs 2.7)
  if(PYTHONLIBS_FOUND AND NOT DISABLE_MATPLOTLIB)
    add_definitions(-DHAVE_PYTHONLIBS=1)
    message(STATUS "PYTHON VERSION: " ${PYTHONLIBS_VERSION_STRING})
    message(STATUS "PYTHON INCLUDE: " ${PYTHON_INCLUDE_DIRS})
    message(STATUS "PYTHON LIBRARIES: " ${PYTHON_LIBRARIES})
  endif()
  include_directories(${PYTHON_INCLUDE_DIRS})

  add_executable(${PROJECT_NAME}_tests
    src/tests/test_utils.cpp
    src/tests/test_se3.cpp
    src/tests/test_kabsch.cpp
    src/tests/test_state_update.cpp
    src/tests/test_object_init_multiframe.cpp
    src/tests/test_levenberg_marquardt.cpp
    src/tests/test_object_lm.cpp
    src/tests/test_object_lm_multiframe.cpp)

  target_link_libraries(${PROJECT_NAME}_tests
    ${PROJECT_NAME}_mapping
    ${PROJECT_NAME}_estimator
    ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} GTest::GTest
    GTest::Main)

  add_test(
    ${PROJECT_NAME}_tests
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

  # for line coverage 
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage -std=c++11 -g")
  # add_custom_target(cov COMMAND ${PROJECT_SOURCE_DIR}/build_scripts/generate_coverage_report -r "make")

endif() # BUILD_TESTING
