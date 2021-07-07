cmake_minimum_required(VERSION 2.8)
project(ORBSLAM)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
# set(CMAKE_C_COMPILER "/usr/bin/gcc-10")

find_package(OpenCV 3.0 REQUIRED)
find_package(Pangolin REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}/ThirdParty/ORBSLAM2
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

FILE(GLOB SRC_ORBSLAM2_FILES
     "src/*.cc") 

add_library(${PROJECT_NAME} SHARED
${SRC_ORBSLAM2_FILES}
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS} 
${Pangolin_LIBRARIES}
${CERES_LIBRARIES}
DBoW2
g2o
)

