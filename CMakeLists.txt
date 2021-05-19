cmake_minimum_required(VERSION 2.8)
project(myorbslam)
set(CMAKE_CXX_FLAGS "-std=c++11   ")
set(CMAKE_BUILD_TYPES "Debug")

find_package(OpenCV 3 REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${Pangolin_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED 
src/System.cc 
src/Tracking.cc 
src/FrameDrawer.cc
src/Map.cc
src/Frame.cc
src/MapDrawer.cc
src/KeyFrameDatabase.cc
src/ORBextractor.cc
src/Viewer.cc
src/MapPoint.cc
src/KeyFrame.cc
src/ORBmatcher.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${Pangolin_LIBRARIES}
)


#add_executable(orb_extractor orb_extractor.cpp)
#target_link_libraries(orb_extractor ${OpenCV_LIBS})

add_executable(myorbslam2 myorbslam2.cpp)
target_link_libraries(myorbslam2 ${PROJECT_NAME})

add_executable(myorbslam2backup myorbslam2backup.cpp)
target_link_libraries(myorbslam2backup ${OpenCV_LIBS})