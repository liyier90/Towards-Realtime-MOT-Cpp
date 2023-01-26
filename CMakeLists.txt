cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(jde)

find_package(Eigen3 REQUIRED NO_MODULE
    PATHS ./third_party/eigen
    NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED
    PATHS ./third_party/opencv4
    NO_DEFAULT_PATH)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCES
    src/main.cpp
    src/JDETracker.cpp
    src/kalmanFilter.cpp
    src/lapjv.cpp
    src/STrack.cpp
    src/utils.cpp
)

add_executable(jde ${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(jde
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(jde
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
set_property(TARGET jde PROPERTY CXX_STANDARD 14)

