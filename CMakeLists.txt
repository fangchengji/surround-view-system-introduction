cmake_minimum_required(VERSION 3.18)

project(SurroundViewStitching C CXX CUDA)

set (CMAKE_CXX_FLAGS "-std=c++14 -fPIC")
set (CMAKE_BUILD_TYPE Debug)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Opencv
find_package(OpenCV REQUIRED)
message(STATUS "OPENCV_INCLUDE_DIRS" ${OppenCV_INCLUDE_DIRS})

# cuda
find_package(CUDA REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
message (STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/)

include_directories(${PROJECT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
file(GLOB HEADERS src/*.h src/*.hpp)
file(GLOB SOURCES src/*.cpp src/*.hpp src/*.cu)
# cuda_add_library(GPU_PREPROCESS STATIC ${HEADERS} ${SOURCES})
add_library(SurroundViewStitching STATIC ${HEADERS} ${SOURCES})

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(SurroundViewStitching
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
# test stitch
add_executable(test_stitch src/test_stitch.cpp)
 
set_property(TARGET test_stitch 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_stitch PRIVATE SurroundViewStitching ${OpenCV_LIBS})

# test undist
add_executable(test_undist src/test_undist.cpp)
 
set_property(TARGET test_undist 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_undist PRIVATE SurroundViewStitching ${OpenCV_LIBS})
