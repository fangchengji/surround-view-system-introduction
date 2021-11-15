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
 
# test program
add_executable(test src/test.cpp)
 
set_property(TARGET test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test PRIVATE SurroundViewStitching ${OpenCV_LIBS})
