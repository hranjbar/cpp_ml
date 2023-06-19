cmake_minimum_required(VERSION 3.20)

project(ONNX_Runtime_Depth_Estimation_EXAMPLE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
message(STATUS "cmake source path: ${CMAKE_SOURCE_DIR}")

set(ONNXRUNTIME_ROOT_PATH ~)
set(ONNXRUNTIME_INCLUDE_DIRS
  ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/include/onnxruntime
  ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
  ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/include/onnxruntime/core/session
)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/build/Linux/Release/libonnxruntime.so)

find_package(OpenCV 4 REQUIRED)
message(STATUS "opencv version: ${OpenCV_VERSION}")

add_executable(image_classifier ${CMAKE_SOURCE_DIR}/main.cpp 
	${CMAKE_SOURCE_DIR}/image_classifier.cpp)
target_include_directories(image_classifier PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
#message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
target_link_libraries(image_classifier PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBRARIES})
