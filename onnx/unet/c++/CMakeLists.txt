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

add_executable(unet_inference ${CMAKE_SOURCE_DIR}/main.cpp 
	${CMAKE_SOURCE_DIR}/Unet.cpp)
target_include_directories(unet_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
#message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
target_link_libraries(unet_inference PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBRARIES})
