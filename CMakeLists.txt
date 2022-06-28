cmake_minimum_required(VERSION 3.19)
project(TRT)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include")
set(OpenCV_DIR "C:/repos/opencv/build")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
# find_package(MLPACK REQUIRED)

set(SOURCE_FILES main.cpp
        buffers.cpp buffers.h
        InferenceEngine.cpp InferenceEngine.h
        yolo.hpp
        utils.cpp utils.h
        macros.h cuda_utils.h
        Testing/VideoStream.cpp Testing/VideoStream.h)

add_executable(TRT ${SOURCE_FILES})

# if(MLPACK_FOUND)
#     message("-MLPACK found")
#     target_include_directories(${PROJECT_NAME} PUBLIC ${MLPACK_INCLUDE_DIRS})
#     target_link_libraries(${PROJECT_NAME} PUBLIC ${MLPACK_LIBS})
# endif()

find_package(OpenCV REQUIRED core imgproc imgcodecs highgui cudawarping cudaarithm cudaimgproc)

if (OpenCV_FOUND)
    message("- OpenCV   FOUND = OK")
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
endif ()

find_package(CUDA 11 REQUIRED)
if (CUDA_FOUND)
    message("- CUDA     FOUND = OK")
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})
endif()
# include_directories(${TRT_DIR}\\include)
# message(${TRT_DIR}\\include)
# set(TRT_INCLUDE "${TRT_DIR}/include")
# include_directories(${TRT_INCLUDE})
# set(TRT_LIB "${TRT_DIR}/lib")
# # set(GLOB_TRT_LIB "${TRT_LIB}\\*.lib")
# # message(${GLOB_TRT_LIB})
# file(GLOB LIST_TRT_LIB "${TRT_LIB}/*.lib")
# foreach(search ${LIST_TRT_LIB})
#      message(${search})
# endforeach()
# link_directories(${TRT_DIR})
# target_link_libraries(${PROJECT_NAME} PUBLIC ${LIST_TRT_LIB} ${CMAKE_THREAD_LIBS_INIT})
find_package(TensorRT REQUIRED)
if (TensorRT_FOUND)
    message("- TensorRT FOUND = OK")
    target_include_directories(${PROJECT_NAME} PUBLIC ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${TensorRT_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()



