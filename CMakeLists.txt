cmake_minimum_required(VERSION 3.2)

project(libStructedLightCamera)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INSTALL_PREFIX C:/Users/LiuYunhuang/Desktop/softWare/WindowsVCLib/libStructedLightCamera)

file(GLOB_RECURSE CPUHeader ./StructedLightCamera.h ./include/*CPU.h ./include/CameraControl/*.h
    ./include/MatrixsInfo.h ./include/PhaseSolver.h ./include/Restructor.h)
file(GLOB_RECURSE GPUHeader ./include/*GPU.h ./include/*.cuh)
file(GLOB MAIN ./Example.cpp)
file(GLOB_RECURSE CPUSource ./StructedLightCamera.cpp ./src/*CPU.cpp ./src/*.c
    ./src/CameraControl/*.cpp ./src/MatrixsInfo.cpp ./src/PhaseSolver.cpp ./src/Restructor.cpp)
file(GLOB_RECURSE GPUSource ./src/*.cu ./src/*GPU.cpp)
source_group("CPUHeader" FILES ${CPUHeader})
source_group("GPUHeader" FILES ${GPUHeader})
source_group("CPUSource" FILES ${CPUSource})
source_group("GPUSource" FILES ${GPUSource})
source_group("MAIN" FILES ${MAIN})

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    )

link_directories(
    ./lib)

find_package(CUDA)
if(CUDA_FOUND)
   add_definitions(-DCUDA)
   include_directories(${CUDA_INCLUDE_DIRS})
   #[[
   cuda_add_executable(
       ${PROJECT_NAME}
       ${MAIN}
       ${CPUHeader}
       ${CPUSource}
       ${GPUHeader}
       ${GPUSource}
       )
   target_link_libraries(
       ${PROJECT_NAME}
       ${OpenCV_LIBRARIES}
       ${CUDA_LIBRARIES}
       MVSDKmd
       cyusbserial
       setupapi
       )
    ]]
   
   CUDA_COMPILE(CUDA_OBJS ${GPUHeader} ${GPUSource})
   set(CMAKE_DEBUG_POSTFIX "d")
   add_library(libStructedLightCamera SHARED  ${CPUSource} ${GPUSource} ${CUDA_OBJS} ${CPUHeader} ${GPUHeader})
   set_target_properties(libStructedLightCamera PROPERTIES DEBUG_POSTFIX "d")
   set_target_properties(libStructedLightCamera PROPERTIES CLEAN_DIRECT_OUTPUT 1)
   set_target_properties(libStructedLightCamera PROPERTIES PUBLIC_HEADER ./StructedLightCamera.h)
   target_link_libraries(
       libStructedLightCamera
       ${OpenCV_LIBRARIES}
       ${CUDA_LIBRARIES}
       MVSDKmd
       cyusbserial
       setupapi
       )

   add_library(libStructedLightCameraStatic STATIC  ${CPUSource} ${GPUSource} ${CUDA_OBJS} ${CPUHeader} ${GPUHeader})
   set_target_properties(libStructedLightCameraStatic PROPERTIES DEBUG_POSTFIX "d") 
   # 指定静态库的输出名称
   set_target_properties(libStructedLightCameraStatic PROPERTIES OUTPUT_NAME "libStructedLightCamera")
   # 使动态库和静态库同时存在
   set_target_properties(libStructedLightCameraStatic PROPERTIES CLEAN_DIRECT_OUTPUT 1)
   set_target_properties(libStructedLightCameraStatic PROPERTIES PUBLIC_HEADER ./StructedLightCamera.h)
   target_link_libraries(
       libStructedLightCameraStatic
       ${OpenCV_LIBRARIES}
       ${CUDA_LIBRARIES}
       MVSDKmd
       cyusbserial
       setupapi
       )
else()
    add_executable(
        ${PROJECT_NAME}
        ${CPUHeader}
        ${CPUSource}
        )

   add_library(libStructedLightCamera SHARED  ${CPUSource}  ${CPUHeader})
   add_library(libStructedLightCameraStatic STATIC  ${CPUSource} ${CPUHeader})
   set(CMAKE_DEBUG_POSTFIX "d")
   set_target_properties(libStructedLightCamera PROPERTIES DEBUG_POSTFIX "d") 
   set_target_properties(libStructedLightCameraStatic PROPERTIES DEBUG_POSTFIX "d") 
   # 指定静态库的输出名称
   set_target_properties(libStructedLightCameraStatic PROPERTIES OUTPUT_NAME "libStructedLightCamera")
   # 使动态库和静态库同时存在
   set_target_properties(libStructedLightCamera PROPERTIES CLEAN_DIRECT_OUTPUT 1)
   set_target_properties(libStructedLightCameraStatic PROPERTIES CLEAN_DIRECT_OUTPUT 1)
   set_target_properties(libStructedLightCamera PROPERTIES PUBLIC_HEADER ./StructedLightCamera.h)
   set_target_properties(libStructedLightCameraStatic PROPERTIES PUBLIC_HEADER ./StructedLightCamera.h)
   target_link_libraries(
        libStructedLightCamera
        ${OpenCV_LIBRARIES}
        MVSDKmd
        cyusbserial
        setupapi
        )
   target_link_libraries(
        libStructedLightCameraStatic
        ${OpenCV_LIBRARIES}
        MVSDKmd
        cyusbserial
        setupapi
        )
endif()

install(
    TARGETS 
    ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
    )


#[[Install include file]]
file(GLOB CameraControl ./include/CameraControl/*.h)
file(GLOB CameraControl_CameraSDK ./include/CameraControl/CameraSDK/*.h)
file(GLOB CameraControl_CameraUtility ./include/CameraControl/CameraUtility/*.h)
file(GLOB CameraControl_ProjectorSDK ./include/CameraControl/ProjectorSDK/*.h)
file(GLOB CameraControl_Restructor ./include/Restructor/*.h)
file(GLOB CameraControl_RestructorCUDA ./include/Restructor/cuda/*.cuh)
source_group("CameraControl_CameraSDK" FILES ${CameraControl_CameraSDK})
source_group("CameraControl_CameraUtility" FILES ${CameraControl_CameraUtility})
source_group("CameraControl_ProjectorSDK" FILES ${CameraControl_ProjectorSDK})
source_group("CameraControl_Restructor" FILES ${CameraControl_Restructor})
source_group("CameraControl_RestructorCUDA" FILES ${CameraControl_RestructorCUDA})

install(
    FILES
    ./lib/MVSDKmd.lib
    DESTINATION
    lib
)

install(
    FILES
    ./lib/cyusbserial.lib
    DESTINATION
    lib
)

install(
    FILES
    ./bin/cyusbserial.dll
    DESTINATION
    bin
)

install(
    FILES
    ${CameraControl}
    DESTINATION
    include/CameraControl
    )
install(
    FILES
    ${CameraControl_CameraSDK}
    DESTINATION
    include/CameraControl/CameraSDK
    )
install(
    FILES
    ${CameraControl_CameraUtility}
    DESTINATION
    include/CameraControl/CameraUtility)
install(
    FILES
    ${CameraControl_ProjectorSDK}
    DESTINATION
    include/CameraControl/ProjectorSDK)
install(
    FILES
    ${CameraControl_Restructor}
    DESTINATION
    include/Restructor)
install(
    FILES
    ${CameraControl_RestructorCUDA}
    DESTINATION
    include/Restructor/cuda/include)

install(
    FILES
    ./StructedLightCameraConfig.cmake
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}
)
