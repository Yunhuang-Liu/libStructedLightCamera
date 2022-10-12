cmake_minimum_required(VERSION 3.2)

project(libStructedLightCamera)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INSTALL_PREFIX C:/Users/LiuYunhuang/Desktop/softWare/WindowsVCLib/libStructedLightCamera)

file(GLOB_RECURSE CPUHeader ./include/structedLightCamera.h ./include/*CPU.h ./include/device/*.h
    ./include/wrapCreator.h ./include/matrixsInfo.h ./include/phaseSolver.h ./include/restructor.h
    ./include/rectifier.h)
file(GLOB_RECURSE GPUHeader ./include/*GPU.h ./include/*.cuh)
file(GLOB_RECURSE CPUSource ./src/structedLightCamera.cc ./src/*CPU.cc ./src/*.c
    ./src/device/*.cc ./src/matrixsInfo.cc ./src/phaseSolver.cc ./src/restructor.cc)
file(GLOB_RECURSE GPUSource ./src/*.cu ./src/*GPU.cc)
source_group("CPUHeader" FILES ${CPUHeader})
source_group("GPUHeader" FILES ${GPUHeader})
source_group("CPUSource" FILES ${CPUSource})
source_group("GPUSource" FILES ${GPUSource})
source_group("MAIN" FILES ${MAIN})

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
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
   set_target_properties(libStructedLightCamera PROPERTIES PUBLIC_HEADER ./include/structedLightCamera.h)
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
   set_target_properties(libStructedLightCameraStatic PROPERTIES PUBLIC_HEADER ./include/structedLightCamera.h)
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
   set_target_properties(libStructedLightCamera PROPERTIES PUBLIC_HEADER ./include/structedLightCamera.h)
   set_target_properties(libStructedLightCameraStatic PROPERTIES PUBLIC_HEADER ./include/structedLightCamera.h)
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

file(GLOB DEVICE ./include/device/*.h)
file(GLOB DEVICE_CAMERA ./include/device/camera/*.h)
file(GLOB DEVICE_PROJECTOR ./include/device/projector/*.h)
file(GLOB PHASESOLVER ./include/phaseSolver/*.h)
file(GLOB WRAPCREATOR ./include/wrapCreator/*.h)
file(GLOB RECTIFIER ./include/rectifier/*.h)
file(GLOB TOOL ./include/tool/*.h)
file(GLOB RESTRUCTOR ./include/restructor/*.h)
file(GLOB RESTRUCTOR ./include/restructor/*.h)
file(GLOB CUDAHEADER ./include/cuda/*.cuh)
source_group("Device" FILES ${DEVICE})
source_group("Device_Camera" FILES ${DEVICE_CAMERA})
source_group("Device_Projector" FILES ${DEVICE_PROJECTOR})
source_group("PhaseSolver" FILES ${PHASESOLVER})
source_group("WrapCerator" FILES ${WRAPCREATOR})
source_group("Rectifier" FILES ${RECTIFIER})
source_group("Tool" FILES ${TOOL})
source_group("Restructor" FILES ${RESTRUCTOR})
source_group("CudaHeader" FILES ${CUDAHEADER})

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
    ${DEVICE}
    DESTINATION
    include/device
    )
install(
    FILES
    ${DEVICE_CAMERA}
    DESTINATION
    include/device/camera
    )
install(
    FILES
    ${DEVICE_PROJECTOR}
    DESTINATION
    include/device/projector)
install(
    FILES
    ${PHASESOLVER}
    DESTINATION
    include/phaseSolver)
install(
    FILES
    ${WRAPCREATOR}
    DESTINATION
    include/wrapCreator)
install(
    FILES
    ${RECTIFIER}
    DESTINATION
    include/rectifier)
install(
    FILES
    ${TOOL}
    DESTINATION
    include/tool)
install(
    FILES
    ${RESTRUCTOR}
    DESTINATION
    include/restructor)
install(
    FILES
    ${CUDAHEADER}
    DESTINATION
    include/cuda)

install(
    FILES
    ./cmake/StructedLightCameraConfig.cmake
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}
)