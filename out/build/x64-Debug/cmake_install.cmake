# Install script for directory: C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/LiuYunhuang/Desktop/softWare/WindowsVCLib/libStructedLightCamera")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/out/build/x64-Debug/libStructedLightCamerad.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/out/build/x64-Debug/libStructedLightCamerad.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./StructedLightCamera.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./lib/MVSDKmd.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./lib/cyusbserial.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./bin/cyusbserial.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CameraControl" TYPE FILE FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/CameraControl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CameraControl/CameraSDK" TYPE FILE FILES
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/CameraSDK/IMVApi.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/CameraSDK/IMVDefines.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CameraControl/CameraUtility" TYPE FILE FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/CameraUtility/CammeraUnilty.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CameraControl/ProjectorSDK" TYPE FILE FILES
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/ProjectorSDK/CyUSBSerial.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/ProjectorSDK/ProjectorControl.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/ProjectorSDK/cypress_i2c.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/ProjectorSDK/dlpc347x_internal_patterns.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/ProjectorSDK/dlpc34xx.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/ProjectorSDK/dlpc_common.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/CameraControl/ProjectorSDK/dlpc_common_private.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Restructor" TYPE FILE FILES
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/DividedSpaceTimeMulUsedMaster_GPU.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/FourStepSixGrayCodeMaster_CPU.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/FourStepSixGrayCodeMaster_GPU.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/MatrixsInfo.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/PhaseSolver.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/Restructor.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/Restructor_CPU.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/Restructor_GPU.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/ShiftGrayCodeUnwrapMaster_GPU.h"
    "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./include/Restructor/ThreeStepFiveGrayCodeMaster_CPU.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "C:/Users/LiuYunhuang/Desktop/softWare/WindowsVCLib/libStructedLightCamera/StructedLightCameraConfig.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "C:/Users/LiuYunhuang/Desktop/softWare/WindowsVCLib/libStructedLightCamera" TYPE FILE FILES "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/./StructedLightCameraConfig.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/LiuYunhuang/Desktop/WindowsCode/libStructedLightCamera/out/build/x64-Debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
