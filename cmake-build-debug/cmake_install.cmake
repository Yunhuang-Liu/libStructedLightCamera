# Install script for directory: /home/lyh/桌面/devolope/libStructedLightCamera

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
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

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libStructedLightCamera" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libStructedLightCamera")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libStructedLightCamera"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/lyh/桌面/devolope/libStructedLightCamera/cmake-build-debug/libStructedLightCamera")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libStructedLightCamera" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libStructedLightCamera")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libStructedLightCamera"
         OLD_RPATH "/usr/local/cuda-11.4/lib64:/usr/local/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libStructedLightCamera")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/device" TYPE FILE FILES "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/cameraControl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/device/camera" TYPE FILE FILES
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/camera/IMVApi.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/camera/IMVDefines.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/camera/cammeraUnilty.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/device/projector" TYPE FILE FILES
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/API.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/CyUSBSerial.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/common.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/cypress_i2c.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/dlpc347x_internal_patterns.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/dlpc34xx.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/dlpc34xx_dual.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/dlpc_common.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/dlpc_common_private.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/hidapi.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/projectorControl.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/device/projector/usb.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/phaseSolver" TYPE FILE FILES
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/phaseSolver/dividedSpaceTimeMulUsedMaster_GPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/phaseSolver/fourFloorFouStepMaster_GPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/phaseSolver/fourStepRefPlainMaster_GPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/phaseSolver/fourStepSixGrayCodeMaster_CPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/phaseSolver/fourStepSixGrayCodeMaster_GPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/phaseSolver/nStepNGrayCodeMaster_CPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/phaseSolver/phaseSolver.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/phaseSolver/shiftGrayCodeUnwrapMaster_GPU.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/wrapCreator" TYPE FILE FILES
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/wrapCreator/wrapCreator.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/wrapCreator/wrapCreator_CPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/wrapCreator/wrapCreator_GPU.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/rectifier" TYPE FILE FILES
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/rectifier/rectifier.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/rectifier/rectifier_CPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/rectifier/rectifier_GPU.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/tool" TYPE FILE FILES
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/tool/matrixsInfo.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/tool/tool.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/restructor" TYPE FILE FILES
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/restructor/Restructor.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/restructor/Restructor_CPU.h"
    "/home/lyh/桌面/devolope/libStructedLightCamera/include/restructor/Restructor_GPU.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libStructedLightCamera/cuda" TYPE FILE FILES "/home/lyh/桌面/devolope/libStructedLightCamera/include/cuda/cudaTypeDef.cuh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/libStructedLightCamera" TYPE FILE FILES "/home/lyh/桌面/devolope/libStructedLightCamera/cmake/StructedLightCameraConfig.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/lyh/桌面/devolope/libStructedLightCamera/cmake/StructedLightCamera.pc")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/lyh/桌面/devolope/libStructedLightCamera/cmake-build-debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
