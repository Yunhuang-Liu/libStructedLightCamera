##################################
#   Find StructedLightCamera
##################################
#   This sets the following variables:
# StructedLightCamera_FOUND             -True if StructedLight Was found
# StructedLightCamera_INCLUDE_DIRS    -Directories containing the StructedLightCamera include files
# StructedLightCamera_LIBRARY            -Libraries needed to use StructedLightCamera

#[[
find_path(
    StructedLightCamera_INCLUDE_DIR
    structedLightCamera.h
    usr/local/include/libStructedLightCamera
)]]#

set(StructedLightCamera_INCLUDE_DIR usr/local/include/libStructedLightCamera)

find_library(
    StructedLightCamera_LIBRARY_Debug
    liblibStructedLightCamera.so
    usr/local/lib
)

find_library(
    StructedLightCamera_LIBRARY_Release
    liblibStructedLightCamera.so
    usr/local/lib
)

find_library(
    DAHUASDK_DIR
    libMVSDK.so
    usr/local/lib
)

find_library(
    TISDK_DIR
    libcyusbserial.so
    usr/local/lib
)

set(StructedLightCamera_INCLUDE_DIRS ${StructedLightCamera_INCLUDE_DIR})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(StructedLightCamera_LIBRARIES ${StructedLightCamera_LIBRARY_Debug} ${DAHUASDK_DIR} ${TISDK_DIR})
else()
    set(StructedLightCamera_LIBRARIES ${StructedLightCamera_LIBRARY_Release} ${DAHUASDK_DIR} ${TISDK_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    StructedLightCamera
    DEFAULT_MSG
    StructedLightCamera_INCLUDE_DIR
    StructedLightCamera_LIBRARIES
    DAHUASDK_DIR 
    TISDK_DIR
)

mark_as_advanced(
    StructedLightCamera_LIBRARIES
    StructedLightCamera_INCLUDE_DIRS
)
