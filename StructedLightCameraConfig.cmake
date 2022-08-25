##################################
#   Find StructedLightCamera
##################################
#   This sets the following variables:
# StructedLightCamera_FOUND             -True if StructedLight Was found
# StructedLightCamera_INCLUDE_DIR    -Directories containing the StructedLightCamera include files
# StructedLightCamera_LIBRARY            -Libraries needed to use StructedLightCamera

find_path(
    StructedLightCamera_INCLUDE_DIR
    StructedLightCamera.h
    ${StructedLightCamera_DIR}/include
)

find_library(
    StructedLightCamera_LIBRARY_Debug
    libStructedLightCamerad.lib
    ${StructedLightCamera_DIR}/lib
)

find_library(
    StructedLightCamera_LIBRARY_Release
    libStructedLightCamera.lib
    ${StructedLightCamera_DIR}/lib
)

find_library(
    DAHUASDK_DIR
    MVSDKmd.lib
    ${StructedLightCamera_DIR}/lib
)

find_library(
    TISDK_DIR
    cyusbserial.lib
    ${StructedLightCamera_DIR}/lib
)

set(StructedLightCamera_INCLUDE_DIRS ${StructedLightCamera_INCLUDE_DIR})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(StructedLightCamera_LIBRARIES ${StructedLightCamera_LIBRARY_Debug} ${DAHUASDK_DIR} ${TISDK_DIR} setupapi)
else()
    set(StructedLightCamera_LIBRARIES ${StructedLightCamera_LIBRARY_Release} ${DAHUASDK_DIR} ${TISDK_DIR} setupapi)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    StructedLightCamera
    DEFAULT_MSG
    StructedLightCamera_INCLUDE_DIRS 
    StructedLightCamera_LIBRARIES
    DAHUASDK_DIR 
    TISDK_DIR
)

mark_as_advanced(
    StructedLightCamera_LIBRARIES
    StructedLightCamera_INCLUDE_DIR
)
