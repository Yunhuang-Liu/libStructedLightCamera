#include <device/camera/cammeraUnilty.h>

namespace sl {
    namespace device {
        //灰度相机取流回调函数
        static void grayFrameCallback(IMV_Frame *pFrame, void *pUser) {
            CammeraUnilty *pCammerWidget = (CammeraUnilty *) pUser;
            CFrameInfo frameInfo;
            frameInfo.m_nWidth = (int) pFrame->frameInfo.width;
            frameInfo.m_nHeight = (int) pFrame->frameInfo.height;
            frameInfo.m_nBufferSize = (int) pFrame->frameInfo.size;
            frameInfo.m_nPaddingX = (int) pFrame->frameInfo.paddingX;
            frameInfo.m_nPaddingY = (int) pFrame->frameInfo.paddingY;
            frameInfo.m_ePixelType = pFrame->frameInfo.pixelFormat;
            frameInfo.m_pImageBuf = (unsigned char *) malloc(sizeof(unsigned char) *
                                                             frameInfo.m_nBufferSize);
            frameInfo.m_nTimeStamp = pFrame->frameInfo.timeStamp;
            if (pFrame->pData != NULL) {
                memcpy(frameInfo.m_pImageBuf, pFrame->pData, frameInfo.m_nBufferSize);
                pCammerWidget->imgs[pCammerWidget->index] = cv::Mat(frameInfo.m_nHeight, frameInfo.m_nWidth, CV_8U,
                                                                    (uint8_t *) frameInfo.m_pImageBuf);
                ++pCammerWidget->index;
            }
        }

        //彩色相机取流回调函数
        static void colorFrameCallback(IMV_Frame *pFrame, void *pUser) {
            CammeraUnilty *pCammerWidget = (CammeraUnilty *) pUser;
            CFrameInfo frameInfo;
            frameInfo.m_nWidth = (int) pFrame->frameInfo.width;
            frameInfo.m_nHeight = (int) pFrame->frameInfo.height;
            frameInfo.m_nBufferSize = (int) pFrame->frameInfo.size;
            frameInfo.m_nPaddingX = (int) pFrame->frameInfo.paddingX;
            frameInfo.m_nPaddingY = (int) pFrame->frameInfo.paddingY;
            frameInfo.m_ePixelType = pFrame->frameInfo.pixelFormat;
            frameInfo.m_pImageBuf = (unsigned char *) malloc(sizeof(unsigned char) *
                                                             frameInfo.m_nBufferSize);
            frameInfo.m_nTimeStamp = pFrame->frameInfo.timeStamp;
            // 内存申请失败，直接返回
            if (frameInfo.m_pImageBuf != NULL) {
                memcpy(frameInfo.m_pImageBuf, pFrame->pData, frameInfo.m_nBufferSize);
                pCammerWidget->imgs[pCammerWidget->index].create(frameInfo.m_nHeight,
                                                                 frameInfo.m_nWidth, CV_8UC3);
                IMV_PixelConvertParam stPixelConvertParam;
                stPixelConvertParam.nWidth = frameInfo.m_nWidth;
                stPixelConvertParam.nHeight = frameInfo.m_nHeight;
                stPixelConvertParam.ePixelFormat = frameInfo.m_ePixelType;
                stPixelConvertParam.pSrcData = frameInfo.m_pImageBuf;
                stPixelConvertParam.nSrcDataLen = frameInfo.m_nBufferSize;
                stPixelConvertParam.nPaddingX = frameInfo.m_nPaddingX;
                stPixelConvertParam.nPaddingY = frameInfo.m_nPaddingY;
                stPixelConvertParam.eBayerDemosaic = demosaicNearestNeighbor;
                stPixelConvertParam.eDstPixelFormat = gvspPixelBGR8;
                stPixelConvertParam.pDstBuf = pCammerWidget->imgs[pCammerWidget->index].data;
                stPixelConvertParam.nDstBufSize = frameInfo.m_nBufferSize * 3;
                IMV_PixelConvert(pCammerWidget->m_devHandle, &stPixelConvertParam);
                ++pCammerWidget->index;
            }
            free((void *) frameInfo.m_pImageBuf);
        }

        CammeraUnilty::CammeraUnilty() : m_currentCameraKey(""), m_devHandle(NULL), index(0), cameraType(LeftCamera) {
            this->imgs.resize(16);
        }

        CammeraUnilty::~CammeraUnilty() {
        }

        bool CammeraUnilty::SetExposeTime(double dExposureTime) {
            if (!m_devHandle) {
                return false;
            }

            int ret = IMV_OK;

            ret = IMV_SetDoubleFeatureValue(m_devHandle, "ExposureTime", dExposureTime);
            if (IMV_OK != ret) {
                printf("set ExposureTime value = %0.2f fail, ErrorCode[%d]\n", dExposureTime, ret);
                return false;
            }

            return true;
        }

        bool CammeraUnilty::SetAdjustPlus(double dGainRaw) {
            if (!m_devHandle) {
                return false;
            }

            int ret = IMV_OK;

            ret = IMV_SetDoubleFeatureValue(m_devHandle, "GainRaw", dGainRaw);
            if (IMV_OK != ret) {
                printf("set GainRaw value = %0.2f fail, ErrorCode[%d]\n", dGainRaw, ret);
                return false;
            }

            return true;
        }

        bool CammeraUnilty::CameraOpen(void) {
            int ret = IMV_OK;

            if (m_devHandle) {
                printf("m_devHandle is already been create!\n");
                return false;
            }
            const char *cameraKey = m_currentCameraKey.data();

            ret = IMV_CreateHandle(&m_devHandle, modeByCameraKey, (void *) cameraKey);

            if (IMV_OK != ret) {
                printf("create devHandle failed! cameraKey[%s], ErrorCode[%d]\n", cameraKey, ret);
                return false;
            }

            ret = IMV_Open(m_devHandle);
            if (IMV_OK != ret) {
                printf("open camera failed! ErrorCode[%d]\n", ret);
                return false;
            }
            IMV_ClearFrameBuffer(m_devHandle);
            IMV_SetBufferCount(m_devHandle, 4);
            IMV_SetIntFeatureValue(m_devHandle, "AcquisitionFrameRate", 2000);
            return true;
        }

        bool CammeraUnilty::CameraClose(void) {
            if (!m_devHandle) {
                return false;
            }

            int ret = IMV_OK;

            if (!m_devHandle) {
                printf("close camera fail. No camera.\n");
                return false;
            }

            if (false == IMV_IsOpen(m_devHandle)) {
                printf("camera is already close.\n");
                return false;
            }

            ret = IMV_Close(m_devHandle);
            if (IMV_OK != ret) {
                printf("close camera failed! ErrorCode[%d]\n", ret);
                return false;
            }

            ret = IMV_DestroyHandle(m_devHandle);
            if (IMV_OK != ret) {
                printf("destroy devHandle failed! ErrorCode[%d]\n", ret);
                return false;
            }

            m_devHandle = NULL;

            return true;
        }

        bool CammeraUnilty::CameraStart() {
            if (!m_devHandle) {
                return false;
            }

            int ret = IMV_OK;

            if (IMV_IsGrabbing(m_devHandle)) {
                printf("camera is already grebbing.\n");
                return false;
            }
            if (this->cameraType == CammeraUnilty::LeftCamera) {
                ret = IMV_AttachGrabbing(m_devHandle, grayFrameCallback, this);
            } else if (this->cameraType == CammeraUnilty::RightCamera) {
                ret = IMV_AttachGrabbing(m_devHandle, grayFrameCallback, this);
            } else {
                ret = IMV_AttachGrabbing(m_devHandle, colorFrameCallback, this);
            }

            if (IMV_OK != ret) {
                printf("Attach grabbing failed! ErrorCode[%d]\n", ret);
                return false;
            }

            ret = IMV_StartGrabbing(m_devHandle);
            if (IMV_OK != ret) {
                printf("start grabbing failed! ErrorCode[%d]\n", ret);
                return false;
            }

            return true;
        }

        bool CammeraUnilty::CameraStop() {
            if (!m_devHandle) {
                return false;
            }

            int ret = IMV_OK;
            if (!IMV_IsGrabbing(m_devHandle)) {
                printf("camera is already stop grubbing.\n");
                return false;
            }

            ret = IMV_StopGrabbing(m_devHandle);
            if (IMV_OK != ret) {
                printf("Stop grubbing failed! ErrorCode[%d]\n", ret);
                return false;
            }
            return true;
        }

        bool CammeraUnilty::CameraChangeTrig(ETrigType trigType) {
            if (!m_devHandle) {
                return false;
            }

            int ret = IMV_OK;

            if (trigContinous == trigType) {
                ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerMode", "Off");
                if (IMV_OK != ret) {
                    printf("set TriggerMode value = Off fail, ErrorCode[%d]\n", ret);
                    return false;
                }
            } else if (trigSoftware == trigType) {
                ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerMode", "On");
                if (IMV_OK != ret) {
                    printf("set TriggerMode value = On fail, ErrorCode[%d]\n", ret);
                    return false;
                }

                ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerSource", "Software");
                if (IMV_OK != ret) {
                    printf("set TriggerSource value = Software fail, ErrorCode[%d]\n", ret);
                    return false;
                }
            } else if (trigLine == trigType) {
                ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerMode", "On");
                if (IMV_OK != ret) {
                    printf("set TriggerMode value = On fail, ErrorCode[%d]\n", ret);
                    return false;
                }

                ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerSource", "Line2");
                if (IMV_OK != ret) {
                    printf("set TriggerSource value = Line1 fail, ErrorCode[%d]\n", ret);
                    return false;
                }
            }
            return true;
        }

        bool CammeraUnilty::ExecuteSoftTrig(void) {
            if (!m_devHandle) {
                return false;
            }

            int ret = IMV_OK;

            ret = IMV_ExecuteCommandFeature(m_devHandle, "TriggerSoftware");
            if (IMV_OK != ret) {
                printf("ExecuteSoftTrig fail, ErrorCode[%d]\n", ret);
                return false;
            }

            printf("ExecuteSoftTrig success.\n");
            return true;
        }

        void CammeraUnilty::SetCamera(const std::string &strKey) {
            m_currentCameraKey = strKey;
        }

        void CammeraUnilty::setROI(const int width, const int height, const int offsetX, const int offsetY) {
            if (m_devHandle == nullptr) {
                std::cout << "Camera handle is invaliable!" << std::endl;
                return;
            }
            IMV_SetIntFeatureValue(m_devHandle, "Width", width);
            IMV_SetIntFeatureValue(m_devHandle, "Height", height);
            IMV_SetIntFeatureValue(m_devHandle, "OffsetX", offsetX);
            IMV_SetIntFeatureValue(m_devHandle, "OffsetY", offsetY);
        }

        void CammeraUnilty::clearBuffer() {
            if (m_devHandle == nullptr) {
                std::cout << "Camera handle is invaliable!" << std::endl;
                return;
            }
            IMV_ClearFrameBuffer(m_devHandle);
        }

        void CammeraUnilty::setWhiteBlanceRGB(float r, float g, float b) {
            if (m_devHandle == nullptr) {
                std::cout << "Camera handle is invaliable!" << std::endl;
                return;
            }
            IMV_SetEnumFeatureSymbol(m_devHandle, "BalanceRatioSelector", "Red");
            IMV_SetDoubleFeatureValue(m_devHandle, "BalanceRatio", r);
            IMV_SetEnumFeatureSymbol(m_devHandle, "BalanceRatioSelector", "Green");
            IMV_SetDoubleFeatureValue(m_devHandle, "BalanceRatio", g);
            IMV_SetEnumFeatureSymbol(m_devHandle, "BalanceRatioSelector", "Blue");
            IMV_SetDoubleFeatureValue(m_devHandle, "BalanceRatio", b);
        }

        void CammeraUnilty::setAutoExposure(const bool isAutoExp) {
            if (m_devHandle == nullptr) {
                std::cout << "Camera handle is invaliable!" << std::endl;
                return;
            }
            if (isAutoExp) {
                IMV_SetEnumFeatureSymbol(m_devHandle, "ExposureAuto", "Continuous");
            } else {
                IMV_SetEnumFeatureSymbol(m_devHandle, "ExposureAuto", "Off");
            }
        }

        void CammeraUnilty::setBrightness(const int brightness) {
            if (m_devHandle == nullptr) {
                std::cout << "Camera handle is invaliable!" << std::endl;
                return;
            }
            IMV_SetIntFeatureValue(m_devHandle, "Brightness", brightness);
        }

        void CammeraUnilty::setAutoWhiteBlance(const bool isAutoBlance) {
            if (m_devHandle == nullptr) {
                std::cout << "Camera handle is invaliable!" << std::endl;
                return;
            }
            if (isAutoBlance) {
                IMV_SetEnumFeatureSymbol(m_devHandle, "Balance White Auto", "Continuous");
            } else {
                IMV_SetEnumFeatureSymbol(m_devHandle, "Balance White Auto", "Off");
            }
        }

        void CammeraUnilty::setPixelFormat(const std::string pixelFormat) {
            IMV_SetEnumFeatureSymbol(m_devHandle, "PixelFormat", pixelFormat.data());
        }

        bool CammeraUnilty::isTriggerLine() {
            uint64_t triggerType;
            IMV_GetEnumFeatureValue(m_devHandle, "TriggerSource", &triggerType);
            if (2 == triggerType) {
                return true;
            } else {
                return false;
            }
        }
    }// namespace device
}// namespace sl