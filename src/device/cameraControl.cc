#include <device/CameraControl.h>

namespace sl {
    namespace device {
        CameraControl::CameraControl(const DLPC34XX_ControllerDeviceId_e projectorModuleType,
                                     CameraUsedState state_) : cameraLeft(nullptr),
                                                               cameraRight(nullptr), cameraColor(nullptr),
                                                               projector(nullptr), cameraUsedState(state_) {
            projector.reset(new ProjectorControl(projectorModuleType));
            int state = IMV_OK;
            IMV_DeviceList devices;
            state = IMV_EnumDevices(&devices, interfaceTypeAll);
            if (state) {
                std::cout << "Search camera failed!" << std::endl;
            }
            if (nullptr == cameraLeft) {
                cameraLeft.reset(new CammeraUnilty());
                bool isSuccessSearch = false;
                for (int i = 0; i < devices.nDevNum; i++) {
                    if (std::string(devices.pDevInfo[i].cameraName) == "Left") {
                        cameraLeft->SetCamera(devices.pDevInfo[i].cameraKey);
                        if (cameraUsedState == CameraUsedState::LeftColorRightGray)
                            cameraLeft->cameraType = CammeraUnilty::ColorCamera;
                        else
                            cameraLeft->cameraType = CammeraUnilty::LeftCamera;
                        if (!cameraLeft->CameraOpen()) {
                            std::cout << "Open left camera failed" << std::endl;
                            return;
                        }
                        cameraLeft->SetExposeTime(3500);
                        cameraLeft->CameraChangeTrig(cameraLeft->trigLine);
                        cameraLeft->CameraStart();
                        isSuccessSearch = true;
                    }
                }
                if (!isSuccessSearch) {
                    std::cout << "Search camera failed!There is dosen't has a camera which is named Left!" << std::endl;
                }
            }
            if (nullptr == cameraRight) {
                cameraRight.reset(new CammeraUnilty());
                bool isSuccessSearch = false;
                for (int i = 0; i < devices.nDevNum; i++) {
                    if (std::string(devices.pDevInfo[i].cameraName) == "Right") {
                        cameraRight->SetCamera(devices.pDevInfo[i].cameraKey);
                        cameraRight->cameraType = CammeraUnilty::RightCamera;
                        if (!cameraRight->CameraOpen()) {
                            std::cout << "Open right camera failed" << std::endl;
                            return;
                        }
                        cameraRight->SetExposeTime(3500);
                        cameraRight->CameraChangeTrig(cameraRight->trigLine);
                        cameraRight->CameraStart();
                        isSuccessSearch = true;
                    }
                }
                if (!isSuccessSearch) {
                    std::cout << "Search camera failed!There is dosen't has a camera which is named Right!" << std::endl;
                }
            }
            if (cameraUsedState == CameraUsedState::LeftGrayRightGrayExColor) {
                if (nullptr == cameraColor) {
                    cameraColor.reset(new CammeraUnilty());
                    bool isSuccessSearch = false;
                    for (int i = 0; i < devices.nDevNum; i++) {
                        if (std::string(devices.pDevInfo[i].cameraName) == "Color") {
                            cameraColor->SetCamera(devices.pDevInfo[i].cameraKey);
                            cameraColor->cameraType = CammeraUnilty::ColorCamera;
                            if (!cameraColor->CameraOpen()) {
                                std::cout << "Open color camera failed" << std::endl;
                                return;
                            }
                            cameraColor->setPixelFormat("BayerRG8");
                            cameraColor->SetExposeTime(20400);
                            cameraColor->CameraChangeTrig(cameraColor->trigLine);
                            cameraColor->CameraStart();
                            isSuccessSearch = true;
                        }
                    }
                    if (!isSuccessSearch) {
                        std::cout << "Search camera failed!There is dosen't has a camera which is named Color!" << std::endl;
                    }
                }
            }
        }

        CameraControl::CameraControl(const int numLutEntries, CameraUsedState state_) : cameraLeft(nullptr), cameraRight(nullptr), cameraColor(nullptr),
                                                                                        projector(nullptr), cameraUsedState(state_) {
            projector.reset(new ProjectorControl(numLutEntries));
            int state = IMV_OK;
            IMV_DeviceList devices;
            state = IMV_EnumDevices(&devices, interfaceTypeAll);
            if (state) {
                std::cout << "Search camera failed!" << std::endl;
            }
            if (nullptr == cameraLeft) {
                cameraLeft.reset(new CammeraUnilty());
                bool isSuccessSearch = false;
                for (int i = 0; i < devices.nDevNum; i++) {
                    if (std::string(devices.pDevInfo[i].cameraName) == "Left") {
                        cameraLeft->SetCamera(devices.pDevInfo[i].cameraKey);
                        if (cameraUsedState == CameraUsedState::LeftColorRightGray)
                            cameraLeft->cameraType = CammeraUnilty::ColorCamera;
                        else
                            cameraLeft->cameraType = CammeraUnilty::LeftCamera;
                        if (!cameraLeft->CameraOpen()) {
                            std::cout << "Open left camera failed" << std::endl;
                            return;
                        }
                        cameraLeft->SetExposeTime(3500);
                        cameraLeft->CameraChangeTrig(cameraLeft->trigLine);
                        cameraLeft->CameraStart();
                        isSuccessSearch = true;
                    }
                }
                if (!isSuccessSearch) {
                    std::cout << "Search camera failed!There is dosen't has a camera which is named Left!" << std::endl;
                }
            }
            if (nullptr == cameraRight) {
                cameraRight.reset(new CammeraUnilty());
                bool isSuccessSearch = false;
                for (int i = 0; i < devices.nDevNum; i++) {
                    if (std::string(devices.pDevInfo[i].cameraName) == "Right") {
                        cameraRight->SetCamera(devices.pDevInfo[i].cameraKey);
                        cameraRight->cameraType = CammeraUnilty::RightCamera;
                        if (!cameraRight->CameraOpen()) {
                            std::cout << "Open right camera failed" << std::endl;
                            return;
                        }
                        cameraRight->SetExposeTime(3500);
                        cameraRight->CameraChangeTrig(cameraRight->trigLine);
                        cameraRight->CameraStart();
                        isSuccessSearch = true;
                    }
                }
                if (!isSuccessSearch) {
                    std::cout << "Search camera failed!There is dosen't has a camera which is named Right!" << std::endl;
                }
            }
            if (cameraUsedState == CameraUsedState::LeftGrayRightGrayExColor) {
                if (nullptr == cameraColor) {
                    cameraColor.reset(new CammeraUnilty());
                    bool isSuccessSearch = false;
                    for (int i = 0; i < devices.nDevNum; i++) {
                        if (std::string(devices.pDevInfo[i].cameraName) == "Color") {
                            cameraColor->SetCamera(devices.pDevInfo[i].cameraKey);
                            cameraColor->cameraType = CammeraUnilty::ColorCamera;
                            if (!cameraColor->CameraOpen()) {
                                std::cout << "Open color camera failed" << std::endl;
                                return;
                            }
                            cameraColor->setPixelFormat("BayerRG8");
                            cameraColor->SetExposeTime(20400);
                            cameraColor->CameraChangeTrig(cameraRight->trigLine);
                            cameraColor->CameraStart();
                            isSuccessSearch = true;
                        }
                    }
                    if (!isSuccessSearch) {
                        std::cout << "Search camera failed!There is dosen't has a camera which is named Color!" << std::endl;
                    }
                }
            }
        }

        void CameraControl::getOneFrameImgs(RestructedFrame &imgsOneFrame) {
            if (imgsOneFrame.leftImgs.size() < projector->elementSize)
                imgsOneFrame.leftImgs.resize(projector->elementSize);
            if (imgsOneFrame.rightImgs.size() < projector->elementSize)
                imgsOneFrame.rightImgs.resize(projector->elementSize);

            while (cameraLeft->imgQueue.size() < projector->elementSize ||
                   cameraRight->imgQueue.size() < projector->elementSize) {
                std::cout << "left grub ,right grub :" << cameraLeft->imgQueue.size() << "," << cameraRight->imgQueue.size() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            for (int i = 0; i < projector->elementSize; ++i) {
                imgsOneFrame.leftImgs[i] = cameraLeft->imgQueue.front();
                cameraLeft->imgQueue.pop();
                imgsOneFrame.rightImgs[i] = cameraRight->imgQueue.front();
                cameraRight->imgQueue.pop();
            }

            if (cameraUsedState == CameraUsedState::LeftGrayRightGrayExColor) {
                while (cameraColor->imgQueue.size() < projector->elementSize) {
                    std::cout << "color grub :" << cameraColor->imgQueue.size() << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                if (imgsOneFrame.colorImgs.size() < projector->elementSize)
                    imgsOneFrame.colorImgs.resize(projector->elementSize);

                for (int i = 0; i < projector->elementSize; ++i) {
                    imgsOneFrame.colorImgs[i] = cameraColor->imgQueue.front();
                    cameraColor->imgQueue.pop();
                }
            }
        }

        void CameraControl::triggerColorCameraSoftCaputure(const int exposureTime) {
            if (cameraUsedState == CameraUsedState::LeftColorRightGray) {
                const int exposureTimeTrigLine = cameraLeft->exposureTime;

                if (cameraLeft->isTriggerLine()) {
                    cameraLeft->CameraChangeTrig(CammeraUnilty::trigSoftware);
                    cameraLeft->SetExposeTime(exposureTime);
                }

                cameraLeft->ExecuteSoftTrig();

                while (cameraLeft->imgQueue.size() < 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                cameraLeft->CameraChangeTrig(CammeraUnilty::trigLine);
                cameraLeft->SetExposeTime(exposureTimeTrigLine);
            } 
            else {
                const int exposureTimeTrigLine = cameraColor->exposureTime;

                if (cameraColor->isTriggerLine()) {
                    cameraColor->CameraChangeTrig(CammeraUnilty::trigSoftware);
                    cameraColor->SetExposeTime(exposureTime);
                }

                cameraColor->ExecuteSoftTrig();

                while (cameraColor->imgQueue.size() < 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                cameraColor->CameraChangeTrig(CammeraUnilty::trigLine);
                cameraColor->SetExposeTime(exposureTimeTrigLine);
            }
        }

        void CameraControl::setCameraExposure(const int grayExposure,
                                              const int colorExposure) {
            if (cameraUsedState == LeftColorRightGray)
                cameraLeft->SetExposeTime(colorExposure);
            else
                cameraLeft->SetExposeTime(grayExposure);

            cameraRight->SetExposeTime(grayExposure);

            if (cameraUsedState == LeftGrayRightGrayExColor)
                cameraColor->SetExposeTime(colorExposure);
        }

        void CameraControl::loadFirmware(const std::string firmwarePath) {
            projector->LoadFirmware(firmwarePath);
        }

        void CameraControl::closeCamera() {
            if (nullptr != cameraLeft) {
                cameraLeft->CameraClose();
            }

            if (nullptr != cameraRight) {
                cameraLeft->CameraClose();
            }

            if (nullptr != cameraColor) {
                cameraLeft->CameraClose();
            }
        }

        std::vector<int> CameraControl::getFrameFps() {
            std::vector<int> fps;
            if (nullptr != cameraLeft) {
                IMV_StreamStatisticsInfo info;
                IMV_GetStatisticsInfo(cameraLeft->m_devHandle, &info);
                fps.emplace_back(info.u3vStatisticsInfo.fps);
            }

            if (nullptr != cameraRight) {
                IMV_StreamStatisticsInfo info;
                IMV_GetStatisticsInfo(cameraRight->m_devHandle, &info);
                fps.emplace_back(info.u3vStatisticsInfo.fps);
            }

            if (nullptr != cameraColor) {
                IMV_StreamStatisticsInfo info;
                IMV_GetStatisticsInfo(cameraColor->m_devHandle, &info);
                fps.emplace_back(info.u3vStatisticsInfo.fps);
            }

            return fps;
        }
        
        void CameraControl::project(const bool isContinues) {
            projector->projecte(isContinues);
        }

        void CameraControl::stopProject() {
            projector->stopProject();
        }
    }// namespace device
}// namespace sl