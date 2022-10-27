#include <device/CameraControl.h>

namespace sl {
    namespace device {
        CameraControl::CameraControl(const DLPC34XX_ControllerDeviceId_e projectorModuleType,
                                     CameraUsedState state_) : cameraLeft(nullptr),
                                                               cameraRight(nullptr), cameraColor(nullptr),
                                                               projector(nullptr), cameraUsedState(state_) {
            projector = new ProjectorControl(projectorModuleType);
            int state = IMV_OK;
            IMV_DeviceList devices;
            state = IMV_EnumDevices(&devices, interfaceTypeAll);
            if (state) {
                std::cout << "Search camera failed!" << std::endl;
            }
            if (nullptr == cameraLeft) {
                cameraLeft = new CammeraUnilty();
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
                cameraRight = new CammeraUnilty();
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
                    cameraColor = new CammeraUnilty();
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
                            cameraColor->imgs.resize(4);
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
            projector = new ProjectorControl(numLutEntries);
            int state = IMV_OK;
            IMV_DeviceList devices;
            state = IMV_EnumDevices(&devices, interfaceTypeAll);
            if (state) {
                std::cout << "Search camera failed!" << std::endl;
            }
            if (nullptr == cameraLeft) {
                cameraLeft = new CammeraUnilty();
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
                cameraRight = new CammeraUnilty();
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
                    cameraColor = new CammeraUnilty();
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
                            cameraColor->imgs.resize(4);
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
            auto start = std::chrono::system_clock::now();
            projector->projecteOnce();
            while (cameraLeft->index < cameraLeft->imgs.size() ||
                   cameraRight->index < cameraRight->imgs.size()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            auto end = std::chrono::system_clock::now();
            auto timeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * static_cast<float>(std::chrono::milliseconds::period::num) / std::chrono::milliseconds::period::den;
            std::cout << "capture rate: " << 4.f / timeUsed << std::endl;
            const int imgsNumGray = cameraLeft->imgs.size();
            imgsOneFrame.leftImgs.resize(imgsNumGray);
            imgsOneFrame.rightImgs.resize(imgsNumGray);
            for (int i = 0; i < cameraLeft->imgs.size(); i++) {
                imgsOneFrame.leftImgs[i] = cameraLeft->imgs[i];
                imgsOneFrame.rightImgs[i] = cameraRight->imgs[i];
            }
            cameraLeft->index = 0;
            cameraRight->index = 0;
            if (cameraUsedState == CameraUsedState::LeftGrayRightGrayExColor) {
                while (cameraColor->index < cameraColor->imgs.size()) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                imgsOneFrame.colorImgs.resize(cameraColor->imgs.size());
                for (int i = 0; i < cameraColor->imgs.size(); i++) {
                    imgsOneFrame.colorImgs[i] = cameraColor->imgs[i];
                }
                cameraColor->index = 0;
            }
        }

        void CameraControl::setCaptureImgsNum(const int GrayImgsNum,
                                              const int ColorImgsNum) {
            if (cameraUsedState == CameraUsedState::LeftColorRightGray)
                cameraLeft->imgs.resize(ColorImgsNum);
            else
                cameraLeft->imgs.resize(GrayImgsNum);
            if (cameraUsedState == CameraUsedState::LeftGrayRightGrayExColor)
                cameraColor->imgs.resize(ColorImgsNum);
            cameraRight->imgs.resize(GrayImgsNum);
        }

        void CameraControl::triggerColorCameraSoftCaputure() {
            if (cameraUsedState == CameraUsedState::LeftColorRightGray) {
                if (cameraLeft->isTriggerLine()) {
                    cameraLeft->CameraChangeTrig(CammeraUnilty::trigSoftware);
                    cameraLeft->SetExposeTime(100000);
                }
                cameraLeft->ExecuteSoftTrig();
                while (cameraLeft->index < 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } else {
                if (cameraColor->isTriggerLine()) {
                    cameraColor->CameraChangeTrig(CammeraUnilty::trigSoftware);
                    cameraColor->SetExposeTime(100000);
                }
                cameraColor->ExecuteSoftTrig();
                while (cameraColor->index < 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
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
    }// namespace device
}// namespace sl