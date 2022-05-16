#include <CameraControl/CameraControl.h>


CameraControl::CameraControl(const DLPC34XX_ControllerDeviceId_e projectorModuleType) : cameraLeft(nullptr),
    cameraRight(nullptr),cameraColor(nullptr),projector(nullptr){
    projector = new ProjectorControl(projectorModuleType);
    int state = IMV_OK;
    IMV_DeviceList devices;
    state = IMV_EnumDevices(&devices, interfaceTypeAll);
    if (state) {
        std::cout<<"Search camera failed!"<<std::endl;
    }
    if(nullptr == cameraLeft){
        cameraLeft = new CammeraUnilty();
        bool isSuccessSearch = false;
        for (int i = 0; i < devices.nDevNum; i++) {
            if (std::string(devices.pDevInfo[i].cameraName) == "Left") {
                cameraLeft->SetCamera(devices.pDevInfo[i].cameraKey);
                cameraLeft->cameraType = CammeraUnilty::LeftCamera;
                if (!cameraLeft->CameraOpen()) {
                    std::cout<<"Open left camera failed"<<std::endl;
                    return;
                }
                cameraLeft->SetExposeTime(3500);
                cameraLeft->CameraChangeTrig(cameraLeft->trigLine);
                cameraLeft->CameraStart();
                isSuccessSearch = true;
            }
        }
        if (!isSuccessSearch) {
            std::cout<<"Search camera failed!There is dosen't has a camera which is named Left!"<<std::endl;
        }
    }
    if(nullptr == cameraRight){
        cameraRight = new CammeraUnilty();
        bool isSuccessSearch = false;
        for (int i = 0; i < devices.nDevNum; i++) {
            if (std::string(devices.pDevInfo[i].cameraName) == "Right") {
                cameraRight->SetCamera(devices.pDevInfo[i].cameraKey);
                cameraRight->cameraType = CammeraUnilty::RightCamera;
                if (!cameraRight->CameraOpen()) {
                    std::cout<< "Open right camera failed" <<std::endl;
                    return;
                }
                cameraRight->SetExposeTime(3500);
                cameraRight->CameraChangeTrig(cameraRight->trigLine);
                cameraRight->CameraStart();
                isSuccessSearch = true;
            }
        }
        if (!isSuccessSearch) {
            std::cout<<"Search camera failed!There is dosen't has a camera which is named Right!"<<std::endl;
        }
    }
    if (nullptr == cameraColor) {
        cameraColor = new CammeraUnilty();
        bool isSuccessSearch = false;
        for (int i = 0; i < devices.nDevNum; i++) {
            if (std::string(devices.pDevInfo[i].cameraName) == "Color") {
                cameraColor->SetCamera(devices.pDevInfo[i].cameraKey);
                cameraColor->cameraType = CammeraUnilty::ColorCamera;
                if (!cameraColor->CameraOpen()) {
                    std::cout<<"Open color camera failed"<<std::endl;
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
            std::cout<< "Search camera failed!There is dosen't has a camera which is named Color!" <<std::endl;
        }
    }
}

void CameraControl::getOneFrameImgs(RestructedFrame& imgsOneFrame){
    projector->projecteOnce();
    while(cameraLeft->index < cameraLeft->imgs.size() || cameraRight->index < cameraRight->imgs.size() || cameraColor->index <cameraColor->imgs.size()){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const int imgsNumGray = cameraLeft->imgs.size();
    imgsOneFrame.leftImgs.resize(imgsNumGray);
    imgsOneFrame.rightImgs.resize(imgsNumGray);
    imgsOneFrame.colorImgs.resize(cameraColor->imgs.size());
    /*
    std::vector<std::thread> threadsConvert;
    threadsConvert.resize(imgsNumGray);
    for (int i = 0; i < imgsNumGray; i++) {
        threadsConvert[i] = std::move(std::thread([&,i] {
            cameraLeft->imgs[i].convertTo(imgsOneFrame.leftImgs[i], CV_32FC1);
            cameraRight->imgs[i].convertTo(imgsOneFrame.rightImgs[i], CV_32FC1);
            }));
    }
    */
    for (int i = 0; i < cameraLeft->imgs.size(); i++) {
        imgsOneFrame.leftImgs[i] = cameraLeft->imgs[i];
        imgsOneFrame.rightImgs[i] = cameraRight->imgs[i];
    }
    for (int i = 0; i < cameraColor->imgs.size(); i++) {
        imgsOneFrame.colorImgs[i] = cameraColor->imgs[i];
    }
    /*
    for (int i = 0; i < imgsNumGray; i++) {
        if (threadsConvert[i].joinable()) {
            threadsConvert[i].join();
        }
    }
    */
    cameraLeft->index = 0;
    cameraRight->index = 0;
    cameraColor->index = 0;
}

void CameraControl::setCaptureImgsNum(const int GrayImgsNum, const int ColorImgsNum){
    cameraColor->imgs.resize(ColorImgsNum);
    cameraLeft->imgs.resize(GrayImgsNum);
    cameraRight->imgs.resize(GrayImgsNum);
}

void CameraControl::triggerColorCameraSoftCaputure() {
    if (cameraColor->isTriggerLine()) {
        cameraColor->CameraChangeTrig(CammeraUnilty::trigSoftware);
        cameraColor->SetExposeTime(100000);
    }
    cameraColor->ExecuteSoftTrig();
    while (cameraColor->index < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void CameraControl::setCameraExposure(const int grayExposure, const int colorExposure) {
    cameraLeft->SetExposeTime(grayExposure);
    cameraRight->SetExposeTime(grayExposure);
    cameraColor->SetExposeTime(colorExposure);
}

void CameraControl::loadFirmware(const std::string firmwarePath){
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