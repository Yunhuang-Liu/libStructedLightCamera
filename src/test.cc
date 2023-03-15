#include "structedLightCamera.h"

int main() {
    /*
    sl::device::ProjectorControl projector(DLPC34XX_CDI_DLPC3479);
    projector.projecte(true);
    projector.stopProject();
     */
    /*
    sl::device::CameraControl camera(DLPC34XX_CDI_DLPC3479, 10, sl::device::CameraControl::LeftGrayRightGray);
    camera.project(false);
    sl::device::RestructedFrame frame;
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    camera.getOneFrameImgs(frame);
    for(int i = 0; i < frame.leftImgs.size(); ++i)
        cv::imwrite(std::to_string(i) + ".bmp", frame.leftImgs[i]);
    */
    sl::tool::MatrixsInfo* info = new sl::tool::MatrixsInfo("/home/lyh/桌面/devolope/StructedLightStudio/system/calibrationFile/intrinsic.yml", "/home/lyh/桌面/devolope/StructedLightStudio/system/calibrationFile/extrinsic.yml");
    StructedLightCamera::AcceleratedMethod accelMethod(StructedLightCamera::AcceleratedMethod::CPU);
    StructedLightCamera::AlgorithmType algorithm(StructedLightCamera::AlgorithmType::FourStepSixGrayCode);
    StructedLightCamera::SLCameraSet cameraSet(StructedLightCamera::ChipControlCore::DLPC3479, sl::device::CameraControl::LeftGrayRightGray);
    sl::restructor::RestructParamater param(-500, 500, 750, 1250, 16);
    StructedLightCamera* camera = new StructedLightCamera(info->getInfo(), algorithm, accelMethod, cameraSet, param);
    //TODO(@Liu Yunhuang):needed to set exposure class
    camera->setExposureTime(8000,8000);
    std::vector<cv::Mat> depth, color;
    camera->getOneFrame(depth, color, true);

    cv::imshow("depth", depth);
    cv::imshow("color", color);
}