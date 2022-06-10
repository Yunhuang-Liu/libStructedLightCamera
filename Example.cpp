#include "StructedLightCamera.h"

int main(){
    /*
    MatrixsInfo* matrixInfo = new MatrixsInfo("../systemFile/calibrationFiles/intrinsic.yml","../systemFile/calibrationFiles/extrinsic.yml");
    const Info& calibrationInfo = matrixInfo->getInfo();
    cv::cuda::setDevice(0);
    cv::Mat leftRef = cv::imread("../systemFile/refImg/left.tif", cv::IMREAD_UNCHANGED);
    cv::Mat rightRef = cv::imread("../systemFile/refImg/right.tif", cv::IMREAD_UNCHANGED);
    RestructorType::RestructParamater params(-500, 500, 150, 350, 16);
    params.block = dim3(32, 16, 1);
    StructedLightCamera* camera = new StructedLightCamera(calibrationInfo, StructedLightCamera::DevidedSpaceTimeMulUsed, StructedLightCamera::GPU ,params, leftRef, rightRef);
    camera->setExposureTime(3000, 20400);
    std::vector<cv::cuda::GpuMat> depthImgs;
    std::vector<cv::cuda::GpuMat> colorImgs;
    camera->getOneFrame(depthImgs, colorImgs);
    */
    cv::cuda::setDevice(0);
    CameraControl* camera = new CameraControl(8, CameraControl::CameraUsedState::LeftColorRightGray);
    camera->setCameraExposure(2000, 2000);
    camera->setCaptureImgsNum(8, 8);
    RestructedFrame frame;
    camera->getOneFrameImgs(frame);
}
