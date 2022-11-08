#include <structedLightCamera.h>

using namespace sl;
using namespace sl::device;

StructedLightCamera::StructedLightCamera(const sl::tool::Info& infoCalibraion, const AlgorithmType algorithmType_, const AcceleratedMethod acceleratedMethod_, const SLCameraSet cameraSet, const restructor::RestructParamater params, const cv::Mat& leftRefImg, const cv::Mat& rightRefImg) :
    restructor(nullptr),phaseSolverLeft(nullptr),phaseSolverRight(nullptr),camera(nullptr),calibrationInfo(infoCalibraion),
    algorithmType(algorithmType_),acceleratedMethod(acceleratedMethod_){
    int grayCaptureImgNum;
    int colorCaptureImgNum;
    int trigNum;
    switch (algorithmType) {
        case AlgorithmType::ThreeStepFiveGrayCode : {
            phaseSolverLeft = new phaseSolver::ThreeStepFiveGrayCodeMaster_CPU();
            phaseSolverRight = new phaseSolver::ThreeStepFiveGrayCodeMaster_CPU();
            grayCaptureImgNum = 8;
            colorCaptureImgNum = 1;
            trigNum = 8;
            break;
        }
        case AlgorithmType::FourStepSixGrayCode : {
            if(AcceleratedMethod::CPU == acceleratedMethod){
                phaseSolverLeft = new phaseSolver::FourStepSixGrayCodeMaster_CPU(leftRefImg);
                phaseSolverRight = new phaseSolver::FourStepSixGrayCodeMaster_CPU(rightRefImg);
            }
            #ifdef CUDA
            else{
                phaseSolverLeft = new phaseSolver::FourStepSixGrayCodeMaster_GPU(params.block);
                phaseSolverRight = new phaseSolver::FourStepSixGrayCodeMaster_GPU(params.block);
            }
            #endif
            grayCaptureImgNum = 10;
            colorCaptureImgNum = 1;
            trigNum = 10;
            break;
        }
        #ifdef CUDA
        case AlgorithmType::DevidedSpaceTimeMulUsed : {
            phaseSolverLeft = new phaseSolver::DividedSpaceTimeMulUsedMaster_GPU(leftRefImg,params.block);
            phaseSolverRight = new phaseSolver::DividedSpaceTimeMulUsedMaster_GPU(rightRefImg, params.block);
            grayCaptureImgNum = 16;
            colorCaptureImgNum = 4;
            trigNum = 16;
            break;
        }
        case AlgorithmType::ShiftGrayCodeTimeMulUsed : {
            phaseSolverLeft = new phaseSolver::ShiftGrayCodeUnwrapMaster_GPU(params.block);
            phaseSolverRight = new phaseSolver::ShiftGrayCodeUnwrapMaster_GPU(params.block);
            grayCaptureImgNum = 16;
            colorCaptureImgNum = 4;
            trigNum = 16;
            break;
        }
        #endif
    }

    if (cameraSet.chipCore == DLP3010)
        camera = new CameraControl(DLPC34XX_ControllerDeviceId_e::DLPC34XX_CDI_DLPC3478, cameraSet.cameraSet);
    else
        camera = new CameraControl(trigNum, cameraSet.cameraSet);

    switch (acceleratedMethod){
        case AcceleratedMethod::CPU : {
            restructor = new restructor::Restructor_CPU(calibrationInfo, params.minDisparity, params.maxDisparity,
                params.minDepth, params.maxDepth, params.threads);
            break;
        }
        #ifdef CUDA
        case AcceleratedMethod::GPU : {
            restructor = new restructor::Restructor_GPU(calibrationInfo, params.minDisparity, params.maxDisparity,
                params.minDepth, params.maxDepth,params.block);
            break;
        }
        #endif
    }
}

#ifdef CUDA
void StructedLightCamera::remapImg(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& remap_x, const cv::cuda::GpuMat& remap_y, cv::cuda::GpuMat& outImg) {
    cv::cuda::remap(src, outImg, remap_x, remap_y, cv::INTER_LINEAR);
}

void StructedLightCamera::getOneFrame(std::vector<cv::cuda::GpuMat>& depthImg, std::vector<cv::cuda::GpuMat>& colorImg) {
    RestructedFrame restructedFrame;
    camera->getOneFrameImgs(restructedFrame);
    if (remap_x_L.empty()) {
        cv::initUndistortRectifyMap(calibrationInfo.M1, calibrationInfo.D1, calibrationInfo.R1, calibrationInfo.P1, restructedFrame.leftImgs[0].size(), CV_32FC1, remap_x_L, remap_y_L);
        cv::initUndistortRectifyMap(calibrationInfo.M2, calibrationInfo.D2, calibrationInfo.R2, calibrationInfo.P2, restructedFrame.leftImgs[0].size(), CV_32FC1, remap_x_R, remap_y_R);
    }
    if (remap_x_deice_L.empty()) {
        remap_x_deice_L.upload(remap_x_L);
        remap_y_deice_L.upload(remap_y_L);
        remap_x_deice_R.upload(remap_x_R);
        remap_y_deice_R.upload(remap_y_R);
    }
    cv::cuda::Stream leftSolvePhase(cudaStreamNonBlocking);
    cv::cuda::Stream rightSolvePhase(cudaStreamNonBlocking);
    phaseSolverLeft->changeSourceImg(restructedFrame.leftImgs, leftSolvePhase);
    phaseSolverRight->changeSourceImg(restructedFrame.rightImgs, rightSolvePhase);
    std::vector<cv::cuda::GpuMat> unwrapImgLeft_dev;
    std::vector<cv::cuda::GpuMat> unwrapImgRight_dev;
    phaseSolverLeft->getUnwrapPhaseImg(unwrapImgLeft_dev, leftSolvePhase);
    phaseSolverRight->getUnwrapPhaseImg(unwrapImgRight_dev, rightSolvePhase);
    leftSolvePhase.waitForCompletion();
    rightSolvePhase.waitForCompletion();
    /*
    auto timeSolvePhase = std::chrono::steady_clock::now();
    auto timeUsedSolvePhase = std::chrono::duration_cast<std::chrono::milliseconds>(timeSolvePhase - timeInit).count() * static_cast<float>(std::chrono::milliseconds::period::num) / std::chrono::milliseconds::period::den;
    std::cout << "解相用时:" << timeUsedSolvePhase << std::endl;
    */
    const int imgNums = unwrapImgLeft_dev.size();
    cv::Size imgSize = unwrapImgLeft_dev[0].size();
    std::vector<cv::cuda::GpuMat> unwrapRec_dev_L(imgNums);
    std::vector<cv::cuda::GpuMat> unwrapRec_dev_R(imgNums);
    for (int i = 0; i < imgNums; i++) {
        unwrapRec_dev_L[i] = cv::cuda::createContinuous(imgSize, CV_32FC1);
        unwrapRec_dev_R[i] = cv::cuda::createContinuous(imgSize, CV_32FC1);
        remapImg(unwrapImgLeft_dev[i], remap_x_deice_L, remap_y_deice_L, unwrapRec_dev_L[i]);
        remapImg(unwrapImgRight_dev[i], remap_x_deice_R, remap_y_deice_R, unwrapRec_dev_R[i]);
    }
    std::vector<cv::cuda::Stream> stream_Restructor(imgNums);
    depthImg.resize(imgNums);
    colorImg.resize(imgNums);
    for (int i = 0; i < imgNums; i++) {
        stream_Restructor[i] = cv::cuda::Stream(cudaStreamNonBlocking);
        restructor->restruction(unwrapRec_dev_L[i], unwrapRec_dev_R[i], i, stream_Restructor[i],true);
    }
    for (int i = 0; i < imgNums; i++) {
        stream_Restructor[i].waitForCompletion();
        restructor->download(i, depthImg[i]);
    }
}
#endif

void StructedLightCamera::getOneFrame(std::vector<cv::Mat>& depthImg, std::vector<cv::Mat>& colorImg){
    if (AcceleratedMethod::CPU == acceleratedMethod || AlgorithmType::FourStepSixGrayCode == algorithmType) {
        camera->triggerColorCameraSoftCaputure(200000);
    }
    RestructedFrame restructedFrame;
    camera->getOneFrameImgs(restructedFrame);
    phaseSolverLeft->changeSourceImg(restructedFrame.leftImgs);
    phaseSolverRight->changeSourceImg(restructedFrame.rightImgs);
    if(remap_x_L.empty()){
        cv::initUndistortRectifyMap(calibrationInfo.M1, calibrationInfo.D1, calibrationInfo.R1, calibrationInfo.P1, restructedFrame.leftImgs[0].size(), CV_32FC1, remap_x_L, remap_y_L);
        cv::initUndistortRectifyMap(calibrationInfo.M2, calibrationInfo.D2, calibrationInfo.R2, calibrationInfo.P2, restructedFrame.leftImgs[0].size(), CV_32FC1, remap_x_R, remap_y_R);
    }
    cv::Mat unwrapLeftImg;
    cv::Mat unwrapRightImg;
    phaseSolverLeft->getUnwrapPhaseImg(unwrapLeftImg);
    phaseSolverRight->getUnwrapPhaseImg(unwrapRightImg);
    std::thread threadRemap = std::thread([&] {
        cv::remap(unwrapLeftImg, unwrapLeftImg, remap_x_L, remap_y_L, cv::INTER_LINEAR);
        });
    cv::remap(unwrapRightImg, unwrapRightImg, remap_x_R, remap_y_R, cv::INTER_LINEAR);
    if (threadRemap.joinable()) {
        threadRemap.join();
    }
    depthImg.resize(1);
    colorImg.resize(1);
    restructor->restruction(unwrapLeftImg, unwrapRightImg, depthImg[0],true);
}

void StructedLightCamera::setExposureTime(const int grayExposureTime, const int colorExposureTime) {
    camera->setCameraExposure(grayExposureTime, colorExposureTime);
}

void StructedLightCamera::closeCamera() {
    camera->closeCamera();
}
