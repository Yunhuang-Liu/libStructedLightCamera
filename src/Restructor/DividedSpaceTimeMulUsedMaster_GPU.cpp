#include "Restructor/DividedSpaceTimeMulUsedMaster_GPU.h"

namespace PhaseSolverType {
DividedSpaceTimeMulUsedMaster_GPU::DividedSpaceTimeMulUsedMaster_GPU(
    std::vector<cv::Mat> &imgs,const cv::Mat& refImgWhite, const dim3 block_) : 
    refImgWhite_device(refImgWhite), block(block_) {
    rows = refImgWhite.rows;
    cols = refImgWhite.cols;
    wrapImg1_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg1_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg1_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg2_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg2_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg2_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg3_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg3_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg3_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg4_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg4_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg4_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    averageImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    averageImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    averageImg_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    averageImg_4_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    conditionImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    conditionImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    conditionImg_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    conditionImg_4_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    unwrapImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    unwrapImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    unwrapImg_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    unwrapImg_4_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    floor_K_device = cv::cuda::createContinuous(rows, cols, CV_8UC1);
    img1_1_device.upload(imgs[0]);
    img1_2_device.upload(imgs[1]);
    img1_3_device.upload(imgs[2]);
    img1_4_device.upload(imgs[3]);
    img2_1_device.upload(imgs[4]);
    img2_2_device.upload(imgs[5]);
    img2_3_device.upload(imgs[6]);
    img2_4_device.upload(imgs[7]);
    img3_1_device.upload(imgs[8]);
    img3_2_device.upload(imgs[9]);
    img3_3_device.upload(imgs[10]);
    img3_4_device.upload(imgs[11]);
    img4_1_device.upload(imgs[12]);
    img4_2_device.upload(imgs[13]);
    img4_3_device.upload(imgs[14]);
    img4_4_device.upload(imgs[15]);
}

DividedSpaceTimeMulUsedMaster_GPU::~DividedSpaceTimeMulUsedMaster_GPU() {

}

void DividedSpaceTimeMulUsedMaster_GPU::getWrapPhaseImg(
    cv::cuda::Stream& stream) {
    PhaseSolverType::cudaFunc::solvePhasePrepare_DevidedSpace(
    img1_4_device, img1_2_device, img1_3_device, img1_1_device,
    img2_4_device, img2_2_device, img2_3_device, img2_1_device,
    img3_4_device, img3_2_device, img3_3_device, img3_1_device,
    img4_4_device, img4_2_device, img4_3_device, img4_1_device,
    rows, cols,
    wrapImg1_1_device, wrapImg1_2_device, wrapImg1_3_device, 
    conditionImg_1_device,
    wrapImg2_1_device, wrapImg2_2_device, wrapImg2_3_device, 
    conditionImg_2_device,
    wrapImg3_1_device, wrapImg3_2_device, wrapImg3_3_device, 
    conditionImg_3_device,
    wrapImg4_1_device, wrapImg4_2_device, wrapImg4_3_device, 
    conditionImg_4_device,
    floor_K_device,block, stream);
}

void DividedSpaceTimeMulUsedMaster_GPU::getUnwrapPhaseImg(
    std::vector<cv::cuda::GpuMat>& unwrapImg, cv::cuda::Stream& stream) {
    unwrapImg.resize(4, cv::cuda::GpuMat(rows, cols, CV_32FC1));
    getWrapPhaseImg(stream);
    //cudaDeviceSynchronize();
    PhaseSolverType::cudaFunc::solvePhase_DevidedSpace(refImgWhite_device, 
        rows, cols,
        wrapImg1_1_device, wrapImg1_2_device, wrapImg1_3_device, 
        conditionImg_1_device, unwrapImg[0],
        wrapImg2_1_device, wrapImg2_2_device, wrapImg2_3_device, 
        conditionImg_2_device, unwrapImg[1],
        wrapImg3_1_device, wrapImg3_2_device, wrapImg3_3_device, 
        conditionImg_3_device, unwrapImg[2],
        wrapImg4_1_device, wrapImg4_2_device, wrapImg4_3_device, 
        conditionImg_4_device, unwrapImg[3], 
        floor_K_device,block,stream);
        //cudaDeviceSynchronize();
}

DividedSpaceTimeMulUsedMaster_GPU::DividedSpaceTimeMulUsedMaster_GPU(
    const cv::Mat &refImgWhite_, const dim3 block_) : 
    refImgWhite_device(refImgWhite_), block(block_) {
    rows = refImgWhite_.rows;
    cols = refImgWhite_.cols;
    wrapImg1_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg1_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg1_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg2_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg2_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg2_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg3_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg3_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg3_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg4_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg4_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    wrapImg4_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    averageImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    averageImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    averageImg_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    averageImg_4_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    conditionImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    conditionImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    conditionImg_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    conditionImg_4_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    unwrapImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    unwrapImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    unwrapImg_3_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    unwrapImg_4_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
    floor_K_device = cv::cuda::createContinuous(rows, cols, CV_8UC1);
}

void DividedSpaceTimeMulUsedMaster_GPU::changeSourceImg(
    std::vector<cv::Mat> &imgs) {
    img1_1_device.upload(imgs[0]);
    img1_2_device.upload(imgs[1]);
    img1_3_device.upload(imgs[2]);
    img1_4_device.upload(imgs[3]);
    img2_1_device.upload(imgs[4]);
    img2_2_device.upload(imgs[5]);
    img2_3_device.upload(imgs[6]);
    img2_4_device.upload(imgs[7]);
    img3_1_device.upload(imgs[8]);
    img3_2_device.upload(imgs[9]);
    img3_3_device.upload(imgs[10]);
    img3_4_device.upload(imgs[11]);
    img4_1_device.upload(imgs[12]);
    img4_2_device.upload(imgs[13]);
    img4_3_device.upload(imgs[14]);
    img4_4_device.upload(imgs[15]);
}

void DividedSpaceTimeMulUsedMaster_GPU::changeSourceImg(
    std::vector<cv::Mat>& imgs, cv::cuda::Stream& stream) {
    img1_1_device.upload(imgs[0], stream);
    img1_2_device.upload(imgs[1], stream);
    img1_3_device.upload(imgs[2], stream);
    img1_4_device.upload(imgs[3], stream);
    img2_1_device.upload(imgs[4], stream);
    img2_2_device.upload(imgs[5], stream);
    img2_3_device.upload(imgs[6], stream);
    img2_4_device.upload(imgs[7], stream);
    img3_1_device.upload(imgs[8], stream);
    img3_2_device.upload(imgs[9], stream);
    img3_3_device.upload(imgs[10], stream);
    img3_4_device.upload(imgs[11], stream);
    img4_1_device.upload(imgs[12], stream);
    img4_2_device.upload(imgs[13], stream);
    img4_3_device.upload(imgs[14], stream);
    img4_4_device.upload(imgs[15], stream);
}

void DividedSpaceTimeMulUsedMaster_GPU::changeSourceImg(
    std::vector<cv::cuda::GpuMat>& imgs) {
    img1_1_device = imgs[0];
    img1_2_device = imgs[1];
    img1_3_device = imgs[2];
    img1_4_device = imgs[3];
    img2_1_device = imgs[4];
    img2_2_device = imgs[5];
    img2_3_device = imgs[6];
    img2_4_device = imgs[7];
    img3_1_device = imgs[8];
    img3_2_device = imgs[9];
    img3_3_device = imgs[10];
    img3_4_device = imgs[11];
    img4_1_device = imgs[12];
    img4_2_device = imgs[13];
    img4_3_device = imgs[14];
    img4_4_device = imgs[15];
}

void DividedSpaceTimeMulUsedMaster_GPU::getTextureImg(
    std::vector<cv::cuda::GpuMat>& textureImg) {
    textureImg.clear();
    textureImg.resize(4, cv::cuda::GpuMat(averageImg_1_device.size(), 
        averageImg_1_device.type()));
    textureImg[0] = averageImg_1_device;
    textureImg[1] = averageImg_2_device;
    textureImg[2] = averageImg_3_device;
    textureImg[3] = averageImg_4_device;
}
}
