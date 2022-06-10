#include <Restructor/FourStepSixGrayCodeMaster_GPU.h>

namespace PhaseSolverType {
    FourStepSixGrayCodeMaster_GPU::FourStepSixGrayCodeMaster_GPU(std::vector<cv::Mat>& imgs,const dim3 block_) :
        block(block_) {
        if(imgs.size() > 0){
            imgs_device.resize(imgs.size());
            for(int i=0;i<imgs_device.size();i++){
                imgs_device[i].upload(imgs[i]);
            }
        }
    }

    FourStepSixGrayCodeMaster_GPU::~FourStepSixGrayCodeMaster_GPU() {

    }

    void FourStepSixGrayCodeMaster_GPU::getWrapPhaseImg() {

    }

    void FourStepSixGrayCodeMaster_GPU::getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>& unwrapImg, cv::cuda::Stream& pStream) {

        PhaseSolverType::cudaFunc::solvePhasePrepare_FourStepSixGray(imgs_device[0], imgs_device[1], imgs_device[2], imgs_device[3], rows, cols, wrapImg_device, averageImg_device, conditionImg_device, block, pStream);
        unwrapImg.resize(1);
        for(int i=0;i<unwrapImg.size();i++){
            unwrapImg[i].create(rows,cols,CV_32FC1);
        }
        PhaseSolverType::cudaFunc::solvePhase_FourStepSixGray(imgs_device[4], imgs_device[5], imgs_device[6], imgs_device[7], imgs_device[8], imgs_device[9], rows, cols, averageImg_device, conditionImg_device, wrapImg_device, unwrapImg[0], block, pStream);
    }

    void FourStepSixGrayCodeMaster_GPU::changeSourceImg(std::vector<cv::Mat>& imgs){
        rows = imgs[0].rows;
        cols = imgs[0].cols;
        imgs_device.resize(imgs.size());
        for(int i=0;i<imgs_device.size();i++){
            imgs_device[i].upload(imgs[i]);
        }
        if (wrapImg_device.empty()) {
            wrapImg_device.create(rows, cols, CV_32FC1);
            averageImg_device.create(rows, cols, CV_32FC1);
            conditionImg_device.create(rows, cols, CV_32FC1);
        }
    }

    void FourStepSixGrayCodeMaster_GPU::changeSourceImg(std::vector<cv::Mat>& imgs, cv::cuda::Stream& stream) {
        rows = imgs[0].rows;
        cols = imgs[0].cols;
        imgs_device.resize(imgs.size());
        for (int i = 0; i < imgs_device.size(); i++) {
            imgs_device[i].upload(imgs[i], stream);
        }
        if (wrapImg_device.empty()) {
            wrapImg_device.create(rows, cols, CV_32FC1);
            averageImg_device.create(rows, cols, CV_32FC1);
            conditionImg_device.create(rows, cols, CV_32FC1);
        }
    }

    void FourStepSixGrayCodeMaster_GPU::changeSourceImg(std::vector<cv::cuda::GpuMat>& imgs) {
        imgs_device.resize(imgs.size());
        for (int i = 0; i < imgs_device.size(); i++) {
            imgs_device[i] = imgs[i];
        }
    }

    FourStepSixGrayCodeMaster_GPU::FourStepSixGrayCodeMaster_GPU(const dim3 block_) : block(block_){

    }

    void FourStepSixGrayCodeMaster_GPU::getTextureImg(std::vector<cv::cuda::GpuMat>& textureImg) {
        textureImg.clear();
        textureImg.resize(1, cv::cuda::GpuMat(averageImg_device.size(), averageImg_device.type()));
        textureImg[0] = averageImg_device;
    }
}
