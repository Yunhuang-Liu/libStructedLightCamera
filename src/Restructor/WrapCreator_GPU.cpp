#include "Restructor/WrapCreator_GPU.h"

namespace WrapCreat{
WrapCreator_GPU::WrapCreator_GPU() {

}

WrapCreator_GPU::~WrapCreator_GPU() {

}

void WrapCreator_GPU::getWrapImg(
    const std::vector<cv::Mat>& imgs, cv::cuda::GpuMat& wrapImg, 
    cv::cuda::GpuMat& conditionImg, const WrapParameter parameter) {
    const int rows = imgs[0].rows;
    const int cols = imgs[0].cols; 

    //This maybe not safe,yet cv::Mat is a count pointer,so that it will works safe.
    std::vector<cv::cuda::GpuMat> imgs_device(imgs.size());
    for(int i=0;i<imgs.size();i++)
        imgs_device[i].upload(imgs[i]);

    wrapImg.create(rows,cols,CV_32FC1);
    conditionImg.create(rows,cols,CV_32FC1);
        
    WrapCreat::cudaFunc::getWrapImg(imgs_device, wrapImg, conditionImg, parameter.block);
}

void WrapCreator_GPU::getWrapImg(
    const std::vector<cv::Mat>& imgs, cv::cuda::GpuMat& wrapImg, 
    cv::cuda::GpuMat& conditionImg, 
    const cv::cuda::Stream& cvStream, const WrapParameter parameter){
    const int rows = imgs[0].rows;
    const int cols = imgs[0].cols; 

    //This maybe not safe,yet cv::Mat is a count pointer,so that it will works safe.
    std::vector<cv::cuda::GpuMat> imgs_device;
    for(int i=0;i<imgs.size();i++)
        imgs_device[i].upload(imgs[i]);

    wrapImg.create(rows,cols,CV_32FC1);
    conditionImg.create(rows,cols,CV_32FC1);
        
    WrapCreat::cudaFunc::getWrapImgSync(imgs_device, wrapImg, 
        conditionImg, cvStream, parameter.block);
}
}