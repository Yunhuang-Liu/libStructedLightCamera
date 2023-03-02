#include "../../include/wrapCreator/wrapCreator_GPU.h"

namespace sl {
    namespace wrapCreator {
        WrapCreator_GPU::WrapCreator_GPU() {
        }

        WrapCreator_GPU::~WrapCreator_GPU() {
        }

        void WrapCreator_GPU::getWrapImg(
                const std::vector<cv::cuda::GpuMat> &imgs, cv::cuda::GpuMat &wrapImg,
                cv::cuda::GpuMat &conditionImg, const bool isCounter,
                cv::cuda::Stream &cvStream, const WrapParameter parameter) {
            const int rows = imgs[0].rows;
            const int cols = imgs[0].cols;

            wrapImg.create(rows, cols, CV_32FC1);
            conditionImg.create(rows, cols, CV_32FC1);

            wrapCreator::cudaFunc::getWrapImgSync(imgs, wrapImg,
                                                  conditionImg, isCounter, parameter.block, cvStream);
        }
    }// namespace wrapCreator
}// namespace sl
