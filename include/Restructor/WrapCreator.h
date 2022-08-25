/**
 * @file WrapCreator.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  包裹相位求解器
 * @version 0.1
 * @date 2022-08-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <opencv2/opencv.hpp>
#ifdef CUDA
    #include <cuda_runtime.h>
#endif

namespace WrapCreat{
    namespace cudaFunc{
        void getWrapImgSync(const std::vector<cv::cuda::GpuMat>& imgs, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& conditionImg,const cv::cuda::Stream& cvStream, const dim3 block);
        void getWrapImg(const std::vector<cv::cuda::GpuMat>& imgs, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& conditionImg, const dim3 block);
    }

    //包裹求解器参数
    struct WrapParameter
    {
       WrapParameter() : threads(16){}
       WrapParameter(const int threads_) : threads(threads_){} 
       int threads;
        #ifdef CUDA
            WrapParameter(const dim3 block_) : block(block_){}
            dim3 block;
        #endif
    };
    
    //包裹求解器
    class WraperCreator{
        public:
            WraperCreator(){}
            virtual ~WraperCreator(){}
            virtual void getWrapImg(const std::vector<cv::Mat>& imgs, cv::Mat& wrapImg, cv::Mat& conditionImg, const WrapParameter parameter = WrapParameter()) = 0;
            #ifdef CUDA
                virtual void getWrapImg(const std::vector<cv::Mat>& imgs, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& conditionImg, const cv::cuda::Stream& cvStream, const WrapParameter parameter = WrapParameter()) = 0;
                virtual void getWrapImg(const std::vector<cv::Mat>& imgs, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& conditionImg, const WrapParameter parameter = WrapParameter()) = 0;
            #endif
        private:
    };
}
