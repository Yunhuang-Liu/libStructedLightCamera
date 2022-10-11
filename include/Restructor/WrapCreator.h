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

#ifndef RESTRUCTOR_WRAPERCREATOR_H
#define RESTRUCTOR_WRAPERCREATOR_H

#include <opencv2/opencv.hpp>
#ifdef CUDA
#include <cuda_runtime.h>
#endif

/** @brief 结构光库 */
namespace sl {
    /** @brief 包裹生成库 */
    namespace wrapCreator {
        #ifdef CUDA
        /** @brief cuda函数库 */
        namespace cudaFunc {
            void getWrapImgSync(const std::vector<cv::cuda::GpuMat> &imgs,
                                cv::cuda::GpuMat &wrapImg,
                                cv::cuda::GpuMat &conditionImg, const bool isCounter = false,
                                const cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
                                const dim3 block = dim3(32, 8));
        }
        #endif

        /** @brief 包裹求解器 */
        class WrapCreator {
        public:
            /** @brief 包裹求解器控制参数 */
            struct WrapParameter {
                WrapParameter() : threads(16) {
                    #ifdef CUDA
                    block = dim3(32, 8);
                    #endif
                }
                WrapParameter(const int threads_) : threads(threads_) {}
                //线程数
                int threads;  
                #ifdef CUDA
                WrapParameter(const dim3 block_) : block(block_) {}
                //设备端线程分配
                dim3 block;
                #endif
            };
            WrapCreator() {}
            virtual ~WrapCreator() {}
            /**
             * @brief 获取包裹相位
             * 
             * @param imgs            输入，图片
             * @param wrapImg         输出，包裹图片
             * @param conditionImg    输出，调制图片
             * @param isCounter       输入，连续帧下前两帧和后两帧颠倒：false:0-1-2-3  true:2-3-0-1
             * @param parameter       输入，算法加速控制参数
             */
            virtual void getWrapImg(const std::vector<cv::Mat> &imgs, cv::Mat &wrapImg,
                                    cv::Mat &conditionImg, const bool isCounter = false,
                                    const WrapParameter parameter = WrapParameter()) = 0;
            #ifdef CUDA
            /**
             * @brief 获取包裹相位
             * 
             * @param imgs            输入，图片
             * @param wrapImg         输出，包裹图片
             * @param conditionImg    输出，调制图片
             * @param isCounter       输入，连续帧下前两帧和后两帧颠倒：false:0-1-2-3  true:2-3-0-1
             * @param cvStream        输入，异步流
             * @param parameter       输入，算法加速控制参数
             */
            virtual void getWrapImg(const std::vector<cv::cuda::GpuMat> &imgs,
                                    cv::cuda::GpuMat &wrapImg,
                                    cv::cuda::GpuMat &conditionImg, const bool isCounter = false,
                                    const cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
                                    const WrapParameter parameter = WrapParameter()) = 0;
            #endif
        };
    }// namespace wrapCreator
}// namespace sl
#endif // RESTRUCTOR_WRAPERCREATOR_H