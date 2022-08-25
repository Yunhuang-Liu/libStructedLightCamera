/**
 * @file WrapCreator_CPU.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  CPU加速包裹相位求解器
 * @version 0.1
 * @date 2022-08-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef WrapCreator_CPU_H
#define WrapCreator_CPU_H

#include "WrapCreator.h"
#include <immintrin.h>
#include <thread>

namespace WrapCreat{
    //CPU加速包裹相位求解器
    class WrapCreator_CPU : public WrapCreator{
        public:
            WrapCreator_CPU();
            ~WrapCreator_CPU();
            /**
             * @brief                   获取包裹相位
             * 
             * @param imgs              输入，相移图片
             * @param wrapImg           输出，包裹图片
             * @param conditionImg      输出，背景图片
             * @param parameter         输入，加速参数
             */
            void getWrapImg(const std::vector<cv::Mat>& imgs, cv::Mat& wrapImg, cv::Mat& conditionImg, const WrapParameter parameter = WrapParameter()) override;
        protected:
            #ifdef CUDA
                void getWrapImg(const std::vector<cv::Mat>& imgs, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& conditionImg, const WrapParameter parameter = WrapParameter()) override {};
                void getWrapImg(const std::vector<cv::Mat>& imgs, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& conditionImg, const cv::cuda::Stream& cvStream, const WrapParameter parameter = WrapParameter()) override {};
            #endif
        private:
            /**
             * @brief                   三步相移求取包裹相位
             * 
             * @param imgs              输入，图片
             * @param wrapImg           输出，包裹图片
             * @param conditionImg      输出，背景图片
             * @param region            输出，算法有效区域
             */
            void thread_ThreeStepWrap(const std::vector<cv::Mat>& imgs,cv::Mat& wrapImg, cv::Mat& conditionImg, const cv::Size region);
            /**
             * @brief                   四步相移求取包裹相位
             * 
             * @param imgs              输入，图片
             * @param wrapImg           输出，包裹图片
             * @param conditionImg      输出，背景图片
             * @param region            输出，算法有效区域
             */
            void thread_FourStepWrap(const std::vector<cv::Mat>& imgs,cv::Mat& wrapImg, cv::Mat& conditionImg, const cv::Size region);
    };
}

#endif