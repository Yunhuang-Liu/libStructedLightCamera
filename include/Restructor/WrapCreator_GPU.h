/**
 * @file WrapCreator_GPU.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  GPU加速包裹相位求解器
 * @version 0.1
 * @date 2022-08-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef WrapCreator_GPU_H
#define WrapCreator_GPU_H

#include "WrapCreator.h"

namespace WrapCreat{
    //GPU加速包裹求解器
    class WrapCreator_GPU : public WrapCreator{
        public:
            WrapCreator_GPU();
            ~WrapCreator_GPU();
            /**
             * @brief                   求取包裹相位
             *  
             * @param imgs              输入，相移图片          
             * @param wrapImg           输出，包裹图片
             * @param conditionImg      输出，背景图片
             * @param parameter         输入，加速参数
             */
            void getWrapImg(const std::vector<cv::Mat>& imgs, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& conditionImg, const WrapParameter parameter = WrapParameter()) override;
            /**
             * @brief                   求取包裹相位
             * 
             * @param imgs              输入，相移图片
             * @param wrapImg           输入，包裹图片
             * @param conditionImg      输入，背景图片
             * @param cvStream          输入，非阻塞流
             * @param parameter         输入，加速参数
             */
            void getWrapImg(const std::vector<cv::Mat>& imgs, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& conditionImg, const cv::cuda::Stream& cvStream, const WrapParameter parameter = WrapParameter()) override;
        protected:
            void getWrapImg(const std::vector<cv::Mat>& imgs, cv::Mat& wrapImg, cv::Mat& conditionImg, const WrapParameter parameter = WrapParameter()) override {};
        private:
    };
}

#endif