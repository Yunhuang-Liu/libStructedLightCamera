/**
 * @file rectifier.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  极线校正器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef RECTIFIER_RECTIFIER_H_
#define RECTIFIER_RECTIFIER_H_

#include "../tool/matrixsInfo.h"

#ifdef CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda/std/functional>
#include <cuda/std/cmath>
#include <device_functions.h>
#endif

/** @brief 结构光库 */
namespace sl {
    /** @brief 极线校正库 */
    namespace rectifier {
        /** @brief 极线校正器 */
        class Rectifier {
        public:
            Rectifier(){};
            virtual ~Rectifier(){};
            /**
             * @brief           使用标定参数初始化极线校正器
             * 
             * @param info      输入，相机标定参数
             */
            virtual void initialize(const tool::Info &info) = 0;
            /**
             * @brief           对图片进行极线校正
             * 
             * @param img       输入，图片
             * @param isLeft    输入，是否进行左相机校正：true 左，false 右
             */
            virtual void remapImg(cv::Mat &imgInput, cv::Mat &imgOutput, const bool isLeft = true) = 0;
            #ifdef CUDA
            virtual void remapImg(cv::Mat &imgInput, cv::cuda::GpuMat &imgOutput,
                                  cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(), const bool isLeft = true) = 0;
            #endif
        private:
        };
    }// namespace rectifier
}// namespace sl
#endif // RECTIFIER_RECTIFIER_H_
