/**
 * @file tool.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  工具库
 * @version 0.1
 * @date 2021-12-10
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef TOOL_TOOL_H_
#define TOOL_TOOL_H_

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <tool/matrixsInfo.h>

/** @brief 结构光库 **/
namespace sl {
    /** @brief 工具库 **/
    namespace tool {
        /**
         * @brief 全图像相位高度映射
         *
         * @param phase         输入，相位图
         * @param intrinsic     输入，内参
         * @param coefficient   输入，八参数
         * @param depth         输入，深度图
         * @param threads       输入，使用的线程数
         */
        void phaseHeightMapEigCoe(const cv::Mat& phase, const cv::Mat &intrinsic, const cv::Mat &coefficient,
                                  cv::Mat& depth, const int threads = 16);
        /**
         * @brief 区域图像相位高度映射
         *
         * @param phase         输入，相位图
         * @param intrinsic     输入，内参
         * @param coefficient   输入，八参数
         * @param rowBegin      输入，区域行起始位置
         * @param rowEnd        输入，区域行结束位置
         * @param depth         输入，深度图
         */
        void phaseHeightMapEigCoeRegion(const cv::Mat& phase, const cv::Mat &intrinsic, const cv::Mat &coefficient,
                                        const int rowBegin, const int rowEnd, cv::Mat& depth);
    }// tool
}// namespace sl

#endif// TOOL_TOOL_H_

