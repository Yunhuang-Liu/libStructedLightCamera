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

#include <tool/matrixsInfo.h>

/** @brief 结构光库 **/
namespace sl {
    /** @brief 工具库 **/
    namespace tool {
        #ifdef CUDA
        /** @brief cuda函数库 */
        namespace cudaFunc {
            /**
             * @brief          由相移图片计算纹理图片           
             * 
             * @param imgs     输入，相移图片     
             * @param texture  输出，纹理图片
             * @param block         输入，线程块
             * @param stream        输入，异步流
             */
            void averageTexture(std::vector<cv::cuda::GpuMat> &imgs, cv::cuda::GpuMat &texture,
                                const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
            /**
             * @brief               全图像相位高度映射（CUDA加速优化）
             *
             * @param phase         输入，相位图
             * @param intrinsic     输入，内参
             * @param coefficient   输入，八参数
             * @param minDepth      输入，最小深度
             * @param maxDepth      输入，最大深度
             * @param depth         输出，深度图
             * @param block         输入，线程块
             * @param stream        输入，异步流
             */
            void phaseHeightMapEigCoe(const cv::cuda::GpuMat &phase, const Eigen::Matrix3f &intrinsic, const Eigen::Vector<float, 8> &coefficient,
                                      const float minDepth, const float maxDepth,
                                      cv::cuda::GpuMat &depth, 
                                      const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
            /**
             * @brief                     反向映射同时细化深度（CUDA加速优化）
             * @warning                   默认第一个相机为细化深度相机
             *
             * @param phase               输入，相位图
             * @param depth               输入，深度图
             * @param firstWrap           输入，第一个辅助相机的包裹相位
             * @param firstCondition      输入，第一个辅助相机的调制图像
             * @param secondWrap          输入，第二个辅助相机的包裹相位
             * @param secondCondition     输入，第二个辅助相机的调制图像
             * @param intrinsicInvD       输入，深度相机的内参逆矩阵
             * @param intrinsicF          输入，第一个辅助相机的内参
             * @param intrinsicS          输入，第二个辅助相机的内参
             * @param RDtoFirst           输入，深度相机到第一个辅助相机的旋转矩阵
             * @param TDtoFirst           输入，深度相机到第一个辅助相机的平移矩阵
             * @param RDtoSecond          输入，深度相机到第二个辅助相机的旋转矩阵
             * @param TDtoSecond          输入，深度相机到第二个辅助相机的平移矩阵
             * @param PL                  输入，深度相机的投影矩阵（世界坐标系重合）
             * @param PR                  输入，第一个辅助相机的投影矩阵（以深度相机坐标系为世界坐标系）
             * @param threshod            输入，去除背景所用阈值（应当为2个像素相位差）
             * @param epiline             输入，深度相机上的点在第一个辅助相机的极线
             * @param depthRefine         输出，细化的深度图
             * @param block               输入，线程块
             * @param stream              输入，异步流
             */
            void reverseMappingRefine(const cv::cuda::GpuMat &phase, const cv::cuda::GpuMat &depth,
                                      const cv::cuda::GpuMat &firstWrap, const cv::cuda::GpuMat &firstCondition,
                                      const cv::cuda::GpuMat &secondWrap, const cv::cuda::GpuMat &secondCondition,
                                      const Eigen::Matrix3f &intrinsicInvD, const Eigen::Matrix3f &intrinsicF, const Eigen::Matrix3f &intrinsicS,
                                      const Eigen::Matrix3f &RDtoFirst, const Eigen::Vector3f &TDtoFirst,
                                      const Eigen::Matrix3f &RDtoSecond, const Eigen::Vector3f &TDtoSecond,
                                      const Eigen::Matrix4f &PL, const Eigen::Matrix4f &PR, const float threshod, 
                                      const cv::cuda::GpuMat &epilineA, const cv::cuda::GpuMat &epilineB, const cv::cuda::GpuMat &epilineC,
                                      cv::cuda::GpuMat &depthRefine,
                                      const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
            /**
             * @brief                     反向投影映射纹理（CUDA加速优化）
             *
             * @param depth               输入，相位图
             * @param textureSrc          输入，深度图
             * @param intrinsicInvD       输入，第一个辅助相机的包裹相位
             * @param intrinsicT          输入，第一个辅助相机的调制图像
             * @param rotateDToT          输入，第二个辅助相机的包裹相位
             * @param translateDtoT       输入，第二个辅助相机的调制图像
             * @param textureMapped       输入，深度相机的内参逆矩阵
             * @param block               输入，线程块
             * @param stream              输入，异步流
             */
            void reverseMappingTexture(const cv::cuda::GpuMat &depth, const cv::cuda::GpuMat &textureSrc,
                                       const Eigen::Matrix3f &intrinsicInvD, const Eigen::Matrix3f &intrinsicT,
                                       const Eigen::Matrix3f &rotateDToT, const Eigen::Vector3f &translateDtoT,
                                       cv::cuda::GpuMat &textureMapped,
                                       const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
        }
        #endif
        /**
         * @brief 全图像相位高度映射
         *
         * @param phase         输入，相位图
         * @param intrinsic     输入，内参
         * @param coefficient   输入，八参数
         * @param minDepth      输入，最小深度
         * @param maxDepth      输入，最大深度
         * @param depth         输入，深度图
         * @param threads       输入，使用的线程数
         */
        void phaseHeightMapEigCoe(const cv::Mat& phase, const cv::Mat &intrinsic, const cv::Mat &coefficient,
                                  const float minDepth, const float maxDepth,
                                  cv::Mat& depth, const int threads = 16);
        /**
         * @brief 区域图像相位高度映射
         *
         * @param phase         输入，相位图
         * @param intrinsic     输入，内参
         * @param coefficient   输入，八参数
         * @param minDepth      输入，最小深度
         * @param maxDepth      输入，最大深度
         * @param rowBegin      输入，区域行起始位置
         * @param rowEnd        输入，区域行结束位置
         * @param depth         输入，深度图
         */
        void phaseHeightMapEigCoeRegion(const cv::Mat& phase, const cv::Mat &intrinsic, const cv::Mat &coefficient,
                                        const float minDepth, const float maxDepth,
                                        const int rowBegin, const int rowEnd, cv::Mat& depth);
    }// tool
}// namespace sl

#endif// TOOL_TOOL_H_

