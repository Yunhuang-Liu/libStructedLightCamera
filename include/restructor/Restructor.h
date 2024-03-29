/**
 * @file restructor.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  重建器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef RESTRUCTOR_RESTRUCTOR_H_
#define RESTRUCTOR_RESTRUCTOR_H_

#include <vector>
#include <string>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#ifdef CUDA
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#endif

/** @brief 结构光库 */
namespace sl {
    /** @brief 重建库 */
    namespace restructor {
        /** @brief 重建器控制参数 **/
        struct RestructParamater {
            RestructParamater() : minDisparity(-500), maxDisparity(500), minDepth(170),
                                  maxDepth(220), threads(16) {
                #ifdef CUDA
                block = dim3(32, 8);
                #endif
            }
            RestructParamater(const int minDisparity_, const int maxDisparity_,
                              const float minDepth_, const float maxDepth_) : minDisparity(minDisparity_), maxDisparity(maxDisparity_),
                                                                              minDepth(minDepth_), maxDepth(maxDepth_), threads(16) {
                #ifdef CUDA
                block = dim3(32, 8);
                #endif
            }
            RestructParamater(const int minDisparity_, const int maxDisparity_) : minDisparity(minDisparity_), maxDisparity(maxDisparity_),
                                                                                  minDepth(170), maxDepth(220), threads(16) {
                #ifdef CUDA
                block = dim3(32, 8);
                #endif
            }
            RestructParamater(const int minDisparity_, const int maxDisparity_,
                              const float minDepth_, const float maxDepth_, const int threads_) : minDisparity(minDisparity_), maxDisparity(maxDisparity_),
                                                                                                  minDepth(minDepth_), maxDepth(maxDepth_), threads(threads_) {
                #ifdef CUDA
                block = dim3(32, 8);
                #endif
            }
            //额外的深度控制参数方便于人的操作习惯，视差值不够直观
            /** \最小视差 **/
            int minDisparity;
            /** \最大视差 **/
            int maxDisparity;
            /** \最小深度 **/
            float minDepth;
            /** \最大深度 **/
            float maxDepth;
            /** \线程数 **/
            int threads;
            #ifdef CUDA
            dim3 block;
            #endif
        };
        #ifdef CUDA
        /** @brief cuda函数库 */
        namespace cudaFunc {
            //深度和纹理映射，CUDA主机端调用函数,使用专用彩色相机捕获纹理（进行纹理映射）
            void depthColorMap(const cv::cuda::GpuMat &leftImg_,
                               const cv::cuda::GpuMat &rightImg_, const int rows, const int cols,
                               const int minDisparity, const int maxDisparity, const float minDepth,
                               const float maxDepth, const Eigen::Matrix4f &Q,
                               const Eigen::Matrix3f &M3, const Eigen::Matrix3f &R,
                               const Eigen::Vector3f &T, const Eigen::Matrix3f &R1_inv,
                               cv::cuda::GpuMat &depthMap, 
                               const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
            //深度映射，CUDA主机端调用函数，不进行纹理映射
            void depthMap(const cv::cuda::GpuMat &leftImg_,
                          const cv::cuda::GpuMat &rightImg_, const int rows, const int cols,
                          const int minDisparity, const int maxDisparity, const float minDepth,
                          const float maxDepth, const Eigen::Matrix4f &Q,
                          const Eigen::Matrix3f &M1, const Eigen::Matrix3f &R1_inv,
                          cv::cuda::GpuMat &depthMap, 
                          const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
        }// namespace cudaFunc
        #endif
        /** @brief 重建器 */
        class Restructor {
        public:
            Restructor();
            virtual ~Restructor();
            /**
             * @brief 重建
             * 
             * @param leftAbsImg 输入，左绝对相位
             * @param rightAbsImg 输入，右绝对相位
             * @param depthImgOut 输入/输出，深度图
             */
            virtual void restruction(const cv::Mat &leftAbsImg,
                                     const cv::Mat &rightAbsImg,
                                     cv::Mat &depthImgOut,
                                     const bool isColor = false) = 0;
             #ifdef CUDA
            /**
             * @brief 获取深度图
             * 
             * @param index 输入，图片索引
             * @param depthImg 输入，深度图
             */
            virtual void download(const int index, cv::cuda::GpuMat &depthImg) = 0;
            /**
             * @brief 重建,仅获取深度图的状态下将不会进行逆极线校正
             * 
             * @param leftAbsImg 输入，左绝对相位
             * @param rightAbsImg 输入，右绝对相位
             * @param colorImg 输入，原始纹理
             * @param sysIndex 输入，图片索引
             * @param stream 输入，异步流
             */
            virtual void restruction(const cv::cuda::GpuMat &leftAbsImg,
                                     const cv::cuda::GpuMat &rightAbsImg,
                                     const int sysIndex,
                                     cv::cuda::Stream &stream,
                                     const bool isColor = false) = 0;
            #endif
        protected:
            /**
             * @brief 获取深度和纹理，便于梳理
             */
            void getDepthColorMap();
        };
    }// namespace restructor
}// namespace sl
#endif // RESTRUCTOR_RESTRUCTOR_H_
