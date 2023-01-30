/**
 * @file restructor_GPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  GPU重建器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef RESTRUCTOR_RESTRUCTOR_GPU_H
#define RESTRUCTOR_RESTRUCTOR_GPU_H

#include <restructor/restructor.h>
#include <tool/matrixsInfo.h>

#include <limits>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/** @brief 结构光库 */
namespace sl {
    /** @brief 重建库 */
    namespace restructor {
        /** @brief GPU加速重建器 */
        class Restructor_GPU : public Restructor {
        public:
            /**
             * @brief 构造函数
             * 
             * @param calibrationInfo 输入，标定信息
             * @param minDisparity 输入，最小视差
             * @param maxDisparity 输入，最大视差
             * @param minDepth 输入，最小深度值
             * @param maxDepth 输入，最大深度值
             * @param block 输入，Block尺寸
             */
            Restructor_GPU(const tool::Info &calibrationInfo, const int minDisparity = -500,
                           const int maxDisparity = 500, const float minDepth = 170,
                           const float maxDepth = 220, const dim3 block = dim3(32, 8));
            /**
             * @brief 析构函数
             */
            ~Restructor_GPU();
            /**
             * @brief 重建
             * 
             * @param leftAbsImg 输入，左绝对相位
             * @param rightAbsImg 输入，右绝对相位
             * @param sysIndex 输入，图片索引
             * @param stream 输入，异步流
             * @param isMap  输入，是否映射到左相机
             * @param isColor 输入，是否映射彩色纹理
             */
            void restruction(const cv::cuda::GpuMat &leftAbsImg,
                             const cv::cuda::GpuMat &rightAbsImg,
                             const int sysIndex,
                             cv::cuda::Stream &stream,
                             const bool isMap = false,
                             const bool isColor = false) override;
            /**
             * @brief 获取深度纹理
             * 
             * @param index 输入，图片索引
             * @param depthImg 输入/输出，深度图
             * @param colorImg 输入/输出，纹理图
             */
            void download(const int index, cv::cuda::GpuMat &depthImg) override;

        protected:
            /**
             * @brief 映射深度纹理
             * 
             * @param leftImg 输入，左绝对相位
             * @param rightImg 输入，右绝对相位
             * @param depthImg 输入/输出，深度图
             * @param pStream 输入，异步流
             */
            void getDepthColorMap(const cv::cuda::GpuMat &leftImg,
                                  const cv::cuda::GpuMat &rightImg,
                                  cv::cuda::GpuMat &depthImg,
                                  cv::cuda::Stream &pStream);
            /**
             * @brief 映射深度纹理
             * 
             * @param leftImg 输入，左绝对相位
             * @param rightImg 输入，右绝对相位
             * @param depthImg 输入/输出，深度图
             * @param isMap     输入，是否映射到左相机
             * @param pStream 输入，异步流
             */
            void getDepthMap(const cv::cuda::GpuMat &leftImg,
                             const cv::cuda::GpuMat &rightImg,
                             cv::cuda::GpuMat &depthImg,
                             const bool isMap,
                             cv::cuda::Stream &pStream);

        private:
            /** \最小深度值 **/
            const float minDepth;
            /** \最大深度值 **/
            const float maxDepth;
            /** \block尺寸 **/
            const dim3 block;
            /** \标定信息 **/
            const tool::Info &calibrationInfo;
            //CPU端函数
            void restruction(const cv::Mat &leftAbsImg,
                             const cv::Mat &rightAbsImg,
                             cv::Mat &depthImgOut,
                             const bool isMap = false,
                             const bool isColor = false) override {}
            /** \深度图 **/
            std::vector<cv::cuda::GpuMat> depthImg_device;
            /** \深度映射矩阵 **/
            Eigen::Matrix4f Q;
            /** \深度映射矩阵 **/
            Eigen::Matrix3f R1_inv;
            /** \灰度相机到彩色相机旋转矩阵 **/
            Eigen::Matrix3f R;
            /** \灰度相机到彩色相机位移矩阵 **/
            Eigen::Vector3f T;
            /** \点云相机内参矩阵 **/
            Eigen::Matrix3f M1;
            /** \彩色相机内参矩阵 **/
            Eigen::Matrix3f M3;
            /** \最小视差值 **/
            int minDisparity;
            /** \最大视差值 **/
            int maxDisparity;
        };
    }// namespace restructor
}// namespace sl
#endif // RESTRUCTOR_RESTRUCTOR_GPU_H
