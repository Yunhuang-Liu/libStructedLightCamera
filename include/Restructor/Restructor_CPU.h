/**
 * @file Restructor_CPU_GrayPhase.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  CPU重建器(SIMD:AVX(256bit),多线程)
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef Restructor_CPU_H
#define Restructor_CPU_H
#include "Restructor.h"
#include <immintrin.h>
#include "MatrixsInfo.h"
#include <limits>

namespace RestructorType {
    class Restructor_CPU : public Restructor{
        public:
        /**
             * @brief 构造函数
             * @param calibrationInfo 输入，标定信息
             * @param minDisparity 输入，最小视差
             * @param maxDisparity 输入，最大视差
             * @param minDepth 输入，最小深度值
             * @param maxDepth 输入，最大深度值
             * @param threads 输入，线程数
             */
            Restructor_CPU(const Info& calibrationInfo, const int minDisparity = -500, const int maxDisparity = 500, 
                           const float minDepth = 170, const float maxDepth = 220, const int threads = 16);
            /**
             * @brief 析构函数
             */
            ~Restructor_CPU();
            /**
             * @brief 重建
             * @param leftAbsImg 输入，左绝对相位
             * @param rightAbsImg 输入，右绝对相位
             * @param colorImg 输入，原始彩色纹理
             * @param depthImgOut 输入/输出，深度图
             * @param colorImgOut 输入/输出，纹理图
             */
            void restruction(const cv::Mat& leftAbsImg, const cv::Mat& rightAbsImg, cv::Mat& depthImgOut, const cv::Mat& colorImg = cv::Mat(1, 1, CV_8UC3), cv::Mat& colorImgOut = cv::Mat(1, 1, CV_8UC3)) override;
        protected:
            /**
             * @brief 映射深度纹理
             * @param leftAbsImg 输入，左绝对相位
             * @param rightAbsImg 输入，右绝对相位
             * @param colorImg 输入，原始彩色图片
             * @param depthImgOut 输入/输出 深度图
             * @param colorImgOut 输入/输出 纹理图
             */
            void getDepthColorMap(const cv::Mat& leftAbsImg, const cv::Mat& rightAbsImg, const cv::Mat& colorImg,
                                  cv::Mat& depthImgOut, cv::Mat& colorImgOut);
        private:
            #ifdef CUDA
            //GPU端函数
            void download(const int index, cv::cuda::GpuMat& depthImg, cv::cuda::GpuMat& colorImg = cv::cuda::GpuMat(1, 1, CV_8UC3)) {};
            void restruction(const cv::cuda::GpuMat& leftAbsImg, const cv::cuda::GpuMat& rightAbsImg,
                const int sysIndex, cv::cuda::Stream& stream, const cv::Mat& colorImg = cv::Mat(1, 1, CV_8UC3)) override {}
            #endif
            /** \标定信息 **/
            const Info& calibrationInfo;
            /** \使用线程数 **/
            const int threads;
            /** \最小视差值 **/
            const int minDisparity;
            /** \最小深度 **/
            const float minDepth;
            /** \最大深度 **/
            const float maxDepth;
            /** \最大视差值 **/
            const int maxDisparity;
            /** \重投影直接获取点云(多线程入口函数) **/
            void thread_DepthColorMap(const cv::Mat& leftAbsImg, const cv::Mat& righAbstImg,const cv::Mat& colorImg,
                                       cv::Mat& depthImgOut, cv::Mat& colorImgOut, const cv::Point2i region);
    };
}
#endif // !Restructor_CPU_H
