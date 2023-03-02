/**
 * @file restructor_CPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  CPU重建器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef RESTRUCTOR_RESTRUCTOR_CPU_H_
#define RESTRUCTOR_RESTRUCTOR_CPU_H_

#include "Restructor.h"
#include "../tool/matrixsInfo.h"

#include <immintrin.h>
#include <limits>

/** @brief 结构光库 */
namespace sl {
    /** @brief 重建库 */
    namespace restructor {
        /** @brief CPU加速重建器 */
        class Restructor_CPU : public Restructor {
        public:
            /**
             * @brief 构造函数
             * 
             * @param calibrationInfo 输入，标定信息
             * @param minDisparity 输入，最小视差
             * @param maxDisparity 输入，最大视差
             * @param minDepth 输入，最小深度值
             * @param maxDepth 输入，最大深度值
             * @param threads 输入，线程数
             */
            Restructor_CPU(const tool::Info &calibrationInfo, const int minDisparity = -500,
                           const int maxDisparity = 500, const float minDepth = 170,
                           const float maxDepth = 220, const int threads = 16);
            /**
             * @brief 析构函数
             */
            ~Restructor_CPU();
            /**
             * @brief 重建
             * 
             * @param leftAbsImg 输入，左绝对相位
             * @param rightAbsImg 输入，右绝对相位
             * @param depthImgOut 输入/输出，深度图
             */
            void restruction(const cv::Mat &leftAbsImg, const cv::Mat &rightAbsImg,
                             cv::Mat &depthImgOut, const bool isMap = false, const bool isColor = false) override;

        protected:
            /**
             * @brief 映射深度纹理
             * 
             * @param leftAbsImg 输入，左绝对相位
             * @param rightAbsImg 输入，右绝对相位
             * @param colorImg 输入，原始彩色图片
             * @param depthImgOut 输入/输出 深度图
             * @param colorImgOut 输入/输出 纹理图
             */
            void getDepthColorMap(const cv::Mat &leftAbsImg, const cv::Mat &rightAbsImg,
                                  cv::Mat &depthImgOut, const bool isMap = false, const bool isColor = false);

        private:
#ifdef CUDA
            //GPU端函数
            void download(const int index, cv::cuda::GpuMat &depthImg){};
            void restruction(const cv::cuda::GpuMat &leftAbsImg,
                             const cv::cuda::GpuMat &rightAbsImg,
                             const int sysIndex,
                             cv::cuda::Stream &stream,
                             const bool isMap = false,
                             const bool isColor = false) override {}
#endif
            /** \存储线程锁 **/
            std::mutex mutexMap;
            /** \标定信息 **/
            const tool::Info &calibrationInfo;
            /** \使用线程数 **/
            const int threads;
            /** \最小视差值 **/
            int minDisparity;
            /** \最大视差值 **/
            int maxDisparity;
            /** \最小深度 **/
            const float minDepth;
            /** \最大深度 **/
            const float maxDepth;
            /** \重投影直接获取点云(多线程入口函数) **/
            void thread_DepthColorMap(const cv::Mat &leftAbsImg,
                                      const cv::Mat &righAbstImg,
                                      cv::Mat &depthImgOut,
                                      const cv::Point2i region,
                                      const bool isMap = false,
                                      const bool isColor = false);
        };
    }// namespace restructor
}// namespace sl
#endif // RESTRUCTOR_RESTRUCTOR_CPU_H_
